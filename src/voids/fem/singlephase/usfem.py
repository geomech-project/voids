from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, splu

from voids.fem.singlephase._common import (
    BrinkmanNondimensionalization,
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    _build_context,
    _brinkman_nondimensional_coefficients,
    _brinkman_nondimensional_scales,
    _mpi_metadata,
    _nondimensional_metadata,
    _pressure_boundary_load,
    _require_dolfinx_core,
    _require_dolfinx_petsc,
    _resolve_linear_backend,
    _resolve_brinkman_nondimensionalization,
    _result_from_velocity_pressure,
    _solve_block_problem_petsc,
    _solve_with_form_builder,
    _standalone_pressure_gauge_bc,
    _thread_environment_metadata,
    _validate_pressure_drop,
    _velocity_side_wall_bcs,
)
from voids.linalg.cudss import NvmathCudssFactor


def _petsc_mat_to_csr(matrix: Any) -> csr_matrix:
    matrix.assemble()
    converted = matrix.convert("aij")
    indptr, indices, data = converted.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=converted.getSize()).copy()


def _drop_relative_by_row(matrix: csr_matrix, drop_rel: float) -> csr_matrix:
    if drop_rel <= 0.0:
        return matrix
    csr = matrix.tocsr()
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    keep = np.zeros(data.shape, dtype=bool)
    for row in range(csr.shape[0]):
        start = indptr[row]
        end = indptr[row + 1]
        if start == end:
            continue
        row_abs = np.abs(data[start:end])
        threshold = float(drop_rel) * float(row_abs.max())
        row_cols = indices[start:end]
        keep[start:end] = (row_abs >= threshold) | (row_cols == row)
    counts = np.add.reduceat(keep.astype(np.int64), indptr[:-1])
    new_indptr = np.empty_like(indptr)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])
    dropped = csr_matrix((data[keep], indices[keep], new_indptr), shape=csr.shape)
    dropped.eliminate_zeros()
    return dropped


def _assemble_usfem_block_csr_system(
    context: Any,
    *,
    forms: list[list[Any]],
    rhs: list[Any],
    bcs: list[Any],
) -> dict[str, Any]:
    if context.mesh.comm.size != 1:
        raise NotImplementedError(
            "The USFEM Schurdiag/cuDSS preset is a single-process iterative "
            "solver. Use PETSc direct_parallel backends for MPI-distributed "
            "reference solves."
        )
    api = _require_dolfinx_petsc(context.api)
    fem = api.fem
    petsc = cast(Any, api.petsc)
    block_forms = [[fem.form(form) for form in row] for row in forms]
    matrix_nest = petsc.assemble_matrix(
        block_forms,
        bcs=bcs,
        kind=petsc.PETSc.Mat.Type.NEST,
    )
    matrix_nest.assemble()

    vector = petsc.assemble_vector(
        [fem.form(rhs_part) for rhs_part in rhs],
        kind=petsc.PETSc.Vec.Type.MPI,
    )
    petsc.apply_lifting(vector, block_forms, [bcs, bcs])
    vector.ghostUpdate(
        addv=petsc.PETSc.InsertMode.ADD,
        mode=petsc.PETSc.ScatterMode.REVERSE,
    )
    petsc.set_bc(vector, bcs)
    return {
        "A00": _petsc_mat_to_csr(matrix_nest.getNestSubMatrix(0, 0)),
        "A01": _petsc_mat_to_csr(matrix_nest.getNestSubMatrix(0, 1)),
        "A10": _petsc_mat_to_csr(matrix_nest.getNestSubMatrix(1, 0)),
        "A11": _petsc_mat_to_csr(matrix_nest.getNestSubMatrix(1, 1)),
        "rhs": np.asarray(vector.array.copy(), dtype=float),
    }


def _usfem_iterative_control(
    controls: dict[str, Any],
    name: str,
    default: Any,
    cast_type: Any,
) -> Any:
    return cast_type(controls.get(name, default))


def _solve_usfem_schurdiag_cudss(
    context: Any,
    *,
    forms: list[list[Any]],
    rhs: list[Any],
    bcs: list[Any],
    solution_functions: list[Any],
    options: FEniCSSolverOptions,
) -> tuple[list[Any], float, dict[str, Any]]:
    controls = dict(options.iterative_solver_controls)
    gmres_rtol = _usfem_iterative_control(controls, "gmres_rtol", 1.0e-8, float)
    gmres_atol = _usfem_iterative_control(controls, "gmres_atol", 0.0, float)
    gmres_maxiter = _usfem_iterative_control(controls, "gmres_maxiter", 1000, int)
    gmres_restart = _usfem_iterative_control(controls, "gmres_restart", 200, int)
    velocity_solver = str(controls.get("velocity_solver", "amg"))
    schur_drop_rel = _usfem_iterative_control(controls, "schurdiag_drop_rel", 0.0, float)
    error_if_not_converged = bool(controls.get("error_if_not_converged", True))

    start = perf_counter()
    assembly_start = perf_counter()
    blocks = _assemble_usfem_block_csr_system(
        context,
        forms=forms,
        rhs=rhs,
        bcs=bcs,
    )
    assembly_seconds = perf_counter() - assembly_start

    a00 = blocks["A00"].tocsc()
    a01 = blocks["A01"].tocsr()
    a10 = blocks["A10"].tocsr()
    a11 = blocks["A11"].tocsr()
    rhs_array = np.asarray(blocks["rhs"], dtype=float)
    system = bmat([[a00, a01], [a10, a11]], format="csr")
    n_u = int(a00.shape[0])

    diagonal = np.asarray(a00.diagonal(), dtype=float)
    inverse_diagonal = np.zeros_like(diagonal)
    nonzero_diagonal = np.abs(diagonal) > 1.0e-300
    inverse_diagonal[nonzero_diagonal] = 1.0 / diagonal[nonzero_diagonal]
    schurdiag_start = perf_counter()
    schurdiag = (a11 - a10 @ diags(inverse_diagonal, format="csr") @ a01).tocsr()
    schurdiag_original_nnz = int(schurdiag.nnz)
    schurdiag = _drop_relative_by_row(schurdiag, schur_drop_rel).tocsr()
    schurdiag_seconds = perf_counter() - schurdiag_start

    pressure_factor_start = perf_counter()
    pressure_factor = NvmathCudssFactor(
        schurdiag,
        controls=options.nvmath_cudss_controls,
    )
    pressure_factor_seconds = perf_counter() - pressure_factor_start

    velocity_setup_start = perf_counter()
    if velocity_solver == "exact":
        velocity_lu = splu(a00)

        def solve_velocity(vector: np.ndarray) -> np.ndarray:
            return np.asarray(velocity_lu.solve(vector), dtype=float)

        velocity_levels = 0
        velocity_operator_complexity = 1.0
    elif velocity_solver == "amg":
        try:
            import pyamg  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dependency
            pressure_factor.close()
            raise ImportError(
                "The USFEM Schurdiag/cuDSS preset with velocity_solver='amg' "
                "requires pyamg. Install pyamg or use velocity_solver='exact'."
            ) from exc
        velocity_hierarchy = pyamg.smoothed_aggregation_solver(
            a00.tocsr(),
            symmetry="nonsymmetric",
        )
        velocity_preconditioner = velocity_hierarchy.aspreconditioner(cycle="V")

        def solve_velocity(vector: np.ndarray) -> np.ndarray:
            return np.asarray(velocity_preconditioner @ vector, dtype=float)

        velocity_levels = int(len(velocity_hierarchy.levels))
        velocity_operator_complexity = float(velocity_hierarchy.operator_complexity())
    else:
        pressure_factor.close()
        raise ValueError("velocity_solver must be either 'amg' or 'exact'")
    velocity_setup_seconds = perf_counter() - velocity_setup_start

    def apply_preconditioner(residual: np.ndarray) -> np.ndarray:
        r_u = residual[:n_u]
        r_p = residual[n_u:]
        z_u = solve_velocity(r_u)
        z_p = pressure_factor.solve(r_p - a10 @ z_u)
        return np.concatenate([z_u, z_p])

    residual_history: list[float] = []
    preconditioner = LinearOperator(system.shape, matvec=apply_preconditioner, dtype=float)
    linear_solve_start = perf_counter()
    try:
        solution_array, info = gmres(
            system,
            rhs_array,
            M=preconditioner,
            rtol=gmres_rtol,
            atol=gmres_atol,
            restart=gmres_restart,
            maxiter=gmres_maxiter,
            callback=residual_history.append,
            callback_type="pr_norm",
        )
    finally:
        pressure_factor_metadata = pressure_factor.metadata()
        pressure_factor.close()
    linear_solve_seconds = perf_counter() - linear_solve_start
    solve_seconds = perf_counter() - start
    residual = np.asarray(system @ solution_array - rhs_array, dtype=float)
    relative_residual = float(
        np.linalg.norm(residual) / max(float(np.linalg.norm(rhs_array)), 1.0e-300)
    )
    info = int(info)
    if info != 0 and error_if_not_converged:
        raise RuntimeError(
            "USFEM Schurdiag/cuDSS GMRES did not converge: "
            f"info={info}, relative_residual={relative_residual:.3e}, "
            f"iterations={len(residual_history)}"
        )

    velocity, pressure = solution_functions
    if n_u != velocity.x.array.size:
        raise RuntimeError(
            "USFEM Schurdiag/cuDSS velocity block size does not match the "
            f"velocity function size: {n_u} != {velocity.x.array.size}"
        )
    n_p = int(pressure.x.array.size)
    if solution_array.size != n_u + n_p:
        raise RuntimeError(
            "USFEM Schurdiag/cuDSS returned an incompatible solution vector size "
            f"{solution_array.size}; expected {n_u + n_p}."
        )
    velocity.x.array[:] = solution_array[:n_u].real
    pressure.x.array[:] = solution_array[n_u : n_u + n_p].real
    velocity.x.scatter_forward()
    pressure.x.scatter_forward()
    return (
        [velocity, pressure],
        solve_seconds,
        {
            "usfem_schurdiag_cudss_assembly_seconds": assembly_seconds,
            "usfem_schurdiag_cudss_linear_solve_seconds": linear_solve_seconds,
            "usfem_schurdiag_cudss_schurdiag_seconds": schurdiag_seconds,
            "usfem_schurdiag_cudss_pressure_factor_seconds": pressure_factor_seconds,
            "usfem_schurdiag_cudss_velocity_setup_seconds": velocity_setup_seconds,
            "usfem_schurdiag_cudss_velocity_solver": velocity_solver,
            "usfem_schurdiag_cudss_velocity_levels": velocity_levels,
            "usfem_schurdiag_cudss_velocity_operator_complexity": (velocity_operator_complexity),
            "usfem_schurdiag_cudss_gmres_info": info,
            "usfem_schurdiag_cudss_gmres_iterations": len(residual_history),
            "usfem_schurdiag_cudss_relative_residual": relative_residual,
            "usfem_schurdiag_cudss_preconditioned_residual_last": (
                residual_history[-1] if residual_history else np.nan
            ),
            "usfem_schurdiag_cudss_ndof": int(rhs_array.size),
            "usfem_schurdiag_cudss_nnz": int(a00.nnz + a01.nnz + a10.nnz + a11.nnz),
            "usfem_schurdiag_cudss_velocity_dofs": int(a00.shape[0]),
            "usfem_schurdiag_cudss_pressure_dofs": int(a11.shape[0]),
            "usfem_schurdiag_cudss_schurdiag_original_nnz": schurdiag_original_nnz,
            "usfem_schurdiag_cudss_schurdiag_nnz": int(schurdiag.nnz),
            "usfem_schurdiag_cudss_schurdiag_drop_rel": float(schur_drop_rel),
            "usfem_schurdiag_cudss_gmres_rtol": gmres_rtol,
            "usfem_schurdiag_cudss_gmres_atol": gmres_atol,
            "usfem_schurdiag_cudss_gmres_maxiter": gmres_maxiter,
            "usfem_schurdiag_cudss_gmres_restart": gmres_restart,
            **pressure_factor_metadata,
        },
    )


def _paper_tau(context: Any, h: Any, gamma: Any, nu_eff: Any, *, m_t: float) -> Any:
    ufl = context.api.ufl
    fem = context.api.fem
    one = fem.Constant(context.mesh, 1.0)
    zero = fem.Constant(context.mesh, 0.0)
    four = fem.Constant(context.mesh, 4.0)
    m_t_value = fem.Constant(context.mesh, float(m_t))
    gamma_positive = ufl.gt(gamma, zero)
    pe_t = ufl.conditional(
        gamma_positive,
        four * nu_eff / (gamma * h * h * m_t_value),
        zero,
    )
    denominator = ufl.conditional(
        gamma_positive,
        gamma * h * h * ufl.max_value(one, pe_t) + four * nu_eff / m_t_value,
        four * nu_eff / m_t_value,
    )
    return h * h / denominator


def _interior_pressure_tau(
    context: Any,
    h_f: Any,
    gamma: Any,
    nu_eff: Any,
    *,
    alpha_edge: float,
) -> Any:
    ufl = context.api.ufl
    fem = context.api.fem
    tiny = fem.Constant(context.mesh, 1.0e-12)
    two = fem.Constant(context.mesh, 2.0)
    twelve = fem.Constant(context.mesh, 12.0)
    alpha = fem.Constant(context.mesh, float(alpha_edge))
    nu_max = ufl.max_value(nu_eff("+"), nu_eff("-"))
    gamma_max = ufl.max_value(
        ufl.max_value(gamma("+"), gamma("-")),
        fem.Constant(context.mesh, 0.0),
    )
    alpha_f = ufl.sqrt(gamma_max * h_f * h_f / nu_max)
    return alpha * ufl.conditional(
        ufl.gt(alpha_f, tiny),
        h_f / (nu_max * alpha_f * alpha_f) * (1.0 - (two / alpha_f) * ufl.tanh(alpha_f / two)),
        h_f / (twelve * nu_max),
    )


def _validate_usfem_controls(*, tau_factor: float, m_t: float, alpha_edge: float) -> None:
    if tau_factor <= 0.0:
        raise ValueError("tau_factor must be positive")
    if m_t <= 0.0:
        raise ValueError("m_t must be positive")
    if alpha_edge <= 0.0:
        raise ValueError("alpha_edge must be positive")


def _usfem_stabilization_terms(
    context: Any,
    *,
    tau_factor: float,
    m_t: float,
    alpha_edge: float,
    gamma: Any | None = None,
    nu_eff: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    ufl = context.api.ufl
    fem = context.api.fem
    gamma = context.coefficients["gamma"] if gamma is None else gamma
    nu_eff = context.coefficients["nu_eff"] if nu_eff is None else nu_eff
    h = ufl.CellDiameter(context.mesh)
    h_f = ufl.avg(h)
    tau = fem.Constant(context.mesh, float(tau_factor)) * _paper_tau(
        context,
        h,
        gamma,
        nu_eff,
        m_t=m_t,
    )
    tau_f = _interior_pressure_tau(
        context,
        h_f,
        gamma,
        nu_eff,
        alpha_edge=alpha_edge,
    )
    return gamma, nu_eff, tau, tau_f


def solve_brinkman_usfem(
    problem: FEMMapProblem,
    *,
    flow_axis: str = "x",
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    tau_factor: float = 1.0,
    m_t: float = 1.0 / 3.0,
    alpha_edge: float = 1.0,
    options: FEniCSSolverOptions | None = None,
    nondimensional: bool | BrinkmanNondimensionalization = False,
) -> FEMSinglePhaseResult:
    """Solve a stabilized Darcy-Brinkman micro-continuum model.

    The formulation uses CG1 velocity and DG1 pressure fields. It augments the
    Brinkman weak form with a residual-based cell stabilization term and an
    interior pressure-jump penalty. The coefficients are intended for
    porosity/permeability maps obtained from a segmented image.
    """

    _validate_usfem_controls(tau_factor=tau_factor, m_t=m_t, alpha_edge=alpha_edge)
    nondimensional_options = _resolve_brinkman_nondimensionalization(nondimensional)
    scales = None
    if nondimensional_options is not None:
        _validate_pressure_drop(pressure_inlet, pressure_outlet)
        context_for_scales = _build_context(
            problem,
            flow_axis=flow_axis,
            api=_require_dolfinx_core(),
        )
        scales = _brinkman_nondimensional_scales(
            context_for_scales,
            problem,
            pressure_inlet=pressure_inlet,
            pressure_outlet=pressure_outlet,
            velocity_scale=nondimensional_options.velocity_scale,
        )

    def form_builder(context, u, p, v, q):
        ufl = context.api.ufl
        coefficient_kwargs: dict[str, Any] = {}
        if scales is not None:
            gamma, nu_eff = _brinkman_nondimensional_coefficients(context, problem, scales)
            coefficient_kwargs = {"gamma": gamma, "nu_eff": nu_eff}
        gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
            context,
            tau_factor=tau_factor,
            m_t=m_t,
            alpha_edge=alpha_edge,
            **coefficient_kwargs,
        )
        residual_u = gamma * u + ufl.grad(p) - nu_eff * ufl.div(ufl.grad(u))
        residual_vq = gamma * v - ufl.grad(q) - nu_eff * ufl.div(ufl.grad(v))
        return (
            nu_eff * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
            + ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
            + tau_f * ufl.jump(p) * ufl.jump(q) * context.dS
            - tau * ufl.inner(residual_u, residual_vq) * context.dx
        )

    result = _solve_with_form_builder(
        problem,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        options=options,
        velocity_degree=1,
        pressure_family="DG",
        method="Darcy-Brinkman USFEM CG1 x DG1",
        formulation="brinkman_usfem_p1dg1",
        prefix_suffix=f"brinkman_usfem_{flow_axis}",
        form_builder=form_builder,
        boundary_pressure_inlet=None if scales is None else 1.0,
        boundary_pressure_outlet=None if scales is None else 0.0,
        velocity_scale=1.0 if scales is None else scales.velocity_scale,
        pressure_scale=1.0 if scales is None else scales.pressure_scale,
        extra_metadata=_nondimensional_metadata(scales),
    )
    result.metadata.update(
        {
            "tau_factor": float(tau_factor),
            "m_t": float(m_t),
            "alpha_edge": float(alpha_edge),
        }
    )
    return result


def solve_brinkman_usfem_block(
    problem: FEMMapProblem,
    *,
    flow_axis: str = "x",
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    tau_factor: float = 1.0,
    m_t: float = 1.0 / 3.0,
    alpha_edge: float = 1.0,
    options: FEniCSSolverOptions | None = None,
    matrix_kind: Literal["mpi", "nest"] = "mpi",
    preconditioner: Literal["none", "diagonal"] = "none",
    nondimensional: bool | BrinkmanNondimensionalization = False,
) -> FEMSinglePhaseResult:
    """Solve USFEM through explicit velocity/pressure block forms.

    This experimental path expands the same CG1 velocity/DG1 pressure USFEM
    bilinear form into separate ``(u, p)`` blocks. ``matrix_kind="mpi"``
    assembles the block forms into an ordinary monolithic PETSc matrix and is
    useful for parity checks against :func:`solve_brinkman_usfem`.
    ``matrix_kind="nest"`` preserves the block structure for PETSc field-split
    experiments. Pass ``preconditioner="diagonal"`` only for iterative
    field-split experiments, not for monolithic direct solves. Results from the
    nested mode should be treated as experimental until compared against a
    direct reference on the same coefficient map.
    """

    if matrix_kind not in {"mpi", "nest"}:
        raise ValueError("matrix_kind must be either 'mpi' or 'nest'")
    if preconditioner not in {"none", "diagonal"}:
        raise ValueError("preconditioner must be either 'none' or 'diagonal'")
    _validate_pressure_drop(pressure_inlet, pressure_outlet)
    _validate_usfem_controls(tau_factor=tau_factor, m_t=m_t, alpha_edge=alpha_edge)
    nondimensional_options = _resolve_brinkman_nondimensionalization(nondimensional)

    solver_options = options or FEniCSSolverOptions()
    api = _require_dolfinx_core()
    selected_linear_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_linear_backend != "petsc":
        raise NotImplementedError(
            "solve_brinkman_usfem_block currently supports only the PETSc backend; "
            "use solve_brinkman_usfem for serial direct solves."
        )
    uses_schurdiag_cudss = solver_options.solver_preset == "iterative_schurdiag_cudss_experimental"
    if uses_schurdiag_cudss and preconditioner != "none":
        raise ValueError(
            "preconditioner must be 'none' for the "
            "iterative_schurdiag_cudss_experimental preset; the preset defines "
            "its own lower-Schur preconditioner."
        )
    api = _require_dolfinx_petsc(api)
    context = _build_context(problem, flow_axis=flow_axis, api=api)
    scales = None
    if nondimensional_options is not None:
        scales = _brinkman_nondimensional_scales(
            context,
            problem,
            pressure_inlet=pressure_inlet,
            pressure_outlet=pressure_outlet,
            velocity_scale=nondimensional_options.velocity_scale,
        )

    ufl = context.api.ufl
    fem = context.api.fem
    velocity_element = context.api.basix_ufl.element(
        "Lagrange",
        context.mesh.basix_cell(),
        1,
        shape=(context.mesh.geometry.dim,),
    )
    pressure_element = context.api.basix_ufl.element("DG", context.mesh.basix_cell(), 1)
    velocity_space = fem.functionspace(context.mesh, velocity_element)
    pressure_space = fem.functionspace(context.mesh, pressure_element)
    velocity = fem.Function(velocity_space)
    velocity.name = "u"
    pressure = fem.Function(pressure_space)
    pressure.name = "p"

    u = ufl.TrialFunction(velocity_space)
    p = ufl.TrialFunction(pressure_space)
    v = ufl.TestFunction(velocity_space)
    q = ufl.TestFunction(pressure_space)
    coefficient_kwargs: dict[str, Any] = {}
    if scales is not None:
        gamma_scaled, nu_eff_scaled = _brinkman_nondimensional_coefficients(
            context,
            problem,
            scales,
        )
        coefficient_kwargs = {"gamma": gamma_scaled, "nu_eff": nu_eff_scaled}
    gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
        context,
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
        **coefficient_kwargs,
    )

    def residual_velocity_part(w):
        return gamma * w - nu_eff * ufl.div(ufl.grad(w))

    a00 = (
        nu_eff * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
        + ufl.inner(gamma * u, v) * context.dx
        - tau
        * ufl.inner(
            residual_velocity_part(u),
            residual_velocity_part(v),
        )
        * context.dx
    )
    a01 = (
        -p * ufl.div(v) * context.dx
        - tau * ufl.inner(ufl.grad(p), residual_velocity_part(v)) * context.dx
    )
    a10 = (
        q * ufl.div(u) * context.dx
        + tau * ufl.inner(residual_velocity_part(u), ufl.grad(q)) * context.dx
    )
    a11 = (
        tau_f * ufl.jump(p) * ufl.jump(q) * context.dS
        + tau * ufl.inner(ufl.grad(p), ufl.grad(q)) * context.dx
    )
    rhs_velocity = _pressure_boundary_load(
        context,
        v,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet if scales is None else 1.0,
        pressure_outlet=pressure_outlet if scales is None else 0.0,
    )
    rhs_pressure = fem.Constant(context.mesh, 0.0) * q * context.dx

    bcs = _velocity_side_wall_bcs(context, velocity_space, flow_axis=flow_axis)
    bcs.append(_standalone_pressure_gauge_bc(context, pressure_space))
    preconditioner_forms = None
    if preconditioner == "diagonal":
        preconditioner_forms = [[a00, None], [None, a11]]

    if uses_schurdiag_cudss:
        solution, solve_seconds, solver_metadata = _solve_usfem_schurdiag_cudss(
            context,
            forms=[[a00, a01], [a10, a11]],
            rhs=[rhs_velocity, rhs_pressure],
            bcs=bcs,
            solution_functions=[velocity, pressure],
            options=solver_options,
        )
    else:
        solution, solve_seconds, solver_metadata = _solve_block_problem_petsc(
            context,
            forms=[[a00, a01], [a10, a11]],
            rhs=[rhs_velocity, rhs_pressure],
            bcs=bcs,
            solution_functions=[velocity, pressure],
            options=solver_options,
            prefix_suffix=f"brinkman_usfem_block_{flow_axis}",
            matrix_kind=matrix_kind,
            preconditioner_forms=preconditioner_forms,
        )
    if scales is not None:
        solution[0].x.array[:] *= scales.velocity_scale
        solution[0].x.scatter_forward()
        solution[1].x.array[:] *= scales.pressure_scale
        solution[1].x.scatter_forward()
    result = _result_from_velocity_pressure(
        context,
        solution[0],
        solution[1],
        method="Darcy-Brinkman USFEM block CG1 x DG1",
        formulation="brinkman_usfem_p1dg1_block",
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        viscosity=problem.viscosity,
        solve_seconds=solve_seconds,
        metadata={
            "linear_backend": selected_linear_backend,
            "solver_preset": solver_options.solver_preset,
            "velocity_degree": 1,
            "pressure_family": "DG",
            "porosity_floor": problem.porosity_floor,
            "permeability_floor": problem.permeability_floor,
            "petsc_options": dict(solver_options.petsc_options),
            "petsc_options_prefix": solver_options.petsc_options_prefix,
            "nvmath_cudss_controls": dict(solver_options.nvmath_cudss_controls),
            "iterative_solver_controls": dict(solver_options.iterative_solver_controls),
            "block_matrix_kind": matrix_kind,
            "block_preconditioner": preconditioner,
            "thread_environment": _thread_environment_metadata(),
            **_nondimensional_metadata(scales),
            **_mpi_metadata(context),
            **solver_metadata,
        },
    )
    result.metadata.update(
        {
            "tau_factor": float(tau_factor),
            "m_t": float(m_t),
            "alpha_edge": float(alpha_edge),
        }
    )
    return result


__all__ = ["solve_brinkman_usfem", "solve_brinkman_usfem_block"]
