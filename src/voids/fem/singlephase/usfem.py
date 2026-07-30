from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Literal, cast
import warnings

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, splu

from voids.fem.singlephase._common import (
    BrinkmanNondimensionalization,
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    _FEMContext,
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
    _thread_environment_metadata,
    _validate_pressure_drop,
    _velocity_side_wall_bcs,
)
from voids.fem.singlephase._typing import (
    FEMFunction as _FEMFunction,
    PETScCSRConvertible as _PETScCSRConvertible,
    UFLAlgebra as _UFLAlgebra,
    UFLExpression as _UFLExpression,
    USFEMBlockCSRSystem as _USFEMBlockCSRSystem,
)
from voids.linalg.cudss import NvmathCudssFactor


_ControlScalar = str | bytes | int | float | bool
USFEMFacetLaw = Literal["classic", "reaction_diffusion", "shifted"]
USFEMFacetSizeMode = Literal["cell_diameter", "facet_measure"]


def _ufl_constant(context: _FEMContext, value: float) -> _UFLExpression:
    return cast(_UFLExpression, context.api.fem.Constant(context.mesh, float(value)))


def _petsc_mat_to_csr(matrix: _PETScCSRConvertible) -> csr_matrix:
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
    counts = np.fromiter(
        (np.count_nonzero(keep[indptr[row] : indptr[row + 1]]) for row in range(csr.shape[0])),
        dtype=np.int64,
        count=csr.shape[0],
    )
    new_indptr = np.empty_like(indptr)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])
    dropped = csr_matrix((data[keep], indices[keep], new_indptr), shape=csr.shape)
    dropped.eliminate_zeros()
    return dropped


def _assemble_usfem_block_csr_system(
    context: _FEMContext,
    *,
    forms: Sequence[Sequence[object]],
    rhs: Sequence[object],
    bcs: Sequence[object],
) -> _USFEMBlockCSRSystem:
    if context.mesh.comm.size != 1:
        raise NotImplementedError(
            "The USFEM Schurdiag/cuDSS preset is a single-process iterative "
            "solver. Use PETSc direct_parallel backends for MPI-distributed "
            "reference solves."
        )
    api = _require_dolfinx_petsc(context.api)
    fem = api.fem
    petsc = api.petsc
    if petsc is None:  # pragma: no cover - guarded by _require_dolfinx_petsc
        raise RuntimeError("PETSc bindings are required to assemble USFEM block matrices")
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
    return _USFEMBlockCSRSystem(
        a00=_petsc_mat_to_csr(matrix_nest.getNestSubMatrix(0, 0)),
        a01=_petsc_mat_to_csr(matrix_nest.getNestSubMatrix(0, 1)),
        a10=_petsc_mat_to_csr(matrix_nest.getNestSubMatrix(1, 0)),
        a11=_petsc_mat_to_csr(matrix_nest.getNestSubMatrix(1, 1)),
        rhs=np.asarray(vector.array.copy(), dtype=float),
    )


def _usfem_float_control(
    controls: Mapping[str, object],
    name: str,
    default: _ControlScalar,
) -> float:
    value = controls.get(name, default)
    if not isinstance(value, str | bytes | int | float | bool):
        raise TypeError(f"{name} must be convertible to float")
    return float(value)


def _usfem_int_control(
    controls: Mapping[str, object],
    name: str,
    default: _ControlScalar,
) -> int:
    value = controls.get(name, default)
    if not isinstance(value, str | bytes | int | float | bool):
        raise TypeError(f"{name} must be convertible to int")
    return int(value)


def _solve_usfem_schurdiag_cudss(
    context: _FEMContext,
    *,
    forms: Sequence[Sequence[object]],
    rhs: Sequence[object],
    bcs: Sequence[object],
    solution_functions: Sequence[_FEMFunction],
    options: FEniCSSolverOptions,
) -> tuple[list[_FEMFunction], float, dict[str, object]]:
    controls = dict(options.iterative_solver_controls)
    gmres_rtol = _usfem_float_control(controls, "gmres_rtol", 1.0e-8)
    gmres_atol = _usfem_float_control(controls, "gmres_atol", 0.0)
    gmres_maxiter = _usfem_int_control(controls, "gmres_maxiter", 1000)
    gmres_restart = _usfem_int_control(controls, "gmres_restart", 200)
    velocity_solver = str(controls.get("velocity_solver", "amg"))
    schur_drop_rel = _usfem_float_control(controls, "schurdiag_drop_rel", 0.0)
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

    a00 = blocks.a00.tocsc()
    a01 = blocks.a01.tocsr()
    a10 = blocks.a10.tocsr()
    a11 = blocks.a11.tocsr()
    rhs_array = np.asarray(blocks.rhs, dtype=float)
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
        pressure_factor_metadata: dict[str, object] = dict(pressure_factor.metadata())
        pressure_factor.close()
    linear_solve_seconds = perf_counter() - linear_solve_start
    solve_seconds = perf_counter() - start
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


def _paper_tau(
    context: _FEMContext,
    h: _UFLExpression,
    gamma: _UFLExpression,
    nu_eff: _UFLExpression,
    *,
    m_t: float,
) -> _UFLExpression:
    ufl = cast(_UFLAlgebra, context.api.ufl)
    one = _ufl_constant(context, 1.0)
    zero = _ufl_constant(context, 0.0)
    four = _ufl_constant(context, 4.0)
    m_t_value = _ufl_constant(context, float(m_t))
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
    context: _FEMContext,
    h_f: _UFLExpression,
    gamma: _UFLExpression,
    nu_eff: _UFLExpression,
    *,
    alpha_edge: float,
    facet_law: USFEMFacetLaw,
) -> _UFLExpression:
    ufl = cast(_UFLAlgebra, context.api.ufl)
    tiny = _ufl_constant(context, 1.0e-12)
    two = _ufl_constant(context, 2.0)
    twelve = _ufl_constant(context, 12.0)
    alpha = _ufl_constant(context, float(alpha_edge))
    nu_max = ufl.max_value(nu_eff("+"), nu_eff("-"))
    gamma_max = ufl.max_value(
        ufl.max_value(gamma("+"), gamma("-")),
        _ufl_constant(context, 0.0),
    )
    if facet_law == "classic":
        return alpha * h_f / (twelve * nu_max)
    if facet_law == "shifted":
        return alpha * h_f / (twelve * (nu_max + gamma_max * h_f * h_f))
    alpha_f = ufl.sqrt(gamma_max * h_f * h_f / nu_max)
    # Here alpha_f is fixed by the local reaction-diffusion problem; it is
    # unrelated to the free alpha_edge multiplier used by the other laws.
    return ufl.conditional(
        ufl.gt(alpha_f, tiny),
        h_f / (nu_max * alpha_f * alpha_f) * (1.0 - (two / alpha_f) * ufl.tanh(alpha_f / two)),
        h_f / (twelve * nu_max),
    )


def _cap_tau_gamma_product(
    context: _FEMContext,
    tau: _UFLExpression,
    gamma: _UFLExpression,
    *,
    tau_gamma_cap: float | None,
) -> _UFLExpression:
    if tau_gamma_cap is None:
        return tau
    ufl = cast(_UFLAlgebra, context.api.ufl)
    gamma_safe = ufl.max_value(gamma, _ufl_constant(context, 1.0e-300))
    return ufl.min_value(
        tau,
        _ufl_constant(context, float(tau_gamma_cap)) / gamma_safe,
    )


def _facet_size_expression(
    context: _FEMContext,
    mode: USFEMFacetSizeMode,
) -> _UFLExpression:
    ufl = cast(_UFLAlgebra, context.api.ufl)
    if mode == "cell_diameter":
        return ufl.avg(ufl.CellDiameter(context.mesh))
    facet_measure = ufl.FacetArea(context.mesh)
    if context.mesh.topology.dim == 2:
        return ufl.avg(facet_measure)
    return ufl.avg(ufl.sqrt(facet_measure))


def _p1dg0_max_uncapped_tau_gamma(
    problem: FEMMapProblem,
    *,
    tau_factor: float,
    m_t: float,
) -> float:
    permeability = np.maximum(
        np.asarray(problem.permeability_map.values, dtype=float),
        float(problem.permeability_floor),
    )
    if problem.porosity_map is None:
        porosity = np.ones_like(permeability)
    else:
        porosity = np.maximum(
            np.asarray(problem.porosity_map.values, dtype=float),
            float(problem.porosity_floor),
        )
    cell_diameter_squared = float(
        np.sum(np.square(np.asarray(problem.permeability_map.cell_size, dtype=float)))
    )
    gamma = float(problem.viscosity) / permeability
    nu_eff = float(problem.viscosity) / porosity
    pe_t = 4.0 * nu_eff / (gamma * cell_diameter_squared * float(m_t))
    denominator = gamma * cell_diameter_squared * np.maximum(1.0, pe_t) + 4.0 * nu_eff / float(m_t)
    return float(np.max(float(tau_factor) * gamma * cell_diameter_squared / denominator))


def _validate_usfem_controls(
    *,
    tau_factor: float,
    m_t: float,
    alpha_edge: float,
    facet_law: USFEMFacetLaw = "reaction_diffusion",
    facet_size_mode: USFEMFacetSizeMode = "cell_diameter",
    tau_gamma_cap: float | None = None,
) -> None:
    if tau_factor < 0.0 or not np.isfinite(tau_factor):
        raise ValueError("tau_factor must be nonnegative and finite")
    if m_t <= 0.0 or not np.isfinite(m_t):
        raise ValueError("m_t must be positive and finite")
    if alpha_edge <= 0.0 or not np.isfinite(alpha_edge):
        raise ValueError("alpha_edge must be positive and finite")
    if facet_law not in {"classic", "reaction_diffusion", "shifted"}:
        raise ValueError("facet_law must be one of 'classic', 'reaction_diffusion', or 'shifted'")
    if facet_law == "reaction_diffusion" and not np.isclose(alpha_edge, 1.0):
        warnings.warn(
            "alpha_edge is ignored by the parameter-free reaction_diffusion facet law",
            RuntimeWarning,
            stacklevel=2,
        )
    if facet_size_mode not in {"cell_diameter", "facet_measure"}:
        raise ValueError("facet_size_mode must be either 'cell_diameter' or 'facet_measure'")
    if tau_gamma_cap is not None and (
        tau_gamma_cap <= 0.0 or tau_gamma_cap >= 1.0 or not np.isfinite(tau_gamma_cap)
    ):
        raise ValueError("tau_gamma_cap must satisfy 0 < tau_gamma_cap < 1")


def _usfem_stabilization_terms(
    context: _FEMContext,
    *,
    tau_factor: float,
    m_t: float,
    alpha_edge: float,
    facet_law: USFEMFacetLaw = "reaction_diffusion",
    gamma: _UFLExpression | None = None,
    nu_eff: _UFLExpression | None = None,
    facet_size: _UFLExpression | None = None,
    tau_gamma_cap: float | None = None,
) -> tuple[_UFLExpression, _UFLExpression, _UFLExpression, _UFLExpression]:
    ufl = cast(_UFLAlgebra, context.api.ufl)
    gamma = cast(_UFLExpression, context.coefficients["gamma"]) if gamma is None else gamma
    nu_eff = cast(_UFLExpression, context.coefficients["nu_eff"]) if nu_eff is None else nu_eff
    h = ufl.CellDiameter(context.mesh)
    h_f = ufl.avg(h) if facet_size is None else facet_size
    tau = _cap_tau_gamma_product(
        context,
        _ufl_constant(context, float(tau_factor))
        * _paper_tau(
            context,
            h,
            gamma,
            nu_eff,
            m_t=m_t,
        ),
        gamma,
        tau_gamma_cap=tau_gamma_cap,
    )
    tau_f = _interior_pressure_tau(
        context,
        h_f,
        gamma,
        nu_eff,
        alpha_edge=alpha_edge,
        facet_law=facet_law,
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
    facet_law: USFEMFacetLaw = "reaction_diffusion",
    facet_size_mode: USFEMFacetSizeMode = "cell_diameter",
    tau_gamma_cap: float | None = None,
    pressure_degree: Literal[0, 1] = 1,
    options: FEniCSSolverOptions | None = None,
    nondimensional: bool | BrinkmanNondimensionalization = False,
) -> FEMSinglePhaseResult:
    """Solve a stabilized Darcy-Brinkman micro-continuum model.

    The formulation uses CG1 velocity and discontinuous pressure fields. DG1 is
    the backward-compatible default; pass ``pressure_degree=0`` for the
    low-order CG1 x DG0 pair used in manufactured-solution and vug studies. It
    augments the Brinkman weak form with a residual-based cell stabilization term and an
    interior pressure-jump penalty. The coefficients are intended for
    porosity/permeability maps obtained from a segmented image.

    Set ``tau_factor=0`` to disable the cell residual while retaining the
    pressure-jump term. In high-contrast CG1 x DG0 studies,
    ``tau_gamma_cap`` can enforce ``gamma * tau_K <= tau_gamma_cap`` and avoid
    near-cancellation of the physical drag. ``facet_size_mode="facet_measure"``
    uses edge length in 2D and the square root of facet area in 3D; the latter
    is a measure-based length, not an exact triangular-facet diameter.
    """

    _validate_usfem_controls(
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
        facet_law=facet_law,
        facet_size_mode=facet_size_mode,
        tau_gamma_cap=tau_gamma_cap,
    )
    if pressure_degree not in {0, 1}:
        raise ValueError("pressure_degree must be either 0 or 1")
    uncapped_max_tau_gamma = (
        _p1dg0_max_uncapped_tau_gamma(
            problem,
            tau_factor=tau_factor,
            m_t=m_t,
        )
        if pressure_degree == 0
        else None
    )
    if (
        uncapped_max_tau_gamma is not None
        and tau_factor > 0.0
        and tau_gamma_cap is None
        and uncapped_max_tau_gamma >= 0.9
    ):
        warnings.warn(
            "The uncapped CG1 x DG0 cell term has estimated "
            f"max(gamma * tau_K)={uncapped_max_tau_gamma:.3g}; it can nearly "
            "cancel physical drag. Set tau_gamma_cap below 1 or use "
            "tau_factor=0 as an explicit sensitivity branch.",
            RuntimeWarning,
            stacklevel=2,
        )
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

    def form_builder(
        context: _FEMContext,
        u: _UFLExpression,
        p: _UFLExpression,
        v: _UFLExpression,
        q: _UFLExpression,
    ) -> _UFLExpression:
        ufl = cast(_UFLAlgebra, context.api.ufl)
        dx = cast(_UFLExpression, context.dx)
        dS = cast(_UFLExpression, context.dS)
        gamma_override: _UFLExpression | None = None
        nu_eff_override: _UFLExpression | None = None
        if scales is not None:
            gamma_raw, nu_eff_raw = _brinkman_nondimensional_coefficients(
                context,
                problem,
                scales,
            )
            gamma_override = cast(_UFLExpression, gamma_raw)
            nu_eff_override = cast(_UFLExpression, nu_eff_raw)
        gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
            context,
            tau_factor=tau_factor,
            m_t=m_t,
            alpha_edge=alpha_edge,
            facet_law=facet_law,
            gamma=gamma_override,
            nu_eff=nu_eff_override,
            facet_size=_facet_size_expression(context, facet_size_mode),
            tau_gamma_cap=tau_gamma_cap,
        )
        residual_u = gamma * u + ufl.grad(p) - nu_eff * ufl.div(ufl.grad(u))
        residual_vq = gamma * v - ufl.grad(q) - nu_eff * ufl.div(ufl.grad(v))
        return (
            nu_eff * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            + ufl.inner(gamma * u, v) * dx
            - p * ufl.div(v) * dx
            + q * ufl.div(u) * dx
            + tau_f * ufl.jump(p) * ufl.jump(q) * dS
            - tau * ufl.inner(residual_u, residual_vq) * dx
        )

    result = _solve_with_form_builder(
        problem,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        options=options,
        velocity_degree=1,
        pressure_family="DG",
        pressure_degree=pressure_degree,
        method=f"Darcy-Brinkman USFEM CG1 x DG{pressure_degree}",
        formulation=f"brinkman_usfem_p1dg{pressure_degree}",
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
            "alpha_edge_active": facet_law in {"classic", "shifted"},
            "facet_law": facet_law,
            "facet_size_mode": facet_size_mode,
            "tau_gamma_cap": (None if tau_gamma_cap is None else float(tau_gamma_cap)),
            "p1dg0_uncapped_max_tau_gamma": uncapped_max_tau_gamma,
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
    facet_law: USFEMFacetLaw = "reaction_diffusion",
    facet_size_mode: USFEMFacetSizeMode = "cell_diameter",
    tau_gamma_cap: float | None = None,
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
    _validate_usfem_controls(
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
        facet_law=facet_law,
        facet_size_mode=facet_size_mode,
        tau_gamma_cap=tau_gamma_cap,
    )
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

    ufl = cast(_UFLAlgebra, context.api.ufl)
    fem = context.api.fem
    dx = cast(_UFLExpression, context.dx)
    dS = cast(_UFLExpression, context.dS)
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
    gamma_override: _UFLExpression | None = None
    nu_eff_override: _UFLExpression | None = None
    if scales is not None:
        gamma_scaled, nu_eff_scaled = _brinkman_nondimensional_coefficients(
            context,
            problem,
            scales,
        )
        gamma_override = cast(_UFLExpression, gamma_scaled)
        nu_eff_override = cast(_UFLExpression, nu_eff_scaled)
    gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
        context,
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
        facet_law=facet_law,
        gamma=gamma_override,
        nu_eff=nu_eff_override,
        facet_size=_facet_size_expression(context, facet_size_mode),
        tau_gamma_cap=tau_gamma_cap,
    )

    def residual_velocity_part(w: _UFLExpression) -> _UFLExpression:
        return gamma * w - nu_eff * ufl.div(ufl.grad(w))

    a00 = (
        nu_eff * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        + ufl.inner(gamma * u, v) * dx
        - tau
        * ufl.inner(
            residual_velocity_part(u),
            residual_velocity_part(v),
        )
        * dx
    )
    a01 = -p * ufl.div(v) * dx - tau * ufl.inner(ufl.grad(p), residual_velocity_part(v)) * dx
    a10 = q * ufl.div(u) * dx + tau * ufl.inner(residual_velocity_part(u), ufl.grad(q)) * dx
    a11 = tau_f * ufl.jump(p) * ufl.jump(q) * dS + tau * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    rhs_velocity = _pressure_boundary_load(
        context,
        v,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet if scales is None else 1.0,
        pressure_outlet=pressure_outlet if scales is None else 0.0,
    )
    rhs_pressure = _ufl_constant(context, 0.0) * q * dx

    bcs = _velocity_side_wall_bcs(context, velocity_space, flow_axis=flow_axis)
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
            "pressure_constraint": "natural_traction",
            "returned_pressure_normalization": "zero_mean",
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
            "alpha_edge_active": facet_law in {"classic", "shifted"},
            "facet_law": facet_law,
            "facet_size_mode": facet_size_mode,
            "tau_gamma_cap": (None if tau_gamma_cap is None else float(tau_gamma_cap)),
        }
    )
    return result


__all__ = [
    "USFEMFacetLaw",
    "USFEMFacetSizeMode",
    "solve_brinkman_usfem",
    "solve_brinkman_usfem_block",
]
