from __future__ import annotations

from typing import Any, Literal

from voids.fem.singlephase._common import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    _build_context,
    _mpi_metadata,
    _pressure_boundary_load,
    _require_dolfinx_core,
    _require_dolfinx_petsc,
    _resolve_linear_backend,
    _result_from_velocity_pressure,
    _solve_block_problem_petsc,
    _solve_with_form_builder,
    _standalone_pressure_gauge_bc,
    _thread_environment_metadata,
    _validate_pressure_drop,
    _velocity_side_wall_bcs,
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
) -> tuple[Any, Any, Any, Any]:
    ufl = context.api.ufl
    fem = context.api.fem
    gamma = context.coefficients["gamma"]
    nu_eff = context.coefficients["nu_eff"]
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
) -> FEMSinglePhaseResult:
    """Solve a stabilized Darcy-Brinkman micro-continuum model.

    The formulation uses CG1 velocity and DG1 pressure fields. It augments the
    Brinkman weak form with a residual-based cell stabilization term and an
    interior pressure-jump penalty. The coefficients are intended for
    porosity/permeability maps obtained from a segmented image.
    """

    _validate_usfem_controls(tau_factor=tau_factor, m_t=m_t, alpha_edge=alpha_edge)

    def form_builder(context, u, p, v, q):
        ufl = context.api.ufl
        gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
            context,
            tau_factor=tau_factor,
            m_t=m_t,
            alpha_edge=alpha_edge,
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

    solver_options = options or FEniCSSolverOptions()
    api = _require_dolfinx_core()
    selected_linear_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_linear_backend != "petsc":
        raise NotImplementedError(
            "solve_brinkman_usfem_block currently supports only the PETSc backend; "
            "use solve_brinkman_usfem for serial direct solves."
        )
    api = _require_dolfinx_petsc(api)
    context = _build_context(problem, flow_axis=flow_axis, api=api)

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
    gamma, nu_eff, tau, tau_f = _usfem_stabilization_terms(
        context,
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
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
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
    )
    rhs_pressure = fem.Constant(context.mesh, 0.0) * q * context.dx

    bcs = _velocity_side_wall_bcs(context, velocity_space, flow_axis=flow_axis)
    bcs.append(_standalone_pressure_gauge_bc(context, pressure_space))
    preconditioner_forms = None
    if preconditioner == "diagonal":
        preconditioner_forms = [[a00, None], [None, a11]]

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
            "block_matrix_kind": matrix_kind,
            "block_preconditioner": preconditioner,
            "thread_environment": _thread_environment_metadata(),
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
