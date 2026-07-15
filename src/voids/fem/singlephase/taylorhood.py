from __future__ import annotations

from voids.fem.singlephase._common import (
    BrinkmanNondimensionalization,
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    _brinkman_nondimensional_coefficients,
    _brinkman_nondimensional_scales,
    _nondimensional_metadata,
    _resolve_brinkman_nondimensionalization,
    _validate_pressure_drop,
    _require_dolfinx_core,
    _build_context,
    _solve_with_form_builder,
)


def solve_darcy_taylor_hood(
    problem: FEMMapProblem,
    *,
    flow_axis: str = "x",
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    options: FEniCSSolverOptions | None = None,
) -> FEMSinglePhaseResult:
    """Solve the mixed Darcy-Darcy comparison model with Taylor-Hood elements.

    The weak form uses a CG2 velocity approximation and a CG1 pressure
    approximation:

    ``(mu / K u, v) - (p, div v) + (q, div u) = boundary pressure work``.

    Side boundaries impose only zero normal velocity on faces transverse to the
    flow axis. Inlet and outlet pressures are applied as natural traction terms.
    """

    def form_builder(context, u, p, v, q):
        ufl = context.api.ufl
        gamma = context.coefficients["gamma"]
        return (
            ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
        )

    return _solve_with_form_builder(
        problem,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        options=options,
        velocity_degree=2,
        pressure_family="Lagrange",
        method="Darcy-Darcy Taylor-Hood CG2 x CG1",
        formulation="darcy_taylor_hood_p2p1",
        prefix_suffix=f"darcy_taylor_hood_{flow_axis}",
        form_builder=form_builder,
    )


def solve_brinkman_taylor_hood(
    problem: FEMMapProblem,
    *,
    flow_axis: str = "x",
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    options: FEniCSSolverOptions | None = None,
    nondimensional: bool | BrinkmanNondimensionalization = False,
) -> FEMSinglePhaseResult:
    """Solve the Darcy-Brinkman micro-continuum model with Taylor-Hood elements.

    The weak form uses CG2 velocity and CG1 pressure:

    ``(mu / phi grad u, grad v) + (mu / K u, v)
    - (p, div v) + (q, div u) = boundary pressure work``.

    ``K`` and ``phi`` are piecewise-constant maps supplied through
    :class:`~voids.fem.singlephase.FEMMapProblem`.
    """

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
        if scales is None:
            gamma = context.coefficients["gamma"]
            nu_eff = context.coefficients["nu_eff"]
        else:
            gamma, nu_eff = _brinkman_nondimensional_coefficients(context, problem, scales)
        return (
            nu_eff * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
            + ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
        )

    return _solve_with_form_builder(
        problem,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        options=options,
        velocity_degree=2,
        pressure_family="Lagrange",
        method="Darcy-Brinkman Taylor-Hood CG2 x CG1",
        formulation="brinkman_taylor_hood_p2p1",
        prefix_suffix=f"brinkman_taylor_hood_{flow_axis}",
        form_builder=form_builder,
        boundary_pressure_inlet=None if scales is None else 1.0,
        boundary_pressure_outlet=None if scales is None else 0.0,
        velocity_scale=1.0 if scales is None else scales.velocity_scale,
        pressure_scale=1.0 if scales is None else scales.pressure_scale,
        extra_metadata=_nondimensional_metadata(scales),
    )


__all__ = [
    "solve_brinkman_taylor_hood",
    "solve_darcy_taylor_hood",
]
