from __future__ import annotations

from typing import Any

from voids.fem.singlephase._common import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    _solve_with_form_builder,
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

    if tau_factor <= 0.0:
        raise ValueError("tau_factor must be positive")
    if m_t <= 0.0:
        raise ValueError("m_t must be positive")
    if alpha_edge <= 0.0:
        raise ValueError("alpha_edge must be positive")

    def form_builder(context, u, p, v, q):
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


__all__ = ["solve_brinkman_usfem"]
