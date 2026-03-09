from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
from scipy import sparse

from voids.core.network import Network
from voids.geom.hydraulic import (
    throat_conductance as _throat_conductance,
    throat_conductance_with_sensitivities as _throat_conductance_with_sensitivities,
)
from voids.graph.connectivity import connected_components, induced_subnetwork
from voids.linalg.assemble import assemble_pressure_system
from voids.linalg.bc import apply_dirichlet_rowcol
from voids.linalg.diagnostics import residual_norm
from voids.linalg.solve import solve_linear_system
from voids.physics.thermo import TabulatedWaterViscosityModel


@dataclass(slots=True)
class FluidSinglePhase:
    """Single-phase fluid properties used by the flow solver.

    Attributes
    ----------
    viscosity :
        Reference dynamic viscosity of the fluid. For constant-viscosity solves,
        this is the viscosity used everywhere. When ``viscosity_model`` is
        supplied, this scalar remains the reporting/reference viscosity used in
        permeability calculations unless left as ``None``.
    density :
        Optional fluid density. It is stored for bookkeeping but is not used by the
        incompressible Darcy-scale solver in ``v0.1``.
    viscosity_model :
        Optional pressure-temperature viscosity model used to evaluate local pore
        and throat viscosities during conductance assembly.
    """

    viscosity: float | None = None
    density: float | None = None
    viscosity_model: TabulatedWaterViscosityModel | None = None

    def __post_init__(self) -> None:
        if self.viscosity is None and self.viscosity_model is None:
            raise ValueError("Provide either a constant viscosity or a viscosity_model")

    @property
    def has_variable_viscosity(self) -> bool:
        """Return whether a pressure-dependent viscosity model is active."""

        return self.viscosity_model is not None

    def reference_viscosity(
        self,
        *,
        pressure: float | None = None,
        pin: float | None = None,
        pout: float | None = None,
    ) -> float:
        """Return the scalar reference viscosity used for reporting.

        Notes
        -----
        If ``self.viscosity`` is specified explicitly it always takes precedence.
        Otherwise, when ``viscosity_model`` is active, the midpoint viscosity
        between ``pin`` and ``pout`` is used.
        """

        if self.viscosity is not None:
            return float(self.viscosity)
        if self.viscosity_model is None:
            raise ValueError("No viscosity or viscosity_model is available")
        if pressure is not None:
            return float(
                self.viscosity_model.evaluate(
                    np.asarray([pressure], dtype=float),
                    pin=float(pressure),
                    pout=float(pressure),
                )[0]
            )
        if pin is None or pout is None:
            raise ValueError(
                "Need explicit pressure or both pin and pout when no constant reference viscosity "
                "is stored on the fluid"
            )
        return float(self.viscosity_model.reference_viscosity(pin=float(pin), pout=float(pout)))


@dataclass(slots=True)
class PressureBC:
    """Dirichlet pressure boundary conditions.

    Attributes
    ----------
    inlet_label, outlet_label :
        Names of pore labels identifying fixed-pressure pores.
    pin, pout :
        Pressure values imposed on inlet and outlet pores.

    Notes
    -----
    For the current incompressible single-phase formulation, the physically
    relevant quantity is the imposed pressure drop ``pin - pout``. Adding the
    same constant offset to both values is therefore only a gauge choice.
    """

    inlet_label: str
    outlet_label: str
    pin: float
    pout: float


@dataclass(slots=True)
class SinglePhaseOptions:
    """Numerical and constitutive options for the single-phase solver.

    Attributes
    ----------
    conductance_model :
        Name of the hydraulic conductance model passed to
        :func:`voids.geom.hydraulic.throat_conductance`.
    solver :
        Linear solver backend name.
    check_mass_balance :
        If ``True``, compute a normalized divergence residual on free pores.
    regularization :
        Optional diagonal shift added to the matrix before Dirichlet elimination.
    nonlinear_max_iterations :
        Maximum number of Picard iterations used when viscosity depends on
        pressure.
    nonlinear_pressure_tolerance :
        Relative infinity-norm pressure-change tolerance for the Picard loop.
    nonlinear_relaxation :
        Under-relaxation factor applied to the Picard pressure update.
    solver_parameters :
        Optional linear-solver configuration dictionary passed to the inner
        SciPy linear solves. For Krylov methods, this can include
        ``{"preconditioner": "pyamg"}``.
    nonlinear_solver :
        Nonlinear strategy used when viscosity depends on pressure. Supported
        values are ``"picard"`` and ``"newton"``.
    nonlinear_line_search_reduction :
        Backtracking reduction factor used by the damped Newton update.
    nonlinear_line_search_max_steps :
        Maximum number of backtracking steps attempted by the damped Newton
        update.
    """

    conductance_model: str = "generic_poiseuille"
    solver: str = "direct"
    check_mass_balance: bool = True
    regularization: float | None = None
    nonlinear_max_iterations: int = 25
    nonlinear_pressure_tolerance: float = 1.0e-10
    nonlinear_relaxation: float = 1.0
    solver_parameters: dict[str, Any] = field(default_factory=dict)
    nonlinear_solver: str = "picard"
    nonlinear_line_search_reduction: float = 0.5
    nonlinear_line_search_max_steps: int = 8


@dataclass(slots=True)
class SinglePhaseResult:
    """Results returned by :func:`solve`.

    Attributes
    ----------
    pore_pressure :
        Pressure solution at pores.
    throat_flux :
        Volumetric flux on each throat, positive when flowing from
        ``throat_conns[:, 0]`` to ``throat_conns[:, 1]``.
    throat_conductance :
        Throat conductance values used during assembly.
    total_flow_rate :
        Net inlet flow rate associated with the imposed pressure drop.
    permeability :
        Dictionary containing the apparent permeability for the simulated axis.
    residual_norm :
        Algebraic residual norm of the solved linear system.
    mass_balance_error :
        Normalized divergence residual on free pores.
    pore_viscosity :
        Final pore-wise dynamic viscosity values used by the conductance model.
    throat_viscosity :
        Final throat-wise dynamic viscosity values used by the conductance model.
    reference_viscosity :
        Scalar viscosity used when reporting apparent permeability.
    solver_info :
        Backend-specific diagnostic information.
    """

    pore_pressure: np.ndarray
    throat_flux: np.ndarray
    throat_conductance: np.ndarray
    total_flow_rate: float
    permeability: dict[str, float] | None
    residual_norm: float
    mass_balance_error: float
    pore_viscosity: np.ndarray | None = None
    throat_viscosity: np.ndarray | None = None
    reference_viscosity: float | None = None
    solver_info: dict[str, Any] = field(default_factory=dict)


def _make_dirichlet_vector(net: Network, bc: PressureBC) -> tuple[np.ndarray, np.ndarray]:
    """Construct Dirichlet values and mask from labeled pores.

    Parameters
    ----------
    net :
        Network carrying pore labels.
    bc :
        Pressure boundary-condition specification.

    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(values, mask)`` where ``values`` contains prescribed pressures and
        ``mask`` selects constrained pores.

    Raises
    ------
    KeyError
        If the requested labels are missing.
    ValueError
        If one label is empty or if inlet and outlet labels overlap.
    """

    if bc.inlet_label not in net.pore_labels:
        raise KeyError(f"Missing pore label '{bc.inlet_label}'")
    if bc.outlet_label not in net.pore_labels:
        raise KeyError(f"Missing pore label '{bc.outlet_label}'")
    inlet = np.asarray(net.pore_labels[bc.inlet_label], dtype=bool)
    outlet = np.asarray(net.pore_labels[bc.outlet_label], dtype=bool)
    if inlet.sum() == 0 or outlet.sum() == 0:
        raise ValueError("BC labels must contain at least one pore each")
    if np.any(inlet & outlet):
        raise ValueError("Inlet and outlet labels overlap")
    mask = inlet | outlet
    values = np.zeros(net.Np, dtype=float)
    values[inlet] = float(bc.pin)
    values[outlet] = float(bc.pout)
    return values, mask


def _inlet_total_flow(net: Network, q: np.ndarray, inlet_mask: np.ndarray) -> float:
    """Compute net volumetric flow entering through inlet pores.

    Parameters
    ----------
    net :
        Network topology.
    q :
        Throat flux array with sign convention ``q_t > 0`` for flow from pore ``i`` to pore ``j``.
    inlet_mask :
        Boolean pore mask identifying inlet pores.

    Returns
    -------
    float
        Net inlet flow rate.

    Notes
    -----
    The implementation sums:

    - ``+q_t`` for throats leaving an inlet pore toward a non-inlet pore
    - ``-q_t`` for throats entering an inlet pore from a non-inlet pore

    Internal inlet-inlet throats are ignored because their contributions cancel.
    """

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    total = 0.0
    total += float(q[inlet_mask[i] & ~inlet_mask[j]].sum())
    total += float((-q[~inlet_mask[i] & inlet_mask[j]]).sum())
    return total


def _mass_balance_error(net: Network, q: np.ndarray, fixed_mask: np.ndarray) -> float:
    """Compute a normalized mass-balance residual on unconstrained pores.

    Parameters
    ----------
    net :
        Network topology.
    q :
        Throat flux array.
    fixed_mask :
        Boolean mask selecting Dirichlet pores.

    Returns
    -------
    float
        Quantity
        ``||div(q)||_2 / max(||q||_2, 1)``
        evaluated only on free pores.
    """

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    div = np.zeros(net.Np, dtype=float)
    np.add.at(div, i, q)
    np.add.at(div, j, -q)
    free = ~fixed_mask
    denom = max(float(np.linalg.norm(q)), 1.0)
    return float(np.linalg.norm(div[free]) / denom)


def _active_bc_component_mask(net: Network, fixed_mask: np.ndarray) -> np.ndarray:
    """Select pores in components touched by at least one Dirichlet pore."""

    _, comp_labels = connected_components(net)
    active_ids = np.unique(comp_labels[np.asarray(fixed_mask, dtype=bool)])
    return np.isin(comp_labels, active_ids)


def _validate_options(options: SinglePhaseOptions) -> None:
    """Validate nonlinear solver controls."""

    if options.nonlinear_max_iterations < 1:
        raise ValueError("nonlinear_max_iterations must be at least 1")
    if options.nonlinear_pressure_tolerance <= 0.0:
        raise ValueError("nonlinear_pressure_tolerance must be positive")
    if not (0.0 < options.nonlinear_relaxation <= 1.0):
        raise ValueError("nonlinear_relaxation must lie in the interval (0, 1]")
    if options.nonlinear_solver not in {"picard", "newton"}:
        raise ValueError("nonlinear_solver must be either 'picard' or 'newton'")
    if not (0.0 < options.nonlinear_line_search_reduction < 1.0):
        raise ValueError("nonlinear_line_search_reduction must lie in the interval (0, 1)")
    if options.nonlinear_line_search_max_steps < 1:
        raise ValueError("nonlinear_line_search_max_steps must be at least 1")


def _solve_active_linear_system(
    active_net: Network,
    g_active: np.ndarray,
    *,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[np.ndarray, dict[str, Any], Any, np.ndarray]:
    """Assemble and solve the active pressure subsystem for a given conductance field."""

    A = assemble_pressure_system(active_net, g_active)
    b = np.zeros(active_net.Np, dtype=float)
    if options.regularization is not None:
        A = A.copy().tocsr()
        A.setdiag(A.diagonal() + float(options.regularization))
    A_bc, b_bc = apply_dirichlet_rowcol(A, b, values=active_values, mask=active_fixed_mask)
    p_active, solver_info = solve_linear_system(
        A_bc,
        b_bc,
        method=options.solver,
        solver_parameters=options.solver_parameters,
    )
    return p_active, solver_info, A_bc, b_bc


def _assemble_active_system(
    active_net: Network,
    g_active: np.ndarray,
    *,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[Any, np.ndarray]:
    """Assemble the Dirichlet-eliminated active pressure system."""

    A = assemble_pressure_system(active_net, g_active)
    b = np.zeros(active_net.Np, dtype=float)
    if options.regularization is not None:
        A = A.copy().tocsr()
        A.setdiag(A.diagonal() + float(options.regularization))
    return apply_dirichlet_rowcol(A, b, values=active_values, mask=active_fixed_mask)


def _assemble_variable_viscosity_system(
    active_net: Network,
    pore_pressure: np.ndarray,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]:
    """Assemble the current pressure-dependent system and associated viscosity fields."""

    pore_mu, throat_mu = _evaluate_viscosity_fields(
        active_net,
        pore_pressure,
        fluid=fluid,
        bc=bc,
    )
    g_active = _throat_conductance(
        active_net,
        viscosity=None,
        model=options.conductance_model,
        pore_viscosity=pore_mu,
        throat_viscosity=throat_mu,
    )
    A_bc, b_bc = _assemble_active_system(
        active_net,
        g_active,
        active_values=active_values,
        active_fixed_mask=active_fixed_mask,
        options=options,
    )
    return pore_mu, throat_mu, g_active, A_bc, b_bc


def _evaluate_viscosity_fields(
    active_net: Network,
    pore_pressure: np.ndarray,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pore-wise and throat-wise viscosities for the current pressure field."""

    if fluid.viscosity_model is None:
        mu = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
        return (
            np.full(active_net.Np, mu, dtype=float),
            np.full(active_net.Nt, mu, dtype=float),
        )

    conns = active_net.throat_conns
    throat_pressure = 0.5 * (pore_pressure[conns[:, 0]] + pore_pressure[conns[:, 1]])
    pore_viscosity = fluid.viscosity_model.evaluate(pore_pressure, pin=bc.pin, pout=bc.pout)
    throat_viscosity = fluid.viscosity_model.evaluate(
        throat_pressure,
        pin=bc.pin,
        pout=bc.pout,
    )
    return (
        np.asarray(pore_viscosity, dtype=float),
        np.asarray(throat_viscosity, dtype=float),
    )


def _evaluate_viscosity_fields_with_derivatives(
    active_net: Network,
    pore_pressure: np.ndarray,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return viscosity fields and their local pressure derivatives."""

    if fluid.viscosity_model is None:
        mu = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
        return (
            np.full(active_net.Np, mu, dtype=float),
            np.full(active_net.Nt, mu, dtype=float),
            np.zeros(active_net.Np, dtype=float),
            np.zeros(active_net.Nt, dtype=float),
        )

    conns = active_net.throat_conns
    throat_pressure = 0.5 * (pore_pressure[conns[:, 0]] + pore_pressure[conns[:, 1]])
    pore_viscosity, pore_dmu = fluid.viscosity_model.evaluate_with_derivative(
        pore_pressure,
        pin=bc.pin,
        pout=bc.pout,
    )
    throat_viscosity, throat_dmu = fluid.viscosity_model.evaluate_with_derivative(
        throat_pressure,
        pin=bc.pin,
        pout=bc.pout,
    )
    return (
        np.asarray(pore_viscosity, dtype=float),
        np.asarray(throat_viscosity, dtype=float),
        np.asarray(pore_dmu, dtype=float),
        np.asarray(throat_dmu, dtype=float),
    )


def _nonlinear_residual_and_jacobian(
    active_net: Network,
    pore_pressure: np.ndarray,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the nonlinear residual and exact local Jacobian."""

    pore_mu, throat_mu, pore_dmu, throat_dmu = _evaluate_viscosity_fields_with_derivatives(
        active_net,
        pore_pressure,
        fluid=fluid,
        bc=bc,
    )
    g_active, dg_dpi, dg_dpj = _throat_conductance_with_sensitivities(
        active_net,
        viscosity=None,
        model=options.conductance_model,
        pore_viscosity=pore_mu,
        throat_viscosity=throat_mu,
        pore_dviscosity_dpressure=pore_dmu,
        throat_dviscosity_dpressure=throat_dmu,
    )

    conns = active_net.throat_conns
    i_idx = conns[:, 0]
    j_idx = conns[:, 1]
    delta_p = pore_pressure[i_idx] - pore_pressure[j_idx]
    q = g_active * delta_p
    residual = np.zeros(active_net.Np, dtype=float)
    np.add.at(residual, i_idx, q)
    np.add.at(residual, j_idx, -q)

    dq_dpi = g_active + delta_p * dg_dpi
    dq_dpj = -g_active + delta_p * dg_dpj
    rows = np.concatenate([i_idx, i_idx, j_idx, j_idx])
    cols = np.concatenate([i_idx, j_idx, i_idx, j_idx])
    data = np.concatenate([dq_dpi, dq_dpj, -dq_dpi, -dq_dpj])
    jacobian = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(active_net.Np, active_net.Np),
    ).tocsr()

    if np.any(active_fixed_mask):
        fixed = np.flatnonzero(active_fixed_mask)
        residual[fixed] = pore_pressure[fixed] - active_values[fixed]
        jacobian = jacobian.tolil()
        jacobian[fixed, :] = 0.0
        jacobian[fixed, fixed] = 1.0
        jacobian = jacobian.tocsr()
    return residual, jacobian, g_active, pore_mu, throat_mu


def _solve_with_variable_viscosity(
    active_net: Network,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any, np.ndarray]:
    """Solve the active subsystem with pressure-dependent viscosity."""

    ref_mu = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
    g_ref = _throat_conductance(
        active_net,
        viscosity=ref_mu,
        model=options.conductance_model,
    )
    p_active, solver_info, A_bc, b_bc = _solve_active_linear_system(
        active_net,
        g_ref,
        active_values=active_values,
        active_fixed_mask=active_fixed_mask,
        options=options,
    )

    free = ~active_fixed_mask
    nonlinear_change = float("inf")
    iterations = 0
    for iterations in range(1, options.nonlinear_max_iterations + 1):
        pore_mu, throat_mu = _evaluate_viscosity_fields(
            active_net,
            p_active,
            fluid=fluid,
            bc=bc,
        )
        g_active = _throat_conductance(
            active_net,
            viscosity=None,
            model=options.conductance_model,
            pore_viscosity=pore_mu,
            throat_viscosity=throat_mu,
        )
        p_candidate, solver_info, A_bc, b_bc = _solve_active_linear_system(
            active_net,
            g_active,
            active_values=active_values,
            active_fixed_mask=active_fixed_mask,
            options=options,
        )
        p_updated = (
            float(options.nonlinear_relaxation) * p_candidate
            + (1.0 - float(options.nonlinear_relaxation)) * p_active
        )
        p_updated[active_fixed_mask] = active_values[active_fixed_mask]
        if np.any(free):
            scale = max(float(np.linalg.norm(p_updated[free], ord=np.inf)), 1.0)
            nonlinear_change = float(
                np.linalg.norm(p_updated[free] - p_active[free], ord=np.inf) / scale
            )
        else:
            nonlinear_change = 0.0
        p_active = p_updated
        if nonlinear_change <= options.nonlinear_pressure_tolerance:
            break
    else:
        warnings.warn(
            "Pressure-dependent viscosity Picard iteration reached the iteration limit; "
            "using the last iterate.",
            RuntimeWarning,
            stacklevel=2,
        )

    pore_mu, throat_mu = _evaluate_viscosity_fields(
        active_net,
        p_active,
        fluid=fluid,
        bc=bc,
    )
    g_active = _throat_conductance(
        active_net,
        viscosity=None,
        model=options.conductance_model,
        pore_viscosity=pore_mu,
        throat_viscosity=throat_mu,
    )
    p_final, solver_info, A_bc, b_bc = _solve_active_linear_system(
        active_net,
        g_active,
        active_values=active_values,
        active_fixed_mask=active_fixed_mask,
        options=options,
    )
    if np.any(free):
        scale = max(float(np.linalg.norm(p_final[free], ord=np.inf)), 1.0)
        nonlinear_change = float(np.linalg.norm(p_final[free] - p_active[free], ord=np.inf) / scale)
    else:
        nonlinear_change = 0.0
    solver_info = {
        **solver_info,
        "nonlinear_iterations": int(iterations),
        "nonlinear_pressure_change": nonlinear_change,
        "nonlinear_solver": "picard",
        "viscosity_backend": (
            fluid.viscosity_model.backend_name if fluid.viscosity_model is not None else "constant"
        ),
        "viscosity_temperature": (
            float(fluid.viscosity_model.temperature)
            if fluid.viscosity_model is not None
            else np.nan
        ),
    }
    return p_final, g_active, pore_mu, throat_mu, solver_info, A_bc, b_bc


def _solve_with_variable_viscosity_newton(
    active_net: Network,
    *,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    active_values: np.ndarray,
    active_fixed_mask: np.ndarray,
    options: SinglePhaseOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any, np.ndarray]:
    """Solve the active subsystem with damped Newton for pressure-dependent viscosity."""

    ref_mu = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
    g_ref = _throat_conductance(
        active_net,
        viscosity=ref_mu,
        model=options.conductance_model,
    )
    p_initial, _, _, _ = _solve_active_linear_system(
        active_net,
        g_ref,
        active_values=active_values,
        active_fixed_mask=active_fixed_mask,
        options=options,
    )
    residual0, J0, g0, pore_mu0, throat_mu0 = _nonlinear_residual_and_jacobian(
        active_net,
        p_initial,
        fluid=fluid,
        bc=bc,
        active_values=active_values,
        active_fixed_mask=active_fixed_mask,
        options=options,
    )

    state: dict[str, Any] = {
        "pore_mu": pore_mu0,
        "throat_mu": throat_mu0,
        "conductance": g0,
        "residual": residual0,
        "jacobian": J0,
    }

    def _refresh_state(pore_pressure: np.ndarray) -> dict[str, Any]:
        residual, jacobian, g_active, pore_mu, throat_mu = _nonlinear_residual_and_jacobian(
            active_net,
            pore_pressure,
            fluid=fluid,
            bc=bc,
            active_values=active_values,
            active_fixed_mask=active_fixed_mask,
            options=options,
        )
        state.update(
            {
                "pore_mu": pore_mu,
                "throat_mu": throat_mu,
                "conductance": g_active,
                "residual": residual,
                "jacobian": jacobian,
            }
        )
        return state

    p_active = p_initial.copy()
    free = ~active_fixed_mask
    residual = residual0
    jacobian = J0
    g_active = g0
    pore_mu = pore_mu0
    throat_mu = throat_mu0
    nonlinear_change = float("inf")
    residual_measure = float(np.linalg.norm(residual[free], ord=np.inf)) if np.any(free) else 0.0
    iterations = 0
    linear_solver_info: dict[str, Any] = {"method": options.solver, "info": 0}

    for iterations in range(1, options.nonlinear_max_iterations + 1):
        if np.any(free):
            scale = max(float(np.linalg.norm(p_active[free], ord=np.inf)), 1.0)
            nonlinear_change = float(residual_measure / scale)
        else:
            nonlinear_change = 0.0
        if nonlinear_change <= options.nonlinear_pressure_tolerance:
            break

        delta, linear_solver_info = solve_linear_system(
            jacobian,
            -residual,
            method=options.solver,
            solver_parameters=options.solver_parameters,
        )
        alpha = 1.0
        accepted = False
        trial_state = state
        trial_pressure = p_active
        trial_residual_measure = residual_measure
        for _ in range(options.nonlinear_line_search_max_steps):
            trial_pressure = p_active + alpha * delta
            trial_pressure[active_fixed_mask] = active_values[active_fixed_mask]
            trial_state = _refresh_state(trial_pressure)
            trial_residual = trial_state["residual"]
            trial_residual_measure = (
                float(np.linalg.norm(trial_residual[free], ord=np.inf)) if np.any(free) else 0.0
            )
            if trial_residual_measure < residual_measure:
                accepted = True
                break
            alpha *= float(options.nonlinear_line_search_reduction)
        if not accepted:
            alpha = float(options.nonlinear_line_search_reduction)
            trial_pressure = p_active + alpha * delta
            trial_pressure[active_fixed_mask] = active_values[active_fixed_mask]
            trial_state = _refresh_state(trial_pressure)
            trial_residual = trial_state["residual"]
            trial_residual_measure = (
                float(np.linalg.norm(trial_residual[free], ord=np.inf)) if np.any(free) else 0.0
            )

        p_active = trial_pressure
        state = trial_state
        residual = state["residual"]
        jacobian = state["jacobian"]
        g_active = state["conductance"]
        pore_mu = state["pore_mu"]
        throat_mu = state["throat_mu"]
        residual_measure = trial_residual_measure
    else:
        warnings.warn(
            "Pressure-dependent viscosity Newton iteration reached the iteration limit; "
            "using the last iterate.",
            RuntimeWarning,
            stacklevel=2,
        )

    if np.any(free):
        scale = max(float(np.linalg.norm(p_active[free], ord=np.inf)), 1.0)
        nonlinear_change = float(residual_measure / scale)
    else:
        nonlinear_change = 0.0
    solver_info = {
        **linear_solver_info,
        "nonlinear_solver": "newton",
        "nonlinear_iterations": int(iterations),
        "nonlinear_pressure_change": nonlinear_change,
        "viscosity_backend": (
            fluid.viscosity_model.backend_name if fluid.viscosity_model is not None else "constant"
        ),
        "viscosity_temperature": (
            float(fluid.viscosity_model.temperature)
            if fluid.viscosity_model is not None
            else np.nan
        ),
    }
    A_bc = jacobian
    b_bc = jacobian @ p_active - residual
    return p_active, g_active, pore_mu, throat_mu, solver_info, A_bc, b_bc


def solve(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseResult:
    """Solve steady incompressible single-phase flow on a pore network.

    Parameters
    ----------
    net :
        Network containing topology, geometry, and sample metadata.
    fluid :
        Fluid properties. Constant viscosity is supported directly; when
        ``fluid.viscosity_model`` is provided the solver performs a nonlinear
        outer iteration so conductances can depend on the evolving pressure
        field. Built-in nonlinear strategies are Picard iteration and a damped
        Newton method using the exact Jacobian of the tabulated viscosity model.
    bc :
        Pressure boundary conditions.
    axis :
        Macroscopic flow axis used when converting total flow to apparent permeability.
    options :
        Optional solver and conductance settings.

    Returns
    -------
    SinglePhaseResult
        Pressure, flux, conductance, and derived transport metrics.

    Raises
    ------
    ValueError
        If the imposed pressure drop is zero, if the viscosity inputs are
        invalid, or if a thermodynamic viscosity model is used with non-positive
        boundary pressures.

    Notes
    -----
    The solver assembles a graph-Laplacian system

    ``A p = b``

    with throat fluxes

    ``q_t = g_t * (p_i - p_j)``

    where ``g_t`` is the hydraulic conductance of throat ``t``. After solving for
    pore pressure, the apparent permeability is computed from Darcy's law:

    ``K = |Q| * mu * L / (A * |delta_p|)``

    where ``Q`` is total inlet flow rate, ``mu`` is a scalar reference viscosity,
    ``L`` is the sample length along ``axis``, and ``A`` is the corresponding
    cross-sectional area. If ``fluid.viscosity`` is provided explicitly that
    value is used as the reference viscosity. Otherwise, when
    ``fluid.viscosity_model`` is active, the midpoint viscosity between
    ``pin`` and ``pout`` is used.

    Thermodynamic viscosity backends interpret pressure as absolute pressure in
    Pa. In that case, unlike the constant-viscosity solver, adding a constant
    offset to both boundary pressures changes the local viscosity field.

    Connected components that do not touch any Dirichlet pore are excluded from the
    linear solve because they form floating pressure blocks. Returned pressures and
    fluxes on those excluded components are reported as ``nan``.
    """

    options = options or SinglePhaseOptions()
    _validate_options(options)

    values, fixed_mask = _make_dirichlet_vector(net, bc)
    active_pores = _active_bc_component_mask(net, fixed_mask)
    active_net, active_idx, active_throats = induced_subnetwork(net, active_pores)
    active_values = values[active_idx]
    active_fixed_mask = fixed_mask[active_idx]
    if fluid.viscosity_model is not None and min(float(bc.pin), float(bc.pout)) <= 0.0:
        raise ValueError(
            "Thermodynamic viscosity models require positive absolute boundary pressures in Pa"
        )

    reference_viscosity = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
    if reference_viscosity <= 0.0:
        raise ValueError("Fluid viscosity must be positive")
    if fluid.viscosity_model is None:
        g_active = _throat_conductance(
            active_net,
            viscosity=reference_viscosity,
            model=options.conductance_model,
        )
        p_active, solver_info, A_bc, b_bc = _solve_active_linear_system(
            active_net,
            g_active,
            active_values=active_values,
            active_fixed_mask=active_fixed_mask,
            options=options,
        )
        pore_mu_active = np.full(active_net.Np, reference_viscosity, dtype=float)
        throat_mu_active = np.full(active_net.Nt, reference_viscosity, dtype=float)
        solver_info = {
            **solver_info,
            "nonlinear_iterations": 0,
            "nonlinear_pressure_change": 0.0,
            "nonlinear_solver": "none",
            "viscosity_backend": "constant",
            "viscosity_temperature": np.nan,
        }
    else:
        nonlinear_solver = (
            _solve_with_variable_viscosity_newton
            if options.nonlinear_solver == "newton"
            else _solve_with_variable_viscosity
        )
        p_active, g_active, pore_mu_active, throat_mu_active, solver_info, A_bc, b_bc = (
            nonlinear_solver(
                active_net,
                fluid=fluid,
                bc=bc,
                active_values=active_values,
                active_fixed_mask=active_fixed_mask,
                options=options,
            )
        )

    p = np.full(net.Np, np.nan, dtype=float)
    p[active_idx] = p_active

    g = np.full(net.Nt, np.nan, dtype=float)
    g[active_throats] = g_active

    pore_mu = np.full(net.Np, np.nan, dtype=float)
    pore_mu[active_idx] = pore_mu_active

    throat_mu = np.full(net.Nt, np.nan, dtype=float)
    throat_mu[active_throats] = throat_mu_active

    q = np.full(net.Nt, np.nan, dtype=float)
    i_active = active_net.throat_conns[:, 0]
    j_active = active_net.throat_conns[:, 1]
    q_active = g_active * (p_active[i_active] - p_active[j_active])
    q[active_throats] = q_active

    inlet_mask = np.asarray(active_net.pore_labels[bc.inlet_label], dtype=bool)
    Q = _inlet_total_flow(active_net, q_active, inlet_mask)
    dP = float(bc.pin - bc.pout)
    if abs(dP) == 0.0:
        raise ValueError("Pressure drop pin-pout must be nonzero")
    L = net.sample.length_for_axis(axis)
    Axs = net.sample.area_for_axis(axis)
    K = abs(Q) * reference_viscosity * L / (Axs * abs(dP))

    res = residual_norm(A_bc, p_active, b_bc)
    mbe = (
        _mass_balance_error(active_net, q_active, active_fixed_mask)
        if options.check_mass_balance
        else float("nan")
    )

    return SinglePhaseResult(
        pore_pressure=p,
        throat_flux=q,
        throat_conductance=g,
        total_flow_rate=Q,
        permeability={axis: float(K)},
        residual_norm=res,
        mass_balance_error=mbe,
        pore_viscosity=pore_mu,
        throat_viscosity=throat_mu,
        reference_viscosity=reference_viscosity,
        solver_info=solver_info,
    )
