from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voids.core.network import Network
from voids.geom.hydraulic import (
    _conduit_lengths_available,
    _get_entity_area,
    _get_entity_shape_factor,
    _resolve_pore_throat_viscosities,
    _segment_conductance_valvatne_blunt,
)
from voids.io.openpnm import to_openpnm_dict, to_openpnm_network
from voids.io.porespy import from_porespy
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    SinglePhaseResult,
    solve,
)


@dataclass(slots=True)
class SinglePhaseCrosscheckSummary:
    """Summary of a solver-to-reference comparison.

    Attributes
    ----------
    reference :
        Name of the reference implementation or workflow.
    axis :
        Flow axis used in the comparison.
    permeability_abs_diff, permeability_rel_diff :
        Absolute and relative differences between apparent permeabilities.
    total_flow_abs_diff, total_flow_rel_diff :
        Absolute and relative differences between total flow rates.
    details :
        Auxiliary metadata useful for debugging and reporting.
    """

    reference: str
    axis: str
    permeability_abs_diff: float
    permeability_rel_diff: float
    total_flow_abs_diff: float
    total_flow_rel_diff: float
    details: dict[str, Any]


@dataclass(slots=True)
class ConduitConductanceAudit:
    """Per-throat single-phase conduit conductance breakdown.

    The arrays are defined on throat order and expose the exact pore1-core-pore2
    decomposition used by the `voids` Valvatne-Blunt conduit model. The three
    segment conductances match the Imperial `pnflow` `SPConductance(area, mu)`
    semantics, meaning lengths are accounted for separately in the equivalent
    pore-to-pore resistance sum.
    """

    model: str
    throat_index: np.ndarray
    pore1_index: np.ndarray
    pore2_index: np.ndarray
    pore1_is_boundary: np.ndarray
    pore2_is_boundary: np.ndarray
    pore1_shape_factor: np.ndarray
    throat_shape_factor: np.ndarray
    pore2_shape_factor: np.ndarray
    pore1_area: np.ndarray
    throat_area: np.ndarray
    pore2_area: np.ndarray
    pore1_radius: np.ndarray
    throat_radius: np.ndarray
    pore2_radius: np.ndarray
    pore1_length: np.ndarray
    throat_length: np.ndarray
    pore2_length: np.ndarray
    pore1_conductance: np.ndarray
    throat_conductance: np.ndarray
    pore2_conductance: np.ndarray
    equivalent_conductance: np.ndarray

    def to_columns(self) -> dict[str, np.ndarray | str]:
        """Return a tabulation-friendly column mapping."""

        return {
            "model": self.model,
            "throat_index": self.throat_index,
            "pore1_index": self.pore1_index,
            "pore2_index": self.pore2_index,
            "pore1_is_boundary": self.pore1_is_boundary,
            "pore2_is_boundary": self.pore2_is_boundary,
            "pore1_shape_factor": self.pore1_shape_factor,
            "throat_shape_factor": self.throat_shape_factor,
            "pore2_shape_factor": self.pore2_shape_factor,
            "pore1_area": self.pore1_area,
            "throat_area": self.throat_area,
            "pore2_area": self.pore2_area,
            "pore1_radius": self.pore1_radius,
            "throat_radius": self.throat_radius,
            "pore2_radius": self.pore2_radius,
            "pore1_length": self.pore1_length,
            "throat_length": self.throat_length,
            "pore2_length": self.pore2_length,
            "pore1_conductance": self.pore1_conductance,
            "throat_conductance": self.throat_conductance,
            "pore2_conductance": self.pore2_conductance,
            "equivalent_conductance": self.equivalent_conductance,
        }


def _rel_diff(a: float, b: float) -> float:
    """Compute a symmetric relative difference.

    Parameters
    ----------
    a, b :
        Values to compare.

    Returns
    -------
    float
        Relative difference ``abs(a - b) / max(abs(a), abs(b), 1e-30)``.
    """

    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _summary_from_values(
    *,
    reference: str,
    axis: str,
    k_voids: float,
    k_ref: float,
    q_voids: float,
    q_ref: float,
    details: dict[str, Any],
) -> SinglePhaseCrosscheckSummary:
    """Build a crosscheck summary from scalar transport metrics.

    Parameters
    ----------
    reference :
        Name of the reference implementation.
    axis :
        Flow axis.
    k_voids, k_ref :
        Apparent permeabilities from ``voids`` and the reference.
    q_voids, q_ref :
        Total flow rates from ``voids`` and the reference.
    details :
        Auxiliary metadata to attach to the summary.

    Returns
    -------
    SinglePhaseCrosscheckSummary
        Comparison summary.
    """

    return SinglePhaseCrosscheckSummary(
        reference=reference,
        axis=axis,
        permeability_abs_diff=abs(k_voids - k_ref),
        permeability_rel_diff=_rel_diff(k_voids, k_ref),
        total_flow_abs_diff=abs(q_voids - q_ref),
        total_flow_rel_diff=_rel_diff(q_voids, q_ref),
        details={"k_voids": k_voids, "k_ref": k_ref, "Q_voids": q_voids, "Q_ref": q_ref, **details},
    )


def audit_singlephase_conduit_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    model: str = "valvatne_blunt",
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
) -> ConduitConductanceAudit:
    """Return a per-throat conduit conductance breakdown for `voids`.

    Parameters
    ----------
    net :
        Network with conduit lengths and pore/throat geometry.
    viscosity :
        Scalar or array viscosity passed to the Valvatne-Blunt closure.
    model :
        Conductance model. Currently only the conduit-based Imperial-style
        variants ``"valvatne_blunt"`` and ``"valvatne_blunt_baseline"`` are
        supported.
    pore_viscosity, throat_viscosity :
        Optional separate viscosities for pore and throat segments.

    Returns
    -------
    ConduitConductanceAudit
        Per-throat geometric and conductance decomposition.
    """

    if model not in {"valvatne_blunt", "valvatne_blunt_baseline"}:
        raise ValueError(
            "audit_singlephase_conduit_conductance currently supports only "
            "'valvatne_blunt' and 'valvatne_blunt_baseline'"
        )
    if not _conduit_lengths_available(net):
        raise KeyError("Missing conduit lengths (pore1_length, core_length, pore2_length)")

    mu_p, mu_t = _resolve_pore_throat_viscosities(
        net,
        viscosity,
        pore_viscosity=pore_viscosity,
        throat_viscosity=throat_viscosity,
    )
    conns = np.asarray(net.throat_conns, dtype=np.int64)
    p1_idx = conns[:, 0]
    p2_idx = conns[:, 1]

    pore_area = _get_entity_area(net, "pore")
    throat_area = _get_entity_area(net, "throat")
    pore_shape = _get_entity_shape_factor(net, "pore", area=pore_area)
    throat_shape = _get_entity_shape_factor(net, "throat", area=throat_area)

    pore1_length = np.asarray(net.throat["pore1_length"], dtype=float)
    throat_length = np.asarray(net.throat["core_length"], dtype=float)
    pore2_length = np.asarray(net.throat["pore2_length"], dtype=float)

    unit_length = np.ones(net.Nt, dtype=float)
    cond1 = _segment_conductance_valvatne_blunt(
        pore_area[p1_idx],
        pore_shape[p1_idx],
        unit_length,
        mu_p[p1_idx],
    )
    condt = _segment_conductance_valvatne_blunt(
        throat_area,
        throat_shape,
        unit_length,
        mu_t,
    )
    cond2 = _segment_conductance_valvatne_blunt(
        pore_area[p2_idx],
        pore_shape[p2_idx],
        unit_length,
        mu_p[p2_idx],
    )
    resistance_terms = []
    for length_arr, cond_arr in (
        (pore1_length, cond1),
        (throat_length, condt),
        (pore2_length, cond2),
    ):
        r = np.zeros(net.Nt, dtype=float)
        positive = cond_arr > 0.0
        r[positive] = length_arr[positive] / cond_arr[positive]
        resistance_terms.append(r)
    total_resistance = resistance_terms[0] + resistance_terms[1] + resistance_terms[2]
    geq = np.zeros(net.Nt, dtype=float)
    positive_total = total_resistance > 0.0
    geq[positive_total] = 1.0 / total_resistance[positive_total]

    pore_boundary = np.asarray(
        net.pore_labels.get("boundary", np.zeros(net.Np, dtype=bool)),
        dtype=bool,
    )
    pore_radius = np.asarray(net.pore["radius_inscribed"], dtype=float)
    throat_radius = np.asarray(net.throat["radius_inscribed"], dtype=float)

    return ConduitConductanceAudit(
        model=model,
        throat_index=np.arange(net.Nt, dtype=np.int64),
        pore1_index=p1_idx.copy(),
        pore2_index=p2_idx.copy(),
        pore1_is_boundary=pore_boundary[p1_idx],
        pore2_is_boundary=pore_boundary[p2_idx],
        pore1_shape_factor=np.asarray(pore_shape[p1_idx], dtype=float),
        throat_shape_factor=np.asarray(throat_shape, dtype=float),
        pore2_shape_factor=np.asarray(pore_shape[p2_idx], dtype=float),
        pore1_area=np.asarray(pore_area[p1_idx], dtype=float),
        throat_area=np.asarray(throat_area, dtype=float),
        pore2_area=np.asarray(pore_area[p2_idx], dtype=float),
        pore1_radius=np.asarray(pore_radius[p1_idx], dtype=float),
        throat_radius=np.asarray(throat_radius, dtype=float),
        pore2_radius=np.asarray(pore_radius[p2_idx], dtype=float),
        pore1_length=pore1_length.copy(),
        throat_length=throat_length.copy(),
        pore2_length=pore2_length.copy(),
        pore1_conductance=cond1,
        throat_conductance=condt,
        pore2_conductance=cond2,
        equivalent_conductance=geq,
    )


def _summary_from_results(
    reference: str, axis: str, r0: SinglePhaseResult, r1: SinglePhaseResult
) -> SinglePhaseCrosscheckSummary:
    """Build a summary from two single-phase solver results.

    Parameters
    ----------
    reference :
        Name of the reference workflow.
    axis :
        Flow axis used for extracting permeability.
    r0, r1 :
        Solver results to compare.

    Returns
    -------
    SinglePhaseCrosscheckSummary
        Comparison summary.
    """

    k0 = float((r0.permeability or {}).get(axis, np.nan))
    k1 = float((r1.permeability or {}).get(axis, np.nan))
    q0 = float(r0.total_flow_rate)
    q1 = float(r1.total_flow_rate)
    return _summary_from_values(
        reference=reference, axis=axis, k_voids=k0, k_ref=k1, q_voids=q0, q_ref=q1, details={}
    )


def crosscheck_singlephase_roundtrip_openpnm_dict(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseCrosscheckSummary:
    """Cross-check ``voids`` after a dict roundtrip through OpenPNM-style keys.

    Parameters
    ----------
    net :
        Network to solve and round-trip.
    fluid :
        Fluid properties.
    bc :
        Pressure boundary conditions.
    axis :
        Flow axis used in the permeability calculation.
    options :
        Optional solver configuration.

    Returns
    -------
    SinglePhaseCrosscheckSummary
        Comparison between the original ``voids`` solve and the round-tripped solve.

    Notes
    -----
    This path does not require OpenPNM itself. It checks whether exporting to the
    flat OpenPNM/PoreSpy naming convention and importing back into ``voids`` changes
    any transport-relevant fields.
    """

    options = options or SinglePhaseOptions()
    r0 = solve(net, fluid=fluid, bc=bc, axis=axis, options=options)
    op_dict = to_openpnm_dict(net)
    net_rt = from_porespy(op_dict, sample=net.sample, provenance=net.provenance)
    r1 = solve(net_rt, fluid=fluid, bc=bc, axis=axis, options=options)
    return _summary_from_results("openpnm_dict_roundtrip", axis, r0, r1)


def _openpnm_phase_factory(op, pn):
    """Construct a compatible OpenPNM phase object.

    Parameters
    ----------
    op :
        Imported OpenPNM module.
    pn :
        OpenPNM network object.

    Returns
    -------
    Any
        Phase object compatible with the installed OpenPNM version.

    Raises
    ------
    RuntimeError
        If no known phase constructor works.
    """

    for factory in (
        lambda: op.phase.Phase(network=pn),
        lambda: op.phases.GenericPhase(network=pn),
    ):
        try:
            return factory()
        except Exception:
            continue
    raise RuntimeError("Unable to construct OpenPNM phase object")


def _get_openpnm_pressure(sf):
    """Extract pore pressure from an OpenPNM StokesFlow result.

    Parameters
    ----------
    sf :
        OpenPNM StokesFlow algorithm object.

    Returns
    -------
    numpy.ndarray
        One-dimensional pore-pressure array.

    Raises
    ------
    RuntimeError
        If pressure cannot be retrieved from the current OpenPNM API.
    """

    for getter in (
        lambda: sf["pore.pressure"],
        lambda: sf.soln["pore.pressure"],
    ):
        try:
            arr = np.asarray(getter(), dtype=float)
            if arr.ndim == 1:
                return arr
        except Exception:
            continue
    raise RuntimeError("Unable to extract pore pressures from OpenPNM StokesFlow result")


def crosscheck_singlephase_with_openpnm(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseCrosscheckSummary:
    """Cross-check ``voids`` against OpenPNM StokesFlow.

    Parameters
    ----------
    net :
        Network to simulate.
    fluid :
        Fluid properties.
    bc :
        Pressure boundary conditions.
    axis :
        Flow axis used for apparent permeability.
    options :
        Optional solver configuration.

    Returns
    -------
    SinglePhaseCrosscheckSummary
        Comparison between ``voids`` and OpenPNM.

    Raises
    ------
    ImportError
        If OpenPNM is not installed.
    RuntimeError
        If the installed OpenPNM API is incompatible with the adapter.
    ValueError
        If the imposed pressure drop is zero.

    Notes
    -----
    The comparison injects the ``voids``-computed ``throat.hydraulic_conductance``
    into OpenPNM. That means the crosscheck isolates differences in system assembly,
    boundary-condition handling, sign conventions, and linear-solver behavior,
    rather than differences in geometric conductance modeling.
    """

    try:
        import openpnm as op
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "OpenPNM is not installed. Use the 'test' pixi environment or install openpnm."
        ) from exc

    options = options or SinglePhaseOptions()
    r_voids = solve(net, fluid=fluid, bc=bc, axis=axis, options=options)
    g = np.asarray(r_voids.throat_conductance, dtype=float)

    pn = to_openpnm_network(net, copy_properties=False, copy_labels=True)
    phase = _openpnm_phase_factory(op, pn)
    phase["throat.hydraulic_conductance"] = g

    sf = op.algorithms.StokesFlow(network=pn, phase=phase)
    inlet_mask = np.asarray(net.pore_labels[bc.inlet_label], dtype=bool)
    outlet_mask = np.asarray(net.pore_labels[bc.outlet_label], dtype=bool)
    inlet = np.where(inlet_mask)[0]
    outlet = np.where(outlet_mask)[0]

    if hasattr(sf, "set_value_BC"):
        sf.set_value_BC(pores=inlet, values=float(bc.pin))
        sf.set_value_BC(pores=outlet, values=float(bc.pout))
    elif hasattr(sf, "set_BC"):
        sf.set_BC(pores=inlet, bctype="value", bcvalues=float(bc.pin))
        sf.set_BC(pores=outlet, bctype="value", bcvalues=float(bc.pout))
    else:  # pragma: no cover
        raise RuntimeError("OpenPNM StokesFlow object does not expose a recognizable BC API")

    sf.run()
    p_ref = _get_openpnm_pressure(sf)

    q_rate = np.asarray(sf.rate(pores=inlet), dtype=float)
    q_ref_raw = float(q_rate.sum())
    q_ref = q_ref_raw
    if np.isfinite(q_ref) and np.isfinite(r_voids.total_flow_rate):
        if np.isclose(abs(q_ref), abs(r_voids.total_flow_rate), rtol=1e-8, atol=1e-14):
            q_ref = float(np.copysign(abs(q_ref), r_voids.total_flow_rate))

    dP = float(bc.pin - bc.pout)
    if abs(dP) == 0.0:
        raise ValueError("Pressure drop pin-pout must be nonzero")
    L = net.sample.length_for_axis(axis)
    Axs = net.sample.area_for_axis(axis)
    mu_ref = fluid.reference_viscosity(pin=bc.pin, pout=bc.pout)
    k_ref = abs(q_ref_raw) * mu_ref * L / (Axs * abs(dP))
    k_voids = float((r_voids.permeability or {}).get(axis, np.nan))

    return _summary_from_values(
        reference="openpnm_stokesflow",
        axis=axis,
        k_voids=k_voids,
        k_ref=float(k_ref),
        q_voids=float(r_voids.total_flow_rate),
        q_ref=float(q_ref),
        details={
            "openpnm_version": getattr(op, "__version__", "unknown"),
            "q_ref_raw": q_ref_raw,
            "n_inlet_pores": int(inlet.size),
            "n_outlet_pores": int(outlet.size),
            "conductance_model": options.conductance_model,
            "solver_voids": options.solver,
            "p_ref_min": float(np.min(p_ref)) if p_ref.size else np.nan,
            "p_ref_max": float(np.max(p_ref)) if p_ref.size else np.nan,
        },
    )
