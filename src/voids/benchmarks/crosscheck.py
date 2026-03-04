from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voids.core.network import Network
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
    k_ref = abs(q_ref_raw) * fluid.viscosity * L / (Axs * abs(dP))
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
