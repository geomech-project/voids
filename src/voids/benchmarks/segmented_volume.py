from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voids.benchmarks._shared import (
    make_benchmark_pressure_bc,
    resolve_benchmark_pressures,
)
from voids.benchmarks.crosscheck import (
    SinglePhaseCrosscheckSummary,
    crosscheck_singlephase_with_openpnm,
)
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions
from voids.image.network_extraction import (
    NetworkExtractionResult,
    extract_spanning_pore_network,
)


def _as_binary_volume(phases: np.ndarray) -> np.ndarray:
    """Validate and normalize a binary segmented volume.

    Parameters
    ----------
    phases :
        Input array encoded as ``void=1`` and ``solid=0``.

    Returns
    -------
    numpy.ndarray
        Integer array with the same shape as ``phases``.

    Raises
    ------
    ValueError
        If ``phases`` is not a 2D/3D binary volume.
    """

    arr = np.asarray(phases)
    if arr.ndim not in {2, 3}:
        raise ValueError("phases must be a 2D or 3D binary segmented volume")

    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0, 1, False, True))):
        raise ValueError("phases must be binary with void=1 and solid=0")
    return np.asarray(arr, dtype=int)


@dataclass(slots=True)
class SegmentedVolumeCrosscheckResult:
    """Store extraction, porosity, and solver cross-check outputs.

    Attributes
    ----------
    extract :
        Result of importing the segmented volume into a `voids` network and
        pruning it to the requested spanning axis.
    fluid :
        Fluid properties used in the permeability solve.
    bc :
        Pressure boundary conditions imposed on the extracted network.
    options :
        Solver and conductance options used for the comparison.
    image_porosity :
        Void fraction of the segmented binary image.
    absolute_porosity, effective_porosity :
        Porosity diagnostics computed from the pruned extracted network.
    summary :
        Comparison summary between `voids` and OpenPNM.

    Notes
    -----
    This high-level benchmark follows the same public pressure-BC convention as
    `benchmark_segmented_volume_with_xlb`: the preferred user input is
    ``delta_p``, while optional ``pin`` / ``pout`` values can still be used to
    preserve an absolute pressure gauge. The applied physical pressures and
    pressure drop are recorded explicitly in
    :meth:`SegmentedVolumeCrosscheckResult.to_record`.
    """

    extract: NetworkExtractionResult
    fluid: FluidSinglePhase
    bc: PressureBC
    options: SinglePhaseOptions
    image_porosity: float
    absolute_porosity: float
    effective_porosity: float
    summary: SinglePhaseCrosscheckSummary

    def to_record(self) -> dict[str, Any]:
        """Return scalar diagnostics suitable for tabulation."""

        details = dict(self.summary.details)
        return {
            "flow_axis": self.summary.axis,
            "phi_image": float(self.image_porosity),
            "phi_abs": float(self.absolute_porosity),
            "phi_eff": float(self.effective_porosity),
            "Np": int(self.extract.net.Np),
            "Nt": int(self.extract.net.Nt),
            "k_voids": float(details["k_voids"]),
            "k_openpnm": float(details["k_ref"]),
            "k_abs_diff": float(self.summary.permeability_abs_diff),
            "k_rel_diff": float(self.summary.permeability_rel_diff),
            "Q_voids": float(details["Q_voids"]),
            "Q_openpnm": float(details["Q_ref"]),
            "Q_abs_diff": float(self.summary.total_flow_abs_diff),
            "Q_rel_diff": float(self.summary.total_flow_rel_diff),
            "n_inlet_pores": int(details["n_inlet_pores"]),
            "n_outlet_pores": int(details["n_outlet_pores"]),
            "conductance_model": str(
                details.get("conductance_model", self.options.conductance_model)
            ),
            "solver_voids": str(details.get("solver_voids", self.options.solver)),
            "p_inlet_physical": float(self.bc.pin),
            "p_outlet_physical": float(self.bc.pout),
            "dp_physical": float(self.bc.pin - self.bc.pout),
            "backend": str(self.extract.backend),
            "backend_version": self.extract.backend_version,
            "openpnm_version": details.get("openpnm_version"),
        }


def benchmark_segmented_volume_with_openpnm(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    fluid: FluidSinglePhase | None = None,
    delta_p: float | None = None,
    pin: float | None = None,
    pout: float | None = None,
    options: SinglePhaseOptions | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
) -> SegmentedVolumeCrosscheckResult:
    """Benchmark an extracted segmented volume against OpenPNM.

    Parameters
    ----------
    phases :
        Binary segmented image encoded as ``void=1`` and ``solid=0``.
    voxel_size :
        Edge length of one voxel in the declared length unit.
    flow_axis :
        Requested transport axis. When omitted, the longest image axis is used.
    fluid :
        Fluid properties. Defaults to water-like viscosity `1e-3 Pa s`.
    delta_p :
        Preferred physical pressure drop for the benchmark, typically in Pa.
        When provided alone, the wrapper uses `pout = 0` and `pin = delta_p` as
        a gauge choice. When combined with one of `pin` or `pout`, the missing
        value is inferred. When combined with both, consistency with
        ``pin - pout`` is enforced.
    pin, pout :
        Optional absolute physical inlet and outlet pressures. They are kept for
        backward compatibility and for cases where the user wants to preserve a
        particular pressure reference level. For the current incompressible
        benchmark, only the pressure drop ``pin - pout`` affects the reported
        permeability.
    options :
        Solver controls. Defaults to the image-workflow baseline
        ``valvatne_blunt`` with the direct linear solver.
    length_unit, pressure_unit :
        Units attached to the extracted sample geometry.
    extraction_kwargs :
        Extra keyword arguments forwarded to `porespy.networks.snow2`.
    provenance_notes :
        Optional metadata attached to the extracted network provenance.
    strict :
        Forwarded to :func:`voids.image.extract_spanning_pore_network`.

    Returns
    -------
    SegmentedVolumeCrosscheckResult
        Extraction metadata, porosity diagnostics, and the OpenPNM comparison.

    Raises
    ------
    ValueError
        If the image is invalid, the pressure specification is inconsistent, or
        the implied pressure drop is not positive.

    Notes
    -----
    This helper uses :func:`voids.benchmarks.crosscheck_singlephase_with_openpnm`,
    which injects the `voids` throat hydraulic conductances into OpenPNM. The
    resulting comparison isolates extraction consistency, boundary-condition
    handling, and linear-solver agreement; it does not benchmark independent
    conductance models between packages.

    Unlike the XLB high-level benchmark, no fluid-density-based unit conversion
    is needed here because both sides solve the same extracted pore network
    directly under the same physical pressure BC.

    The absolute pressure offset is numerically immaterial for this current
    incompressible benchmark. For example, ``delta_p=1``, ``pin=1``/``pout=0``,
    and ``delta_p=1`` with ``pin=101326``/``pout=101325`` all impose the same
    permeability-driving pressure drop.
    """

    arr = _as_binary_volume(phases)
    image_phi = float(arr.mean())
    pin_used, pout_used, _ = resolve_benchmark_pressures(
        delta_p=delta_p,
        pin=pin,
        pout=pout,
    )

    notes = dict(provenance_notes or {})
    notes.setdefault("benchmark_kind", "segmented_volume_openpnm")

    extract = extract_spanning_pore_network(
        arr,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        length_unit=length_unit,
        pressure_unit=pressure_unit,
        extraction_kwargs=extraction_kwargs,
        provenance_notes=notes,
        strict=strict,
    )

    fluid_used = fluid or FluidSinglePhase(viscosity=1.0e-3)
    options_used = options or SinglePhaseOptions(
        conductance_model="valvatne_blunt",
        solver="direct",
    )
    axis = extract.flow_axis
    bc = make_benchmark_pressure_bc(axis, pin=pin_used, pout=pout_used)
    summary = crosscheck_singlephase_with_openpnm(
        extract.net,
        fluid=fluid_used,
        bc=bc,
        axis=axis,
        options=options_used,
    )

    return SegmentedVolumeCrosscheckResult(
        extract=extract,
        fluid=fluid_used,
        bc=bc,
        options=options_used,
        image_porosity=image_phi,
        absolute_porosity=float(absolute_porosity(extract.net)),
        effective_porosity=float(effective_porosity(extract.net, axis=axis)),
        summary=summary,
    )


__all__ = [
    "SegmentedVolumeCrosscheckResult",
    "benchmark_segmented_volume_with_openpnm",
]
