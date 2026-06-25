"""Segmented-volume benchmarks that consume the XLB LBM backend.

The direct-image XLB adapter lives in :mod:`voids.lbm.singlephase.xlb`. This
module composes that backend with `voids` network extraction and single-phase
PNM solves for benchmark comparisons. Low-level XLB symbols are re-exported here
for backward compatibility with older notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voids.benchmarks._shared import (
    make_benchmark_pressure_bc,
    resolve_benchmark_pressures,
)
from voids.image.network_extraction import (
    NetworkExtractionResult,
    extract_spanning_pore_network,
)
from voids.lbm.singlephase import xlb as _xlb_backend
from voids.lbm.singlephase.xlb import (
    DEFAULT_PRESSURE_DROP_LATTICE,
    DEFAULT_REFERENCE_DENSITY_LATTICE,
    DEFAULT_STOKES_PRESSURE_DROP_LATTICE,
    ISOTHERMAL_LATTICE_CS2,
    MAX_RECOMMENDED_DENSITY_DROP_LATTICE,
    XLBConvergenceWarning,
    XLBDirectSimulationResult,
    XLBOptions,
    _as_binary_volume,
    _axis_to_index as _axis_to_index,
    _couple_xlb_options_to_physical_pressure_drop,
    _import_xlb,
    _mask_to_indices as _mask_to_indices,
    _physical_pressure_drop_to_lattice as _physical_pressure_drop_to_lattice,
    _rel_diff,
    _resolve_lattice_pressure_bc as _resolve_lattice_pressure_bc,
    _superficial_velocity_profile as _superficial_velocity_profile,
)
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    SinglePhaseResult,
    solve,
)


def solve_binary_volume_with_xlb(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    options: XLBOptions | None = None,
) -> XLBDirectSimulationResult:
    """Backward-compatible wrapper for the LBM XLB direct-image solver."""

    _xlb_backend._import_xlb = _import_xlb
    return _xlb_backend.solve_binary_volume_with_xlb(
        phases,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        options=options,
    )


@dataclass(slots=True)
class SegmentedVolumeXLBResult:
    """Store extraction, porosity, and direct-image XLB benchmark outputs.

    Attributes
    ----------
    bc :
        Physical pressure BC used on the extracted-network `voids` solve.
    xlb_options :
        XLB options actually used for the direct-image solve. For the high-level
        benchmark wrapper these are pressure-coupled so they match the resolved
        physical pressure drop used on the `voids` side.
    xlb_result :
        Direct-image XLB result, including resolved lattice pressure diagnostics.
    """

    extract: NetworkExtractionResult
    fluid: FluidSinglePhase
    bc: PressureBC
    options: SinglePhaseOptions
    xlb_options: XLBOptions
    image_porosity: float
    absolute_porosity: float
    effective_porosity: float
    voids_result: SinglePhaseResult
    xlb_result: XLBDirectSimulationResult
    permeability_abs_diff: float
    permeability_rel_diff: float

    def to_record(self) -> dict[str, float | int | str | bool | None]:
        """Return scalar diagnostics suitable for tabulation."""

        k_voids = float((self.voids_result.permeability or {}).get(self.extract.flow_axis, np.nan))
        return {
            "flow_axis": self.extract.flow_axis,
            "phi_image": float(self.image_porosity),
            "phi_abs": float(self.absolute_porosity),
            "phi_eff": float(self.effective_porosity),
            "Np": int(self.extract.net.Np),
            "Nt": int(self.extract.net.Nt),
            "k_voids": k_voids,
            "k_xlb": float(self.xlb_result.permeability),
            "k_abs_diff": float(self.permeability_abs_diff),
            "k_rel_diff": float(self.permeability_rel_diff),
            "voids_mass_balance_error": float(self.voids_result.mass_balance_error),
            "conductance_model": str(self.options.conductance_model),
            "solver_voids": str(self.options.solver),
            "p_inlet_physical": float(self.bc.pin),
            "p_outlet_physical": float(self.bc.pout),
            "dp_physical": float(self.bc.pin - self.bc.pout),
            "extract_backend": str(self.extract.backend),
            "extract_backend_version": self.extract.backend_version,
            "xlb_backend": str(self.xlb_result.backend),
            "xlb_backend_version": self.xlb_result.backend_version,
            "xlb_formulation": str(self.xlb_result.formulation),
            "xlb_velocity_set": str(self.xlb_result.velocity_set),
            "xlb_collision_model": str(self.xlb_result.collision_model),
            "xlb_streaming_scheme": str(self.xlb_result.streaming_scheme),
            "xlb_steps": int(self.xlb_result.n_steps),
            "xlb_converged": bool(self.xlb_result.converged),
            "xlb_convergence_metric": float(self.xlb_result.convergence_metric),
            "xlb_lattice_viscosity": float(self.xlb_result.lattice_viscosity),
            "xlb_p_inlet": float(self.xlb_result.lattice_pressure_inlet),
            "xlb_p_outlet": float(self.xlb_result.lattice_pressure_outlet),
            "xlb_rho_inlet": float(self.xlb_result.lattice_density_inlet),
            "xlb_rho_outlet": float(self.xlb_result.lattice_density_outlet),
            "xlb_dp_lattice": float(self.xlb_result.lattice_pressure_drop),
            "xlb_buffer_cells": int(self.xlb_result.inlet_outlet_buffer_cells),
            "xlb_u_superficial_lattice": float(self.xlb_result.superficial_velocity_lattice),
            "xlb_u_max_lattice": float(self.xlb_result.max_speed_lattice),
            "xlb_mach_max": float(self.xlb_result.max_mach_lattice),
            "xlb_re_voxel_max": float(self.xlb_result.reynolds_voxel_max),
        }


def benchmark_segmented_volume_with_xlb(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    fluid: FluidSinglePhase | None = None,
    delta_p: float | None = None,
    pin: float | None = None,
    pout: float | None = None,
    options: SinglePhaseOptions | None = None,
    xlb_options: XLBOptions | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
) -> SegmentedVolumeXLBResult:
    """Benchmark a segmented volume against a direct-image XLB solve.

    The `voids` side solves on the extracted pore network. The XLB side solves
    directly on the binary segmented image through
    :func:`voids.lbm.singlephase.xlb.solve_binary_volume_with_xlb`. The wrapper
    enforces a shared physical pressure drop before comparing permeability.
    """

    arr = _as_binary_volume(phases)
    image_phi = float(arr.mean())
    pin_used, pout_used, delta_p_physical = resolve_benchmark_pressures(
        delta_p=delta_p,
        pin=pin,
        pout=pout,
    )

    notes = dict(provenance_notes or {})
    notes.setdefault("benchmark_kind", "segmented_volume_xlb")

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

    fluid_used = fluid or FluidSinglePhase(viscosity=1.0e-3, density=1.0e3)
    options_used = options or SinglePhaseOptions(
        conductance_model="valvatne_blunt",
        solver="direct",
    )
    xlb_options_used = xlb_options or XLBOptions()

    axis = extract.flow_axis
    inlet_count = int(
        np.asarray(extract.net.pore_labels.get(f"inlet_{axis}min", []), dtype=bool).sum()
    )
    outlet_count = int(
        np.asarray(extract.net.pore_labels.get(f"outlet_{axis}max", []), dtype=bool).sum()
    )
    if extract.net.Np == 0 or inlet_count == 0 or outlet_count == 0:
        raise ValueError(
            "The extracted spanning network is empty or lacks non-empty inlet/outlet pore labels "
            f"for axis '{axis}', so the XLB benchmark cannot be compared against `voids` on this case."
        )

    if fluid_used.density is None or fluid_used.density <= 0.0:
        raise ValueError(
            "benchmark_segmented_volume_with_xlb requires `fluid.density` to map the shared "
            "physical pressure drop into lattice pressure units"
        )
    xlb_options_coupled = _couple_xlb_options_to_physical_pressure_drop(
        xlb_options_used,
        delta_p_physical=delta_p_physical,
        voxel_size=voxel_size,
        fluid=fluid_used,
    )

    bc = make_benchmark_pressure_bc(axis, pin=pin_used, pout=pout_used)
    voids_result = solve(
        extract.net,
        fluid=fluid_used,
        bc=bc,
        axis=axis,
        options=options_used,
    )
    xlb_result = solve_binary_volume_with_xlb(
        arr,
        voxel_size=voxel_size,
        flow_axis=axis,
        options=xlb_options_coupled,
    )

    k_voids = float((voids_result.permeability or {}).get(axis, np.nan))
    k_xlb = float(xlb_result.permeability)

    return SegmentedVolumeXLBResult(
        extract=extract,
        fluid=fluid_used,
        bc=bc,
        options=options_used,
        xlb_options=xlb_options_coupled,
        image_porosity=image_phi,
        absolute_porosity=float(absolute_porosity(extract.net)),
        effective_porosity=float(effective_porosity(extract.net, axis=axis)),
        voids_result=voids_result,
        xlb_result=xlb_result,
        permeability_abs_diff=abs(k_voids - k_xlb),
        permeability_rel_diff=_rel_diff(k_voids, k_xlb),
    )


__all__ = [
    "DEFAULT_PRESSURE_DROP_LATTICE",
    "DEFAULT_REFERENCE_DENSITY_LATTICE",
    "DEFAULT_STOKES_PRESSURE_DROP_LATTICE",
    "ISOTHERMAL_LATTICE_CS2",
    "MAX_RECOMMENDED_DENSITY_DROP_LATTICE",
    "SegmentedVolumeXLBResult",
    "XLBConvergenceWarning",
    "XLBDirectSimulationResult",
    "XLBOptions",
    "benchmark_segmented_volume_with_xlb",
    "solve_binary_volume_with_xlb",
]
