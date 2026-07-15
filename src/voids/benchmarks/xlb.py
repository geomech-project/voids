"""Segmented-volume benchmarks that consume the XLB LBM backend.

The direct-image XLB adapter lives in :mod:`voids.lbm.singlephase.xlb`. This
module composes that backend with `voids` network extraction and single-phase
PNM solves for benchmark comparisons. Low-level XLB symbols are re-exported here
for backward compatibility with older notebooks.
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

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
from voids.image.porosity import PorosityMap
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    SinglePhaseResult,
    solve,
)
from voids.visualization.fields import plot_vector_midplanes, write_structured_vector_field


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


def _write_csv_records(
    path: str | Path,
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def export_xlb_direct_simulation_artifacts(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axes: Sequence[str],
    options: XLBOptions,
    output_dir: str | Path,
    output_prefix: str,
    sample_name: str,
    m2_per_md: float,
    directional_path: str | Path | None = None,
    status_path: str | Path | None = None,
    field_outputs_path: str | Path | None = None,
    quiver_stride: int = 8,
) -> list[dict[str, object]]:
    """Run direct-image XLB solves and export benchmark artifacts.

    This helper is intentionally process-friendly: notebooks can call it from a
    short-lived Python worker so JAX/XLB GPU allocations are released before
    other GPU solvers, such as cuDSS-backed FEM, run in the main notebook kernel.
    """

    arr = _as_binary_volume(phases)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    directional_destination = (
        Path(directional_path)
        if directional_path is not None
        else destination / f"{output_prefix}_xlb_lbm_directional.csv"
    )
    status_destination = (
        Path(status_path)
        if status_path is not None
        else destination / f"{output_prefix}_xlb_lbm_status.json"
    )
    field_outputs_destination = (
        Path(field_outputs_path)
        if field_outputs_path is not None
        else destination / f"{output_prefix}_xlb_lbm_field_outputs.csv"
    )

    method = "Direct-image LBM DNS (XLB, Stokes-limit preset)"
    family = "direct_image_dns"
    grid = PorosityMap(
        values=np.asarray(arr, dtype=float),
        cell_size=(float(voxel_size),) * arr.ndim,
        metadata={
            "field_role": "voxel_grid_for_direct_image_lbm_exports",
            "phase_convention": "1=void_or_pore, 0=solid",
        },
    )

    directional_columns = [
        "family",
        "formulation",
        "method",
        "solver_backend",
        "axis",
        "K_m2",
        "K_mD",
        "solve_seconds",
        "xlb_steps",
        "xlb_converged",
        "xlb_convergence_metric",
        "xlb_mach_max",
        "xlb_re_voxel_max",
        "xlb_lattice_viscosity",
        "xlb_pressure_drop_lattice",
        "warning_count",
        "warnings",
    ]
    field_columns = ["family", "formulation", "method", "axis", "field", "kind", "path"]

    directional_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    status: dict[str, object] = {
        "status": "ok",
        "options": asdict(options),
        "runs": [],
    }

    def write_progress() -> None:
        _write_csv_records(
            directional_destination,
            directional_rows,
            columns=directional_columns,
        )
        _write_csv_records(
            field_outputs_destination,
            field_rows,
            columns=field_columns,
        )
        status_destination.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        for axis in flow_axes:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = solve_binary_volume_with_xlb(
                    arr,
                    voxel_size=voxel_size,
                    flow_axis=axis,
                    options=options,
                )
            formulation = f"xlb_{result.formulation}"
            solve_seconds = float(time.perf_counter() - start)
            directional_rows.append(
                {
                    "family": family,
                    "formulation": formulation,
                    "method": method,
                    "solver_backend": f"xlb:{result.backend}",
                    "axis": axis,
                    "K_m2": float(result.permeability),
                    "K_mD": float(result.permeability / m2_per_md),
                    "solve_seconds": solve_seconds,
                    "xlb_steps": int(result.n_steps),
                    "xlb_converged": bool(result.converged),
                    "xlb_convergence_metric": float(result.convergence_metric),
                    "xlb_mach_max": float(result.max_mach_lattice),
                    "xlb_re_voxel_max": float(result.reynolds_voxel_max),
                    "xlb_lattice_viscosity": float(result.lattice_viscosity),
                    "xlb_pressure_drop_lattice": float(result.lattice_pressure_drop),
                    "warning_count": int(len(caught)),
                    "warnings": "; ".join(str(item.message) for item in caught),
                }
            )

            vtu_path = destination / f"{output_prefix}_xlb_lbm_velocity_{axis}.vtu"
            write_structured_vector_field(
                result.velocity_lattice,
                grid,
                vtu_path,
                extra_cell_data={"axial_velocity_lattice": result.axial_velocity_lattice},
            )
            field_rows.append(
                {
                    "family": family,
                    "formulation": formulation,
                    "method": method,
                    "axis": axis,
                    "field": "velocity",
                    "kind": "paraview_vtu",
                    "path": str(vtu_path),
                }
            )

            plot_path = destination / f"{output_prefix}_xlb_lbm_velocity_midplanes_{axis}.png"
            plot_vector_midplanes(
                result.velocity_lattice,
                title=f"{sample_name} XLB/LBM velocity, flow {axis}",
                path=plot_path,
                quiver_stride=quiver_stride,
                colorbar_label="velocity magnitude [lattice units]",
            )
            field_rows.append(
                {
                    "family": family,
                    "formulation": formulation,
                    "method": method,
                    "axis": axis,
                    "field": "velocity",
                    "kind": "midplane_quiver_png",
                    "path": str(plot_path),
                }
            )

            runs = status["runs"]
            if isinstance(runs, list):
                runs.append(
                    {
                        "axis": axis,
                        "status": "ok",
                        "solve_seconds": solve_seconds,
                    }
                )
            write_progress()
    except Exception as exc:
        status["status"] = "failed"
        status["message"] = f"{type(exc).__name__}: {exc}"
        write_progress()
        raise

    return directional_rows


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
    "export_xlb_direct_simulation_artifacts",
    "solve_binary_volume_with_xlb",
]
