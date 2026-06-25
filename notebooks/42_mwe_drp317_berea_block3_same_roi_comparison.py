# %% [markdown]
# # MWE 42 - DRP-317 Berea block-3 same-ROI comparison
#
# This notebook repeats the DRP-317 Berea comparison with a finer
# `(3, 3, 3)` porosity/permeability-map block while keeping the same image ROI
# for every model family:
#
# - extracted pore-network models from the binary ROI
# - direct-image lattice-Boltzmann DNS on the binary ROI using XLB
# - a cell-centered TPFA Darcy-Darcy solve on `K(phi)`
# - direct FEniCSx FEM coefficient-field solves on the same `phi` and
#   `K(phi)` maps:
#   - Darcy-Brinkman USFEM `CG1 x DG1`
#   - Darcy-Brinkman Taylor-Hood `CG2 x CG1`
#   - Darcy-Darcy Taylor-Hood `CG2 x CG1`
# - the experimental Berea scalar permeability reference
#
# The default ROI is `75^3` voxels. Attempts at `78^3` and `84^3` with the
# full direct FEM comparison exceeded the original 20 min/direction runtime
# criterion before producing directional results, so `75^3` is kept as the
# largest fully observed feasible ROI for this notebook. The practical cap used
# for the run below is 35 min/direction.

# %%
from __future__ import annotations

import json
import os
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.fem.singlephase import (
    FEMMapProblem,
    FEniCSSolverOptions,
    solve_brinkman_taylor_hood,
    solve_brinkman_usfem,
    solve_darcy_taylor_hood,
)
from voids.fvm.singlephase import solve_tpfa
from voids.image import extract_spanning_pore_network, infer_sample_axes
from voids.image.porosity import (
    PorosityMap,
    load_permeability_map_hdf5,
    load_porosity_map_hdf5,
    permeability_map_from_porosity,
    porosity_map_from_binary,
    save_permeability_map_hdf5,
    save_porosity_map_hdf5,
)
from voids.lbm.singlephase import XLBOptions, solve_binary_volume_with_xlb
from voids.paths import data_path, project_root
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    solve,
)
from voids.visualization.fields import (
    plot_scalar_midplanes,
    plot_vector_midplanes,
    reference_pressure_to_outlet,
    reconstruct_tpfa_cell_velocity,
    sample_dolfinx_function_on_grid,
    write_dolfinx_function_xdmf,
    write_structured_vector_field,
)

plt.ioff()

# %%
# User-editable inputs
sample_name = "DRP-317 Berea"
sample_stem = "Berea_2d25um"
raw_relpath = Path("drp-317") / "Berea_2d25um_binary.raw"
voxel_size_um = 2.25
voxel_size_m = voxel_size_um * 1.0e-6

experimental_porosity_pct = 18.96
experimental_kabs_mD = 121.0
experimental_kabs_rel_error = 0.10

full_shape = (1000, 1000, 1000)
roi_shape = (75, 75, 75)
map_block_shape = (3, 3, 3)
roi_scan_positions = 5
roi_porosity_target = "full_image"  # "full_image" or "experimental"

kozeny_constant = 180.0
solid_permeability_m2 = 1.0e-20
free_flow_permeability_m2 = 1.0e-8
max_permeability_m2 = 1.0e-8

mu_pa_s = 1.0e-3
pressure_gradient_pa_per_m = 1.0e4
pressure_inlet_pa = 1.0
pressure_outlet_pa = 0.0
reference_pressure_pa = 1.0e5
flow_axes = ("x", "y", "z")
resistor_solver_method = "cg"
resistor_solver_parameters: dict[str, Any] = {
    "rtol": 1.0e-10,
    "atol": 0.0,
    "maxiter": 2000,
    "preconditioner": "pyamg",
    "pyamg_solver": "smoothed_aggregation",
}

run_pnm = True
trim_nonpercolating_paths = True
pnm_backends = ("porespy", "prego", "native_maximal_ball")
pnm_backend_configs: dict[str, dict[str, object]] = {
    "porespy": {
        "label": "PoreSpy snow2",
        "extraction_kwargs": {},
        "geometry_repairs": None,
    },
    "prego": {
        "label": "PREGO",
        "extraction_kwargs": {
            "settings": {
                "r_max": 4,
                "sigma": 0.4,
                "peak_footprint": "sphere",
                "growth_mode": "level_queue",
                "distance_map_backend": "scipy",
            },
            "regions_to_network_kwargs": {"accuracy": "standard"},
        },
        "geometry_repairs": None,
    },
    "native_maximal_ball": {
        "label": "Native maximal-ball",
        "extraction_kwargs": {
            "distance_map_backend": "auto",
            "flow_boundary_mode": "direct",
        },
        "geometry_repairs": None,
    },
}

run_fem = True
fem_porosity_floor = 1.0e-3
fem_k_floor = 1.0e-20
# PETSc's built-in LU backend is not a valid scientific fallback for the
# high-contrast mixed FEM systems in this notebook; direct sparse factorizations
# with MUMPS and SuperLU_DIST give stable results for the Taylor-Hood and USFEM
# rows used below.
fem_taylor_hood_solver_backends = ("mumps",)
fem_usfem_solver_backends = ("superlu_dist",)
fem_direct_base_options: dict[str, object] = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_shift_type": "nonzero",
    "pc_factor_shift_amount": 1.0e-12,
    "ksp_error_if_not_converged": True,
}


def fem_direct_solver_options(backend: str) -> FEniCSSolverOptions:
    options = dict(fem_direct_base_options)
    options["pc_factor_mat_solver_type"] = backend
    return FEniCSSolverOptions(petsc_options=options)


# XLB solves the binary ROI directly. These settings keep the run in a
# low-Mach, low-voxel-Reynolds regime for comparison with creeping-flow models.
run_lbm = True
lbm_options = XLBOptions.steady_stokes_defaults()

output_dir = (
    project_root()
    / "notebooks"
    / "outputs"
    / "42_mwe_drp317_berea_block3_same_roi_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_prefix = "drp317_berea_block3_same_roi"
lbm_directional_path = output_dir / f"{output_prefix}_xlb_lbm_directional.csv"
lbm_status_path = output_dir / f"{output_prefix}_xlb_lbm_status.json"
field_output_records: list[dict[str, object]] = []

M2_PER_MD = 9.869233e-16


def _record_field_output(
    *,
    family: str,
    formulation: str,
    method: str,
    axis: str,
    field: str,
    kind: str,
    path: Path,
) -> None:
    field_output_records.append(
        {
            "family": family,
            "formulation": formulation,
            "method": method,
            "axis": axis,
            "field": field,
            "kind": kind,
            "path": str(path),
        }
    )


# %% [markdown]
# ## ROI selection

# %%
raw_path = data_path() / raw_relpath
if not raw_path.exists():
    raise FileNotFoundError(f"Missing DRP-317 Berea RAW file: {raw_path}")

expected_voxels = int(np.prod(np.asarray(full_shape, dtype=np.int64)))
actual_voxels = raw_path.stat().st_size // np.dtype(np.uint8).itemsize
if actual_voxels != expected_voxels:
    raise ValueError(
        f"Configured shape {full_shape} requires {expected_voxels:,} voxels, "
        f"but {raw_path.name} stores {actual_voxels:,}."
    )

if any(s % b != 0 for s, b in zip(roi_shape, map_block_shape, strict=True)):
    raise ValueError(
        f"roi_shape {roi_shape} must be divisible by map_block_shape {map_block_shape}"
    )

raw_image = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=full_shape, order="C")


def raw_to_void(raw: np.ndarray) -> np.ndarray:
    """Return the DRP-317 Berea phase convention used in notebook 18."""

    return np.asarray(raw == 0, dtype=bool)


def candidate_starts(full_edge: int, sub_edge: int, *, count: int) -> list[int]:
    if sub_edge > full_edge:
        raise ValueError(f"ROI edge {sub_edge} exceeds full edge {full_edge}")
    if count <= 1 or sub_edge == full_edge:
        return [0]
    max_origin = full_edge - sub_edge
    return sorted({int(round(value)) for value in np.linspace(0, max_origin, count)})


phi_void_is_zero = 1.0 - float(np.mean(raw_image))
target_porosity = (
    phi_void_is_zero
    if roi_porosity_target == "full_image"
    else 0.01 * experimental_porosity_pct
)

scan_records: list[dict[str, object]] = []
starts_by_axis = [
    candidate_starts(full, sub, count=roi_scan_positions)
    for full, sub in zip(full_shape, roi_shape, strict=True)
]
for origin in product(*starts_by_axis):
    slices = tuple(
        slice(start, start + size)
        for start, size in zip(origin, roi_shape, strict=True)
    )
    block_raw = np.asarray(raw_image[slices], dtype=np.uint8)
    block_porosity = float(raw_to_void(block_raw).mean())
    scan_records.append(
        {
            "origin": tuple(int(value) for value in origin),
            "porosity_pct": 100.0 * block_porosity,
            "target_porosity_pct": 100.0 * target_porosity,
            "abs_porosity_error_pct_points": 100.0
            * abs(block_porosity - target_porosity),
        }
    )

roi_scan = pd.DataFrame(scan_records).sort_values(
    ["abs_porosity_error_pct_points", "porosity_pct"],
    kind="stable",
)
roi_scan = roi_scan.reset_index(drop=True)
roi_origin = tuple(int(value) for value in roi_scan.loc[0, "origin"])
roi_stop = tuple(
    start + size for start, size in zip(roi_origin, roi_shape, strict=True)
)
roi_slices = tuple(
    slice(start, stop) for start, stop in zip(roi_origin, roi_stop, strict=True)
)
roi_raw = np.asarray(raw_image[roi_slices], dtype=np.uint8)
void_roi = raw_to_void(roi_raw)
roi_porosity_pct = 100.0 * float(void_roi.mean())

phase_summary = pd.DataFrame(
    {
        "raw_value": [0, 1],
        "voxel_count": [
            int(np.count_nonzero(roi_raw == 0)),
            int(np.count_nonzero(roi_raw == 1)),
        ],
        "phase": ["void/pore", "solid"],
    }
)
phase_summary["fraction"] = phase_summary["voxel_count"] / float(roi_raw.size)
roi_summary = pd.DataFrame(
    [
        {"quantity": "sample", "value": sample_name, "units": "-"},
        {"quantity": "RAW path", "value": str(raw_path), "units": "-"},
        {"quantity": "phase convention", "value": "0=void/pore, 1=solid", "units": "-"},
        {"quantity": "ROI origin", "value": str(roi_origin), "units": "voxels"},
        {"quantity": "ROI shape", "value": str(roi_shape), "units": "voxels"},
        {"quantity": "voxel size", "value": voxel_size_m, "units": "m"},
        {"quantity": "ROI porosity", "value": roi_porosity_pct, "units": "%"},
        {
            "quantity": "full-image porosity if void==0",
            "value": 100.0 * phi_void_is_zero,
            "units": "%",
        },
        {
            "quantity": "experimental porosity",
            "value": experimental_porosity_pct,
            "units": "%",
        },
        {"quantity": "experimental Kabs", "value": experimental_kabs_mD, "units": "mD"},
        {
            "quantity": "map block shape",
            "value": str(map_block_shape),
            "units": "voxels",
        },
    ]
)
voxel_porosity_map = PorosityMap(
    values=void_roi.astype(float),
    cell_size=(voxel_size_m, voxel_size_m, voxel_size_m),
    origin=(0.0, 0.0, 0.0),
    units={"length": "m"},
    metadata={
        "case": output_prefix,
        "field_role": "voxel_grid_for_direct_image_lbm_exports",
        "phase_convention": "1=void_or_pore, 0=solid",
    },
)

display(roi_scan.head(10))
display(phase_summary)
display(roi_summary)

# %% [markdown]
# ## Porosity and Kozeny-Carman maps

# %%
porosity_map = porosity_map_from_binary(
    void_roi,
    block_shape=map_block_shape,
    voxel_size=(voxel_size_m, voxel_size_m, voxel_size_m),
    strict=True,
    metadata={
        "case": "drp317_berea_block3_same_roi",
        "raw_filename": raw_relpath.name,
        "raw_shape": full_shape,
        "raw_order": "C",
        "roi_origin": roi_origin,
        "roi_shape": roi_shape,
        "phase_convention": "0=void_or_pore, 1=solid",
        "experimental_porosity_pct": experimental_porosity_pct,
        "experimental_kabs_mD": experimental_kabs_mD,
    },
)

characteristic_length_m = min(porosity_map.cell_size)
permeability_map = permeability_map_from_porosity(
    porosity_map,
    characteristic_length=characteristic_length_m,
    kozeny_constant=kozeny_constant,
    solid_permeability=solid_permeability_m2,
    free_flow_permeability=free_flow_permeability_m2,
    max_permeability=max_permeability_m2,
    metadata={
        "closure_note": "Kozeny-Carman coefficient map for same-ROI block-3 Berea comparison",
        "scientific_caveat": "closure field, not a direct pore-scale permeability solve",
    },
)

porosity_h5 = output_dir / "drp317_berea_block3_same_roi_porosity_map.h5"
permeability_h5 = output_dir / "drp317_berea_block3_same_roi_permeability_map.h5"
save_porosity_map_hdf5(porosity_map, porosity_h5)
save_permeability_map_hdf5(permeability_map, permeability_h5)

loaded_porosity = load_porosity_map_hdf5(porosity_h5)
loaded_permeability = load_permeability_map_hdf5(permeability_h5)
assert np.allclose(loaded_porosity.values, porosity_map.values)
assert np.allclose(loaded_permeability.values, permeability_map.values)

map_summary = pd.DataFrame(
    [
        {
            "field": "porosity",
            "shape": str(porosity_map.shape),
            "min": float(np.min(porosity_map.values)),
            "mean": float(np.mean(porosity_map.values)),
            "max": float(np.max(porosity_map.values)),
            "units": "-",
        },
        {
            "field": "permeability",
            "shape": str(permeability_map.shape),
            "min": float(np.min(permeability_map.values)),
            "mean": float(np.mean(permeability_map.values)),
            "max": float(np.max(permeability_map.values)),
            "units": "m^2",
        },
    ]
)
display(map_summary)

# %%
mid = tuple(s // 2 for s in porosity_map.shape)
map_slice_specs = [
    (
        "x-mid",
        np.take(porosity_map.values, mid[0], axis=0),
        np.take(permeability_map.values, mid[0], axis=0),
    ),
    (
        "y-mid",
        np.take(porosity_map.values, mid[1], axis=1),
        np.take(permeability_map.values, mid[1], axis=1),
    ),
    (
        "z-mid",
        np.take(porosity_map.values, mid[2], axis=2),
        np.take(permeability_map.values, mid[2], axis=2),
    ),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
finite_log = np.log10(permeability_map.values[np.isfinite(permeability_map.values)])
for col, (title, porosity_slice, permeability_slice) in enumerate(map_slice_specs):
    im0 = axes[0, col].imshow(
        porosity_slice.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    axes[0, col].set_title(f"{title} porosity")
    fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

    with np.errstate(divide="ignore"):
        log_perm = np.log10(permeability_slice)
    im1 = axes[1, col].imshow(
        log_perm.T,
        origin="lower",
        vmin=float(np.min(finite_log)),
        vmax=float(np.max(finite_log)),
        cmap="magma",
    )
    axes[1, col].set_title(f"{title} log10 K")
    fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

map_figure_path = (
    output_dir / "drp317_berea_block3_same_roi_porosity_permeability_midplanes.png"
)
fig.savefig(map_figure_path, dpi=180)
display(fig)
map_figure_path

# %% [markdown]
# ## Cell-centered TPFA Darcy-Darcy solve

# %%
resistor_results = {
    axis: solve_tpfa(
        permeability_map,
        flow_axis=axis,
        viscosity=mu_pa_s,
        pressure_inlet=pressure_inlet_pa,
        pressure_outlet=pressure_outlet_pa,
        solver_method=resistor_solver_method,
        solver_parameters=resistor_solver_parameters,
    )
    for axis in flow_axes
}

resistor_df = pd.DataFrame(
    [
        {
            "family": "map_resistor",
            "formulation": "darcy_darcy_tpfa_fv",
            "method": "TPFA finite-volume Darcy-Darcy",
            "axis": result.flow_axis,
            "mu_Pa_s": mu_pa_s,
            "pressure_inlet_Pa": pressure_inlet_pa,
            "pressure_outlet_Pa": pressure_outlet_pa,
            "delta_p_Pa": abs(pressure_inlet_pa - pressure_outlet_pa),
            "inlet_flux_m3_s": result.inlet_flow_rate,
            "outlet_flux_m3_s": result.outlet_flow_rate,
            "mass_imbalance_relative": result.mass_balance_error,
            "K_eq_m2": result.permeability,
            "K_eq_mD": result.permeability / M2_PER_MD,
            "matrix_nnz": result.matrix_nnz,
            "solve_seconds": result.solve_seconds,
            "solver_backend": (
                f"{result.solver_method}+{result.solver_info['preconditioner']}"
                if "preconditioner" in result.solver_info
                else result.solver_method
            ),
            "linear_residual_relative": result.residual_relative,
            "solver_info_json": json.dumps(result.solver_info, sort_keys=True),
        }
        for result in resistor_results.values()
    ]
)
display(resistor_df)

for axis, result in resistor_results.items():
    tpfa_velocity = reconstruct_tpfa_cell_velocity(
        result.pressure,
        permeability_map,
        flow_axis=axis,
        viscosity=mu_pa_s,
        pressure_inlet=pressure_inlet_pa,
        pressure_outlet=pressure_outlet_pa,
    )
    tpfa_vtu_path = output_dir / f"{output_prefix}_tpfa_velocity_{axis}.vtu"
    write_structured_vector_field(
        tpfa_velocity,
        porosity_map,
        tpfa_vtu_path,
        extra_cell_data={"pressure": result.pressure},
    )
    _record_field_output(
        family="map_resistor",
        formulation="darcy_darcy_tpfa_fv",
        method="TPFA finite-volume Darcy-Darcy",
        axis=axis,
        field="velocity",
        kind="paraview_vtu",
        path=tpfa_vtu_path,
    )

    pressure_plot_path = (
        output_dir / f"{output_prefix}_tpfa_pressure_midplanes_{axis}.png"
    )
    plotted_pressure = reference_pressure_to_outlet(
        result.pressure,
        flow_axis=axis,
        reference_pressure=reference_pressure_pa,
        pressure_outlet=pressure_outlet_pa,
    )
    pressure_fig = plot_scalar_midplanes(
        plotted_pressure,
        title=f"{sample_name} TPFA pressure, flow {axis}",
        path=pressure_plot_path,
        colorbar_label="pressure [Pa]",
        colorbar_use_offset=False,
    )
    plt.close(pressure_fig)
    _record_field_output(
        family="map_resistor",
        formulation="darcy_darcy_tpfa_fv",
        method="TPFA finite-volume Darcy-Darcy",
        axis=axis,
        field="pressure",
        kind="midplane_png",
        path=pressure_plot_path,
    )

    velocity_plot_path = (
        output_dir / f"{output_prefix}_tpfa_velocity_midplanes_{axis}.png"
    )
    velocity_fig = plot_vector_midplanes(
        tpfa_velocity,
        title=f"{sample_name} TPFA velocity, flow {axis}",
        path=velocity_plot_path,
        quiver_stride=3,
        colorbar_label="velocity magnitude [m/s]",
    )
    plt.close(velocity_fig)
    _record_field_output(
        family="map_resistor",
        formulation="darcy_darcy_tpfa_fv",
        method="TPFA finite-volume Darcy-Darcy",
        axis=axis,
        field="velocity",
        kind="midplane_quiver_png",
        path=velocity_plot_path,
    )

# %% [markdown]
# ## Same-ROI pore-network extraction and single-phase solves

# %%
_, axis_lengths, _, _ = infer_sample_axes(void_roi.shape, voxel_size=voxel_size_m)


def prepare_axis_image(image: np.ndarray, *, axis: str) -> np.ndarray:
    if not trim_nonpercolating_paths:
        return np.asarray(image, dtype=np.int8)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    trimmed = ps.filters.trim_nonpercolating_paths(
        np.asarray(image, dtype=bool), axis=axis_index
    )
    return trimmed.astype(np.int8)


pnm_rows: list[dict[str, object]] = []
pnm_failure_rows: list[dict[str, object]] = []

if run_pnm:
    for backend in pnm_backends:
        backend_config = pnm_backend_configs[backend]
        backend_label = str(backend_config["label"])
        for axis in flow_axes:
            start = time.perf_counter()
            try:
                axis_image = prepare_axis_image(void_roi, axis=axis)
                if not np.any(axis_image):
                    raise RuntimeError(
                        f"No {axis}-percolating void voxels after trimming"
                    )
                extract = extract_spanning_pore_network(
                    axis_image,
                    voxel_size=voxel_size_m,
                    backend=backend,
                    flow_axis=axis,
                    length_unit="m",
                    geometry_repairs=backend_config.get("geometry_repairs"),
                    extraction_kwargs=dict(backend_config.get("extraction_kwargs", {})),
                    provenance_notes={
                        "raw_source": str(raw_relpath).replace("\\", "/"),
                        "roi_origin_voxels": roi_origin,
                        "roi_shape_voxels": roi_shape,
                        "map_block_shape_voxels": map_block_shape,
                        "same_roi_comparison": True,
                        "trim_nonpercolating_paths": trim_nonpercolating_paths,
                    },
                )
                delta_p = pressure_gradient_pa_per_m * axis_lengths[axis]
                result = solve(
                    extract.net,
                    fluid=FluidSinglePhase(viscosity=mu_pa_s),
                    bc=PressureBC(
                        f"inlet_{axis}min",
                        f"outlet_{axis}max",
                        pin=delta_p,
                        pout=0.0,
                    ),
                    axis=axis,
                    options=SinglePhaseOptions(
                        conductance_model="generic_poiseuille", solver="direct"
                    ),
                )
                k_m2 = float(result.permeability[axis])
                pnm_rows.append(
                    {
                        "family": "extracted_pnm",
                        "formulation": "pore_network_model",
                        "method": backend_label,
                        "backend": backend,
                        "axis": axis,
                        "K_eq_m2": k_m2,
                        "K_eq_mD": k_m2 / M2_PER_MD,
                        "n_pores": float(extract.net.Np),
                        "n_throats": float(extract.net.Nt),
                        "mass_balance_error": float(result.mass_balance_error),
                        "solve_seconds": float(time.perf_counter() - start),
                    }
                )
            except Exception as exc:
                pnm_failure_rows.append(
                    {
                        "backend": backend,
                        "method": backend_label,
                        "axis": axis,
                        "failure": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": float(time.perf_counter() - start),
                    }
                )

pnm_df = pd.DataFrame(pnm_rows)
pnm_failure_columns = ["backend", "method", "axis", "failure", "elapsed_seconds"]
pnm_failures = pd.DataFrame(pnm_failure_rows, columns=pnm_failure_columns)
display(pnm_df)
if not pnm_failures.empty:
    display(pnm_failures)

# %% [markdown]
# ## FEniCSx FEM micro-continuum solves

# %%
fem_directional_path = output_dir / f"{output_prefix}_fenicsx_fem_directional.csv"
fem_status_path = output_dir / "drp317_berea_block3_same_roi_fem_status.json"
fem_run_specs: list[dict[str, Any]] = [
    {
        "label": "usfem_brinkman_direct",
        "solver": solve_brinkman_usfem,
        "family": "fem_micro_continuum",
        "formulation": "brinkman_usfem_p1dg1",
        "method": "Darcy-Brinkman micro-continuum USFEM CG1 x DG1",
        "solver_backends": fem_usfem_solver_backends,
    },
    {
        "label": "taylor_hood_brinkman_direct",
        "solver": solve_brinkman_taylor_hood,
        "family": "fem_micro_continuum",
        "formulation": "brinkman_taylor_hood_p2p1",
        "method": "Darcy-Brinkman coefficient-field Taylor-Hood CG2 x CG1",
        "solver_backends": fem_taylor_hood_solver_backends,
    },
    {
        "label": "taylor_hood_darcy_direct",
        "solver": solve_darcy_taylor_hood,
        "family": "fem_coefficient_field",
        "formulation": "darcy_taylor_hood_p2p1",
        "method": "Darcy-Darcy coefficient-field Taylor-Hood CG2 x CG1",
        "solver_backends": fem_taylor_hood_solver_backends,
    },
]

fem_status: dict[str, Any] = {
    "requested": bool(run_fem),
    "backend": "voids.fem.singlephase",
    "direct_solver_backends": {
        "usfem": fem_usfem_solver_backends,
        "taylor_hood": fem_taylor_hood_solver_backends,
    },
    "direct_base_options": fem_direct_base_options,
    "thread_env": {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    },
    "runs": [],
    "status": "not_requested" if not run_fem else "pending",
}

if run_fem:
    fem_problem = FEMMapProblem(
        permeability_map=permeability_map,
        porosity_map=porosity_map,
        viscosity=mu_pa_s,
        porosity_floor=fem_porosity_floor,
        permeability_floor=fem_k_floor,
    )
    fem_rows: list[dict[str, object]] = []
    try:
        for spec in fem_run_specs:
            for axis in flow_axes:
                attempts: list[dict[str, object]] = []
                last_exception: Exception | None = None
                for backend in spec["solver_backends"]:
                    solver_options = fem_direct_solver_options(backend)
                    start = time.perf_counter()
                    try:
                        result = spec["solver"](
                            fem_problem,
                            flow_axis=axis,
                            pressure_inlet=pressure_inlet_pa,
                            pressure_outlet=pressure_outlet_pa,
                            options=solver_options,
                        )
                    except Exception as exc:
                        last_exception = exc
                        attempts.append(
                            {
                                "backend": backend,
                                "wall_seconds": time.perf_counter() - start,
                                "status": "failed",
                                "message": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    wall_seconds = time.perf_counter() - start
                    attempts.append(
                        {
                            "backend": backend,
                            "wall_seconds": wall_seconds,
                            "reported_solve_seconds": result.solve_seconds,
                            "status": "ok",
                        }
                    )
                    row = {
                        "family": spec["family"],
                        "formulation": spec["formulation"],
                        "method": spec["method"],
                        "solver_backend": f"fenicsx:petsc-lu-{backend}",
                        "axis": result.flow_axis,
                        "mu_Pa_s": mu_pa_s,
                        "pressure_inlet_Pa": result.pressure_inlet,
                        "pressure_outlet_Pa": result.pressure_outlet,
                        "delta_p_Pa": result.pressure_drop,
                        "outlet_flux_m3_s": result.flow_rate,
                        "K_eq_m2": result.permeability,
                        "K_eq_mD": result.permeability / M2_PER_MD,
                        "solve_seconds": result.solve_seconds,
                        "wall_seconds": wall_seconds,
                        "solver_options_json": json.dumps(
                            solver_options.petsc_options,
                            sort_keys=True,
                        ),
                        "metadata_json": json.dumps(result.metadata, sort_keys=True),
                    }
                    field_stem = f"{output_prefix}_{spec['formulation']}_{axis}"
                    fem_velocity_path = output_dir / f"{field_stem}_velocity.xdmf"
                    fem_pressure_path = output_dir / f"{field_stem}_pressure.xdmf"
                    write_dolfinx_function_xdmf(
                        result.velocity,
                        fem_velocity_path,
                        name=f"{spec['formulation']}_{axis}_velocity",
                    )
                    write_dolfinx_function_xdmf(
                        result.pressure,
                        fem_pressure_path,
                        name=f"{spec['formulation']}_{axis}_pressure",
                    )
                    _record_field_output(
                        family=str(spec["family"]),
                        formulation=str(spec["formulation"]),
                        method=str(spec["method"]),
                        axis=axis,
                        field="velocity",
                        kind="paraview_xdmf",
                        path=fem_velocity_path,
                    )
                    _record_field_output(
                        family=str(spec["family"]),
                        formulation=str(spec["formulation"]),
                        method=str(spec["method"]),
                        axis=axis,
                        field="pressure",
                        kind="paraview_xdmf",
                        path=fem_pressure_path,
                    )

                    sampled_pressure = sample_dolfinx_function_on_grid(
                        result.pressure,
                        shape=porosity_map.shape,
                        cell_size=porosity_map.cell_size,
                        origin=porosity_map.origin,
                    )
                    sampled_velocity = sample_dolfinx_function_on_grid(
                        result.velocity,
                        shape=porosity_map.shape,
                        cell_size=porosity_map.cell_size,
                        origin=porosity_map.origin,
                    )
                    pressure_plot_path = (
                        output_dir / f"{field_stem}_pressure_midplanes.png"
                    )
                    plotted_pressure = reference_pressure_to_outlet(
                        sampled_pressure,
                        flow_axis=axis,
                        reference_pressure=reference_pressure_pa,
                        pressure_outlet=pressure_outlet_pa,
                    )
                    pressure_fig = plot_scalar_midplanes(
                        plotted_pressure,
                        title=f"{sample_name} {spec['method']} pressure, flow {axis}",
                        path=pressure_plot_path,
                        colorbar_label="pressure [Pa]",
                        colorbar_use_offset=False,
                    )
                    plt.close(pressure_fig)
                    _record_field_output(
                        family=str(spec["family"]),
                        formulation=str(spec["formulation"]),
                        method=str(spec["method"]),
                        axis=axis,
                        field="pressure",
                        kind="midplane_png",
                        path=pressure_plot_path,
                    )

                    velocity_plot_path = (
                        output_dir / f"{field_stem}_velocity_midplanes.png"
                    )
                    velocity_fig = plot_vector_midplanes(
                        sampled_velocity,
                        title=f"{sample_name} {spec['method']} velocity, flow {axis}",
                        path=velocity_plot_path,
                        quiver_stride=3,
                        colorbar_label="velocity magnitude [m/s]",
                    )
                    plt.close(velocity_fig)
                    _record_field_output(
                        family=str(spec["family"]),
                        formulation=str(spec["formulation"]),
                        method=str(spec["method"]),
                        axis=axis,
                        field="velocity",
                        kind="midplane_quiver_png",
                        path=velocity_plot_path,
                    )
                    fem_rows.append(row)
                    fem_status["runs"].append(
                        {
                            "label": spec["label"],
                            "axis": axis,
                            "attempts": attempts,
                            "status": "ok",
                        }
                    )
                    pd.DataFrame(fem_rows).to_csv(fem_directional_path, index=False)
                    fem_status_path.write_text(
                        json.dumps(fem_status, indent=2),
                        encoding="utf-8",
                    )
                    break
                else:
                    fem_status["runs"].append(
                        {
                            "label": spec["label"],
                            "axis": axis,
                            "attempts": attempts,
                            "status": "failed",
                        }
                    )
                    raise RuntimeError(
                        f"All direct FEM solver backends failed for {spec['label']} axis {axis}"
                    ) from last_exception
        fem_status["status"] = "ok"
    except Exception as exc:
        fem_status["status"] = "failed"
        fem_status["message"] = f"{type(exc).__name__}: {exc}"
        fem_status_path.write_text(json.dumps(fem_status, indent=2), encoding="utf-8")
        raise
    fem_df = pd.DataFrame(fem_rows)
    fem_df.to_csv(fem_directional_path, index=False)
elif fem_directional_path.exists():
    fem_df = pd.read_csv(fem_directional_path)
    fem_status["status"] = "loaded_existing"
else:
    fem_df = pd.DataFrame()
    fem_status["status"] = "missing"

display(fem_status)
display(fem_df)

# %% [markdown]
# ## Direct-image LBM DNS
#
# This row solves the binary ROI directly with XLB. It does not use the
# Kozeny-Carman porosity/permeability map, so it is a DNS-style reference row
# rather than another map-based Darcy or Brinkman discretization.

# %%
lbm_status: dict[str, Any] = {
    "requested": bool(run_lbm),
    "status": "not_requested" if not run_lbm else "pending",
    "options": {
        "formulation": lbm_options.formulation,
        "backend": lbm_options.backend,
        "precision_policy": lbm_options.precision_policy,
        "collision_model": lbm_options.collision_model,
        "streaming_scheme": lbm_options.streaming_scheme,
        "lattice_viscosity": lbm_options.lattice_viscosity,
        "pressure_drop_lattice": lbm_options.pressure_drop_lattice,
        "inlet_outlet_buffer_cells": lbm_options.inlet_outlet_buffer_cells,
        "max_steps": lbm_options.max_steps,
        "min_steps": lbm_options.min_steps,
        "check_interval": lbm_options.check_interval,
        "steady_rtol": lbm_options.steady_rtol,
    },
}

if run_lbm:
    lbm_rows: list[dict[str, object]] = []
    lbm_status["status"] = "ok"
    for axis in flow_axes:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = solve_binary_volume_with_xlb(
                np.asarray(void_roi, dtype=int),
                voxel_size=voxel_size_m,
                flow_axis=axis,
                options=lbm_options,
            )
        lbm_rows.append(
            {
                "family": "direct_image_dns",
                "formulation": f"xlb_{result.formulation}",
                "method": "Direct-image LBM DNS (XLB, Stokes-limit preset)",
                "solver_backend": f"xlb:{result.backend}",
                "axis": axis,
                "K_m2": result.permeability,
                "K_mD": result.permeability / M2_PER_MD,
                "solve_seconds": float(time.perf_counter() - start),
                "xlb_steps": result.n_steps,
                "xlb_converged": result.converged,
                "xlb_convergence_metric": result.convergence_metric,
                "xlb_mach_max": result.max_mach_lattice,
                "xlb_re_voxel_max": result.reynolds_voxel_max,
                "xlb_lattice_viscosity": result.lattice_viscosity,
                "xlb_pressure_drop_lattice": result.lattice_pressure_drop,
                "warning_count": len(caught),
                "warnings": "; ".join(str(item.message) for item in caught),
            }
        )
        lbm_vtu_path = output_dir / f"{output_prefix}_xlb_lbm_velocity_{axis}.vtu"
        write_structured_vector_field(
            result.velocity_lattice,
            voxel_porosity_map,
            lbm_vtu_path,
            extra_cell_data={"axial_velocity_lattice": result.axial_velocity_lattice},
        )
        _record_field_output(
            family="direct_image_dns",
            formulation=f"xlb_{result.formulation}",
            method="Direct-image LBM DNS (XLB, Stokes-limit preset)",
            axis=axis,
            field="velocity",
            kind="paraview_vtu",
            path=lbm_vtu_path,
        )

        velocity_plot_path = (
            output_dir / f"{output_prefix}_xlb_lbm_velocity_midplanes_{axis}.png"
        )
        velocity_fig = plot_vector_midplanes(
            result.velocity_lattice,
            title=f"{sample_name} XLB/LBM velocity, flow {axis}",
            path=velocity_plot_path,
            quiver_stride=8,
            colorbar_label="velocity magnitude [lattice units]",
        )
        plt.close(velocity_fig)
        _record_field_output(
            family="direct_image_dns",
            formulation=f"xlb_{result.formulation}",
            method="Direct-image LBM DNS (XLB, Stokes-limit preset)",
            axis=axis,
            field="velocity",
            kind="midplane_quiver_png",
            path=velocity_plot_path,
        )
    lbm_df = pd.DataFrame(lbm_rows)
    lbm_df.to_csv(lbm_directional_path, index=False)
    lbm_status_path.write_text(json.dumps(lbm_status, indent=2), encoding="utf-8")
elif lbm_directional_path.exists():
    lbm_df = pd.read_csv(lbm_directional_path)
    lbm_status["status"] = "loaded_existing"
else:
    lbm_df = pd.DataFrame()
    lbm_status["status"] = "missing"

display(lbm_df)

# %% [markdown]
# ## Compare all rows from the same ROI

# %%
comparison_rows: list[dict[str, object]] = []


for row in resistor_df.to_dict(orient="records"):
    comparison_rows.append(
        {
            "family": row["family"],
            "formulation": row["formulation"],
            "method": row["method"],
            "solver_backend": row.get("solver_backend", ""),
            "axis": row["axis"],
            "K_m2": float(row["K_eq_m2"]),
            "K_mD": float(row["K_eq_mD"]),
            "solve_seconds": float(row["solve_seconds"]),
        }
    )

if not fem_df.empty:
    for row in fem_df.to_dict(orient="records"):
        comparison_rows.append(
            {
                "family": row["family"],
                "formulation": row["formulation"],
                "method": row["method"],
                "solver_backend": row.get("solver_backend", ""),
                "axis": row["axis"],
                "K_m2": float(row["K_eq_m2"]),
                "K_mD": float(row["K_eq_mD"]),
                "solve_seconds": float(row["solve_seconds"]),
            }
        )

if not lbm_df.empty:
    for row in lbm_df.to_dict(orient="records"):
        comparison_rows.append(
            {
                "family": row["family"],
                "formulation": row["formulation"],
                "method": row["method"],
                "solver_backend": row.get("solver_backend", ""),
                "axis": row["axis"],
                "K_m2": float(row["K_m2"]),
                "K_mD": float(row["K_mD"]),
                "solve_seconds": float(row["solve_seconds"]),
            }
        )

if not pnm_df.empty:
    for row in pnm_df.to_dict(orient="records"):
        comparison_rows.append(
            {
                "family": row["family"],
                "formulation": row["formulation"],
                "method": row["method"],
                "solver_backend": "",
                "axis": row["axis"],
                "K_m2": float(row["K_eq_m2"]),
                "K_mD": float(row["K_eq_mD"]),
                "solve_seconds": float(row["solve_seconds"]),
            }
        )

for axis in flow_axes:
    comparison_rows.append(
        {
            "family": "experimental",
            "formulation": "bulk_experiment",
            "method": "Experimental Kabs",
            "solver_backend": "",
            "axis": axis,
            "K_m2": experimental_kabs_mD * M2_PER_MD,
            "K_mD": experimental_kabs_mD,
            "solve_seconds": np.nan,
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
ratio_df = comparison_df.assign(
    K_experimental_mD=experimental_kabs_mD,
    method_over_experiment=comparison_df["K_mD"] / experimental_kabs_mD,
    experiment_over_method=experimental_kabs_mD / comparison_df["K_mD"],
)
display(comparison_df.sort_values(["axis", "family", "method"]))
display(ratio_df.sort_values(["axis", "family", "method"]))

# %%
axis_order = ["x", "y", "z"]
method_order = [
    "Experimental Kabs",
    "Direct-image LBM DNS (XLB, Stokes-limit preset)",
    "Darcy-Brinkman micro-continuum USFEM CG1 x DG1",
    "Darcy-Brinkman coefficient-field Taylor-Hood CG2 x CG1",
    "Darcy-Darcy coefficient-field Taylor-Hood CG2 x CG1",
    "TPFA finite-volume Darcy-Darcy",
    "PoreSpy snow2",
    "PREGO",
    "Native maximal-ball",
]
available_methods = [
    method for method in method_order if method in set(comparison_df["method"])
]

# %% [markdown]
# ### Bulk scalar summaries
#
# The experimental reference is reported as a scalar bulk permeability. The
# directional simulations estimate \(K_x\), \(K_y\), and \(K_z\) on a small,
# anisotropic ROI. The arithmetic and harmonic means below are therefore scalar
# summaries of the directional values, not replacements for the directional
# permeability tensor:
#
# $$
# K_\mathrm{arith} = \frac{K_x + K_y + K_z}{3},
# \qquad
# K_\mathrm{harm} = \frac{3}{1/K_x + 1/K_y + 1/K_z}.
# $$

# %%
bulk_records: list[dict[str, Any]] = []
for method in available_methods:
    if method == "Experimental Kabs":
        continue
    subset = comparison_df[comparison_df["method"] == method].set_index("axis")
    if not all(axis in subset.index for axis in axis_order):
        continue
    values = np.asarray(
        [float(subset.loc[axis, "K_mD"]) for axis in axis_order], dtype=float
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        continue
    first = subset.loc[axis_order[0]]
    arithmetic_mean_mD = float(np.mean(values))
    harmonic_mean_mD = float(len(values) / np.sum(1.0 / values))
    bulk_records.append(
        {
            "family": str(first["family"]),
            "formulation": str(first["formulation"]),
            "method": method,
            "solver_backend": (
                "" if pd.isna(first["solver_backend"]) else str(first["solver_backend"])
            ),
            "K_arithmetic_mean_mD": arithmetic_mean_mD,
            "K_harmonic_mean_mD": harmonic_mean_mD,
            "K_arithmetic_over_experiment": arithmetic_mean_mD / experimental_kabs_mD,
            "K_harmonic_over_experiment": harmonic_mean_mD / experimental_kabs_mD,
            "K_max_over_min": float(np.max(values) / np.min(values)),
        }
    )

bulk_summary_df = pd.DataFrame(bulk_records)
bulk_summary_df = (
    bulk_summary_df.set_index("method")
    .reindex([method for method in available_methods if method != "Experimental Kabs"])
    .dropna(how="all")
    .reset_index()
)
display(bulk_summary_df)

fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
y_positions = np.arange(len(bulk_summary_df), dtype=float)
bar_height = 0.36
ax.barh(
    y_positions - 0.5 * bar_height,
    bulk_summary_df["K_arithmetic_mean_mD"],
    height=bar_height,
    label=r"arithmetic mean of $K_i$",
)
ax.barh(
    y_positions + 0.5 * bar_height,
    bulk_summary_df["K_harmonic_mean_mD"],
    height=bar_height,
    label=r"harmonic mean of $K_i$",
)
ax.axvline(
    experimental_kabs_mD,
    color="black",
    linestyle="--",
    linewidth=1.4,
    label=f"experimental bulk ({experimental_kabs_mD:.0f} mD)",
)
ax.set_xscale("log")
ax.set_yticks(y_positions)
ax.set_yticklabels(bulk_summary_df["method"])
ax.invert_yaxis()
ax.set_xlabel("scalar permeability summary [mD]")
ax.set_title("DRP-317 Berea scalar summaries of directional permeability")
ax.grid(True, axis="x", which="both", alpha=0.25)
ax.legend(fontsize=8, loc="lower right")

bulk_plot_path = output_dir / f"{output_prefix}_bulk_permeability_means.png"
fig.savefig(bulk_plot_path, dpi=200)
display(fig)
bulk_plot_path

# %%
fig, ax = plt.subplots(figsize=(13, 5.4), constrained_layout=True)
bar_width = 0.82 / max(len(available_methods), 1)
x_positions = np.arange(len(axis_order), dtype=float)

for method_index, method in enumerate(available_methods):
    subset = comparison_df[comparison_df["method"] == method]
    values = []
    for axis in axis_order:
        match = subset[subset["axis"] == axis]
        values.append(float(match.iloc[0]["K_mD"]) if not match.empty else np.nan)
    offset = (method_index - 0.5 * (len(available_methods) - 1)) * bar_width
    ax.bar(x_positions + offset, values, width=bar_width, label=method)

ax.set_yscale("log")
ax.set_xticks(x_positions)
ax.set_xticklabels([r"$K_x$", r"$K_y$", r"$K_z$"])
ax.set_ylabel("equivalent permeability [mD]")
ax.set_title("DRP-317 Berea block-3 same-ROI comparison")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

comparison_plot_path = output_dir / "drp317_berea_block3_same_roi_model_comparison.png"
fig.savefig(comparison_plot_path, dpi=200)
display(fig)
comparison_plot_path

# %%
timing_df = comparison_df[np.isfinite(comparison_df["solve_seconds"])].copy()
available_timing_methods = [
    method for method in method_order if method in set(timing_df["method"])
]

fig, ax = plt.subplots(figsize=(13, 5.4), constrained_layout=True)
bar_width = 0.82 / max(len(available_timing_methods), 1)

for method_index, method in enumerate(available_timing_methods):
    subset = timing_df[timing_df["method"] == method]
    values = []
    for axis in axis_order:
        match = subset[subset["axis"] == axis]
        values.append(
            float(match.iloc[0]["solve_seconds"]) if not match.empty else np.nan
        )
    offset = (method_index - 0.5 * (len(available_timing_methods) - 1)) * bar_width
    ax.bar(x_positions + offset, values, width=bar_width, label=method)

ax.set_yscale("log")
ax.set_xticks(x_positions)
ax.set_xticklabels([r"$K_x$", r"$K_y$", r"$K_z$"])
ax.set_ylabel("solve time [s]")
ax.set_title("DRP-317 Berea block-3 same-ROI solver wall time by axis")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

time_plot_path = output_dir / "drp317_berea_block3_same_roi_model_solve_time.png"
fig.savefig(time_plot_path, dpi=200)
display(fig)
time_plot_path

# %%
heatmap_df = (
    comparison_df.pivot_table(
        index="method", columns="axis", values="K_mD", aggfunc="first"
    )
    .reindex(available_methods)
    .reindex(columns=axis_order)
)
with np.errstate(divide="ignore", invalid="ignore"):
    heatmap_values = np.log10(heatmap_df.to_numpy(dtype=float))

fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
finite_values = heatmap_values[np.isfinite(heatmap_values)]
image = ax.imshow(
    heatmap_values,
    cmap="viridis",
    aspect="auto",
    vmin=float(np.min(finite_values)),
    vmax=float(np.max(finite_values)),
)
ax.set_xticks(np.arange(len(axis_order)))
ax.set_xticklabels([r"$K_x$", r"$K_y$", r"$K_z$"])
ax.set_yticks(np.arange(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index)
ax.set_title("DRP-317 Berea block-3 same-ROI equivalent permeability")
fig.colorbar(image, ax=ax, label=r"$\log_{10}(K\,[\mathrm{mD}])$")

for row_index, method in enumerate(heatmap_df.index):
    for col_index, axis in enumerate(axis_order):
        value = heatmap_df.loc[method, axis]
        if not np.isfinite(value):
            continue
        ax.text(
            col_index,
            row_index,
            f"{value:.2g}",
            ha="center",
            va="center",
            color=(
                "white"
                if heatmap_values[row_index, col_index] < np.median(finite_values)
                else "black"
            ),
            fontsize=9,
        )

comparison_heatmap_path = (
    output_dir / "drp317_berea_block3_same_roi_model_comparison_heatmap.png"
)
fig.savefig(comparison_heatmap_path, dpi=200)
display(fig)
comparison_heatmap_path

# %% [markdown]
# ## Save tables

# %%
roi_scan_path = output_dir / "drp317_berea_block3_same_roi_scan.csv"
roi_summary_path = output_dir / "drp317_berea_block3_same_roi_summary.csv"
phase_summary_path = output_dir / "drp317_berea_block3_same_roi_phase_summary.csv"
map_summary_path = output_dir / "drp317_berea_block3_same_roi_map_summary.csv"
resistor_path = output_dir / "drp317_berea_block3_same_roi_map_resistor_directional.csv"
pnm_path = output_dir / "drp317_berea_block3_same_roi_pnm_directional.csv"
pnm_failure_path = output_dir / "drp317_berea_block3_same_roi_pnm_failures.csv"
comparison_path = output_dir / "drp317_berea_block3_same_roi_model_comparison.csv"
ratio_path = output_dir / "drp317_berea_block3_same_roi_model_ratios_to_experiment.csv"
bulk_summary_path = output_dir / f"{output_prefix}_bulk_permeability_means.csv"
field_outputs_path = output_dir / f"{output_prefix}_field_outputs.csv"
roi_scan.to_csv(roi_scan_path, index=False)
roi_summary.to_csv(roi_summary_path, index=False)
phase_summary.to_csv(phase_summary_path, index=False)
map_summary.to_csv(map_summary_path, index=False)
resistor_df.to_csv(resistor_path, index=False)
pnm_df.to_csv(pnm_path, index=False)
pnm_failures.to_csv(pnm_failure_path, index=False)
comparison_df.to_csv(comparison_path, index=False)
ratio_df.to_csv(ratio_path, index=False)
bulk_summary_df.to_csv(bulk_summary_path, index=False)
field_outputs_df = pd.DataFrame(field_output_records)
field_outputs_df.to_csv(field_outputs_path, index=False)
fem_status_path.write_text(json.dumps(fem_status, indent=2), encoding="utf-8")
lbm_status_path.write_text(json.dumps(lbm_status, indent=2), encoding="utf-8")

saved_paths = [
    path
    for path in [
        porosity_h5,
        permeability_h5,
        roi_scan_path,
        roi_summary_path,
        phase_summary_path,
        map_summary_path,
        resistor_path,
        pnm_path,
        pnm_failure_path,
        comparison_path,
        ratio_path,
        bulk_summary_path,
        fem_status_path,
        lbm_directional_path,
        lbm_status_path,
        map_figure_path,
        comparison_plot_path,
        bulk_plot_path,
        time_plot_path,
        comparison_heatmap_path,
        field_outputs_path,
        fem_directional_path,
    ]
    if path.exists()
]
for record in field_output_records:
    field_path = Path(str(record["path"]))
    if field_path.exists():
        saved_paths.append(field_path)

for axis, result in resistor_results.items():
    pressure_path = (
        output_dir / f"drp317_berea_block3_same_roi_map_resistor_pressure_{axis}.npy"
    )
    np.save(pressure_path, result.pressure)
    saved_paths.append(pressure_path)

pd.DataFrame({"saved_path": [str(path) for path in saved_paths]})

# %% [markdown]
# ## Interpretation notes
#
# - All model rows in this notebook use the same cropped binary ROI.
# - The `(3, 3, 3)` block maps use a Kozeny-Carman closure with characteristic
#   length equal to the block edge, `6.75 um`.
# - The direct-image LBM DNS row uses the binary ROI, pressure reservoirs, and
#   bounce-back solids. It does not use the Kozeny-Carman closure field.
# - TPFA and LBM full velocity fields are exported as regular-grid VTU files.
#   FEM pressure and velocity fields are exported as XDMF/HDF5 files. PNM solves
#   are graph-valued and do not produce a full volumetric velocity field.
# - The experimental permeability is a bulk scalar for Berea and is repeated
#   over the three axes only as a visual reference.
# - Because the ROI is small, agreement or disagreement with the bulk experiment
#   should be read as a model/ROI sensitivity check, not a sample-scale
#   validation.
