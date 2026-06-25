# %% [markdown]
# # MWE 38 - Trabecular bone 3-D ROI map and FEM comparison
#
# This notebook compares three model families on the same 3-D ROI:
#
# - extracted pore-network models from
#   `37_mwe_trabecular_bone_slice_pore_network`
# - a cell-centered resistor solve on the Kozeny-Carman permeability map
# - direct-image lattice-Boltzmann DNS on the binary ROI using XLB
# - FEniCSx USFEM Darcy-Brinkman micro-continuum solves from `voids.fem`
#
# Scientific scope and assumptions:
#
# - the binary image defines marrow/pore space as `0` and bone/solid as `1`
# - the porosity map is a 3-D block average of the binary ROI
# - the permeability map is a Kozeny-Carman closure field computed from that
#   porosity map, not a direct pore-scale permeability measurement
# - the Darcy-Brinkman FEM row is the closest micro-continuum model here because
#   it uses both the porosity map and the permeability map

# %%
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

FEM_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for name, value in FEM_THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(name, value)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.image.porosity import (
    load_permeability_map_hdf5,
    load_porosity_map_hdf5,
    permeability_map_from_porosity,
    porosity_map_from_binary,
    save_permeability_map_hdf5,
    save_porosity_map_hdf5,
)
from voids.fem.singlephase import (
    FEMMapProblem,
    FEniCSSolverOptions,
    solve_brinkman_usfem,
)
from voids.fvm.singlephase import solve_tpfa
from voids.lbm.singlephase import XLBOptions, solve_binary_volume_with_xlb
from voids.paths import data_path, project_root

plt.ioff()

# %%
# User-editable inputs
raw_filename = "Trabecular_300_cubo_0_poro_1_osso_2086nm.raw"
raw_shape = (350, 350, 349)
raw_dtype = np.uint8
raw_order = "C"

bone_value = 1
marrow_value = 0
axis_labels = ("x", "y", "z")

roi_shape = (100, 100, 100)
roi_start: tuple[int, int, int] | None = None

voxel_size_nm = 2086.0
voxel_size_m = voxel_size_nm * 1.0e-9

# FEM and resistor coefficient maps use this coarsening. The default 10^3 map
# keeps the 3-D micro-continuum solves light enough for laptop comparisons.
map_block_shape = (10, 10, 10)

# Kozeny-Carman closure parameters. These are model assumptions and should be
# varied in sensitivity studies.
kozeny_constant = 180.0
solid_permeability_m2 = 1.0e-20
free_flow_permeability_m2 = 1.0e-8
max_permeability_m2 = 1.0e-8

mu_pa_s = 3.0e-3
pressure_inlet_pa = 1.0
pressure_outlet_pa = 0.0
flow_axes = ("x", "y", "z")
resistor_solver_method = "cg"
resistor_solver_parameters: dict[str, Any] = {
    "rtol": 1.0e-10,
    "atol": 0.0,
    "maxiter": 2000,
    "preconditioner": "pyamg",
    "pyamg_solver": "smoothed_aggregation",
}

run_fem = True
fem_porosity_floor = 1.0e-3
fem_k_floor = 1.0e-20
fem_solver_backend = "superlu_dist"
fem_direct_options = FEniCSSolverOptions.direct_lu(fem_solver_backend).petsc_options

# XLB solves the binary ROI directly. These settings keep the run in a
# low-Mach, low-voxel-Reynolds regime for comparison with creeping-flow models.
run_lbm = True
lbm_options = XLBOptions.steady_stokes_defaults(
    max_steps=3000,
    min_steps=500,
    check_interval=100,
    steady_rtol=1.0e-3,
    inlet_outlet_buffer_cells=6,
)

save_outputs = True
output_dir = (
    project_root()
    / "notebooks"
    / "outputs"
    / ("38_mwe_trabecular_bone_map_resistor_upscaling")
)
output_dir.mkdir(parents=True, exist_ok=True)

pnm_output_dir = (
    project_root()
    / "notebooks"
    / "outputs"
    / ("37_mwe_trabecular_bone_slice_pore_network")
)
pnm_directional_path = pnm_output_dir / "trabecular_bone_roi_kabs_directional.csv"

output_prefix = "trabecular_bone_roi"
lbm_directional_path = output_dir / f"{output_prefix}_xlb_lbm_directional.csv"
lbm_status_path = output_dir / f"{output_prefix}_xlb_lbm_status.json"

M2_PER_MD = 9.869233e-16


def _candidate_raw_paths() -> list[Path]:
    trabecular_root = data_path() / "trabecular-image"
    return [
        trabecular_root
        / "36_mwe_trabecular_bone_slice_porosity_permeability_maps"
        / raw_filename,
        trabecular_root / raw_filename,
    ]


candidate_raw_paths = _candidate_raw_paths()
raw_path = next(
    (path for path in candidate_raw_paths if path.exists()), candidate_raw_paths[0]
)
raw_path

# %% [markdown]
# ## Load the same 3-D ROI used by notebook 37

# %%
if not raw_path.exists():
    raise FileNotFoundError(
        "Could not find the trabecular RAW volume. Checked:\n"
        + "\n".join(f"- {path}" for path in candidate_raw_paths)
    )

expected_voxels = int(np.prod(np.asarray(raw_shape, dtype=np.int64)))
actual_voxels = raw_path.stat().st_size // np.dtype(raw_dtype).itemsize
if expected_voxels != actual_voxels:
    raise ValueError(
        f"Configured shape {raw_shape} requires {expected_voxels:,} voxels, "
        f"but {raw_path.name} stores {actual_voxels:,}."
    )

roi_shape = tuple(int(v) for v in roi_shape)
if len(roi_shape) != 3 or any(v <= 0 for v in roi_shape):
    raise ValueError("roi_shape must contain three positive integers")
if any(r > s for r, s in zip(roi_shape, raw_shape, strict=True)):
    raise ValueError(f"roi_shape {roi_shape} must fit inside raw_shape {raw_shape}")
if any(s % b != 0 for s, b in zip(roi_shape, map_block_shape, strict=True)):
    raise ValueError(
        f"roi_shape {roi_shape} must be divisible by map_block_shape {map_block_shape}"
    )

if roi_start is None:
    resolved_roi_start = tuple(
        (s - r) // 2 for s, r in zip(raw_shape, roi_shape, strict=True)
    )
else:
    resolved_roi_start = tuple(int(v) for v in roi_start)
roi_stop = tuple(a + b for a, b in zip(resolved_roi_start, roi_shape, strict=True))
if any(a < 0 for a in resolved_roi_start) or any(
    b > s for b, s in zip(roi_stop, raw_shape, strict=True)
):
    raise ValueError(
        f"ROI start {resolved_roi_start} and shape {roi_shape} must stay inside raw_shape {raw_shape}"
    )

raw_image = np.memmap(
    raw_path, dtype=raw_dtype, mode="r", shape=raw_shape, order=raw_order
)
roi_slices = tuple(
    slice(a, b) for a, b in zip(resolved_roi_start, roi_stop, strict=True)
)
roi_image = np.asarray(raw_image[roi_slices])
values, counts = np.unique(roi_image, return_counts=True)
unexpected_values = set(values.tolist()) - {marrow_value, bone_value}
if unexpected_values:
    raise ValueError(
        "This notebook expects a binary segmentation with only the configured "
        f"marrow/bone values {marrow_value!r}/{bone_value!r}; got {sorted(values.tolist())}."
    )

marrow_roi = np.asarray(roi_image == marrow_value, dtype=bool)
phase_summary = pd.DataFrame(
    {
        "raw_value": values.astype(int),
        "voxel_count": counts.astype(np.int64),
        "fraction": counts / counts.sum(),
        "phase": [
            "marrow/pore" if int(value) == marrow_value else "bone/solid"
            for value in values
        ],
    }
)

roi_summary = pd.DataFrame(
    [
        {"quantity": "source RAW path", "value": str(raw_path), "units": "-"},
        {"quantity": "ROI start", "value": str(resolved_roi_start), "units": "voxels"},
        {"quantity": "ROI shape", "value": str(roi_shape), "units": "voxels"},
        {"quantity": "voxel size", "value": voxel_size_m, "units": "m"},
        {
            "quantity": "ROI pore fraction",
            "value": float(marrow_roi.mean()),
            "units": "-",
        },
        {
            "quantity": "map block shape",
            "value": str(map_block_shape),
            "units": "voxels",
        },
    ]
)

display(phase_summary)
display(roi_summary)

# %% [markdown]
# ## Build 3-D porosity and permeability maps
#
# The porosity map is a block average of the segmented marrow/pore phase. The
# permeability map is then computed from that porosity field with the configured
# Kozeny-Carman closure.

# %%
porosity_map = porosity_map_from_binary(
    marrow_roi,
    block_shape=map_block_shape,
    voxel_size=(voxel_size_m, voxel_size_m, voxel_size_m),
    strict=True,
    metadata={
        "case": "trabecular_bone_3d_roi",
        "raw_filename": raw_filename,
        "raw_shape": raw_shape,
        "raw_order": raw_order,
        "roi_start": resolved_roi_start,
        "roi_shape": roi_shape,
        "phase_convention": "0=marrow_or_pore, 1=bone",
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
        "closure_note": "Kozeny-Carman coefficient map for 3-D ROI micro-continuum comparison",
        "scientific_caveat": "closure field, not a direct image-resolved permeability solve",
    },
)

porosity_h5 = output_dir / "trabecular_bone_roi_porosity_map.h5"
permeability_h5 = output_dir / "trabecular_bone_roi_permeability_map.h5"
if save_outputs:
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
for col, (title, poro_slice, perm_slice) in enumerate(map_slice_specs):
    im0 = axes[0, col].imshow(
        poro_slice.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    axes[0, col].set_title(f"{title} porosity")
    fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

    with np.errstate(divide="ignore"):
        log_perm = np.log10(perm_slice)
    im1 = axes[1, col].imshow(
        log_perm.T,
        origin="lower",
        vmin=float(np.min(finite_log)),
        vmax=float(np.max(finite_log)),
        cmap="magma",
    )
    axes[1, col].set_title(f"{title} log10 K")
    fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

map_figure_path = output_dir / "trabecular_bone_roi_porosity_permeability_midplanes.png"
if save_outputs:
    fig.savefig(map_figure_path, dpi=180)
map_figure_path

# %% [markdown]
# ## Cell-centered 3-D TPFA Darcy-Darcy solve
#
# This finite-volume Darcy upscaling solve uses the same `K` field as the FEM
# runs, but it does not include the Brinkman viscous diffusion term.

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

# %% [markdown]
# ## FEniCSx USFEM micro-continuum solves
#
# The USFEM backend in `voids.fem.singlephase` consumes the porosity and
# permeability maps generated above and solves the Darcy-Brinkman
# micro-continuum formulation through the public `voids` API.

# %%
fem_output_dir = output_dir / "fenicsx_usfem_micro_continuum"
fem_output_dir.mkdir(parents=True, exist_ok=True)
fem_directional_path = fem_output_dir / f"{output_prefix}_fenicsx_usfem_directional.csv"
fem_directional_paths = [fem_directional_path]
fem_status_path = output_dir / "trabecular_bone_roi_fem_status.json"

fem_status: dict[str, Any] = {
    "requested": bool(run_fem),
    "backend": "voids.fem.singlephase.solve_brinkman_usfem",
    "solver_backend": fem_solver_backend,
    "direct_options": fem_direct_options,
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
    fem_options = FEniCSSolverOptions(petsc_options=fem_direct_options)
    fem_rows: list[dict[str, object]] = []
    fem_status["status"] = "ok"
    for axis in flow_axes:
        start = time.perf_counter()
        result = solve_brinkman_usfem(
            fem_problem,
            flow_axis=axis,
            pressure_inlet=pressure_inlet_pa,
            pressure_outlet=pressure_outlet_pa,
            options=fem_options,
        )
        wall_seconds = time.perf_counter() - start
        fem_status["runs"].append(
            {
                "axis": axis,
                "status": "ok",
                "solve_seconds": result.solve_seconds,
                "wall_seconds": wall_seconds,
            }
        )
        fem_rows.append(
            {
                "family": "fem_micro_continuum",
                "formulation": result.formulation,
                "method": result.method,
                "solver_backend": f"fenicsx:petsc-lu-{fem_solver_backend}",
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
                "solver_options_json": json.dumps(fem_direct_options, sort_keys=True),
                "metadata_json": json.dumps(result.metadata, sort_keys=True),
            }
        )
    fem_df = pd.DataFrame(fem_rows)
    if save_outputs:
        fem_df.to_csv(fem_directional_path, index=False)
        fem_status_path.write_text(
            json.dumps(fem_status, indent=2),
            encoding="utf-8",
        )
elif fem_directional_path.exists():
    fem_df = pd.read_csv(fem_directional_path)
    fem_status["status"] = "loaded_existing"
else:
    fem_df = pd.DataFrame()

fem_status

# %%
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
                np.asarray(marrow_roi, dtype=int),
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
    lbm_df = pd.DataFrame(lbm_rows)
    if save_outputs:
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
# ## Compare PNM, Darcy-Darcy, FEM, and direct-image DNS rows

# %%
comparison_rows: list[dict[str, object]] = []


for row in resistor_df.to_dict(orient="records"):
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

if pnm_directional_path.exists():
    pnm_directional = pd.read_csv(pnm_directional_path)
    for row in pnm_directional.to_dict(orient="records"):
        comparison_rows.append(
            {
                "family": "extracted_pnm",
                "formulation": "pore_network_model",
                "method": row["backend_label"],
                "solver_backend": "",
                "axis": row["axis"],
                "K_m2": float(row["k_m2"]),
                "K_mD": float(row["k_mD"]),
                "solve_seconds": float(row["solve_seconds"]),
            }
        )
else:
    pnm_directional = pd.DataFrame()
    print(f"PNM directional summary not found: {pnm_directional_path}")

comparison_df = pd.DataFrame(comparison_rows)
display(comparison_df.sort_values(["axis", "family", "method"]))

# %%
if not comparison_df.empty:
    ratio_rows: list[dict[str, object]] = []
    reference = comparison_df[comparison_df["formulation"].eq("brinkman_usfem_p1dg1")]
    for _, row in comparison_df.iterrows():
        axis = str(row["axis"])
        ref_match = reference[reference["axis"].eq(axis)]
        if ref_match.empty:
            continue
        k_ref = float(ref_match.iloc[0]["K_m2"])
        k_row = float(row["K_m2"])
        ratio_rows.append(
            {
                "axis": axis,
                "method": row["method"],
                "family": row["family"],
                "K_method_m2": k_row,
                "K_brinkman_usfem_m2": k_ref,
                "method_over_brinkman_usfem": k_row / k_ref,
                "brinkman_usfem_over_method": k_ref / k_row,
            }
        )
    ratio_df = pd.DataFrame(ratio_rows)
else:
    ratio_df = pd.DataFrame()

display(
    ratio_df.sort_values(["axis", "family", "method"])
    if not ratio_df.empty
    else ratio_df
)

# %% [markdown]
# ## Visual diagnostics

# %%
plot_df = comparison_df.copy()
axis_order = ["x", "y", "z"]
method_order = [
    "Darcy-Brinkman USFEM CG1 x DG1",
    "Direct-image LBM DNS (XLB, Stokes-limit preset)",
    "TPFA finite-volume Darcy-Darcy",
    "Native maximal-ball",
    "PoreSpy snow2",
    "PREGO",
]
available_methods = [
    method for method in method_order if method in set(plot_df["method"])
]

fig, ax = plt.subplots(figsize=(13, 5.4), constrained_layout=True)
bar_width = 0.82 / max(len(available_methods), 1)
x_positions = np.arange(len(axis_order), dtype=float)

for method_index, method in enumerate(available_methods):
    subset = plot_df[plot_df["method"] == method]
    values = []
    for axis in axis_order:
        match = subset[subset["axis"] == axis]
        values.append(float(match.iloc[0]["K_m2"]) if not match.empty else np.nan)
    offset = (method_index - 0.5 * (len(available_methods) - 1)) * bar_width
    ax.bar(x_positions + offset, values, width=bar_width, label=method)

ax.set_yscale("log")
ax.set_xticks(x_positions)
ax.set_xticklabels([r"$K_x$", r"$K_y$", r"$K_z$"])
ax.set_ylabel(r"equivalent permeability [m$^2$]")
ax.set_title("Trabecular 3-D ROI: PNM, TPFA, USFEM, and direct-image DNS comparisons")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

comparison_plot_path = output_dir / "trabecular_bone_roi_model_comparison.png"
if save_outputs:
    fig.savefig(comparison_plot_path, dpi=200)
comparison_plot_path

# %%
timing_df = plot_df[np.isfinite(plot_df["solve_seconds"])].copy()
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
ax.set_title("Trabecular 3-D ROI solver wall time by axis")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

time_plot_path = output_dir / "trabecular_bone_roi_model_solve_time.png"
if save_outputs:
    fig.savefig(time_plot_path, dpi=200)
time_plot_path

# %%
heatmap_df = (
    plot_df.pivot_table(index="method", columns="axis", values="K_mD", aggfunc="first")
    .reindex(available_methods)
    .reindex(columns=axis_order)
)

with np.errstate(divide="ignore", invalid="ignore"):
    heatmap_values = np.log10(heatmap_df.to_numpy(dtype=float))

fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
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
ax.set_title("Trabecular 3-D ROI equivalent permeability")
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
    output_dir / "trabecular_bone_roi_model_comparison_heatmap.png"
)
if save_outputs:
    fig.savefig(comparison_heatmap_path, dpi=200)
comparison_heatmap_path

# %% [markdown]
# ## Save tables

# %%
saved_paths: list[Path] = []

if save_outputs:
    roi_summary_path = output_dir / "trabecular_bone_roi_summary.csv"
    phase_summary_path = output_dir / "trabecular_bone_roi_phase_summary.csv"
    map_summary_path = output_dir / "trabecular_bone_roi_map_summary.csv"
    resistor_path = output_dir / "trabecular_bone_roi_map_resistor_directional.csv"
    comparison_path = output_dir / "trabecular_bone_roi_model_comparison.csv"
    ratio_path = output_dir / "trabecular_bone_roi_model_ratios_to_brinkman_usfem.csv"

    roi_summary.to_csv(roi_summary_path, index=False)
    phase_summary.to_csv(phase_summary_path, index=False)
    map_summary.to_csv(map_summary_path, index=False)
    resistor_df.to_csv(resistor_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    ratio_df.to_csv(ratio_path, index=False)
    fem_status_path.write_text(json.dumps(fem_status, indent=2), encoding="utf-8")
    lbm_status_path.write_text(json.dumps(lbm_status, indent=2), encoding="utf-8")

    for axis, result in resistor_results.items():
        pressure_path = (
            output_dir / f"trabecular_bone_roi_map_resistor_pressure_{axis}.npy"
        )
        np.save(pressure_path, result.pressure)
        saved_paths.append(pressure_path)

    saved_paths.extend(
        path
        for path in [
            porosity_h5,
            permeability_h5,
            roi_summary_path,
            phase_summary_path,
            map_summary_path,
            resistor_path,
            comparison_path,
            ratio_path,
            fem_status_path,
            lbm_directional_path,
            lbm_status_path,
            map_figure_path,
            comparison_plot_path,
            time_plot_path,
            comparison_heatmap_path,
        ]
        if path.exists()
    )
    saved_paths.extend(path for path in fem_directional_paths if path.exists())

pd.DataFrame({"saved_path": [str(path) for path in saved_paths]})

# %% [markdown]
# ## Interpretation notes
#
# - The Darcy-Brinkman FEM row is a micro-continuum solve because it uses the
#   porosity map in the effective viscosity and the permeability map in the
#   Darcy drag term.
# - The TPFA finite-volume Darcy-Darcy row uses the same permeability coefficient
#   field but omits Brinkman viscous diffusion, so it is a comparison baseline
#   rather than the same equation.
# - The direct-image LBM DNS row uses the binary ROI, pressure reservoirs, and
#   bounce-back solids. It does not use the Kozeny-Carman closure field.
# - The extracted PNM rows use the binary ROI directly and construct pore-throat
#   geometry. Agreement or disagreement with FEM rows therefore mixes model
#   family differences with extraction and closure assumptions.
# - The Kozeny-Carman characteristic length and caps are exposed above because
#   they are closure assumptions. They can move the map-based results by orders
#   of magnitude and should not be treated as measured sample properties.
