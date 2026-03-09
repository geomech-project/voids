# %% [markdown]
# # MWE 21 - DRP-317 Bandera Brown porosity and absolute permeability from RAW image
#
# This notebook estimates porosity and absolute permeability for:
#
# - `examples/data/drp-317/BanderaBrown_2d25um_binary.raw`
#
# Provided experimental references from Table 1 of the Scientific Reports paper:
#
# - Absolute permeability: `Kabs = 63 mD`
# - Porosity: `phi = 24.11%`
# - Image resolution: `2.25 um`
# - Paper Table 1 sample label: `D`
#
# Workflow highlights:
#
# 1. Load the RAW binary image with a memory map (full-volume porosity from all voxels).
# 2. Extract a pore network from a configurable analysis subvolume.
# 3. Solve single-phase flow and estimate absolute permeability.
# 4. Compare estimated properties against experimental values.
# 5. Produce an interactive pore-network view and pore-network statistics.
#
# Notes:
#
# - The file size is 1,000,000,000 bytes, interpreted here as a `1000 x 1000 x 1000` uint8 binary volume.
# - Full-volume network extraction at this scale is expensive; by default, extraction runs on a centered ROI.
#

# %% [markdown]
# ## Data source and citation
#
# These DRP-317 rock images come from:
#
# - Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
#   *11 Sandstones: raw, filtered and segmented data* [Dataset].
#   Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
#
# The experimental porosity and permeability targets used for comparison come from:
#
# - Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E., Barbalho, H.,
#   Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
#   *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
#   *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>
#

# %%
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps

from voids.geom import characteristic_size
from voids.graph.metrics import connectivity_metrics, coordination_numbers
from voids.image import extract_spanning_pore_network, infer_sample_axes
from voids.io.hdf5 import save_hdf5
from voids.paths import data_path
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    solve,
)
from voids.physics.thermo import TabulatedWaterViscosityModel
from voids.visualization import plot_network_plotly

# %%
# Inputs
raw_relpath = Path("drp-317") / "BanderaBrown_2d25um_binary.raw"
voxel_size_um = 2.25
voxel_size_m = voxel_size_um * 1.0e-6

experimental_porosity_pct = 24.11
experimental_kabs_mD = 63.0
experimental_kabs_rel_error = 0.10  # +/-10%
paper_table_sample_label = "D"
paper_table_reference = "Table 1"
drp317_dataset_citation = (
    "Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7). "
    "11 Sandstones: raw, filtered and segmented data [Dataset]. "
    "Digital Porous Media Portal. https://www.doi.org/10.17612/f4h1-w124"
)
drp317_dataset_url = (
    "https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-317"
)
drp317_paper_citation = (
    "Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E., Barbalho, H., "
    "Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021). "
    "High accuracy capillary network representation in digital rock reveals "
    "permeability scaling functions. Scientific Reports, 11, 11370. "
    "https://doi.org/10.1038/s41598-021-90090-0"
)
drp317_paper_url = "https://www.nature.com/articles/s41598-021-90090-0#Sec13"

# Paper-reference PNM controls:
# - PoreSpy 1.2.0 extraction
# - cylindrical Poiseuille links
# - 10 kPa/m pressure gradient
# - quadratic mean across the three principal axes
paper_reference_porespy_version = "1.2.0"
conductance_models: tuple[str, ...] = ("generic_poiseuille",)
geometry_repairs: str | None = None
trim_nonpercolating_paths = True
pressure_gradient_pa_per_m = 1.0e4
pressure_reference_pa = 5.0e6
permeability_mean_mode = "quadratic"  # "arithmetic" or "quadratic"
viscosity_backend = "thermo"
viscosity_temperature_k = 298.15
viscosity_pressure_points = 192
nonlinear_solver = "newton"
nonlinear_pressure_tolerance = 1.0e-10

# Phase convention controls:
# - "auto": pick the convention (void==0 or void==1) whose full-image porosity
#   is closest to experimental_porosity_pct.
# - "void_is_zero": raw 0 means void, raw 1 means solid.
# - "void_is_one": raw 1 means void, raw 0 means solid.
phase_convention = "auto"

# Analysis controls
# Use None to run extraction on the full volume (closest to the paper, but expensive).
analysis_shape_voxels: tuple[int, int, int] | None = (300, 300, 300)
# ROI origin strategy:
# - "center": centered ROI
# - "manual": use analysis_origin_voxels
# - "scan": coarse scan and pick the ROI whose image porosity is closest to
#   the chosen target (full-image or experimental porosity)
analysis_origin_strategy = "scan"
analysis_origin_porosity_target = "full_image"  # "full_image" or "experimental"
analysis_origin_scan_positions = 3
analysis_origin_voxels: tuple[int, int, int] | None = None
flow_axis_override: str | None = None  # None -> infer longest axis
max_plot_throats = 3000

# Output controls
save_outputs = True

viscosity_model = TabulatedWaterViscosityModel.from_backend(
    viscosity_backend,
    temperature=viscosity_temperature_k,
    pressure_points=viscosity_pressure_points,
)


# %%
def infer_cubic_shape_from_bytes(path: Path) -> tuple[int, int, int]:
    """Infer cubic shape for a uint8 RAW file from byte size."""
    n_bytes = int(path.stat().st_size)
    side = round(n_bytes ** (1.0 / 3.0))
    if side**3 != n_bytes:
        raise ValueError(
            f"RAW size {n_bytes} is not a perfect cube in uint8 voxels. "
            "Set shape manually in the notebook before loading."
        )
    return (side, side, side)


def roi_origin_centered(
    full_shape: tuple[int, int, int],
    sub_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return centered ROI origin."""
    return tuple((f - s) // 2 for f, s in zip(full_shape, sub_shape))


def roi_slices(
    full_shape: tuple[int, int, int],
    sub_shape: tuple[int, int, int],
    origin: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Build validated ROI slices."""
    slices: list[slice] = []
    for f, s, o in zip(full_shape, sub_shape, origin):
        if s <= 0 or s > f:
            raise ValueError(f"Invalid ROI edge {s} for full edge {f}")
        if o < 0 or (o + s) > f:
            raise ValueError(f"Invalid ROI origin {o} for edge {s} in full edge {f}")
        slices.append(slice(o, o + s))
    return tuple(slices)  # type: ignore[return-value]


def resolve_phase_convention(
    image: np.ndarray,
    *,
    convention: str,
    reference_porosity_pct: float | None,
) -> tuple[str, float, float]:
    """Resolve void/solid convention and return candidate full-volume porosities."""
    raw_mean = float(np.mean(image))
    if 0.0 <= raw_mean <= 1.0:
        phi_if_void_is_one = raw_mean
    else:
        # Fallback for non-binary encodings: treat all nonzero values as void.
        phi_if_void_is_one = float(np.count_nonzero(image)) / float(image.size)
    phi_if_void_is_zero = 1.0 - phi_if_void_is_one

    if convention == "void_is_one":
        selected = "void_is_one"
    elif convention == "void_is_zero":
        selected = "void_is_zero"
    elif convention == "auto":
        if reference_porosity_pct is None:
            selected = "void_is_zero"
        else:
            phi_ref = 0.01 * reference_porosity_pct
            selected = (
                "void_is_zero"
                if abs(phi_if_void_is_zero - phi_ref)
                <= abs(phi_if_void_is_one - phi_ref)
                else "void_is_one"
            )
    else:
        raise ValueError(
            "phase_convention must be one of: 'auto', 'void_is_zero', 'void_is_one'"
        )

    return selected, phi_if_void_is_zero, phi_if_void_is_one


def raw_to_void_image(
    raw_image: np.ndarray,
    *,
    selected_convention: str,
) -> np.ndarray:
    """Convert the raw binary convention into the void=1 image used downstream."""
    raw_arr = np.asarray(raw_image, dtype=np.uint8)
    if selected_convention == "void_is_zero":
        return (raw_arr == 0).astype(np.int8)
    return (raw_arr > 0).astype(np.int8)


def candidate_roi_starts(
    full_edge: int,
    sub_edge: int,
    *,
    n_positions: int,
) -> list[int]:
    """Return evenly spaced candidate ROI starts along one axis."""
    if sub_edge >= full_edge or n_positions <= 1:
        return [0]
    max_origin = full_edge - sub_edge
    return sorted({int(round(v)) for v in np.linspace(0, max_origin, num=n_positions)})


def scan_roi_origins_by_porosity(
    raw_image: np.ndarray,
    *,
    full_shape: tuple[int, int, int],
    sub_shape: tuple[int, int, int],
    selected_convention: str,
    target_porosity: float,
    n_positions: int,
) -> tuple[tuple[int, int, int], pd.DataFrame]:
    """Coarsely scan ROI origins and rank them by porosity mismatch."""
    candidate_axes = [
        candidate_roi_starts(f, s, n_positions=n_positions)
        for f, s in zip(full_shape, sub_shape)
    ]
    records: list[dict[str, object]] = []
    for origin in product(*candidate_axes):
        slx, sly, slz = roi_slices(full_shape, sub_shape, origin)
        raw_block = np.asarray(raw_image[slx, sly, slz], dtype=np.uint8)
        phi_block = float(
            np.mean(
                raw_to_void_image(raw_block, selected_convention=selected_convention)
            )
        )
        records.append(
            {
                "origin": tuple(int(v) for v in origin),
                "porosity_pct": 100.0 * phi_block,
                "target_porosity_pct": 100.0 * target_porosity,
                "abs_porosity_error_pct_points": 100.0
                * abs(phi_block - target_porosity),
            }
        )

    scan = pd.DataFrame(records).sort_values(
        ["abs_porosity_error_pct_points", "porosity_pct"],
        kind="stable",
    )
    scan = scan.reset_index(drop=True)
    best_origin = tuple(int(v) for v in scan.loc[0, "origin"])
    return best_origin, scan


def prepare_axis_image(
    image: np.ndarray,
    *,
    axis: str,
    trim_nonpercolating: bool,
) -> np.ndarray:
    """Optionally trim to axis-percolating paths before network extraction."""
    if not trim_nonpercolating:
        return np.asarray(image, dtype=np.int8)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    trimmed = ps.filters.trim_nonpercolating_paths(
        np.asarray(image, dtype=bool),
        axis=axis_index,
    )
    return trimmed.astype(np.int8)


# %%
examples_data = data_path()
raw_path = examples_data / raw_relpath
full_shape = infer_cubic_shape_from_bytes(raw_path)

im_full = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=full_shape)

print(f"RAW path: {raw_path}")
print(f"Inferred shape: {full_shape}")
print(f"dtype: {im_full.dtype}")
print(
    f"Unique values (first 1e6 voxels): {np.unique(np.asarray(im_full[:100, :100, :100]))}"
)

# %%
selected_phase_convention, phi_void_is_zero, phi_void_is_one = resolve_phase_convention(
    im_full,
    convention=phase_convention,
    reference_porosity_pct=experimental_porosity_pct,
)
phi_image_full = (
    phi_void_is_zero if selected_phase_convention == "void_is_zero" else phi_void_is_one
)

print(f"Phase convention requested: {phase_convention}")
print(f"Phase convention selected: {selected_phase_convention}")
print(f"Full-volume porosity if void==0: {100.0 * phi_void_is_zero:.4f}%")
print(f"Full-volume porosity if void==1: {100.0 * phi_void_is_one:.4f}%")
print(f"Full-volume image porosity used: {100.0 * phi_image_full:.4f}%")

# %% [markdown]
# ## Quick visual check of the full binary image

# %%
mid_x, mid_y, mid_z = (n // 2 for n in full_shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(np.asarray(im_full[mid_x, :, :]), cmap="gray", origin="lower")
axes[0].set_title(f"YZ slice (x={mid_x})")
axes[0].set_xlabel("z")
axes[0].set_ylabel("y")

axes[1].imshow(np.asarray(im_full[:, mid_y, :]), cmap="gray", origin="lower")
axes[1].set_title(f"XZ slice (y={mid_y})")
axes[1].set_xlabel("z")
axes[1].set_ylabel("x")

axes[2].imshow(np.asarray(im_full[:, :, mid_z]), cmap="gray", origin="lower")
axes[2].set_title(f"XY slice (z={mid_z})")
axes[2].set_xlabel("y")
axes[2].set_ylabel("x")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Select analysis volume for network extraction

# %%
if analysis_shape_voxels is None:
    roi_shape = full_shape
    roi_origin = (0, 0, 0)
    roi_target_porosity = phi_image_full
    roi_scan_summary: pd.DataFrame | None = None
else:
    roi_shape = analysis_shape_voxels
    if analysis_origin_strategy == "center":
        roi_origin = roi_origin_centered(full_shape, roi_shape)
        roi_target_porosity = phi_image_full
        roi_scan_summary = None
    elif analysis_origin_strategy == "manual":
        if analysis_origin_voxels is None:
            raise ValueError(
                "Set analysis_origin_voxels when analysis_origin_strategy='manual'"
            )
        roi_origin = analysis_origin_voxels
        roi_target_porosity = phi_image_full
        roi_scan_summary = None
    elif analysis_origin_strategy == "scan":
        if analysis_origin_porosity_target == "full_image":
            roi_target_porosity = phi_image_full
        elif analysis_origin_porosity_target == "experimental":
            roi_target_porosity = 0.01 * experimental_porosity_pct
        else:
            raise ValueError(
                "analysis_origin_porosity_target must be 'full_image' or 'experimental'"
            )
        roi_origin, roi_scan_summary = scan_roi_origins_by_porosity(
            im_full,
            full_shape=full_shape,
            sub_shape=roi_shape,
            selected_convention=selected_phase_convention,
            target_porosity=roi_target_porosity,
            n_positions=analysis_origin_scan_positions,
        )
    else:
        raise ValueError(
            "analysis_origin_strategy must be 'center', 'manual', or 'scan'"
        )

slx, sly, slz = roi_slices(full_shape, roi_shape, roi_origin)
im_analysis_raw = np.asarray(im_full[slx, sly, slz], dtype=np.uint8)
im_analysis = raw_to_void_image(
    im_analysis_raw,
    selected_convention=selected_phase_convention,
)

phi_image_analysis = float(np.mean(im_analysis))

_, axis_lengths, axis_areas, inferred_flow_axis = infer_sample_axes(
    im_analysis.shape, voxel_size=voxel_size_m
)
flow_axis = inferred_flow_axis if flow_axis_override is None else flow_axis_override

print(f"ROI origin: {roi_origin}")
print(f"ROI shape: {im_analysis.shape}")
print(f"ROI image porosity: {100.0 * phi_image_analysis:.4f}%")
print(f"ROI origin strategy: {analysis_origin_strategy}")
if analysis_shape_voxels is not None:
    print(
        "ROI porosity target: "
        f"{100.0 * roi_target_porosity:.4f}% ({analysis_origin_porosity_target})"
    )
if roi_scan_summary is not None:
    print("Top ROI candidates by porosity match:")
    print(roi_scan_summary.head(5).to_string(index=False))
print(f"Flow axis used: {flow_axis}")
print(f"Axis lengths [m]: {axis_lengths}")
print(f"Axis areas [m^2]: {axis_areas}")


# %%
def extract_axis_network(axis: str):
    """Extract one axis-spanning network under the selected paper-like settings."""
    axis_image = prepare_axis_image(
        im_analysis,
        axis=axis,
        trim_nonpercolating=trim_nonpercolating_paths,
    )
    extract_axis = extract_spanning_pore_network(
        axis_image,
        voxel_size=voxel_size_m,
        flow_axis=axis,
        length_unit="m",
        geometry_repairs=geometry_repairs,
        provenance_notes={
            "raw_source": str(raw_relpath).replace("\\", "/"),
            "roi_origin_voxels": roi_origin,
            "roi_shape_voxels": im_analysis.shape,
            "experimental_porosity_pct": experimental_porosity_pct,
            "experimental_kabs_mD": experimental_kabs_mD,
            "paper_table_sample_label": paper_table_sample_label,
            "paper_table_reference": paper_table_reference,
            "dataset_citation": drp317_dataset_citation,
            "dataset_url": drp317_dataset_url,
            "paper_citation": drp317_paper_citation,
            "paper_url": drp317_paper_url,
            "trim_nonpercolating_paths": trim_nonpercolating_paths,
            "conductance_models": list(conductance_models),
            "analysis_origin_strategy": analysis_origin_strategy,
            "analysis_origin_porosity_target": analysis_origin_porosity_target,
            "paper_reference_porespy_version": paper_reference_porespy_version,
        },
    )
    return axis_image, extract_axis


im_flow_analysis, extract = extract_axis_network(flow_axis)

net_full = extract.net_full
net = extract.net
phi_image_analysis_flow = float(np.mean(im_flow_analysis))

print(f"Backend: {extract.backend} {extract.backend_version}")
print(f"Imported full network: Np={net_full.Np}, Nt={net_full.Nt}")
print(f"Axis-spanning network ({flow_axis}): Np={net.Np}, Nt={net.Nt}")
print(
    "Paper PNM reference uses PoreSpy "
    f"{paper_reference_porespy_version}; current backend is {extract.backend_version}"
)
print(f"Conductance models requested: {conductance_models}")
print(f"Geometry repairs: {geometry_repairs}")
print(f"Trim nonpercolating paths: {trim_nonpercolating_paths}")
print(
    f"Viscosity model: {viscosity_model.backend_name} at "
    f"{viscosity_model.temperature:.2f} K"
)
print(
    f"ROI image porosity after {flow_axis}-path trim: "
    f"{100.0 * phi_image_analysis_flow:.4f}%"
)

# %%
phi_abs = absolute_porosity(net)
phi_eff = effective_porosity(net, axis=flow_axis)


def solve_axis_with_fallback(net_axis, axis: str):
    """Solve one axis using the configured conductance models."""
    delta_p = pressure_gradient_pa_per_m * axis_lengths[axis]
    bc_axis = PressureBC(
        f"inlet_{axis}min",
        f"outlet_{axis}max",
        pin=pressure_reference_pa + delta_p,
        pout=pressure_reference_pa,
    )
    last_exc: Exception | None = None
    for model in conductance_models:
        try:
            res_axis = solve(
                net_axis,
                fluid=FluidSinglePhase(viscosity_model=viscosity_model),
                bc=bc_axis,
                axis=axis,
                options=SinglePhaseOptions(
                    conductance_model=model,
                    solver="direct",
                    nonlinear_solver=nonlinear_solver,
                    nonlinear_pressure_tolerance=nonlinear_pressure_tolerance,
                ),
            )
            return res_axis, model
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Single-phase solve failed for all conductance models on axis '{axis}'"
    ) from last_exc


res, used_conductance_model = solve_axis_with_fallback(net, flow_axis)

M2_PER_MD = 9.869233e-16
k_m2 = float(res.permeability[flow_axis])
k_mD = k_m2 / M2_PER_MD

phi_est_pct = 100.0 * phi_abs
phi_eff_pct = 100.0 * phi_eff

porosity_abs_error_pct_points = phi_est_pct - experimental_porosity_pct
porosity_rel_error_pct = (
    100.0 * porosity_abs_error_pct_points / experimental_porosity_pct
)

axes_available = tuple(ax for ax in ("x", "y", "z") if ax in axis_lengths)
directional_records: list[dict[str, float | str]] = []

for ax in axes_available:
    if ax == flow_axis:
        net_ax = net
        res_ax = res
        model_ax = used_conductance_model
    else:
        _, extract_ax = extract_axis_network(ax)
        net_ax = extract_ax.net
        res_ax, model_ax = solve_axis_with_fallback(net_ax, ax)

    k_ax_m2 = float(res_ax.permeability[ax])
    k_ax_mD = k_ax_m2 / M2_PER_MD
    directional_records.append(
        {
            "axis": ax,
            "k_m2": k_ax_m2,
            "k_mD": k_ax_mD,
            "n_pores": float(net_ax.Np),
            "n_throats": float(net_ax.Nt),
            "conductance_model": str(model_ax),
        }
    )

kabs_directional = (
    pd.DataFrame(directional_records).sort_values("axis").reset_index(drop=True)
)
kabs_mean_mD = float(kabs_directional["k_mD"].mean())
kabs_mean_m2 = float(kabs_directional["k_m2"].mean())
kabs_rms_mD = float(
    np.sqrt(np.mean(np.square(kabs_directional["k_mD"].to_numpy(dtype=float))))
)
kabs_rms_m2 = float(
    np.sqrt(np.mean(np.square(kabs_directional["k_m2"].to_numpy(dtype=float))))
)
kabs_mean_abs_error_mD = kabs_mean_mD - experimental_kabs_mD
kabs_mean_rel_error_pct = 100.0 * kabs_mean_abs_error_mD / experimental_kabs_mD
kabs_rms_abs_error_mD = kabs_rms_mD - experimental_kabs_mD
kabs_rms_rel_error_pct = 100.0 * kabs_rms_abs_error_mD / experimental_kabs_mD

if permeability_mean_mode == "quadratic":
    aggregate_label = "Quadratic mean"
    aggregate_kabs_mD = kabs_rms_mD
    aggregate_kabs_m2 = kabs_rms_m2
else:
    aggregate_label = "Arithmetic mean"
    aggregate_kabs_mD = kabs_mean_mD
    aggregate_kabs_m2 = kabs_mean_m2

estimated_properties = pd.DataFrame(
    [
        {
            "property": "Porosity (image full volume)",
            "estimated": 100.0 * phi_image_full,
            "units": "%",
            "experimental": experimental_porosity_pct,
            "abs_error": (100.0 * phi_image_full) - experimental_porosity_pct,
            "rel_error_pct": 100.0
            * ((100.0 * phi_image_full) - experimental_porosity_pct)
            / experimental_porosity_pct,
        },
        {
            "property": "Porosity (network absolute)",
            "estimated": phi_est_pct,
            "units": "%",
            "experimental": experimental_porosity_pct,
            "abs_error": porosity_abs_error_pct_points,
            "rel_error_pct": porosity_rel_error_pct,
        },
        {
            "property": "Porosity (network effective)",
            "estimated": phi_eff_pct,
            "units": "%",
            "experimental": np.nan,
            "abs_error": np.nan,
            "rel_error_pct": np.nan,
        },
    ]
)
for _, row in kabs_directional.iterrows():
    k_dir_mD = float(row["k_mD"])
    estimated_properties.loc[len(estimated_properties)] = {
        "property": f"Absolute permeability K{row['axis']}",
        "estimated": k_dir_mD,
        "units": "mD",
        "experimental": experimental_kabs_mD,
        "abs_error": k_dir_mD - experimental_kabs_mD,
        "rel_error_pct": 100.0
        * (k_dir_mD - experimental_kabs_mD)
        / experimental_kabs_mD,
    }
estimated_properties.loc[len(estimated_properties)] = {
    "property": "Absolute permeability arithmetic mean(Kx,Ky,Kz)",
    "estimated": kabs_mean_mD,
    "units": "mD",
    "experimental": experimental_kabs_mD,
    "abs_error": kabs_mean_abs_error_mD,
    "rel_error_pct": kabs_mean_rel_error_pct,
}
estimated_properties.loc[len(estimated_properties)] = {
    "property": "Absolute permeability quadratic mean(sqrt(mean(K^2)))",
    "estimated": kabs_rms_mD,
    "units": "mD",
    "experimental": experimental_kabs_mD,
    "abs_error": kabs_rms_abs_error_mD,
    "rel_error_pct": kabs_rms_rel_error_pct,
}

print(f"Conductance model used: {used_conductance_model}")
print(
    f"Reference viscosity used for permeability reporting: "
    f"{res.reference_viscosity:.6e} Pa s"
)
print(
    f"Nonlinear iterations ({nonlinear_solver}): "
    f"{res.solver_info.get('nonlinear_iterations', 'n/a')}"
)
print(f"Total flow rate Q: {res.total_flow_rate:.6e} m^3/s")
print(f"K{flow_axis}: {k_m2:.6e} m^2 ({k_mD:.3f} mD)")
print(f"Mass-balance error: {res.mass_balance_error:.3e}")
print()
print("Directional Kabs estimates [mD]:")
print(kabs_directional[["axis", "k_mD", "n_pores", "n_throats", "conductance_model"]])
print(
    f"Arithmetic mean Kabs across axes: {kabs_mean_m2:.6e} m^2 ({kabs_mean_mD:.3f} mD)"
)
print(f"Quadratic mean Kabs across axes: {kabs_rms_m2:.6e} m^2 ({kabs_rms_mD:.3f} mD)")
estimated_properties

# %% [markdown]
# ## Kabs directional comparison (mD)

# %%
experimental_kabs_error_mD = experimental_kabs_mD * experimental_kabs_rel_error

bar_labels = [f"K{ax}" for ax in kabs_directional["axis"]] + [
    aggregate_label,
    "Experimental",
]
bar_values = list(kabs_directional["k_mD"].to_numpy(dtype=float)) + [
    aggregate_kabs_mD,
    experimental_kabs_mD,
]
bar_errors = [0.0] * (len(bar_labels) - 1) + [experimental_kabs_error_mD]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    bar_labels,
    bar_values,
    yerr=bar_errors,
    capsize=6,
    color=["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:gray"][
        : len(bar_labels)
    ],
    edgecolor="black",
    alpha=0.85,
)
ax.axhline(
    experimental_kabs_mD,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Experimental",
)
ax.fill_between(
    [-0.6, len(bar_labels) - 0.4],
    experimental_kabs_mD - experimental_kabs_error_mD,
    experimental_kabs_mD + experimental_kabs_error_mD,
    color="gray",
    alpha=0.15,
    label="Experimental +/-10%",
)
ax.set_ylabel("Absolute permeability [mD]")
ax.set_title(
    f"Estimated Kabs by direction and {aggregate_label.lower()} vs experimental"
)
ax.grid(alpha=0.3, linestyle=":", axis="y")
ax.legend()

for rect, val in zip(bars, bar_values):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height(),
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pore-network statistics

# %%
pore_size_m, pore_size_field = characteristic_size(net.pore, expected_shape=(net.Np,))
throat_size_m, throat_size_field = characteristic_size(
    net.throat, expected_shape=(net.Nt,)
)

pore_size_um = 1.0e6 * pore_size_m
throat_size_um = 1.0e6 * throat_size_m

coord = coordination_numbers(net)
coord_vals, coord_counts = np.unique(coord, return_counts=True)

pore_volume = np.asarray(net.pore.get("region_volume", net.pore["volume"]), dtype=float)
order = np.argsort(pore_size_um)
cum_pore_volume = np.cumsum(pore_volume[order]) / pore_volume.sum()

conn = connectivity_metrics(net)

network_stats = pd.DataFrame(
    [
        {"metric": "Np", "value": float(net.Np), "units": "count"},
        {"metric": "Nt", "value": float(net.Nt), "units": "count"},
        {
            "metric": "Mean coordination",
            "value": float(np.mean(coord)),
            "units": "-",
        },
        {
            "metric": "Max coordination",
            "value": float(np.max(coord)),
            "units": "-",
        },
        {
            "metric": f"Mean pore size ({pore_size_field})",
            "value": float(np.mean(pore_size_um)),
            "units": "um",
        },
        {
            "metric": f"Median pore size ({pore_size_field})",
            "value": float(np.median(pore_size_um)),
            "units": "um",
        },
        {
            "metric": f"Mean throat size ({throat_size_field})",
            "value": float(np.mean(throat_size_um)),
            "units": "um",
        },
        {
            "metric": f"Median throat size ({throat_size_field})",
            "value": float(np.median(throat_size_um)),
            "units": "um",
        },
        {
            "metric": "Connected components",
            "value": float(conn.n_components),
            "units": "count",
        },
        {
            "metric": "Giant component fraction",
            "value": float(conn.giant_component_fraction),
            "units": "-",
        },
        {
            "metric": "Dead-end pore fraction",
            "value": float(conn.dead_end_fraction),
            "units": "-",
        },
    ]
)

network_stats

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(pore_size_um, bins=30, color="tab:blue", alpha=0.85, edgecolor="black")
axes[0, 0].set_title("Pore size distribution")
axes[0, 0].set_xlabel(f"Pore {pore_size_field} [um]")
axes[0, 0].set_ylabel("Count")
axes[0, 0].grid(alpha=0.3, linestyle=":")

axes[0, 1].hist(
    throat_size_um,
    bins=30,
    color="tab:orange",
    alpha=0.85,
    edgecolor="black",
)
axes[0, 1].set_title("Throat size distribution")
axes[0, 1].set_xlabel(f"Throat {throat_size_field} [um]")
axes[0, 1].set_ylabel("Count")
axes[0, 1].grid(alpha=0.3, linestyle=":")

axes[1, 0].bar(
    coord_vals,
    coord_counts,
    color="tab:green",
    alpha=0.85,
    edgecolor="black",
)
axes[1, 0].set_title("Coordination number distribution")
axes[1, 0].set_xlabel("Coordination number")
axes[1, 0].set_ylabel("Pore count")
axes[1, 0].grid(alpha=0.3, linestyle=":", axis="y")

axes[1, 1].plot(pore_size_um[order], cum_pore_volume, color="tab:red", linewidth=2)
axes[1, 1].set_title("Cumulative pore-volume fraction")
axes[1, 1].set_xlabel(f"Pore {pore_size_field} [um]")
axes[1, 1].set_ylabel("Cumulative fraction")
axes[1, 1].grid(alpha=0.3, linestyle=":")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interactive pore network

# %%
fig = plot_network_plotly(
    net,
    point_scalars=res.pore_pressure,
    max_throats=max_plot_throats,
    title=(
        f"DRP-317 Bandera Brown network ({flow_axis}-spanning) - pressure field "
        f"[{used_conductance_model}]"
    ),
    layout_kwargs={"width": 1000, "height": 750},
)
fig.show()

# %% [markdown]
# ## Save outputs

# %%
out_dir = examples_data / "drp-317"
out_net_full_h5 = out_dir / "BanderaBrown_2d25um_network_full_voids.h5"
out_net_span_h5 = out_dir / f"BanderaBrown_2d25um_network_{flow_axis}spanning_voids.h5"
out_props_csv = out_dir / "BanderaBrown_2d25um_estimated_properties.csv"
out_stats_csv = out_dir / "BanderaBrown_2d25um_network_stats.csv"
out_kabs_dir_csv = out_dir / "BanderaBrown_2d25um_kabs_directional.csv"
out_roi_scan_csv = out_dir / "BanderaBrown_2d25um_roi_scan.csv"

if save_outputs:
    save_hdf5(net_full, out_net_full_h5)
    save_hdf5(net, out_net_span_h5)
    estimated_properties.to_csv(out_props_csv, index=False)
    network_stats.to_csv(out_stats_csv, index=False)
    kabs_directional.to_csv(out_kabs_dir_csv, index=False)
    if roi_scan_summary is not None:
        roi_scan_summary.to_csv(out_roi_scan_csv, index=False)

print(f"Saved full network: {out_net_full_h5}")
print(f"Saved spanning network: {out_net_span_h5}")
print(f"Saved estimated properties: {out_props_csv}")
print(f"Saved network stats: {out_stats_csv}")
print(f"Saved directional Kabs: {out_kabs_dir_csv}")
if roi_scan_summary is not None:
    print(f"Saved ROI scan summary: {out_roi_scan_csv}")
