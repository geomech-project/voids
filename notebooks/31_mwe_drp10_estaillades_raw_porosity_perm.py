# %% [markdown]
# # MWE 31 - DRP-10 Estaillades v2 porosity and absolute permeability from RAW image
#
# This notebook estimates porosity and absolute permeability for:
#
# - `examples/data/drp-10/estaillades.raw`
#
# Reference values from Tables 1 and 2 of:
#
# - Muljadi, B. P., Blunt, M. J., Raeini, A. Q., & Bijeljic, B. (2016).
#   *The impact of porous media heterogeneity on non-Darcy flow behaviour
#   from pore-scale simulation*. Advances in Water Resources, 95, 329-340.
#   <https://doi.org/10.1016/j.advwatres.2015.05.019>
#
# Using the Estaillades rows:
#
# - Binary image voxel size: `500 x 500 x 500`
# - Resolution: `3.3113 um`
# - Porosity (Table 1): `10.8%`
# - Darcy permeability (Table 2): `0.172 darcy` (`172 mD`)
#
# Notes:
#
# - This DRP-10 workflow intentionally runs on the **full sample**.
# - No ROI/subvolume selection is applied.

# %% [markdown]
# ## Data source and citation
#
# Dataset source:
#
# - Digital Porous Media Portal (DPM), DRP-10 dataset:
#   <https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-10>
#
# Reference paper:
#
# - Muljadi, B. P., Blunt, M. J., Raeini, A. Q., & Bijeljic, B. (2016).
#   *The impact of porous media heterogeneity on non-Darcy flow behaviour from
#   pore-scale simulation*. Advances in Water Resources, 95, 329-340.
#   <https://doi.org/10.1016/j.advwatres.2015.05.019>

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps
import plotly.graph_objects as go

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
raw_relpath = Path("drp-10") / "estaillades.raw"
raw_shape = (500, 500, 500)
raw_dtype = np.uint8
raw_order = "C"  # DRP-10 Estaillades v2 works with default C-style voxel ordering.
raw_void_value = 0

voxel_size_um = 3.3113
voxel_size_m = voxel_size_um * 1.0e-6

paper_porosity_pct = 10.8
paper_porosity_table_reference = "Table 1"
paper_kabs_table_reference = "Table 2"
paper_kabs_darcy = 0.172
paper_kabs_mD = 1000.0 * paper_kabs_darcy

dataset_citation = (
    "Digital Porous Media Portal. DRP-10 dataset. "
    "https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-10"
)
paper_citation = (
    "Muljadi, B. P., Blunt, M. J., Raeini, A. Q., & Bijeljic, B. (2016). "
    "The impact of porous media heterogeneity on non-Darcy flow behaviour from "
    "pore-scale simulation. Advances in Water Resources, 95, 329-340. "
    "https://doi.org/10.1016/j.advwatres.2015.05.019"
)

# Modeling controls
flow_axis = "x"  # Paper computes absolute permeability left-to-right.
trim_nonpercolating_paths = True
extraction_backend = "native_maximal_ball"
comparison_extraction_backends = ("snow2",)
extraction_backends = tuple(
    dict.fromkeys((extraction_backend, *comparison_extraction_backends))
)
compute_directional_all_axes = True
edt_parallel_threads = min(4, os.cpu_count() or 1)
extraction_kwargs: dict[str, object] = {
    "distance_map_backend": "auto",
    "edt_parallel_threads": edt_parallel_threads,
    "flow_boundary_mode": "direct",
}
geometry_repairs: str | None = None
extraction_backend_configs: dict[str, dict[str, object]] = {
    "native_maximal_ball": {
        "label": "Native maximal-ball",
        "extraction_kwargs": extraction_kwargs,
        "geometry_repairs": geometry_repairs,
    },
    "snow2": {
        "label": "PoreSpy snow2",
        "extraction_kwargs": {},
        "geometry_repairs": "imperial_export",
    },
}
conductance_models: tuple[str, ...] = ("valvatne_blunt",)

pressure_gradient_pa_per_m = 1.0e4
pressure_reference_pa = 5.0e6

viscosity_backend = "thermo"
viscosity_temperature_k = 298.15
viscosity_pressure_points = 192
nonlinear_solver = "newton"
nonlinear_pressure_tolerance = 1.0e-10

# Output controls
save_outputs = True

viscosity_model = TabulatedWaterViscosityModel.from_backend(
    viscosity_backend,
    temperature=viscosity_temperature_k,
    pressure_points=viscosity_pressure_points,
)


def raw_to_void_image(raw_image: np.ndarray, *, void_value: int) -> np.ndarray:
    """Convert raw segmented image to void=1, solid=0 convention."""
    raw_arr = np.asarray(raw_image, dtype=np.uint8)
    return np.asarray(raw_arr == np.uint8(void_value), dtype=np.int8)


def prepare_axis_image(
    image: np.ndarray,
    *,
    axis: str,
    trim_nonpercolating: bool,
) -> np.ndarray:
    """Optionally trim to axis-percolating paths before extraction."""
    if not trim_nonpercolating:
        return np.asarray(image, dtype=np.int8)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    trimmed = ps.filters.trim_nonpercolating_paths(
        np.asarray(image, dtype=bool),
        axis=axis_index,
    )
    return np.asarray(trimmed, dtype=np.int8)


# %%
examples_data = data_path()
raw_path = examples_data / raw_relpath

if not raw_path.exists():
    raise FileNotFoundError(
        "Missing DRP-10 RAW volume at "
        f"{raw_path}. Place `estaillades.raw` under examples/data/drp-10 or set "
        "VOIDS_DATA_PATH to a directory that contains it."
    )

im_full_raw = np.memmap(
    filename=str(raw_path),
    dtype=np.dtype(raw_dtype),
    mode="r",
    shape=raw_shape,
    order=raw_order,
)
im_full = raw_to_void_image(im_full_raw, void_value=raw_void_value)
phi_image_full = float(np.mean(im_full))

_, axis_lengths, axis_areas, inferred_flow_axis = infer_sample_axes(
    im_full.shape, voxel_size=voxel_size_m
)

print(f"RAW path: {raw_path}")
print(f"Shape: {raw_shape}")
print(f"dtype: {im_full_raw.dtype}")
print(f"Memory order: {raw_order}")
print(
    f"Raw unique values (central block): {np.unique(np.asarray(im_full_raw[:50, :50, :50]))}"
)
print(f"Void value convention: raw == {raw_void_value}")
print(f"Full-volume porosity: {100.0 * phi_image_full:.4f}%")
print(f"Inferred longest axis: {inferred_flow_axis}")
print(f"Flow axis used: {flow_axis}")
print(f"Axis lengths [m]: {axis_lengths}")
print(f"Axis areas [m^2]: {axis_areas}")

# %% [markdown]
# ## Quick visual check of the full binary image

# %%
mid_x, mid_y, mid_z = (n // 2 for n in raw_shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(
    np.asarray(im_full[mid_x, :, :]), cmap="gray", origin="lower", vmin=0, vmax=1
)
axes[0].set_title(f"YZ slice (x={mid_x})")
axes[0].set_xlabel("z")
axes[0].set_ylabel("y")

axes[1].imshow(
    np.asarray(im_full[:, mid_y, :]), cmap="gray", origin="lower", vmin=0, vmax=1
)
axes[1].set_title(f"XZ slice (y={mid_y})")
axes[1].set_xlabel("z")
axes[1].set_ylabel("x")

axes[2].imshow(
    np.asarray(im_full[:, :, mid_z]), cmap="gray", origin="lower", vmin=0, vmax=1
)
axes[2].set_title(f"XY slice (z={mid_z})")
axes[2].set_xlabel("y")
axes[2].set_ylabel("x")

fig.tight_layout()

# %%
# 3D rock render (solid phase, not PNM)
# The full 500 x 500 x 500 solid phase is expensive to render directly, so
# this preview downsamples the binary image and renders the outer rock surface.
# A Plotly point-cloud is shown interactively in the notebook and exported to
# HTML for later inspection.


def _rock_surface_boundary_voxels(rock_volume: np.ndarray) -> np.ndarray:
    """Return indices of boundary voxels in a binary solid image."""
    rock = np.asarray(rock_volume, dtype=bool)
    padded = np.pad(rock, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return np.argwhere(rock & ~interior)


def rock_surface_point_cloud(
    image_void: np.ndarray,
    *,
    voxel_size: float,
    stride: int = 4,
    max_points: int = 50000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a downsampled solid-phase boundary point cloud in metric units."""
    if stride < 1:
        raise ValueError("stride must be >= 1")

    rock = np.asarray(image_void[::stride, ::stride, ::stride] == 0, dtype=bool)
    boundary = _rock_surface_boundary_voxels(rock)
    if boundary.size == 0:
        raise RuntimeError(
            "No solid-phase boundary voxels were found in the downsampled volume."
        )

    raw_boundary_points = int(boundary.shape[0])
    if raw_boundary_points > max_points:
        step = int(np.ceil(raw_boundary_points / max_points))
        boundary = boundary[::step]

    spacing = float(stride) * float(voxel_size)
    xyz = spacing * boundary.astype(float)
    meta = {
        "downsampled_shape": rock.shape,
        "stride": int(stride),
        "raw_surface_points": raw_boundary_points,
        "plotted_surface_points": int(boundary.shape[0]),
    }
    return xyz, meta


def plot_rock_surface_plotly(
    image_void: np.ndarray,
    *,
    voxel_size: float,
    stride: int = 4,
    max_points: int = 50000,
) -> tuple[go.Figure, dict[str, object]]:
    """Build an interactive Plotly rock-surface figure."""
    xyz, meta = rock_surface_point_cloud(
        image_void,
        voxel_size=voxel_size,
        stride=stride,
        max_points=max_points,
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=xyz[:, 2],
                    colorscale="Earth",
                    opacity=0.10,
                    showscale=False,
                ),
                hovertemplate=(
                    "x=%{x:.3e} m<br>" "y=%{y:.3e} m<br>" "z=%{z:.3e} m<extra></extra>"
                ),
                name="rock surface",
            )
        ]
    )
    fig.update_layout(
        title="DRP-10 Estaillades v2 rock surface (solid phase)",
        template="plotly_white",
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    meta["backend"] = "plotly"
    return fig, meta


def plot_rock_surface_preview(
    image_void: np.ndarray,
    *,
    voxel_size: float,
    stride: int = 4,
    max_points: int = 50000,
    screenshot_path: Path | None = None,
) -> dict[str, object]:
    """Render the solid phase of a binary void image.

    Uses PyVista when available and falls back to a Matplotlib surface-point
    preview otherwise.
    """
    xyz, meta = rock_surface_point_cloud(
        image_void,
        voxel_size=voxel_size,
        stride=stride,
        max_points=max_points,
    )
    rock_shape = tuple(int(v) for v in meta["downsampled_shape"])
    spacing = float(stride) * float(voxel_size)

    try:
        import pyvista as pv

        rock = np.asarray(image_void[::stride, ::stride, ::stride] == 0, dtype=np.uint8)
        grid = pv.ImageData(
            dimensions=[int(dim) + 1 for dim in rock.shape],
            spacing=(spacing, spacing, spacing),
            origin=(0.0, 0.0, 0.0),
        )
        grid.cell_data["rock"] = rock.ravel(order="F")
        surface = (
            grid.threshold(0.5, scalars="rock")
            .extract_surface(algorithm="dataset_surface")
            .triangulate()
        )

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(surface, color="tan", smooth_shading=True, specular=0.05)
        plotter.view_isometric()
        if screenshot_path is not None:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(screenshot_path), scale=2)
        plotter.close()
        meta["backend"] = "pyvista"
        meta["surface_cells"] = int(surface.n_cells)
        return meta
    except ImportError:
        fig_r = plt.figure(figsize=(8, 8))
        ax_r = fig_r.add_subplot(projection="3d")
        ax_r.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=0.8,
            c="tan",
            alpha=0.08,
            linewidths=0,
        )
        ax_r.set_xlabel("x [m]")
        ax_r.set_ylabel("y [m]")
        ax_r.set_zlabel("z [m]")
        ax_r.set_box_aspect(rock_shape)
        ax_r.view_init(elev=25, azim=40)
        ax_r.set_title("DRP-10 Estaillades v2 rock surface (solid phase)")
        fig_r.tight_layout()
        if screenshot_path is not None:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            fig_r.savefig(screenshot_path, dpi=180)
        meta["backend"] = "matplotlib"
        return meta


rock_render_stride = 4
out_rock_png = examples_data / "drp-10" / "Estaillades_v2_rock_surface.png"
out_rock_html = examples_data / "drp-10" / "Estaillades_v2_rock_surface_plotly.html"
rock_render_error: Exception | None = None
rock_render_meta: dict[str, object] | None = None
fig_rock_plotly: go.Figure | None = None

try:
    rock_render_meta = plot_rock_surface_preview(
        im_full,
        voxel_size=voxel_size_m,
        stride=rock_render_stride,
        screenshot_path=out_rock_png if save_outputs else None,
    )
    fig_rock_plotly, rock_plotly_meta = plot_rock_surface_plotly(
        im_full,
        voxel_size=voxel_size_m,
        stride=rock_render_stride,
        max_points=40000,
    )
    if save_outputs:
        fig_rock_plotly.write_html(out_rock_html, include_plotlyjs="cdn")
    print(f"Rock render backend: {rock_render_meta['backend']}")
    print(f"Rock render stride: {rock_render_stride}")
    print(f"Downsampled shape: {rock_render_meta['downsampled_shape']}")
    print(f"Interactive plotted points: {rock_plotly_meta['plotted_surface_points']}")
    if save_outputs and out_rock_png.exists():
        rock_preview = plt.imread(out_rock_png)
        fig_r, ax_r = plt.subplots(figsize=(8, 8))
        ax_r.imshow(rock_preview)
        ax_r.axis("off")
        ax_r.set_title("DRP-10 Estaillades v2 rock surface (solid phase)")
        fig_r.tight_layout()
        print(f"Saved rock render: {out_rock_png}")
    if save_outputs:
        print(f"Saved interactive rock HTML: {out_rock_html}")
except Exception as exc:
    rock_render_error = exc
    print(
        "3D rock rendering skipped. Install `pyvista` for surface rendering or use "
        "the Matplotlib fallback, and ensure the RAW volume is available. "
        f"Original error: {exc}"
    )

fig_rock_plotly


# %%
def extract_axis_network(axis: str, *, backend: str | None = None):
    """Extract one axis-spanning network from the full sample."""
    selected_backend = extraction_backend if backend is None else backend
    backend_config = extraction_backend_configs[selected_backend]
    backend_extraction_kwargs = dict(backend_config.get("extraction_kwargs", {}))
    backend_geometry_repairs = backend_config.get("geometry_repairs")
    axis_image = prepare_axis_image(
        im_full,
        axis=axis,
        trim_nonpercolating=trim_nonpercolating_paths,
    )
    extract_axis = extract_spanning_pore_network(
        axis_image,
        voxel_size=voxel_size_m,
        backend=selected_backend,
        flow_axis=axis,
        length_unit="m",
        geometry_repairs=backend_geometry_repairs,
        extraction_kwargs=backend_extraction_kwargs,
        provenance_notes={
            "raw_source": str(raw_relpath).replace("\\", "/"),
            "raw_shape_voxels": raw_shape,
            "paper_porosity_table_reference": paper_porosity_table_reference,
            "paper_porosity_pct": paper_porosity_pct,
            "paper_kabs_table_reference": paper_kabs_table_reference,
            "paper_kabs_darcy": paper_kabs_darcy,
            "paper_kabs_mD": paper_kabs_mD,
            "dataset_citation": dataset_citation,
            "paper_citation": paper_citation,
            "trim_nonpercolating_paths": trim_nonpercolating_paths,
            "conductance_models": list(conductance_models),
            "extraction_backend": selected_backend,
            "extraction_kwargs": backend_extraction_kwargs,
            "geometry_repairs": backend_geometry_repairs,
            "full_sample_analysis": True,
        },
    )
    return axis_image, extract_axis


im_flow, extract = extract_axis_network(flow_axis)
net_full = extract.net_full
net = extract.net

phi_image_flow = float(np.mean(im_flow))

print(f"Backend: {extract.backend} {extract.backend_version}")
print(f"Imported full network: Np={net_full.Np}, Nt={net_full.Nt}")
print(f"Axis-spanning network ({flow_axis}): Np={net.Np}, Nt={net.Nt}")
print(f"Trim nonpercolating paths: {trim_nonpercolating_paths}")
print(
    f"Viscosity model: {viscosity_model.backend_name} at "
    f"{viscosity_model.temperature:.2f} K"
)
print(
    f"Full-volume porosity after {flow_axis}-path trim: "
    f"{100.0 * phi_image_flow:.4f}%"
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
k_flow_m2 = float(res.permeability[flow_axis])
k_flow_mD = k_flow_m2 / M2_PER_MD

axes_available = tuple(ax for ax in ("x", "y", "z") if ax in axis_lengths)
if not compute_directional_all_axes:
    axes_available = (flow_axis,)

directional_records: list[dict[str, float | str]] = []
for backend in extraction_backends:
    backend_label = str(extraction_backend_configs[backend]["label"])
    for ax in axes_available:
        if backend == extraction_backend and ax == flow_axis:
            net_ax = net
            res_ax = res
            model_ax = used_conductance_model
        else:
            _, extract_ax = extract_axis_network(ax, backend=backend)
            net_ax = extract_ax.net
            res_ax, model_ax = solve_axis_with_fallback(net_ax, ax)

        k_ax_m2 = float(res_ax.permeability[ax])
        k_ax_mD = k_ax_m2 / M2_PER_MD
        directional_records.append(
            {
                "backend": backend,
                "backend_label": backend_label,
                "axis": ax,
                "k_m2": k_ax_m2,
                "k_mD": k_ax_mD,
                "n_pores": float(net_ax.Np),
                "n_throats": float(net_ax.Nt),
                "conductance_model": str(model_ax),
            }
        )

kabs_directional_by_backend = (
    pd.DataFrame(directional_records)
    .sort_values(["backend", "axis"])
    .reset_index(drop=True)
)
kabs_directional = (
    kabs_directional_by_backend[
        kabs_directional_by_backend["backend"] == extraction_backend
    ]
    .drop(columns=["backend", "backend_label"])
    .reset_index(drop=True)
)
kabs_summary_by_backend = pd.DataFrame(
    [
        {
            "backend": backend,
            "backend_label": str(extraction_backend_configs[backend]["label"]),
            "k_mean_mD": float(group["k_mD"].mean()),
            "k_rms_mD": float(
                np.sqrt(np.mean(np.square(group["k_mD"].to_numpy(dtype=float))))
            ),
            "axis_count": float(group.shape[0]),
        }
        for backend, group in kabs_directional_by_backend.groupby("backend", sort=True)
    ]
)

if flow_axis in tuple(kabs_directional["axis"]):
    k_flow_mD = float(
        kabs_directional.loc[kabs_directional["axis"] == flow_axis, "k_mD"].iloc[0]
    )
    k_flow_m2 = k_flow_mD * M2_PER_MD

kabs_mean_mD = float(kabs_directional["k_mD"].mean())
kabs_rms_mD = float(
    np.sqrt(np.mean(np.square(kabs_directional["k_mD"].to_numpy(dtype=float))))
)
snow2_kabs_directional = kabs_directional_by_backend[
    kabs_directional_by_backend["backend"] == "snow2"
].reset_index(drop=True)

phi_image_pct = 100.0 * phi_image_full
phi_abs_pct = 100.0 * phi_abs
phi_eff_pct = 100.0 * phi_eff

phi_image_abs_error_pct = phi_image_pct - paper_porosity_pct
phi_image_rel_error_pct = 100.0 * phi_image_abs_error_pct / paper_porosity_pct
phi_abs_abs_error_pct = phi_abs_pct - paper_porosity_pct
phi_abs_rel_error_pct = 100.0 * phi_abs_abs_error_pct / paper_porosity_pct
phi_eff_abs_error_pct = phi_eff_pct - paper_porosity_pct
phi_eff_rel_error_pct = 100.0 * phi_eff_abs_error_pct / paper_porosity_pct

k_flow_abs_error_mD = k_flow_mD - paper_kabs_mD
k_flow_rel_error_pct = 100.0 * k_flow_abs_error_mD / paper_kabs_mD

estimated_properties = pd.DataFrame(
    [
        {
            "property": "Porosity (image full volume)",
            "estimated": phi_image_pct,
            "units": "%",
            "paper_reference": paper_porosity_pct,
            "abs_error": phi_image_abs_error_pct,
            "rel_error_pct": phi_image_rel_error_pct,
            "reference_label": "Muljadi et al. (2016) Table 1",
        },
        {
            "property": "Porosity (network absolute)",
            "estimated": phi_abs_pct,
            "units": "%",
            "paper_reference": paper_porosity_pct,
            "abs_error": phi_abs_abs_error_pct,
            "rel_error_pct": phi_abs_rel_error_pct,
            "reference_label": "Muljadi et al. (2016) Table 1",
        },
        {
            "property": "Porosity (network effective)",
            "estimated": phi_eff_pct,
            "units": "%",
            "paper_reference": paper_porosity_pct,
            "abs_error": phi_eff_abs_error_pct,
            "rel_error_pct": phi_eff_rel_error_pct,
            "reference_label": "Muljadi et al. (2016) Table 1",
        },
        {
            "property": f"Absolute permeability K{flow_axis}",
            "estimated": k_flow_mD,
            "units": "mD",
            "paper_reference": paper_kabs_mD,
            "abs_error": k_flow_abs_error_mD,
            "rel_error_pct": k_flow_rel_error_pct,
            "reference_label": "Muljadi et al. (2016) Table 2",
        },
    ]
)

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
print(f"K{flow_axis}: {k_flow_m2:.6e} m^2 ({k_flow_mD:.3f} mD)")
print(f"Mass-balance error: {res.mass_balance_error:.3e}")
print()
print("Directional Kabs estimates by backend [mD]:")
print(
    kabs_directional_by_backend[
        [
            "backend_label",
            "axis",
            "k_mD",
            "n_pores",
            "n_throats",
            "conductance_model",
        ]
    ]
)
print()
print("Kabs summary by backend [mD]:")
print(kabs_summary_by_backend[["backend_label", "k_mean_mD", "k_rms_mD"]])
print(
    f"Primary ({extraction_backend_configs[extraction_backend]['label']}) "
    f"arithmetic mean Kabs: {kabs_mean_mD:.3f} mD"
)
print(
    f"Primary ({extraction_backend_configs[extraction_backend]['label']}) "
    f"quadratic mean Kabs: {kabs_rms_mD:.3f} mD"
)
print(
    f"Table 1 porosity reference: {paper_porosity_pct:.2f}%, "
    f"full-image relative error: {phi_image_rel_error_pct:.2f}%"
)
print(
    f"Table 2 Kabs reference: {paper_kabs_mD:.2f} mD, "
    f"relative error: {k_flow_rel_error_pct:.2f}%"
)
estimated_properties

# %% [markdown]
# ## Kabs comparison (mD)

# %%
paper_kabs_error_mD = (
    0.10 * paper_kabs_mD
)  # +/-10% assumed uncertainty band for display only

bar_data = kabs_directional_by_backend.sort_values(["backend", "axis"])
mean_bar_data = kabs_summary_by_backend.sort_values("backend")
bar_labels = (
    [f"{row.backend_label}\nK{row.axis}" for row in bar_data.itertuples(index=False)]
    + [
        f"{row.backend_label}\nmean Kabs"
        for row in mean_bar_data.itertuples(index=False)
    ]
    + ["Paper\nKabs"]
)
bar_values = (
    list(bar_data["k_mD"].to_numpy(dtype=float))
    + list(mean_bar_data["k_mean_mD"].to_numpy(dtype=float))
    + [paper_kabs_mD]
)
bar_errors = [0.0] * (len(bar_labels) - 1) + [paper_kabs_error_mD]
backend_colors = {
    "native_maximal_ball": "tab:blue",
    "snow2": "tab:orange",
}
bar_colors = (
    [backend_colors.get(str(backend), "tab:purple") for backend in bar_data["backend"]]
    + [
        backend_colors.get(str(backend), "tab:purple")
        for backend in mean_bar_data["backend"]
    ]
    + ["tab:gray"]
)
bar_hatches = [""] * len(bar_data) + ["//"] * len(mean_bar_data) + [""]

fig_k, ax_k = plt.subplots(figsize=(13.5, 4.8))
bars = ax_k.bar(
    bar_labels,
    bar_values,
    yerr=bar_errors,
    capsize=5,
    color=bar_colors,
    edgecolor="black",
    alpha=0.85,
)
for rect, hatch in zip(bars, bar_hatches):
    rect.set_hatch(hatch)
ax_k.axhline(
    paper_kabs_mD, color="black", linestyle="--", linewidth=1.2, label="Paper Kabs"
)
ax_k.fill_between(
    [-0.6, len(bar_labels) - 0.4],
    paper_kabs_mD - paper_kabs_error_mD,
    paper_kabs_mD + paper_kabs_error_mD,
    color="gray",
    alpha=0.15,
    label="Paper Kabs +/-10% (display band)",
)
ax_k.set_ylabel("Absolute permeability [mD]")
ax_k.set_title("DRP-10 Estaillades v2 permeability comparison by backend")
ax_k.grid(alpha=0.3, linestyle=":", axis="y")
ax_k.legend()
ax_k.tick_params(axis="x", labelrotation=20)

for rect, val in zip(bars, bar_values):
    ax_k.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height(),
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig_k.tight_layout()

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
        {"metric": "Mean coordination", "value": float(np.mean(coord)), "units": "-"},
        {"metric": "Max coordination", "value": float(np.max(coord)), "units": "-"},
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
fig_s, axes = plt.subplots(2, 2, figsize=(12, 9))

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

fig_s.tight_layout()

# %% [markdown]
# ## Plotly pore-network image (for docs)

# %%
fig_pnm_plotly = plot_network_plotly(
    net,
    point_scalars=np.asarray(res.pore_pressure, dtype=float),
    max_throats=8000,
    line_opacity=0.30,
    size_scale=0.7,
    title=f"DRP-10 Estaillades v2 network ({flow_axis}-spanning)",
    show_colorbar=True,
)
fig_pnm_plotly.update_layout(
    template="plotly_white",
    scene=dict(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        zaxis_title="z [m]",
        aspectmode="data",
    ),
)
fig_pnm_plotly

# %% [markdown]
# ## Save outputs

# %%
out_dir = examples_data / "drp-10"
out_net_full_h5 = out_dir / "Estaillades_v2_network_full_voids.h5"
out_net_span_h5 = out_dir / f"Estaillades_v2_network_{flow_axis}spanning_voids.h5"
out_props_csv = out_dir / "Estaillades_v2_estimated_properties.csv"
out_stats_csv = out_dir / "Estaillades_v2_network_stats.csv"
out_kabs_dir_csv = out_dir / "Estaillades_v2_kabs_directional_by_backend.csv"
out_kabs_summary_csv = out_dir / "Estaillades_v2_kabs_summary_by_backend.csv"
out_kabs_png = out_dir / "Estaillades_v2_kabs_comparison.png"
out_stats_png = out_dir / "Estaillades_v2_network_stats.png"
out_slices_png = out_dir / "Estaillades_v2_slices.png"
out_pnm_png = out_dir / "Estaillades_v2_network_static.png"
out_pnm_html = out_dir / "Estaillades_v2_network_plotly.html"

pnm_png_saved = False
pnm_png_error: Exception | None = None

if save_outputs:
    save_hdf5(net_full, out_net_full_h5)
    save_hdf5(net, out_net_span_h5)
    estimated_properties.to_csv(out_props_csv, index=False)
    network_stats.to_csv(out_stats_csv, index=False)
    kabs_directional_by_backend.to_csv(out_kabs_dir_csv, index=False)
    kabs_summary_by_backend.to_csv(out_kabs_summary_csv, index=False)
    fig_k.savefig(out_kabs_png, dpi=180)
    fig_s.savefig(out_stats_png, dpi=180)
    fig.savefig(out_slices_png, dpi=180)
    fig_pnm_plotly.write_html(out_pnm_html, include_plotlyjs="cdn")
    try:
        fig_pnm_plotly.write_image(out_pnm_png, width=1400, height=1000, scale=2)
        pnm_png_saved = True
    except Exception as exc:  # kaleido missing or image export backend error
        pnm_png_error = exc

print(f"Saved full network: {out_net_full_h5}")
print(f"Saved spanning network: {out_net_span_h5}")
print(f"Saved estimated properties: {out_props_csv}")
print(f"Saved network stats: {out_stats_csv}")
print(f"Saved directional Kabs: {out_kabs_dir_csv}")
print(f"Saved backend Kabs summary: {out_kabs_summary_csv}")
print(f"Saved Kabs plot: {out_kabs_png}")
print(f"Saved stats plot: {out_stats_png}")
print(f"Saved slice plot: {out_slices_png}")
print(f"Saved Plotly PNM HTML: {out_pnm_html}")
if pnm_png_saved:
    print(f"Saved Plotly static PNM image: {out_pnm_png}")
elif pnm_png_error is not None:
    print(
        "Plotly static PNG export skipped. Install `kaleido` in this environment to "
        f"enable `fig.write_image(...)`. Original error: {pnm_png_error}"
    )

# %%
plt.close("all")
