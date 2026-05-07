# %% [markdown]
# # MWE 15 - External `pnextract` / `pnflow` benchmark against `voids`
#
# This notebook compares `voids` against a committed reference dataset produced earlier with the
# Imperial College `pnextract` + `pnflow` workflow. The reference data now lives under
# `examples/data/external_pnflow_benchmark/`, so this notebook does not invoke `pnextract` or
# `pnflow` and remains runnable even if those binaries are unavailable in the future.
#
# The committed benchmark bundle includes:
#
# - the exact binary input volumes used for the benchmark
# - `pnflow` report files (`*_pnflow.prt`, `*_upscaled.tsv`)
# - extracted-network files (`*_node*.dat`, `*_link*.dat`) for later inspection
#
# Scientific scope and caveats:
# - this is an end-to-end workflow comparison, not a solver-only cross-check
# - any mismatch reflects both extraction differences and constitutive-model differences
# - the `voids` side is evaluated in three modes:
#   - import the saved `pnextract` CNM network, enable explicit
#     `pnflow_solver_box_compat=True`, and solve it with `voids`
#   - re-extract the saved binary volume with `snow2` and solve that network with `voids`
#   - re-extract the saved binary volume with the native maximal-ball backend, using explicit
#     external-reservoir boundary pores on the flow axis, and solve that network with `voids`
# - the external side uses previously saved `pnextract` geometry plus `pnflow`'s internal
#   single-phase model
# - the CNM compatibility switch is benchmark-specific and reproduces checked-in `pnflow`
#   preprocessing, not a generic physical boundary rule for all imported networks
# - the committed input volumes make this benchmark stable against future changes in the synthetic
#   generator implementation
# - we keep `mu = constant` here on purpose because the checked `pnflow` code path uses scalar fluid
#   viscosities rather than a thermodynamic `mu(P, T)` coupling

# %%
from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voids.benchmarks import compare_network_geometry
from voids.image import (
    construct_spanning_network,
    extract_maximal_ball_regions,
    summarize_maximal_ball_extraction_diagnostics,
)
from voids.paths import data_path, project_root
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    solve,
)


def _render_inline_and_close_figure(fig: object) -> None:
    """Render notebook figures inline while keeping script and CI runs non-interactive."""

    get_ipython_shell = globals().get("get_ipython")
    shell = get_ipython_shell() if callable(get_ipython_shell) else None
    if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
        from IPython.display import Image, display

        figure_buffer = io.BytesIO()
        fig.savefig(figure_buffer, format="png", bbox_inches="tight", dpi=160)
        display(Image(data=figure_buffer.getvalue()))
    plt.close(fig)


def _require_match(pattern: str, text: str, *, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {label!r}")
    return match


def _parse_upscaled_metrics(path: Path, *, prefix: str) -> dict[str, float]:
    text = path.read_text()
    permeability = float(
        _require_match(
            rf"{re.escape(prefix)}_permeability:\s*([0-9.eE+-]+);",
            text,
            label=f"{prefix}_permeability",
        ).group(1)
    )
    porosity = float(
        _require_match(
            rf"{re.escape(prefix)}_porosity:\s*([0-9.eE+-]+);",
            text,
            label=f"{prefix}_porosity",
        ).group(1)
    )
    formation_factor = float(
        _require_match(
            rf"{re.escape(prefix)}_formationfactor:\s*([0-9.eE+-]+);",
            text,
            label=f"{prefix}_formationfactor",
        ).group(1)
    )
    return {
        "k_pnflow": permeability,
        "phi_pnflow": porosity,
        "formation_factor_pnflow": formation_factor,
    }


def _parse_pnflow_prt(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    int_patterns = {
        "pnflow_n_pores": r"Number of pores:\s+([0-9]+)",
        "pnflow_n_throats": r"Number of throats:\s+([0-9]+)",
        "pnflow_n_inlet_connections": r"Number of inlet connections:\s+([0-9]+)",
        "pnflow_n_outlet_connections": r"Number of outlet connections:\s+([0-9]+)",
        "pnflow_n_isolated_elements": r"Number of isolated elements:\s+([0-9]+)",
    }
    float_patterns = {
        "phi_pnflow_prt": r"Total porosity:\s+([0-9.eE+-]+)",
        "k_pnflow_prt": r"Absolute permeability:\s+([0-9.eE+-]+)",
    }
    parsed: dict[str, float | int] = {}
    for key, pattern in int_patterns.items():
        value = _require_match(pattern, text, label=key).group(1)
        parsed[key] = int(value)
    for key, pattern in float_patterns.items():
        value = _require_match(pattern, text, label=key).group(1)
        parsed[key] = float(value)
    return parsed


def _load_reference_case(
    reference_root: Path, row: pd.Series
) -> tuple[np.ndarray, dict[str, object]]:
    case = str(row["case"])
    volume = np.load(reference_root / str(row["volume_file"]))
    prt_path = reference_root / str(row["pnflow_prt_file"])
    upscaled_path = reference_root / str(row["pnflow_upscaled_file"])
    metrics: dict[str, object] = {
        **_parse_upscaled_metrics(upscaled_path, prefix=case),
        **_parse_pnflow_prt(prt_path),
        "reference_case_dir": str((reference_root / case).relative_to(project_root())),
    }
    return np.asarray(volume, dtype=bool), metrics


def _construct_voids_from_volume_case(
    volume: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str,
    backend: str,
    extraction_kwargs: dict[str, object] | None = None,
) -> object:
    return construct_spanning_network(
        backend=backend,
        phases=volume.astype(int),
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        extraction_kwargs=extraction_kwargs,
        provenance_notes={"benchmark_kind": "external_pnflow_reference"},
    )


def _summarize_voids_construction_transport(
    construction: object,
    *,
    flow_axis: str,
    fluid: FluidSinglePhase,
    options: SinglePhaseOptions,
    metric_prefix: str,
) -> dict[str, float | int]:
    image = getattr(construction, "image", None)
    if image is None:
        raise ValueError(
            "Image-based construction is required for volume transport summary"
        )
    bc = PressureBC(
        f"inlet_{flow_axis}min",
        f"outlet_{flow_axis}max",
        pin=2.0e5,
        pout=0.0,
    )
    result = solve(
        construction.net,
        fluid=fluid,
        bc=bc,
        axis=flow_axis,
        options=options,
    )
    helper_pore_mask = _flow_reservoir_helper_pore_mask(
        construction.net, flow_axis=flow_axis
    )
    return {
        "phi_image": float(np.asarray(image, dtype=bool).mean()),
        f"phi_abs_{metric_prefix}_voids": float(absolute_porosity(construction.net)),
        f"phi_eff_{metric_prefix}_voids": float(
            effective_porosity(construction.net, axis=flow_axis)
        ),
        f"Np_{metric_prefix}_voids": int(construction.net.Np),
        f"Np_{metric_prefix}_physical_voids": int(
            construction.net.Np - np.count_nonzero(helper_pore_mask)
        ),
        f"Np_{metric_prefix}_reservoir_helper_voids": int(
            np.count_nonzero(helper_pore_mask)
        ),
        f"Nt_{metric_prefix}_voids": int(construction.net.Nt),
        f"k_{metric_prefix}_voids": float(result.permeability[flow_axis]),
        f"Q_{metric_prefix}_voids": float(result.total_flow_rate),
    }


def _construct_voids_on_imported_cnm_case(
    prefix: Path,
    *,
    flow_axis: str,
) -> object:
    return construct_spanning_network(
        backend="pnflow_cnm",
        pnflow_cnm_prefix=prefix,
        pnflow_solver_box_compat=True,
        flow_axis=flow_axis,
        provenance_notes={"benchmark_kind": "external_pnflow_reference"},
    )


def _summarize_imported_cnm_transport(
    construction: object,
    *,
    flow_axis: str,
    fluid: FluidSinglePhase,
    options: SinglePhaseOptions,
) -> dict[str, float | int]:
    bc = PressureBC(
        f"inlet_{flow_axis}min",
        f"outlet_{flow_axis}max",
        pin=2.0e5,
        pout=0.0,
    )
    result = solve(
        construction.net,
        fluid=fluid,
        bc=bc,
        axis=flow_axis,
        options=options,
    )
    return {
        "phi_abs_imported_voids": float(absolute_porosity(construction.net)),
        "phi_eff_imported_voids": float(
            effective_porosity(construction.net, axis=flow_axis)
        ),
        "Np_imported_physical": int(construction.backend_details["n_physical_pores"]),
        "Np_imported_total": int(construction.net.Np),
        "Np_imported_boundary_mirror": int(
            construction.backend_details["n_boundary_mirror_pores"]
        ),
        "Nt_imported_voids": int(construction.net.Nt),
        "k_imported_voids": float(result.permeability[flow_axis]),
        "Q_imported_voids": float(result.total_flow_rate),
    }


def _geometry_comparison_metrics(
    imported_construction: object,
    candidate_construction: object,
    *,
    flow_axis: str,
    candidate_name: str,
) -> dict[str, float | int]:
    imported_full_net = imported_construction.net_full
    n_physical_pores = int(imported_construction.backend_details["n_physical_pores"])
    imported_physical_mask = (
        np.arange(imported_full_net.Np, dtype=np.int64) < n_physical_pores
    )
    candidate_helper_mask = _flow_reservoir_helper_pore_mask(
        candidate_construction.net_full,
        flow_axis=flow_axis,
    )
    candidate_physical_mask = (
        ~candidate_helper_mask if np.any(candidate_helper_mask) else None
    )
    comparison = compare_network_geometry(
        imported_full_net,
        candidate_construction.net_full,
        axis=flow_axis,
        reference_pore_mask=imported_physical_mask,
        candidate_pore_mask=candidate_physical_mask,
        reference_name="imported_physical",
        candidate_name=candidate_name,
    )
    return {
        f"{candidate_name}_geom_pore_count_rel_diff": float(
            comparison.pore_count_rel_diff
        ),
        f"{candidate_name}_geom_throat_count_rel_diff": float(
            comparison.throat_count_rel_diff
        ),
        f"{candidate_name}_geom_mean_coordination_rel_diff": float(
            comparison.mean_coordination_rel_diff
        ),
        f"{candidate_name}_geom_pore_radius_ks": float(comparison.pore_radius_ks),
        f"{candidate_name}_geom_throat_radius_ks": float(comparison.throat_radius_ks),
        f"{candidate_name}_geom_throat_area_ks": float(comparison.throat_area_ks),
        f"{candidate_name}_geom_throat_length_ks": float(comparison.throat_length_ks),
        f"{candidate_name}_geom_throat_core_length_ks": float(
            comparison.throat_core_length_ks
        ),
        f"{candidate_name}_geom_coordination_ks": float(comparison.coordination_ks),
        f"{candidate_name}_geom_throat_face_count_ks": float(
            comparison.throat_face_count_ks
        ),
        f"{candidate_name}_geom_n_components": int(
            comparison.candidate_summary.n_components
        ),
        f"{candidate_name}_geom_dead_end_fraction": float(
            comparison.candidate_summary.dead_end_fraction
        ),
        f"{candidate_name}_geom_overlap_boundary_count": int(
            comparison.candidate_summary.overlapping_boundary_count
        ),
        f"{candidate_name}_geom_support_radius_mean": float(
            comparison.candidate_summary.throat_support_radius_mean
        ),
    }


def _flow_reservoir_helper_pore_mask(net: object, *, flow_axis: str) -> np.ndarray:
    """Return helper reservoir pores, if the extraction backend created them."""

    pore_count = int(getattr(net, "Np"))
    pore_labels = getattr(net, "pore_labels", {})
    helper_mask = np.zeros(pore_count, dtype=bool)
    for side_name in ("min", "max"):
        connected_key = (
            f"boundary_connected_inlet_{flow_axis}{side_name}"
            if side_name == "min"
            else f"boundary_connected_outlet_{flow_axis}{side_name}"
        )
        helper_key = (
            f"inlet_{flow_axis}{side_name}"
            if side_name == "min"
            else f"outlet_{flow_axis}{side_name}"
        )
        if connected_key in pore_labels and helper_key in pore_labels:
            helper_mask |= np.asarray(pore_labels[helper_key], dtype=bool)
    return helper_mask


def _maximal_ball_step_diagnostics_metrics(
    volume: np.ndarray,
    *,
    distance_map_backend: str,
) -> dict[str, float | int]:
    """Summarize native maximal-ball extraction stages for per-case benchmarking."""

    extraction_result = extract_maximal_ball_regions(
        volume,
        distance_map_backend=distance_map_backend,
    )
    diagnostics = summarize_maximal_ball_extraction_diagnostics(
        volume,
        extraction_result,
    )
    return {
        "maxball_diag_retained_ball_count": int(diagnostics.retained_ball_count),
        "maxball_diag_root_region_count": int(diagnostics.root_region_count),
        "maxball_diag_occupied_region_count": int(diagnostics.occupied_region_count),
        "maxball_diag_assigned_void_fraction": float(
            diagnostics.assigned_void_fraction
        ),
        "maxball_diag_unassigned_void_voxel_count": int(
            diagnostics.unassigned_void_voxel_count
        ),
        "maxball_diag_zero_throat_region_count": int(
            diagnostics.zero_throat_region_count
        ),
        "maxball_diag_internal_zero_throat_region_count": int(
            diagnostics.internal_zero_throat_region_count
        ),
        "maxball_diag_boundary_zero_throat_region_count": int(
            diagnostics.boundary_zero_throat_region_count
        ),
        "maxball_diag_touch_radius_side1_mean_voxels": float(
            diagnostics.throat_touch_radius_side1_mean_voxels
        ),
        "maxball_diag_touch_radius_side2_mean_voxels": float(
            diagnostics.throat_touch_radius_side2_mean_voxels
        ),
        "maxball_diag_refined_support_radius_side1_mean_voxels": float(
            diagnostics.throat_refined_support_radius_side1_mean_voxels
        ),
        "maxball_diag_refined_support_radius_side2_mean_voxels": float(
            diagnostics.throat_refined_support_radius_side2_mean_voxels
        ),
    }


def _make_permeability_comparison_figure(
    summary_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    kmin = float(
        min(
            summary_frame["k_imported_voids"].min(),
            summary_frame["k_porespy_voids"].min(),
            summary_frame["k_maxball_voids"].min(),
            summary_frame["k_pnflow"].min(),
        )
    )
    kmax = float(
        max(
            summary_frame["k_imported_voids"].max(),
            summary_frame["k_porespy_voids"].max(),
            summary_frame["k_maxball_voids"].max(),
            summary_frame["k_pnflow"].max(),
        )
    )
    pad = 0.05 * max(kmax - kmin, 1.0e-30)

    axes[0].scatter(
        summary_frame["k_pnflow"],
        summary_frame["k_imported_voids"],
        s=65,
        color="tab:green",
        label="voids on imported pnextract CNM",
    )
    axes[0].scatter(
        summary_frame["k_pnflow"],
        summary_frame["k_porespy_voids"],
        s=65,
        color="tab:blue",
        label="voids on PoreSpy extraction",
    )
    axes[0].scatter(
        summary_frame["k_pnflow"],
        summary_frame["k_maxball_voids"],
        s=65,
        color="tab:red",
        label="voids on native maximal-ball extraction",
    )
    axes[0].plot(
        [kmin - pad, kmax + pad],
        [kmin - pad, kmax + pad],
        color="black",
        lw=1.2,
        linestyle="--",
    )
    for row in summary_frame.itertuples(index=False):
        axes[0].annotate(
            row.case,
            (row.k_pnflow, row.k_imported_voids),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[0].set_xlabel("pnflow permeability [m$^2$]")
    axes[0].set_ylabel("voids permeability [m$^2$]")
    axes[0].set_title("Permeability scatter")
    axes[0].legend()

    x = np.arange(len(summary_frame))
    width = 0.26
    axes[1].bar(
        x - width,
        100.0 * summary_frame["k_rel_diff_imported"],
        width=width,
        color="tab:green",
        label="imported CNM",
    )
    axes[1].bar(
        x,
        100.0 * summary_frame["k_rel_diff_porespy"],
        width=width,
        color="tab:orange",
        label="PoreSpy extraction",
    )
    axes[1].bar(
        x + width,
        100.0 * summary_frame["k_rel_diff_maxball"],
        width=width,
        color="tab:red",
        label="native maximal-ball extraction",
    )
    axes[1].set_xticks(x, summary_frame["case"], rotation=25)
    axes[1].set_xlabel("case")
    axes[1].set_ylabel("relative difference [%]")
    axes[1].set_title("Per-case permeability mismatch")
    axes[1].legend()

    fig.suptitle("External pnflow benchmark", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    _render_inline_and_close_figure(fig)


def _make_porosity_trend_figure(
    summary_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    plot_df = summary_frame.sort_values(["phi_image", "blobiness"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.semilogy(
        plot_df["phi_image"],
        plot_df["k_pnflow"],
        marker="o",
        lw=1.5,
        label="pnflow",
    )
    ax.semilogy(
        plot_df["phi_image"],
        plot_df["k_imported_voids"],
        marker="^",
        lw=1.5,
        label="voids on imported pnextract CNM",
    )
    ax.semilogy(
        plot_df["phi_image"],
        plot_df["k_porespy_voids"],
        marker="s",
        lw=1.5,
        label="voids on PoreSpy extraction",
    )
    ax.semilogy(
        plot_df["phi_image"],
        plot_df["k_maxball_voids"],
        marker="D",
        lw=1.5,
        label="voids on native maximal-ball extraction",
    )
    ax.set_xlabel("image porosity [-]")
    ax.set_ylabel("permeability [m$^2$]")
    ax.set_title("Permeability trend with porosity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    _render_inline_and_close_figure(fig)


examples_data = data_path()
reference_root = examples_data / "external_pnflow_benchmark"
manifest_path = reference_root / "manifest.csv"
report_dir = project_root() / "docs" / "assets" / "verification"
report_dir.mkdir(parents=True, exist_ok=True)
report_csv = report_dir / "pnflow_5_case_results.csv"
maxball_diagnostics_csv = report_dir / "pnflow_maxball_step_diagnostics.csv"
comparison_figure_path = report_dir / "pnflow_permeability_scatter_and_error.png"
porosity_figure_path = report_dir / "pnflow_porosity_vs_permeability.png"

if not manifest_path.exists():
    raise FileNotFoundError(
        "Committed reference data not found. Expected " f"{manifest_path} to exist."
    )

fluid = FluidSinglePhase(viscosity=1.0e-3)
options = SinglePhaseOptions(
    conductance_model="valvatne_blunt",
    solver="direct",
)
manifest_df = pd.read_csv(manifest_path)

# %% [markdown]
# ## Reference dataset provenance
#
# The table below is the committed benchmark manifest. The notebook loads the exact saved volumes
# and previously generated `pnflow` reports from `examples/data/external_pnflow_benchmark/`.

# %%
manifest_df

# %% [markdown]
# ## Representative benchmark input
#
# The committed input volumes are binary segmented images (`True` = void) that were used in the
# original external `pnextract` / `pnflow` run.

# %%
representative_row = manifest_df.loc[manifest_df["case"] == "phi038_b18"].iloc[0]
representative_volume = np.load(
    reference_root / representative_row["volume_file"]
).astype(bool)
mid = representative_volume.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(representative_volume[mid, :, :], cmap="gray", origin="lower")
axes[0].set_title("YZ slice")
axes[0].set_xlabel("z")
axes[0].set_ylabel("y")

axes[1].imshow(representative_volume[:, mid, :], cmap="gray", origin="lower")
axes[1].set_title("XZ slice")
axes[1].set_xlabel("z")
axes[1].set_ylabel("x")

axes[2].imshow(representative_volume[:, :, mid], cmap="gray", origin="lower")
axes[2].set_title("XY slice")
axes[2].set_xlabel("y")
axes[2].set_ylabel("x")

fig.suptitle(
    f"Representative committed input: {representative_row['case']}", fontsize=13
)
plt.tight_layout()
_render_inline_and_close_figure(fig)

print("shape =", representative_volume.shape)
print("void fraction =", float(representative_volume.mean()))

# %% [markdown]
# ## Run `voids` against the committed external references
#
# For each case, the notebook loads the saved external outputs and then evaluates three `voids`
# pathways:
#
# - imported CNM: reuse the saved `pnextract` network and compare the `voids` single-phase solve
#   directly against `pnflow`
# - image extraction with `snow2`: re-extract the saved binary volume with the current `voids`
#   `snow2` workflow and compare that full path against `pnflow`
# - image extraction with native maximal-ball: run the current `voids` maximal-ball path with
#   explicit external-reservoir boundary pores and compare that full path against `pnflow`
#
# This split helps identify whether a mismatch is dominated by extraction or by the single-phase
# solver/geometry interpretation on the same network.

# %%
rows: list[dict[str, object]] = []

for row in manifest_df.itertuples(index=False):
    row_series = pd.Series(row._asdict())
    volume, pnflow_metrics = _load_reference_case(reference_root, row_series)
    case_root = reference_root / str(row_series["case"])
    imported_construction = _construct_voids_on_imported_cnm_case(
        case_root / str(row_series["case"]),
        flow_axis=str(row_series["flow_axis"]),
    )
    imported_metrics = _summarize_imported_cnm_transport(
        imported_construction,
        flow_axis=str(row_series["flow_axis"]),
        fluid=fluid,
        options=options,
    )
    porespy_construction = _construct_voids_from_volume_case(
        volume,
        voxel_size=float(row_series["voxel_size_m"]),
        flow_axis=str(row_series["flow_axis"]),
        backend="porespy",
    )
    porespy_metrics = _summarize_voids_construction_transport(
        porespy_construction,
        flow_axis=str(row_series["flow_axis"]),
        fluid=fluid,
        options=options,
        metric_prefix="porespy",
    )
    maxball_construction = _construct_voids_from_volume_case(
        volume,
        voxel_size=float(row_series["voxel_size_m"]),
        flow_axis=str(row_series["flow_axis"]),
        backend="native_maximal_ball",
        extraction_kwargs={
            "distance_map_backend": "scipy",
            "flow_boundary_mode": "external_reservoir",
        },
    )
    maxball_metrics = _summarize_voids_construction_transport(
        maxball_construction,
        flow_axis=str(row_series["flow_axis"]),
        fluid=fluid,
        options=options,
        metric_prefix="maxball",
    )
    porespy_geometry_metrics = _geometry_comparison_metrics(
        imported_construction,
        porespy_construction,
        flow_axis=str(row_series["flow_axis"]),
        candidate_name="porespy",
    )
    maxball_geometry_metrics = _geometry_comparison_metrics(
        imported_construction,
        maxball_construction,
        flow_axis=str(row_series["flow_axis"]),
        candidate_name="maxball",
    )
    maxball_step_diagnostics = _maximal_ball_step_diagnostics_metrics(
        volume,
        distance_map_backend="scipy",
    )
    rows.append(
        {
            **row._asdict(),
            **pnflow_metrics,
            **imported_metrics,
            **porespy_metrics,
            **maxball_metrics,
            **porespy_geometry_metrics,
            **maxball_geometry_metrics,
            **maxball_step_diagnostics,
        }
    )

summary_df = pd.DataFrame(rows)
summary_df["k_ratio_imported_to_pnflow"] = (
    summary_df["k_imported_voids"] / summary_df["k_pnflow"]
)
summary_df["k_ratio_porespy_to_pnflow"] = (
    summary_df["k_porespy_voids"] / summary_df["k_pnflow"]
)
summary_df["k_ratio_maxball_to_pnflow"] = (
    summary_df["k_maxball_voids"] / summary_df["k_pnflow"]
)
summary_df["k_rel_diff_imported"] = np.abs(
    summary_df["k_imported_voids"] - summary_df["k_pnflow"]
) / np.maximum(
    np.maximum(np.abs(summary_df["k_imported_voids"]), np.abs(summary_df["k_pnflow"])),
    1.0e-30,
)
summary_df["k_rel_diff_porespy"] = np.abs(
    summary_df["k_porespy_voids"] - summary_df["k_pnflow"]
) / np.maximum(
    np.maximum(np.abs(summary_df["k_porespy_voids"]), np.abs(summary_df["k_pnflow"])),
    1.0e-30,
)
summary_df["k_rel_diff_maxball"] = np.abs(
    summary_df["k_maxball_voids"] - summary_df["k_pnflow"]
) / np.maximum(
    np.maximum(np.abs(summary_df["k_maxball_voids"]), np.abs(summary_df["k_pnflow"])),
    1.0e-30,
)
summary_df["phi_abs_diff_imported"] = (
    summary_df["phi_abs_imported_voids"] - summary_df["phi_pnflow"]
)
summary_df["phi_abs_diff_porespy"] = (
    summary_df["phi_abs_porespy_voids"] - summary_df["phi_pnflow"]
)
summary_df["phi_abs_diff_maxball"] = (
    summary_df["phi_abs_maxball_voids"] - summary_df["phi_pnflow"]
)
summary_df["np_rel_diff_imported"] = np.abs(
    summary_df["Np_imported_physical"] - summary_df["pnflow_n_pores"]
) / np.maximum(summary_df["pnflow_n_pores"], 1.0)
summary_df["np_rel_diff_porespy"] = np.abs(
    summary_df["Np_porespy_physical_voids"] - summary_df["pnflow_n_pores"]
) / np.maximum(summary_df["pnflow_n_pores"], 1.0)
summary_df["nt_rel_diff_imported"] = np.abs(
    summary_df["Nt_imported_voids"] - summary_df["pnflow_n_throats"]
) / np.maximum(summary_df["pnflow_n_throats"], 1.0)
summary_df["nt_rel_diff_porespy"] = np.abs(
    summary_df["Nt_porespy_voids"] - summary_df["pnflow_n_throats"]
) / np.maximum(summary_df["pnflow_n_throats"], 1.0)
summary_df["np_rel_diff_maxball"] = np.abs(
    summary_df["Np_maxball_physical_voids"] - summary_df["pnflow_n_pores"]
) / np.maximum(summary_df["pnflow_n_pores"], 1.0)
summary_df["nt_rel_diff_maxball"] = np.abs(
    summary_df["Nt_maxball_voids"] - summary_df["pnflow_n_throats"]
) / np.maximum(summary_df["pnflow_n_throats"], 1.0)

display_columns = [
    "case",
    "seed_used",
    "porosity_target",
    "blobiness",
    "phi_image",
    "phi_abs_imported_voids",
    "phi_abs_porespy_voids",
    "phi_abs_maxball_voids",
    "phi_pnflow",
    "Np_imported_physical",
    "Np_porespy_physical_voids",
    "Np_porespy_reservoir_helper_voids",
    "Np_porespy_voids",
    "Np_maxball_physical_voids",
    "Np_maxball_reservoir_helper_voids",
    "Np_maxball_voids",
    "pnflow_n_pores",
    "Nt_imported_voids",
    "Nt_porespy_voids",
    "Nt_maxball_voids",
    "pnflow_n_throats",
    "k_imported_voids",
    "k_porespy_voids",
    "k_maxball_voids",
    "k_pnflow",
    "k_rel_diff_imported",
    "k_rel_diff_porespy",
    "k_rel_diff_maxball",
    "porespy_geom_pore_count_rel_diff",
    "porespy_geom_throat_count_rel_diff",
    "porespy_geom_throat_radius_ks",
    "porespy_geom_coordination_ks",
    "porespy_geom_n_components",
    "maxball_geom_pore_count_rel_diff",
    "maxball_geom_throat_count_rel_diff",
    "maxball_geom_throat_radius_ks",
    "maxball_geom_coordination_ks",
    "maxball_geom_n_components",
    "maxball_geom_dead_end_fraction",
    "maxball_diag_assigned_void_fraction",
    "maxball_diag_unassigned_void_voxel_count",
    "maxball_diag_zero_throat_region_count",
    "maxball_diag_internal_zero_throat_region_count",
]
summary_df.loc[:, display_columns]

# %%
diagnostic_summary_columns = [
    "k_rel_diff_imported",
    "k_rel_diff_porespy",
    "k_rel_diff_maxball",
    "porespy_geom_pore_count_rel_diff",
    "porespy_geom_throat_count_rel_diff",
    "porespy_geom_coordination_ks",
    "porespy_geom_n_components",
    "maxball_geom_pore_count_rel_diff",
    "maxball_geom_throat_count_rel_diff",
    "maxball_geom_throat_radius_ks",
    "maxball_geom_coordination_ks",
    "maxball_geom_n_components",
    "maxball_geom_dead_end_fraction",
    "maxball_diag_assigned_void_fraction",
    "maxball_diag_zero_throat_region_count",
    "maxball_diag_internal_zero_throat_region_count",
]
summary_df.loc[:, diagnostic_summary_columns].mean(numeric_only=True).to_frame(
    name="mean_over_cases"
)

# %% [markdown]
# ## Save derived comparison artifacts
#
# The committed external reference data under `examples/data/` are treated as inputs. The CSV and
# figures written here remain derived benchmark reports for documentation and review.

# %%
summary_df.to_csv(report_csv, index=False)
print("Saved:", report_csv)

# %%
maxball_step_diagnostic_columns = [
    "case",
    "maxball_diag_retained_ball_count",
    "maxball_diag_root_region_count",
    "maxball_diag_occupied_region_count",
    "maxball_diag_assigned_void_fraction",
    "maxball_diag_unassigned_void_voxel_count",
    "maxball_diag_zero_throat_region_count",
    "maxball_diag_internal_zero_throat_region_count",
    "maxball_diag_boundary_zero_throat_region_count",
    "maxball_diag_touch_radius_side1_mean_voxels",
    "maxball_diag_touch_radius_side2_mean_voxels",
    "maxball_diag_refined_support_radius_side1_mean_voxels",
    "maxball_diag_refined_support_radius_side2_mean_voxels",
]
summary_df.loc[:, maxball_step_diagnostic_columns].to_csv(
    maxball_diagnostics_csv,
    index=False,
)
print("Saved:", maxball_diagnostics_csv)

# %%
_make_permeability_comparison_figure(
    summary_df,
    output_path=comparison_figure_path,
)
print("Saved:", comparison_figure_path)

# %%
_make_porosity_trend_figure(
    summary_df,
    output_path=porosity_figure_path,
)
print("Saved:", porosity_figure_path)

# %% [markdown]
# ## Interpretation
#
# The permeability comparison is the main benchmark output. The porosity and network-size columns
# help diagnose whether a mismatch comes primarily from extraction topology, pore/throat geometry,
# or the transport model itself. The added geometry columns compare each `voids` extraction against
# the imported physical `pnextract` CNM rather than only against the final `pnflow` permeability.
#
# Points to keep in mind when interpreting the numbers:
# - `pnflow` reports porosity after its own extracted-network construction, not the binary-image
#   void fraction
# - imported-CNM and PoreSpy-extracted `voids` porosities are not identical diagnostics:
#   the imported CNM includes the saved `pnextract` conduit volumes, while the PoreSpy path uses
#   the current `voids` extraction and pruning workflow
# - if imported-CNM permeability is much closer to `pnflow` than both image-extraction paths, the
#   dominant remaining mismatch is likely extraction rather than the single-phase solver itself
# - large throat-count relative error, large throat-radius KS distance, or many disconnected
#   components in the candidate extraction are stronger evidence of topology/ownership mismatch than
#   of a boundary-condition or transport-model issue
# - at the current stage, the native maximal-ball path should be treated as a topology diagnostic:
#   it now uses explicit external-reservoir boundary pores on the flow axis and is closer than
#   `snow2` on mean permeability error for this five-case set; after excluding helper reservoir
#   pores from geometry diagnostics, the residual mismatch is more concentrated in conduit
#   area/shape-factor and boundary-reservoir reduction than in physical pore/throat counts
# - a close permeability match on the imported CNM is encouraging, but it still does not prove full
#   geometric equivalence between implementations
# - a large mismatch is not automatically a bug in `voids`; it may reflect different extraction and
#   conductance assumptions in the two workflows
