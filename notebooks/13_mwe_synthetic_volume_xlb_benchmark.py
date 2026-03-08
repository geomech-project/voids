# %% [markdown]
# # MWE 13 - Synthetic porous-volume benchmark against XLB
#
# This notebook generates fifteen synthetic segmented spanning volumes with `voids`, solves them
# directly on the binary image with XLB, extracts pore networks with `snow2`, and compares `Kabs`
# estimates between the direct-image voxel-scale LBM reference and the extracted-network PNM
# workflow.
#
# Scientific scope and assumptions:
# - the benchmark works on binary segmented images directly; it does not include scanner-physics
#   emulation or a grayscale segmentation step
# - the XLB side uses a stable lattice-unit pressure drop and converts only the permeability, not
#   a fully pressure-calibrated physical transient
# - the XLB solve adds short fluid reservoir layers ahead of and behind the sample so the imposed
#   pressure boundary conditions live on clean planar faces rather than directly on a perforated
#   porous inlet/outlet surface
# - the `voids` side uses `snow2` extraction plus the selected hydraulic conductance model, so any
#   mismatch reflects both extraction loss and PNM constitutive simplification
# - side faces orthogonal to the flow axis are treated as sealed sample walls in the direct-image
#   XLB solve, which matches the intended core-flood-style interpretation of the benchmark
# - XLB is an optional dependency; run this notebook in the Pixi `lbm` environment

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voids.benchmarks import XLBOptions, benchmark_segmented_volume_with_xlb
from voids.generators import generate_spanning_blobs_matrix
from voids.physics.singlephase import FluidSinglePhase, SinglePhaseOptions

# %%
flow_axis = "x"
axis_index = 0
voxel_size = 2.0e-6
fluid = FluidSinglePhase(viscosity=1.0e-3)
options = SinglePhaseOptions(
    conductance_model="valvatne_blunt_baseline",
    solver="direct",
)
xlb_options = XLBOptions(
    max_steps=2500,
    min_steps=400,
    check_interval=50,
    steady_rtol=1.0e-3,
    lattice_viscosity=0.10,
    rho_inlet=1.001,
    rho_outlet=1.000,
    inlet_outlet_buffer_cells=6,
)


def _find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "mkdocs.yml").exists() and (candidate / "docs").exists():
            return candidate
    return cwd


project_root = _find_project_root()
report_dir = project_root / "docs" / "assets" / "verification"
report_dir.mkdir(parents=True, exist_ok=True)
report_csv = report_dir / "xlb_15_case_results.csv"
representative_figure_path = report_dir / "xlb_representative_case.png"
comparison_figure_path = report_dir / "xlb_permeability_scatter_and_error.png"
porosity_figure_path = report_dir / "xlb_porosity_vs_permeability.png"

case_specs = [
    {
        "case": "phi028_b12",
        "shape": (24, 24, 24),
        "porosity": 0.28,
        "blobiness": 1.2,
        "seed_start": 101,
    },
    {
        "case": "phi028_b14",
        "shape": (24, 24, 24),
        "porosity": 0.28,
        "blobiness": 1.4,
        "seed_start": 135,
    },
    {
        "case": "phi028_b15",
        "shape": (24, 24, 24),
        "porosity": 0.28,
        "blobiness": 1.5,
        "seed_start": 152,
    },
    {
        "case": "phi032_b12",
        "shape": (24, 24, 24),
        "porosity": 0.32,
        "blobiness": 1.2,
        "seed_start": 271,
    },
    {
        "case": "phi032_b14",
        "shape": (24, 24, 24),
        "porosity": 0.32,
        "blobiness": 1.4,
        "seed_start": 305,
    },
    {
        "case": "phi032_b15",
        "shape": (24, 24, 24),
        "porosity": 0.32,
        "blobiness": 1.5,
        "seed_start": 322,
    },
    {
        "case": "phi036_b12",
        "shape": (24, 24, 24),
        "porosity": 0.36,
        "blobiness": 1.2,
        "seed_start": 441,
    },
    {
        "case": "phi036_b14",
        "shape": (24, 24, 24),
        "porosity": 0.36,
        "blobiness": 1.4,
        "seed_start": 475,
    },
    {
        "case": "phi036_b15",
        "shape": (24, 24, 24),
        "porosity": 0.36,
        "blobiness": 1.5,
        "seed_start": 492,
    },
    {
        "case": "phi040_b12",
        "shape": (24, 24, 24),
        "porosity": 0.40,
        "blobiness": 1.2,
        "seed_start": 611,
    },
    {
        "case": "phi040_b14",
        "shape": (24, 24, 24),
        "porosity": 0.40,
        "blobiness": 1.4,
        "seed_start": 645,
    },
    {
        "case": "phi040_b15",
        "shape": (24, 24, 24),
        "porosity": 0.40,
        "blobiness": 1.5,
        "seed_start": 662,
    },
    {
        "case": "phi044_b12",
        "shape": (24, 24, 24),
        "porosity": 0.44,
        "blobiness": 1.2,
        "seed_start": 781,
    },
    {
        "case": "phi044_b14",
        "shape": (24, 24, 24),
        "porosity": 0.44,
        "blobiness": 1.4,
        "seed_start": 815,
    },
    {
        "case": "phi044_b15",
        "shape": (24, 24, 24),
        "porosity": 0.44,
        "blobiness": 1.5,
        "seed_start": 832,
    },
]
case_specs

# %% [markdown]
# ## Generate binary benchmark cases and compare `voids` against XLB
#
# Each case is a percolating segmented void image. XLB sees the binary image directly, while `voids`
# extracts a spanning pore network from the same image before solving. This notebook also writes a
# CSV summary and figure assets used by the documentation report under `docs/assets/verification/`.

# %%
benchmark_rows: list[dict[str, object]] = []
case_artifacts: dict[str, dict[str, object]] = {}

for case in case_specs:
    segmented, seed_used = generate_spanning_blobs_matrix(
        shape=case["shape"],
        porosity=case["porosity"],
        blobiness=case["blobiness"],
        axis_index=axis_index,
        seed_start=case["seed_start"],
        max_tries=30,
    )

    benchmark = benchmark_segmented_volume_with_xlb(
        segmented,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        fluid=fluid,
        options=options,
        xlb_options=xlb_options,
        provenance_notes={
            "benchmark_case": case["case"],
            "seed_used": seed_used,
        },
    )

    record = benchmark.to_record()
    benchmark_rows.append(
        {
            **case,
            "seed_used": int(seed_used),
            **record,
            "k_ratio_voids_to_xlb": float(record["k_voids"]) / float(record["k_xlb"]),
            "k_rel_diff_pct": 100.0 * float(record["k_rel_diff"]),
        }
    )
    case_artifacts[case["case"]] = {
        "segmented": segmented,
        "benchmark": benchmark,
    }

summary_df = pd.DataFrame(benchmark_rows)
summary_df["k_factor_gap"] = np.maximum(
    summary_df["k_ratio_voids_to_xlb"],
    1.0 / summary_df["k_ratio_voids_to_xlb"],
)
display_columns = [
    "case",
    "seed_used",
    "shape",
    "porosity",
    "blobiness",
    "phi_image",
    "phi_abs",
    "phi_eff",
    "Np",
    "Nt",
    "k_voids",
    "k_xlb",
    "k_rel_diff",
    "k_ratio_voids_to_xlb",
    "xlb_steps",
    "xlb_converged",
    "xlb_convergence_metric",
]
summary_df.loc[:, display_columns]
summary_df.loc[:, display_columns].to_csv(report_csv, index=False)

# %%
summary_stats = pd.Series(
    {
        "n_cases": int(len(summary_df)),
        "mean_rel_diff_pct": float(summary_df["k_rel_diff_pct"].mean()),
        "median_rel_diff_pct": float(summary_df["k_rel_diff_pct"].median()),
        "max_rel_diff_pct": float(summary_df["k_rel_diff_pct"].max()),
        "mean_factor_gap": float(summary_df["k_factor_gap"].mean()),
        "median_factor_gap": float(summary_df["k_factor_gap"].median()),
        "max_factor_gap": float(summary_df["k_factor_gap"].max()),
    },
    name="value",
)
summary_stats.to_frame()

# %% [markdown]
# ## Representative binary slice and XLB axial-velocity field
#
# The XLB result retains the axial lattice velocity field, which lets us inspect whether the
# direct-image reference solution is well aligned with the segmented geometry.

# %%
representative_case = "phi036_b14"
artifact = case_artifacts[representative_case]
segmented = artifact["segmented"]
benchmark = artifact["benchmark"]
axial_velocity = benchmark.xlb_result.axial_velocity_lattice
mid = segmented.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].imshow(segmented[mid, :, :], cmap="gray", origin="lower")
axes[0].set_title(f"{representative_case}: binary segmented image")
axes[0].set_xlabel("z")
axes[0].set_ylabel("y")

im = axes[1].imshow(axial_velocity[mid, :, :], cmap="viridis", origin="lower")
axes[1].set_title(f"{representative_case}: XLB axial velocity")
axes[1].set_xlabel("z")
axes[1].set_ylabel("y")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="u_x [lattice units]")

axes[2].plot(
    benchmark.xlb_result.superficial_velocity_profile_lattice,
    marker="o",
    lw=1.5,
)
axes[2].set_title(f"{representative_case}: superficial velocity by x-slice")
axes[2].set_xlabel("x index")
axes[2].set_ylabel("U_s [lattice units]")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Representative direct-image XLB diagnostics", fontsize=14)
plt.tight_layout()
fig.savefig(representative_figure_path, dpi=160, bbox_inches="tight")
plt.show()

print("Representative XLB converged:", benchmark.xlb_result.converged)
print("Representative XLB steps:", benchmark.xlb_result.n_steps)
print("Representative XLB convergence metric:", benchmark.xlb_result.convergence_metric)

# %% [markdown]
# ## Permeability comparison
#
# Agreement is not expected to be exact here: the direct-image XLB solve resolves the segmented
# pore space voxel-by-voxel, while `voids` first compresses that geometry into an extracted network
# and then uses the selected pore-network conductance model.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

kmin = float(min(summary_df["k_voids"].min(), summary_df["k_xlb"].min()))
kmax = float(max(summary_df["k_voids"].max(), summary_df["k_xlb"].max()))
pad = 0.05 * max(kmax - kmin, 1.0e-30)

axes[0].scatter(summary_df["k_xlb"], summary_df["k_voids"], s=60, color="tab:blue")
axes[0].plot(
    [kmin - pad, kmax + pad],
    [kmin - pad, kmax + pad],
    color="black",
    lw=1.2,
    linestyle="--",
)
axes[0].set_xlabel("XLB permeability [m$^2$]")
axes[0].set_ylabel("voids permeability [m$^2$]")
axes[0].set_title("Permeability scatter")

axes[1].bar(summary_df["case"], summary_df["k_rel_diff_pct"], color="tab:orange")
axes[1].set_xlabel("case")
axes[1].set_ylabel("relative difference [%]")
axes[1].set_title("Per-case permeability mismatch")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
fig.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
plt.show()

summary_df.loc[
    :, ["case", "k_voids", "k_xlb", "k_rel_diff_pct", "xlb_steps", "xlb_converged"]
]

# %% [markdown]
# ## Permeability trend with porosity
#
# This view is useful for checking whether the extracted-network workflow follows
# the same overall porosity-permeability trend as the direct-image XLB reference.

# %%
plot_df = summary_df.sort_values(["phi_image", "blobiness"]).copy()
fig, ax = plt.subplots(figsize=(7.5, 4.8))

ax.semilogy(
    plot_df["phi_image"],
    plot_df["k_xlb"],
    marker="o",
    lw=1.5,
    label="XLB",
)
ax.semilogy(
    plot_df["phi_image"],
    plot_df["k_voids"],
    marker="s",
    lw=1.5,
    label="voids",
)
ax.set_xlabel("image porosity [-]")
ax.set_ylabel("permeability [m$^2$]")
ax.set_title("Permeability trend across the 15-case verification set")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()
fig.savefig(porosity_figure_path, dpi=160, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Notes
#
# Useful follow-up checks if the mismatch is larger than expected:
#
# - tighten XLB convergence controls (`max_steps`, `check_interval`, `steady_rtol`)
# - inspect whether the segmented geometry is too coarse for the direct-image solve
# - compare against multiple conductance models on the `voids` side
# - repeat on larger synthetic samples to reduce inlet/outlet and discretization effects
