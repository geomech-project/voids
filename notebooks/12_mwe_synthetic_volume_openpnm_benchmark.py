# %% [markdown]
# # MWE 12 - Synthetic porous-volume benchmark against OpenPNM
#
# This notebook generates synthetic spanning porous volumes with `voids`, converts them into
# synthetic grayscale observations, segments them, extracts pore networks with `snow2`, and
# compares `Kabs` estimates between `voids` and OpenPNM.
#
# Scientific scope and assumptions:
# - the grayscale model is synthetic and intentionally simple; it does not represent scanner physics
# - the network extraction step uses the segmented image, not the binary ground truth, so we report the
#   segmentation mismatch against the known truth for context
# - the OpenPNM comparison injects the `voids` throat conductances into OpenPNM, so this benchmark
#   isolates extraction consistency, boundary-condition handling, and linear-solver agreement
# - if the goal is to compare independent constitutive models, the OpenPNM side should reconstruct its
#   own conductances from geometry rather than reusing the `voids` values

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voids.benchmarks import benchmark_segmented_volume_with_openpnm
from voids.physics.singlephase import FluidSinglePhase, SinglePhaseOptions
from voids.generators import (
    generate_spanning_blobs_matrix,
    make_synthetic_grayscale,
)
from voids.image import binarize_grayscale_volume


def _find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "mkdocs.yml").exists() and (candidate / "docs").exists():
            return candidate
    return cwd


# %%
flow_axis = "x"
axis_index = 0
voxel_size = 2.0e-6
fluid = FluidSinglePhase(viscosity=1.0e-3)
options = SinglePhaseOptions(
    conductance_model="valvatne_blunt",
    solver="direct",
)
project_root = _find_project_root()
report_dir = project_root / "docs" / "assets" / "verification"
report_dir.mkdir(parents=True, exist_ok=True)
report_csv = report_dir / "openpnm_5_case_results.csv"
segmentation_figure_path = report_dir / "openpnm_representative_segmentation.png"
comparison_figure_path = report_dir / "openpnm_permeability_scatter.png"
porosity_figure_path = report_dir / "openpnm_porosity_pipeline.png"

case_specs = [
    {
        "case": "phi032_b14",
        "shape": (32, 32, 32),
        "porosity": 0.32,
        "blobiness": 1.4,
        "seed_start": 401,
    },
    {
        "case": "phi035_b16",
        "shape": (32, 32, 32),
        "porosity": 0.35,
        "blobiness": 1.6,
        "seed_start": 501,
    },
    {
        "case": "phi038_b18",
        "shape": (32, 32, 32),
        "porosity": 0.38,
        "blobiness": 1.8,
        "seed_start": 601,
    },
    {
        "case": "phi040_b18",
        "shape": (32, 32, 32),
        "porosity": 0.40,
        "blobiness": 1.8,
        "seed_start": 901,
    },
    {
        "case": "phi041_b20",
        "shape": (32, 32, 32),
        "porosity": 0.41,
        "blobiness": 2.0,
        "seed_start": 701,
    },
]
case_specs

# %% [markdown]
# ## Generate, segment, extract, and cross-check
#
# Each case is built from a percolating synthetic void image. We then generate a synthetic grayscale
# realization, segment it with Otsu thresholding, extract the spanning pore network with `snow2`, and
# compare `voids` against OpenPNM on the extracted network.
#
# This high-level benchmark now uses `delta_p` as the preferred physical input. Here we choose
# `delta_p = 1 Pa` and rely on the default gauge choice `pout = 0 Pa`, `pin = delta_p`. For the
# present benchmark we intentionally keep **constant viscosity** so the common-offset gauge choice
# remains irrelevant. A thermodynamic `mu(P, T)` model would require absolute positive pressures and
# would blur the benchmark goal, which is to isolate extraction and solver agreement.

# %%
benchmark_rows: list[dict[str, object]] = []
case_artifacts: dict[str, dict[str, object]] = {}

for case in case_specs:
    truth_void, seed_used = generate_spanning_blobs_matrix(
        shape=case["shape"],
        porosity=case["porosity"],
        blobiness=case["blobiness"],
        axis_index=axis_index,
        seed_start=case["seed_start"],
        max_tries=30,
    )
    grayscale = make_synthetic_grayscale(
        truth_void,
        seed=seed_used + 10_000,
        void_mean=65.0,
        solid_mean=185.0,
        noise_std=10.0,
    )
    segmented, threshold = binarize_grayscale_volume(
        grayscale,
        method="otsu",
        void_phase="dark",
    )

    benchmark = benchmark_segmented_volume_with_openpnm(
        segmented,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        fluid=fluid,
        delta_p=1.0,
        options=options,
        provenance_notes={
            "benchmark_case": case["case"],
            "seed_used": seed_used,
            "segmentation_threshold": float(threshold),
        },
    )

    benchmark_rows.append(
        {
            **case,
            "seed_used": int(seed_used),
            "threshold": float(threshold),
            "phi_truth": float(truth_void.mean()),
            "segmentation_mismatch": float(
                np.mean(segmented.astype(bool) != truth_void)
            ),
            **benchmark.to_record(),
        }
    )
    case_artifacts[case["case"]] = {
        "truth_void": truth_void,
        "grayscale": grayscale,
        "segmented": segmented,
        "benchmark": benchmark,
    }

summary_df = pd.DataFrame(benchmark_rows)
summary_df["k_ratio_voids_to_openpnm"] = summary_df["k_voids"] / summary_df["k_openpnm"]
summary_df["k_rel_diff_ppm"] = 1.0e6 * summary_df["k_rel_diff"]
summary_df["Q_rel_diff_ppm"] = 1.0e6 * summary_df["Q_rel_diff"]

display_columns = [
    "case",
    "seed_used",
    "shape",
    "porosity",
    "blobiness",
    "threshold",
    "phi_truth",
    "phi_image",
    "segmentation_mismatch",
    "phi_abs",
    "phi_eff",
    "Np",
    "Nt",
    "k_voids",
    "k_openpnm",
    "k_rel_diff",
    "Q_rel_diff",
]
summary_df.loc[:, display_columns]

# %% [markdown]
# ## Representative segmentation slices
#
# The synthetic grayscale model is intentionally high-contrast, so Otsu thresholding should recover the
# known binary truth almost exactly for this benchmark suite.

# %%
representative_case = "phi038_b18"
artifact = case_artifacts[representative_case]
truth_void = artifact["truth_void"]
grayscale = artifact["grayscale"]
segmented = artifact["segmented"]
mid = truth_void.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(truth_void[mid, :, :], cmap="gray", origin="lower")
axes[0].set_title(f"{representative_case}: binary truth")
axes[0].set_xlabel("z")
axes[0].set_ylabel("y")

axes[1].imshow(grayscale[mid, :, :], cmap="gray", origin="lower")
axes[1].set_title(f"{representative_case}: synthetic grayscale")
axes[1].set_xlabel("z")
axes[1].set_ylabel("y")

axes[2].imshow(segmented[mid, :, :], cmap="gray", origin="lower")
axes[2].set_title(f"{representative_case}: Otsu segmentation")
axes[2].set_xlabel("z")
axes[2].set_ylabel("y")

fig.suptitle("Representative mid-plane slices", fontsize=14)
plt.tight_layout()
fig.savefig(segmentation_figure_path, dpi=160, bbox_inches="tight")
plt.show()

rep_row = summary_df.loc[summary_df["case"] == representative_case].iloc[0]
print("Representative threshold:", rep_row["threshold"])
print("Representative segmentation mismatch:", rep_row["segmentation_mismatch"])
print("Saved:", segmentation_figure_path)

# %% [markdown]
# ## Comparison plots
#
# The permeability scatter should lie on the identity line because both solvers operate on the same
# extracted network and the same throat conductance values. The porosity comparison shows the effect of
# going from the segmented image to the pruned spanning network used for transport.

# %%
fig, ax = plt.subplots(figsize=(6.4, 5.2))

kmin = float(min(summary_df["k_voids"].min(), summary_df["k_openpnm"].min()))
kmax = float(max(summary_df["k_voids"].max(), summary_df["k_openpnm"].max()))

ax.scatter(
    summary_df["k_voids"],
    summary_df["k_openpnm"],
    s=70,
    color="tab:blue",
)
ax.plot([kmin, kmax], [kmin, kmax], "k--", linewidth=1.5)
for row in summary_df.itertuples(index=False):
    ax.annotate(
        row.case,
        (row.k_voids, row.k_openpnm),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=8,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Kabs from voids [m^2]")
ax.set_ylabel("Kabs from OpenPNM [m^2]")
ax.set_title("Permeability comparison")
ax.grid(alpha=0.3, linestyle=":")

plt.tight_layout()
fig.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("Saved:", comparison_figure_path)

# %%
fig, ax = plt.subplots(figsize=(7.4, 4.8))

ax.plot(
    summary_df["case"],
    summary_df["phi_truth"],
    marker="o",
    linewidth=2,
    label="phi_truth",
)
ax.plot(
    summary_df["case"],
    summary_df["phi_image"],
    marker="s",
    linewidth=2,
    label="phi_segmented",
)
ax.plot(
    summary_df["case"],
    summary_df["phi_abs"],
    marker="^",
    linewidth=2,
    label="phi_abs(network)",
)
ax.plot(
    summary_df["case"],
    summary_df["phi_eff"],
    marker="d",
    linewidth=2,
    label="phi_eff(network)",
)
ax.set_xlabel("Benchmark case")
ax.set_ylabel("Porosity [-]")
ax.set_title("Porosity from image to extracted network")
ax.tick_params(axis="x", rotation=20)
ax.grid(alpha=0.3, linestyle=":")
ax.legend()

plt.tight_layout()
fig.savefig(porosity_figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("Saved:", porosity_figure_path)

# %% [markdown]
# ## Numerical summary
#
# On this benchmark suite, permeability and total-flow differences should remain near machine precision.
# If that stops being true, the first things to inspect are extraction changes, BC labeling, or solver/API
# changes in OpenPNM.

# %%
summary_df.to_csv(report_csv, index=False)
print("Saved:", report_csv)

# %%
max_k_rel = float(summary_df["k_rel_diff"].max())
max_q_rel = float(summary_df["Q_rel_diff"].max())
mean_segmentation_mismatch = float(summary_df["segmentation_mismatch"].mean())

print(f"Max relative permeability difference: {max_k_rel:.3e}")
print(f"Max relative total-flow difference: {max_q_rel:.3e}")
print(f"Mean segmentation mismatch: {mean_segmentation_mismatch:.3e}")
print(
    "OpenPNM versions seen:",
    sorted(summary_df["openpnm_version"].dropna().unique().tolist()),
)

if max_k_rel < 1.0e-10 and max_q_rel < 1.0e-10:
    print("Agreement remains in the machine-precision regime for all benchmark cases.")
else:
    print(
        "Differences exceeded the expected machine-precision regime; inspect the workflow."
    )
