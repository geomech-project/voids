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

# %%
flow_axis = "x"
axis_index = 0
voxel_size = 2.0e-6
fluid = FluidSinglePhase(viscosity=1.0e-3)
options = SinglePhaseOptions(
    conductance_model="valvatne_blunt_baseline",
    solver="direct",
)

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
plt.show()

rep_row = summary_df.loc[summary_df["case"] == representative_case].iloc[0]
print("Representative threshold:", rep_row["threshold"])
print("Representative segmentation mismatch:", rep_row["segmentation_mismatch"])

# %% [markdown]
# ## Comparison plots
#
# The permeability scatter should lie on the identity line because both solvers operate on the same
# extracted network and the same throat conductance values. The porosity comparison shows the effect of
# going from the segmented image to the pruned spanning network used for transport.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

kmin = float(min(summary_df["k_voids"].min(), summary_df["k_openpnm"].min()))
kmax = float(max(summary_df["k_voids"].max(), summary_df["k_openpnm"].max()))

axes[0].scatter(
    summary_df["k_voids"],
    summary_df["k_openpnm"],
    s=70,
    color="tab:blue",
)
axes[0].plot([kmin, kmax], [kmin, kmax], "k--", linewidth=1.5)
for row in summary_df.itertuples(index=False):
    axes[0].annotate(
        row.case,
        (row.k_voids, row.k_openpnm),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=8,
    )
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Kabs from voids [m^2]")
axes[0].set_ylabel("Kabs from OpenPNM [m^2]")
axes[0].set_title("Permeability comparison")
axes[0].grid(alpha=0.3, linestyle=":")

axes[1].plot(
    summary_df["case"],
    summary_df["phi_truth"],
    marker="o",
    linewidth=2,
    label="phi_truth",
)
axes[1].plot(
    summary_df["case"],
    summary_df["phi_image"],
    marker="s",
    linewidth=2,
    label="phi_segmented",
)
axes[1].plot(
    summary_df["case"],
    summary_df["phi_abs"],
    marker="^",
    linewidth=2,
    label="phi_abs(network)",
)
axes[1].plot(
    summary_df["case"],
    summary_df["phi_eff"],
    marker="d",
    linewidth=2,
    label="phi_eff(network)",
)
axes[1].set_xlabel("Benchmark case")
axes[1].set_ylabel("Porosity [-]")
axes[1].set_title("Porosity from image to extracted network")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(alpha=0.3, linestyle=":")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Numerical summary
#
# On this benchmark suite, permeability and total-flow differences should remain near machine precision.
# If that stops being true, the first things to inspect are extraction changes, BC labeling, or solver/API
# changes in OpenPNM.

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
