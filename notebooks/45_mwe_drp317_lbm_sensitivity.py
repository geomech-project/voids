# %% [markdown]
# # MWE 45 - DRP-317 LBM setup sensitivity
#
# This notebook summarizes the direct-image LBM setup study used to choose the
# package-level `XLBOptions.steady_stokes_defaults()` preset. The expensive XLB
# runs are stored as CSV artifacts by the validation workflow; this notebook
# rebuilds the diagnostic tables and figures from those artifacts.
#
# The scientific question is narrow: do convergence tolerance, reservoir buffer
# length, lattice pressure drop, or BGK lattice viscosity explain the large LBM
# overprediction observed on the small DRP-317 Berea and Bentheimer same-ROI
# validation crops?

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voids.paths import project_root

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

plt.ioff()

# %%
output_dir = project_root() / "notebooks" / "outputs" / "45_mwe_drp317_lbm_sensitivity"
asset_dir = project_root() / "docs" / "assets" / "validation"

representative_csv = output_dir / "drp317_lbm_representative_sensitivity.csv"
recommended_csv = output_dir / "drp317_lbm_recommended_all_axes.csv"
default_update_csv = output_dir / "drp317_lbm_default_update_all_axes.csv"

if not representative_csv.exists():
    representative_csv = asset_dir / representative_csv.name
if not recommended_csv.exists():
    recommended_csv = asset_dir / recommended_csv.name
if not default_update_csv.exists():
    default_update_csv = asset_dir / default_update_csv.name

representative = pd.read_csv(representative_csv)
recommended = pd.read_csv(recommended_csv)
default_update = pd.read_csv(default_update_csv)

display(representative)
display(recommended)
display(default_update)

# %% [markdown]
# ## Representative Setup Sweep
#
# The representative sweep varies one numerical choice at a time around the
# selected strict buffer-12 preset. Percent changes are computed relative to the
# `strict_b12` row for each sample/axis.

# %%
config_order = [
    "notebook_current",
    "library_default",
    "strict_b6",
    "strict_b12",
    "strict_b18",
    "strict_b12_dp_half",
    "strict_b12_dp_double",
    "strict_b12_nu005",
    "strict_b12_nu0167",
]
config_labels = {
    "notebook_current": "previous notebook\n6 buf, 1e-3",
    "library_default": "previous library\n6 buf, 5e-4",
    "strict_b6": "strict\n6 buf",
    "strict_b12": "strict\n12 buf",
    "strict_b18": "strict\n18 buf",
    "strict_b12_dp_half": "half dp\n12 buf",
    "strict_b12_dp_double": "double dp\n12 buf",
    "strict_b12_nu005": "nu=0.05\n12 buf",
    "strict_b12_nu0167": "nu=1/6\n12 buf",
}

sensitivity_rows: list[dict[str, object]] = []
for sample, axis in [("Berea", "x"), ("Bentheimer", "y")]:
    subset = representative[
        (representative["sample"] == sample) & (representative["axis"] == axis)
    ].set_index("config")
    reference = float(subset.loc["strict_b12", "K_mD"])
    for config in config_order:
        value = float(subset.loc[config, "K_mD"])
        sensitivity_rows.append(
            {
                "sample": sample,
                "axis": axis,
                "config": config,
                "K_mD": value,
                "relative_change_pct": 100.0 * (value / reference - 1.0),
            }
        )

sensitivity_summary = pd.DataFrame(sensitivity_rows)
display(sensitivity_summary)

# %%
fig, axes = plt.subplots(
    2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True
)
for sample, axis, color in [
    ("Berea", "x", "tab:blue"),
    ("Bentheimer", "y", "tab:orange"),
]:
    subset = (
        representative[
            (representative["sample"] == sample) & (representative["axis"] == axis)
        ]
        .set_index("config")
        .reindex(config_order)
    )
    reference = float(subset.loc["strict_b12", "K_mD"])
    x = np.arange(len(config_order), dtype=float)
    axes[0].plot(
        x, subset["K_mD"], marker="o", label=f"{sample} {axis}-flow", color=color
    )
    axes[1].plot(
        x,
        100.0 * (subset["K_mD"] / reference - 1.0),
        marker="o",
        label=f"{sample} {axis}-flow",
        color=color,
    )

axes[0].set_ylabel("permeability [mD]")
axes[0].set_yscale("log")
axes[0].set_title("DRP-317 representative LBM setup sensitivity")
axes[0].grid(True, which="both", alpha=0.25)
axes[0].legend()
axes[1].axhline(0.0, color="black", linewidth=1.0)
axes[1].set_ylabel("change relative to strict buffer-12 [%]")
axes[1].set_xticks(np.arange(len(config_order)))
axes[1].set_xticklabels(
    [config_labels[item] for item in config_order], rotation=25, ha="right"
)
axes[1].grid(True, axis="y", alpha=0.25)
axes[1].legend()

path = output_dir / "drp317_lbm_representative_sensitivity.png"
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=200)
display(fig)
path

# %% [markdown]
# ## Previous Validation Preset vs Recommended Preset

# %%
fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
plot_rows = default_update.sort_values(["sample", "axis"]).reset_index(drop=True)
x = np.arange(len(plot_rows), dtype=float)
width = 0.36
ax.bar(
    x - width / 2,
    plot_rows["K_mD_previous"],
    width=width,
    label="previous validation preset",
)
ax.bar(x + width / 2, plot_rows["K_mD"], width=width, label="recommended preset")
for sample, experimental_k_mD in {"Berea": 121.0, "Bentheimer": 386.0}.items():
    indices = [index for index, row in plot_rows.iterrows() if row["sample"] == sample]
    ax.hlines(
        experimental_k_mD,
        min(indices) - 0.55,
        max(indices) + 0.55,
        colors="black",
        linestyles="--",
        linewidth=1.0,
    )
    ax.text(
        max(indices) + 0.58,
        experimental_k_mD,
        f"{sample} exp.",
        va="center",
        fontsize=8,
    )
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{row.sample} {row.axis}" for row in plot_rows.itertuples()],
    rotation=25,
    ha="right",
)
ax.set_ylabel("permeability [mD]")
ax.set_title("DRP-317 LBM validation rows: previous vs recommended preset")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend()

path = output_dir / "drp317_lbm_default_update_all_axes.png"
fig.savefig(path, dpi=200)
display(fig)
path

# %% [markdown]
# ## Interpretation
#
# The strict buffer-12 preset is a numerically safer default, but it changes the
# same-ROI validation permeabilities by only about 1-6 %. Pressure-drop
# invariance is good in the representative axes, while BGK viscosity sensitivity
# is the largest observed numerical knob. The remaining LBM overprediction
# relative to the published scalar bulk permeabilities should therefore be read
# as a small-ROI, segmentation/porosity, voxel-boundary, and representativeness
# issue rather than as a simple tolerance-setting failure.
