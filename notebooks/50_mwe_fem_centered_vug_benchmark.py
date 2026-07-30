# %% [markdown]
# # MWE 50 - Body-fitted centered-vug FEM benchmark
#
# This experiment reproduces the centered circular and spherical vug geometry
# with Gmsh physical tags and solves the same mesh with Taylor-Hood and
# `P1/DG1` USFEM. It checks flux parity between formulations and also exposes
# the known `P1/DG0` high-contrast failure mode.
#
# The benchmark is pressure-driven on the unit domain:
#
# \[
# \gamma_\mathrm{matrix}=10^7,\quad
# \gamma_\mathrm{vug}=1,\quad
# \nu=10^{-2},\quad p_L=1,\quad p_R=-1.
# \]

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.examples.mms import (
    CenteredVugBenchmark,
    run_centered_vug_benchmark,
)
from voids.fem.singlephase import FEniCSSolverOptions

plt.ioff()

# %% [markdown]
# ## Reproducible configuration
#
# The report-scale values are approximately `resolution=96` in 2D and
# `resolution=30` in 3D. The smaller defaults below are intended for a quick
# end-to-end run and should not be presented as mesh-converged benchmark values.

# %%
benchmark_resolutions = {2: 8, 3: 6}
reference_methods = ("taylor_hood", "usfem_p1dg1")
run_low_order_2d_diagnostic = True
run_low_order_2d_safeguard_sensitivity = True
solver_options = FEniCSSolverOptions.superlu_direct()

# %% [markdown]
# ## Run like-for-like body-fitted comparisons

# %%
rows = []
results_by_dimension = {}
for dimension, resolution in benchmark_resolutions.items():
    benchmark = CenteredVugBenchmark(
        dimension=dimension,
        resolution=resolution,
        mesh_representation="body_fitted",
    )
    methods = list(reference_methods)
    if dimension == 2 and run_low_order_2d_diagnostic:
        methods.append("usfem_p1dg0")
    dimension_results = {}
    for method in methods:
        print(f"Running {dimension}D {method} at nominal resolution {resolution}")
        result = run_centered_vug_benchmark(
            benchmark,
            method=method,
            options=solver_options,
        )
        dimension_results[method] = result
        rows.append(
            {
                "dimension": dimension,
                "resolution": resolution,
                "method": method,
                "cells": result.metadata["num_cells"],
                "facet_law": result.metadata["facet_law"],
                "facet_size_mode": result.metadata["facet_size_mode"],
                "analytic_vug_fraction": result.metadata["analytic_vug_fraction"],
                "represented_vug_fraction": result.metadata["represented_vug_fraction"],
                "flow_rate": result.flow_rate,
                "permeability_diagnostic": result.permeability,
                "solve_seconds": result.solve_seconds,
            }
        )
    results_by_dimension[dimension] = dimension_results

if run_low_order_2d_safeguard_sensitivity:
    benchmark = CenteredVugBenchmark(
        dimension=2,
        resolution=benchmark_resolutions[2],
        mesh_representation="body_fitted",
    )
    for label, controls in (
        ("usfem_p1dg0_tau0", {"tau_factor": 0.0}),
        ("usfem_p1dg0_cap0.5", {"tau_gamma_cap": 0.5}),
    ):
        print(f"Running 2D {label}")
        result = run_centered_vug_benchmark(
            benchmark,
            method="usfem_p1dg0",
            options=solver_options,
            **controls,
        )
        rows.append(
            {
                "dimension": 2,
                "resolution": benchmark.resolution,
                "method": label,
                "cells": result.metadata["num_cells"],
                "facet_law": result.metadata["facet_law"],
                "facet_size_mode": result.metadata["facet_size_mode"],
                "analytic_vug_fraction": result.metadata["analytic_vug_fraction"],
                "represented_vug_fraction": result.metadata["represented_vug_fraction"],
                "flow_rate": result.flow_rate,
                "permeability_diagnostic": result.permeability,
                "solve_seconds": result.solve_seconds,
            }
        )

summary = pd.DataFrame(rows)
reference_flux = summary.loc[
    summary["method"] == "taylor_hood", ["dimension", "flow_rate"]
].rename(columns={"flow_rate": "taylor_hood_flow_rate"})
summary = summary.merge(reference_flux, on="dimension")
summary["flux_ratio_to_taylor_hood"] = (
    summary["flow_rate"] / summary["taylor_hood_flow_rate"]
)
display(summary)

# %% [markdown]
# ## Formulation parity checks
#
# The 2D quick mesh should already give close `P1/DG1`/Taylor-Hood fluxes.
# The 3D quick mesh is intentionally coarse, so its ratio is reported rather
# than asserted as a mesh-converged value.

# %%
two_dimensional_ratio = summary.loc[
    (summary["dimension"] == 2) & (summary["method"] == "usfem_p1dg1"),
    "flux_ratio_to_taylor_hood",
].item()
assert abs(two_dimensional_ratio - 1.0) < 5.0e-3

parity = summary.loc[
    summary["method"].isin(reference_methods),
    [
        "dimension",
        "resolution",
        "method",
        "represented_vug_fraction",
        "flux_ratio_to_taylor_hood",
    ],
]
display(parity)

# %% [markdown]
# ## Flux comparison

# %%
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
reference_plot_table = summary.loc[summary["method"].isin(reference_methods)].pivot(
    index="dimension",
    columns="method",
    values="flux_ratio_to_taylor_hood",
)
reference_plot_table.plot.bar(ax=axes[0])
axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.38),
    ncol=2,
)
axes[0].axhline(1.0, color="black", linewidth=1.0)
axes[0].set_ylim(0.75, 1.05)
axes[0].set_ylabel("outlet flux / Taylor-Hood outlet flux")
axes[0].set_xlabel("spatial dimension")
axes[0].grid(True, axis="y", alpha=0.3)
axes[0].set_title("Meaningful formulation comparison")

low_order = summary.loc[
    summary["method"].str.startswith("usfem_p1dg0"),
    ["dimension", "method", "flux_ratio_to_taylor_hood"],
]
if low_order.empty:
    axes[1].set_visible(False)
else:
    axes[1].bar(
        low_order["method"],
        low_order["flux_ratio_to_taylor_hood"],
        color="tab:orange",
    )
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("outlet flux / Taylor-Hood outlet flux (log scale)")
    axes[1].set_xlabel("P1/DG0 cell-stabilization branch")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_title("P1/DG0 negative diagnostic")
figure.tight_layout()
display(figure)
plt.close(figure)

# %% [markdown]
# ## Interpretation and limitations
#
# - Gmsh creates separate physical cell groups for the matrix and vug and
#   physical facet groups for all outer faces and the internal interface.
# - Both formulations in a dimension are regenerated deterministically from
#   the same geometry and target size. Their flux comparison is therefore a
#   like-for-like discretization check.
# - The low-order `P1/DG0` row is a documented negative high-contrast result:
#   its reaction-only volumetric term can nearly cancel matrix drag. A small
#   divergence norm would not make that flux physically acceptable.
# - The `tau0` and `cap0.5` rows are explicit sensitivity branches. The cap
#   enforces \(\gamma\tau_K\le 0.5\), so at least half of the physical drag
#   remains. Neither branch should be accepted without mesh refinement and a
#   comparison to the Taylor-Hood reference.
# - The reported permeability is a flow-based diagnostic for this boundary
#   value problem. Quantitative comparison with report tables requires their
#   finer nominal resolutions.
