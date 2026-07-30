# %% [markdown]
# # MWE 49 - Three-dimensional FEM manufactured solutions
#
# This experiment exercises the three-dimensional Brinkman weak forms on a
# tetrahedral unit-cube mesh. The exact velocity is generated from derivatives
# of a polynomial boundary bubble, so it vanishes on the boundary and is
# exactly divergence-free. The forcing is derived automatically from the exact
# velocity and pressure.
#
# The low-order `P1/DG0` branch uses the triangular-face subscale (`face3d`)
# pressure-jump coefficient from the 3D verification study. The face solve and
# its refinement level are explicit numerical inputs.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullLocator

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.examples.mms import (
    available_mms_methods,
    bubble_case_3d,
    run_mms_convergence,
)
from voids.fem.singlephase import FEniCSSolverOptions

plt.ioff()

# %% [markdown]
# ## Reproducible configuration
#
# The default sequence is a quick verification run. Extend it to
# `(4, 6, 8, 10, 12, 16, 20)` for comparison with the full linear-element
# study, or through `30` for the focused low-order sequence.

# %%
viscosity = 1.0e-2
reaction = 1.0
resolutions = (4, 6, 8)
methods = available_mms_methods()
face_refinement = 24
absolute_rate_tolerance = 0.35
solver_options = FEniCSSolverOptions.superlu_direct()

case = bubble_case_3d(
    viscosity=viscosity,
    reaction=reaction,
)

# %% [markdown]
# ## Run the refinement studies

# %%
studies = {}
for method in methods:
    print(f"Running {method} on {resolutions}")
    studies[method] = run_mms_convergence(
        case,
        method=method,
        resolutions=resolutions,
        options=solver_options,
        face_refinement=face_refinement,
        keep_solution=False,
    )

rows = []
for method, study in studies.items():
    rows.extend({"method": method, **row} for row in study.as_dicts())
results = pd.DataFrame(rows)
display(results)

# %% [markdown]
# ## Verify observed rates

# %%
rate_rows = []
for method, study in studies.items():
    study.assert_expected_rates(
        absolute_tolerance=absolute_rate_tolerance,
    )
    rate_rows.append(
        {
            "method": method,
            "facet_law": study.metadata["facet_law"],
            "face_refinement": study.metadata["face_refinement"],
            **{f"observed_{name}": value for name, value in study.last_rates.items()},
            **{
                f"nominal_{name}": value
                for name, value in study.expected_rates.as_dict().items()
            },
        }
    )
rate_table = pd.DataFrame(rate_rows)
display(rate_table)

# %% [markdown]
# ## Convergence plot

# %%
metric_labels = {
    "velocity_l2_error": r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$",
    "velocity_h1_error": r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$",
    "pressure_l2_error": r"$\|p-p_h\|_{L^2}$",
}
figure, axes = plt.subplots(1, 3, figsize=(13, 3.7))
for axis, (metric, label) in zip(axes, metric_labels.items()):
    for method, group in results.groupby("method"):
        axis.loglog(
            group["h"],
            group[metric],
            marker="o",
            label=method,
        )
    tick_values = [1.0 / resolution for resolution in reversed(resolutions)]
    axis.set_xticks(tick_values)
    axis.set_xticklabels([f"1/{resolution}" for resolution in reversed(resolutions)])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel(r"$h=1/n$")
    axis.set_ylabel(label)
    axis.grid(True, which="both", alpha=0.3)
axes[0].legend(fontsize=8)
figure.suptitle(rf"3D Brinkman MMS: $\nu={viscosity:g}$, $\gamma={reaction:g}$")
figure.tight_layout()
display(figure)
plt.close(figure)

# %% [markdown]
# ## Interpretation and limitations
#
# - The `face3d` coefficient is a candidate three-dimensional extension, not a
#   theorem already established for discontinuous pressure on tetrahedra.
# - `face_refinement=24` controls the small reference-triangle finite-element
#   solve used to compute the scalar pressure-jump coefficient.
# - These results verify constant-coefficient smooth solutions. They do not by
#   themselves establish robustness for high-contrast vug coefficients.
# - Divergence is a diagnostic for the raw conforming velocity. Raviart-Thomas
#   conservative recovery is not part of this MMS API yet.
