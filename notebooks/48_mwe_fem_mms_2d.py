# %% [markdown]
# # MWE 48 - Two-dimensional FEM manufactured solutions
#
# This experiment verifies the finite-element Brinkman formulations in `voids`
# against a divergence-free exact solution on the unit square. The body force is
# manufactured automatically from
#
# \[
# -\nu\Delta\mathbf{u}+\gamma\mathbf{u}+\nabla p=\mathbf{f},
# \qquad \nabla\cdot\mathbf{u}=0.
# \]
#
# Success means that the last mesh pair attains the nominal smooth-solution
# rates within the documented tolerance. The default viscosity is deliberately
# moderate so the quick mesh sequence resolves the exponential boundary layer.

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
    boundary_layer_case_2d,
    run_mms_convergence,
)
from voids.fem.singlephase import FEniCSSolverOptions

plt.ioff()

# %% [markdown]
# ## Reproducible configuration
#
# Set `use_reference_boundary_layer=True` to reproduce the sharper
# \(\nu=10^{-2}\) case. That case needs the finer sequence through \(128^2\)
# before velocity rates are asymptotic.

# %%
use_reference_boundary_layer = False
if use_reference_boundary_layer:
    viscosity = 1.0e-2
    resolutions = (16, 32, 64, 128)
else:
    viscosity = 1.0e-1
    resolutions = (8, 16, 32)

reaction = 1.0
methods = available_mms_methods()
absolute_rate_tolerance = 0.35
solver_options = FEniCSSolverOptions.superlu_direct()

case = boundary_layer_case_2d(
    viscosity=viscosity,
    reaction=reaction,
)

# %% [markdown]
# ## Run all shipped FEM formulations
#
# `P1/DG0` and `P1/DG1` use the reaction-diffusion pressure-jump coefficient
# in 2D. Taylor-Hood uses the unstabilized `P2/P1` mixed form.

# %%
studies = {}
for method in methods:
    print(f"Running {method} on {resolutions}")
    studies[method] = run_mms_convergence(
        case,
        method=method,
        resolutions=resolutions,
        options=solver_options,
        keep_solution=False,
    )

rows = []
for method, study in studies.items():
    rows.extend({"method": method, **row} for row in study.as_dicts())
results = pd.DataFrame(rows)
display(results)

# %% [markdown]
# ## Check the finest-pair rates
#
# The assertion is intentionally one-sided: an observed rate must not fall more
# than `absolute_rate_tolerance` below the nominal rate. Superconvergent
# pre-asymptotic slopes are not rejected.

# %%
rate_rows = []
for method, study in studies.items():
    study.assert_expected_rates(
        absolute_tolerance=absolute_rate_tolerance,
    )
    rate_rows.append(
        {
            "method": method,
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
figure.suptitle(rf"2D Brinkman MMS: $\nu={viscosity:g}$, $\gamma={reaction:g}$")
figure.tight_layout()
display(figure)
plt.close(figure)

# %% [markdown]
# ## Interpretation and limitations
#
# - Pressure errors are aligned modulo their domain mean because incompressible
#   pressure is defined only up to a constant.
# - The reported velocity \(H^1\) error is the full norm, not only its
#   seminorm.
# - Passing this refinement test verifies the implementation for this smooth
#   manufactured case. It does not validate heterogeneous coefficients,
#   geometry representation, or a particular porous sample.
# - At \(\nu=10^{-2}\), a sequence ending at \(64^2\) is still pre-asymptotic
#   in velocity; use the reference sequence through \(128^2\).
