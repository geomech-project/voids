# %% [markdown]
# # MWE 51 - Presentation-scale MMS replication
#
# This experiment treats the numerical values in the supplied USFEM
# presentations and reports as regression targets. A replication passes only
# when the case definition, coefficients, finite-element pair, facet law, and
# finest mesh pair agree with the stored profile and every compared scalar lies
# within its documented tolerance.
#
# The default live run reproduces the two-dimensional equal-order boundary-layer
# study. The larger three-dimensional MMS and body-fitted vug runs are opt-in
# because they require substantially more memory and solve time.

# %%
from __future__ import annotations

import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.examples.mms import (
    presentation_mms_references,
    presentation_vug_references,
    run_presentation_mms,
    run_presentation_vug,
)
from voids.fem.singlephase import FEniCSSolverOptions

# %% [markdown]
# ## Reproducible configuration
#
# The 2D default runs through \(128^2\), matching the equal-order report row.
# Set the 3D and vug flags only when report-scale cost is acceptable. The 3D
# linear MMS sequence ends at \(20^3\); the vug case regenerates the body-fitted
# Gmsh mesh with target size \(\sqrt{3}/30\).

# %%
two_dimensional_reference = "2d_boundary_layer_p1dg1"
run_three_dimensional_face3d = False
run_report_scale_vug = False
solver_options = FEniCSSolverOptions.superlu_direct()

# %% [markdown]
# ## Auditable baseline catalogue
#
# These are tracked numerical targets; no ignored `tmp/` file is read at
# runtime. Rates use the final consecutive mesh pair from each supplied study.

# %%
mms_catalogue_rows = []
for reference in presentation_mms_references():
    mms_catalogue_rows.append(
        {
            "name": reference.name,
            "dimension": reference.case.dimension,
            "method": reference.method,
            "facet_law": reference.facet_law,
            "facet_size_mode": reference.facet_size_mode,
            "finest_pair": reference.resolutions[-2:],
            "target_metrics": ", ".join(
                quantity.metric for quantity in reference.quantities
            ),
        }
    )
mms_catalogue = pd.DataFrame(mms_catalogue_rows)
display(mms_catalogue)

vug_catalogue = pd.DataFrame(
    [
        {
            "name": reference.name,
            "method": reference.method,
            "resolution": reference.benchmark.resolution,
            "facet_law": reference.facet_law,
            "facet_size_mode": reference.facet_size_mode,
            "target_flow_rate": next(
                quantity.expected
                for quantity in reference.quantities
                if quantity.metric == "flow_rate"
            ),
        }
        for reference in presentation_vug_references()
    ]
)
display(vug_catalogue)

# %% [markdown]
# ## Replicate the 2D presentation row

# %%
two_dimensional_run = run_presentation_mms(
    two_dimensional_reference,
    options=solver_options,
)
two_dimensional_run.assert_matches()

two_dimensional_errors = pd.DataFrame(two_dimensional_run.result.as_dicts())
two_dimensional_comparison = pd.DataFrame(two_dimensional_run.comparison.as_dicts())
display(two_dimensional_errors)
display(two_dimensional_comparison)

# %% [markdown]
# ## Optional 3D face-subscale replication
#
# The supplied 3D table reports separate rows for the P1/DG0 and P1/DG1
# formulations. Both use the same polynomial exact solution and
# `face_refinement=24`; each live run below executes the complete
# \((4,6,8,10,12,16,20)^3\) sequence.

# %%
three_dimensional_runs = {}
if run_three_dimensional_face3d:
    for reference_name in (
        "3d_bubble_usfem_p1dg0_face3d",
        "3d_bubble_usfem_p1dg1_face3d",
    ):
        print(f"Running {reference_name}")
        replication = run_presentation_mms(
            reference_name,
            options=solver_options,
        )
        replication.assert_matches()
        three_dimensional_runs[reference_name] = replication

if three_dimensional_runs:
    display(
        pd.concat(
            [
                pd.DataFrame(replication.comparison.as_dicts()).assign(
                    reference=reference_name
                )
                for reference_name, replication in three_dimensional_runs.items()
            ],
            ignore_index=True,
        )
    )
else:
    print("3D report-scale runs disabled; set run_three_dimensional_face3d=True.")

# %% [markdown]
# ## Optional report-scale 3D centered-vug replication
#
# This is not an MMS because no exact flow field is available. It is a
# like-for-like body-fitted flux benchmark. Each formulation is checked against
# its supplied flux, and their ratio is also reported.

# %%
vug_runs = {}
if run_report_scale_vug:
    for reference_name in (
        "3d_centered_vug_taylor_hood",
        "3d_centered_vug_p1dg1",
    ):
        print(f"Running {reference_name}")
        replication = run_presentation_vug(
            reference_name,
            options=solver_options,
        )
        replication.assert_matches()
        vug_runs[reference_name] = replication

if vug_runs:
    taylor_hood_flux = vug_runs["3d_centered_vug_taylor_hood"].result.flow_rate
    p1dg1_flux = vug_runs["3d_centered_vug_p1dg1"].result.flow_rate
    display(
        pd.DataFrame(
            [
                {
                    "Taylor-Hood flux": taylor_hood_flux,
                    "P1/DG1 flux": p1dg1_flux,
                    "relative difference": abs(p1dg1_flux - taylor_hood_flux)
                    / abs(taylor_hood_flux),
                }
            ]
        )
    )
else:
    print("3D report-scale vug runs disabled; set run_report_scale_vug=True.")

# %% [markdown]
# ## Interpretation and limitations
#
# - The comparisons use the exact reported finest pair; a quick coarse-grid
#   trend cannot pass as a presentation replication.
# - Gmsh cell counts may vary slightly with Gmsh version. The vug gate therefore
#   checks the represented volume and flux rather than an exact tetrahedron
#   count.
# - The shipped gates cover raw MMS errors/rates and the centered-vug fluxes
#   available from the current `voids` FEM result objects.
# - The presentation's Raviart--Thomas recovered-velocity divergence is not
#   claimed here because conservative recovery is not yet part of the shipped
#   `voids` FEM API. Adding it requires a separate implementation and regression
#   layer; raw divergence and recovered divergence must not be conflated.
