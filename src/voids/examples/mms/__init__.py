"""Manufactured Brinkman solutions and finite-element convergence studies."""

from voids.examples.mms._core import (
    BrinkmanMMSCase,
    ConvergenceExpectation,
    MMSConvergenceLevel,
    MMSConvergenceResult,
    MMSDiscreteSolution,
    MMSFacetLaw,
    MMSFacetSizeMode,
    MMSMethod,
)
from voids.examples.mms._runner import (
    available_mms_methods,
    face3d_pressure_jump_coefficient,
    observed_rate,
    run_mms_convergence,
)
from voids.examples.mms.cases_2d import boundary_layer_case_2d
from voids.examples.mms.cases_3d import bubble_case_3d
from voids.examples.mms.replication import (
    MMSPresentationReference,
    MMSPresentationRun,
    PresentationComparison,
    ReferenceComparison,
    ReferenceQuantity,
    VugPresentationReference,
    VugPresentationRun,
    compare_mms_with_presentation,
    compare_vug_with_presentation,
    presentation_mms_references,
    presentation_vug_references,
    run_presentation_mms,
    run_presentation_vug,
)
from voids.examples.mms.vug import (
    BodyFittedVugMesh,
    CenteredVugBenchmark,
    VugMeshRepresentation,
    make_body_fitted_centered_vug_mesh,
    run_centered_vug_benchmark,
)
from voids.examples.mms.vug_flow import (
    CenteredVugFlowCase2D,
    CenteredVugFlowModel,
    M2_PER_MILLIDARCY,
    run_centered_vug_flow_case,
)

__all__ = [
    "BodyFittedVugMesh",
    "BrinkmanMMSCase",
    "CenteredVugBenchmark",
    "CenteredVugFlowCase2D",
    "CenteredVugFlowModel",
    "ConvergenceExpectation",
    "MMSConvergenceLevel",
    "MMSConvergenceResult",
    "MMSDiscreteSolution",
    "MMSFacetLaw",
    "MMSFacetSizeMode",
    "MMSMethod",
    "MMSPresentationReference",
    "MMSPresentationRun",
    "PresentationComparison",
    "ReferenceComparison",
    "ReferenceQuantity",
    "M2_PER_MILLIDARCY",
    "VugMeshRepresentation",
    "VugPresentationReference",
    "VugPresentationRun",
    "available_mms_methods",
    "boundary_layer_case_2d",
    "bubble_case_3d",
    "compare_mms_with_presentation",
    "compare_vug_with_presentation",
    "face3d_pressure_jump_coefficient",
    "make_body_fitted_centered_vug_mesh",
    "observed_rate",
    "presentation_mms_references",
    "presentation_vug_references",
    "run_centered_vug_benchmark",
    "run_centered_vug_flow_case",
    "run_mms_convergence",
    "run_presentation_mms",
    "run_presentation_vug",
]
