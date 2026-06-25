"""Single-phase finite-element Darcy and Brinkman backends."""

from voids.fem.singlephase._common import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    LinearSolverBackend,
)
from voids.fem.singlephase.taylorhood import (
    solve_brinkman_taylor_hood,
    solve_darcy_taylor_hood,
)
from voids.fem.singlephase.upscaling import (
    FEMUpscalingResult,
    upscale_permeability_fem,
    upscale_principal_permeabilities_fem,
)
from voids.fem.singlephase.usfem import solve_brinkman_usfem

__all__ = [
    "FEMMapProblem",
    "FEMSinglePhaseResult",
    "FEMUpscalingResult",
    "FEniCSSolverOptions",
    "LinearSolverBackend",
    "solve_brinkman_taylor_hood",
    "solve_brinkman_usfem",
    "solve_darcy_taylor_hood",
    "upscale_permeability_fem",
    "upscale_principal_permeabilities_fem",
]
