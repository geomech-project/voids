"""Single-phase finite-volume Darcy-flow backends."""

from voids.fvm.singlephase.tpfa import (
    TPFAResult,
    solve_tpfa,
)
from voids.fvm.singlephase.upscaling import (
    TPFAUpscalingResult,
    upscale_permeability_tpfa,
    upscale_principal_permeabilities_tpfa,
)

__all__ = [
    "TPFAResult",
    "TPFAUpscalingResult",
    "solve_tpfa",
    "upscale_permeability_tpfa",
    "upscale_principal_permeabilities_tpfa",
]
