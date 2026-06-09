from __future__ import annotations

from typing import Any

import numpy as np

from voids.lbm.singlephase.xlb import (
    XLBConvergenceWarning,
    XLBDirectSimulationResult,
    XLBOptions,
    solve_binary_volume_with_xlb,
)


def steady_stokes_options(**overrides: Any) -> XLBOptions:
    """Return XLB options configured for the steady Stokes-limit formulation."""

    return XLBOptions.steady_stokes_defaults(**overrides)


def solve_binary_volume_stokes(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    options: XLBOptions | None = None,
) -> XLBDirectSimulationResult:
    """Solve Stokes-limit flow on a binary image using the XLB backend.

    This is the package-facing LBM namespace for direct-image single-phase
    solves. Benchmark utilities consume this lower-level adapter for
    verification workflows. The input follows the current XLB adapter
    convention: ``void=1`` and ``solid=0``.
    """

    return solve_binary_volume_with_xlb(
        phases,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        options=options or steady_stokes_options(),
    )


__all__ = [
    "XLBConvergenceWarning",
    "XLBDirectSimulationResult",
    "XLBOptions",
    "solve_binary_volume_stokes",
    "steady_stokes_options",
]
