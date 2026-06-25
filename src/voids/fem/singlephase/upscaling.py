from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voids.fem.singlephase._common import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
)
from voids.fem.singlephase.taylorhood import (
    solve_brinkman_taylor_hood,
    solve_darcy_taylor_hood,
)
from voids.fem.singlephase.usfem import solve_brinkman_usfem


FEMBackend = Callable[..., FEMSinglePhaseResult] | str


@dataclass(slots=True)
class FEMUpscalingResult:
    """Principal-direction FEM micro-continuum permeability result."""

    results: dict[str, FEMSinglePhaseResult]
    backend: str

    @property
    def permeability(self) -> dict[str, float]:
        """Return effective permeability by principal axis."""

        return {axis: result.permeability for axis, result in self.results.items()}

    @property
    def solve_seconds(self) -> dict[str, float]:
        """Return wall-clock solve time by principal axis."""

        return {axis: result.solve_seconds for axis, result in self.results.items()}


def _default_axes(ndim: int) -> tuple[str, ...]:
    if ndim == 2:
        return ("x", "y")
    if ndim == 3:
        return ("x", "y", "z")
    raise ValueError("permeability maps must be 2D or 3D")


def _backend_from_name(name: str) -> Callable[..., FEMSinglePhaseResult]:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"th_brinkman", "taylor_hood_brinkman", "brinkman_taylor_hood"}:
        return solve_brinkman_taylor_hood
    if normalized in {"th_darcy", "taylor_hood_darcy", "darcy_taylor_hood", "darcy_darcy"}:
        return solve_darcy_taylor_hood
    if normalized in {"usfem", "usfem_brinkman", "brinkman_usfem"}:
        return solve_brinkman_usfem
    raise ValueError(
        "backend must be one of 'taylor_hood_brinkman', 'taylor_hood_darcy', or 'usfem_brinkman'"
    )


def upscale_permeability_fem(
    problem: FEMMapProblem,
    *,
    backend: FEMBackend = "taylor_hood_brinkman",
    axes: tuple[str, ...] | None = None,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    options: FEniCSSolverOptions | None = None,
    backend_kwargs: dict[str, object] | None = None,
) -> FEMUpscalingResult:
    """Compute principal-direction FEM permeability estimates.

    Parameters
    ----------
    problem :
        Porosity/permeability map problem.
    backend :
        Either a backend name or a compatible solver callable. Supported names
        are ``"taylor_hood_brinkman"``, ``"taylor_hood_darcy"``, and
        ``"usfem_brinkman"``.
    axes :
        Principal axes to solve. By default all axes supported by the map
        dimensionality are solved.
    pressure_inlet, pressure_outlet :
        Natural pressure values imposed on opposite faces of each flow axis.
    options :
        Linear solver options passed to the FEM backend. The default preserves
        PETSc where available and uses a serial SciPy direct solve on native
        Windows when PETSc is unavailable.
    backend_kwargs :
        Additional backend-specific keyword arguments.
    """

    solve_axes = axes or _default_axes(problem.permeability_map.ndim)
    solver = _backend_from_name(backend) if isinstance(backend, str) else backend
    kwargs = dict(backend_kwargs or {})
    results = {
        axis: solver(
            problem,
            flow_axis=axis,
            pressure_inlet=pressure_inlet,
            pressure_outlet=pressure_outlet,
            options=options,
            **kwargs,
        )
        for axis in solve_axes
    }
    backend_name = backend if isinstance(backend, str) else getattr(backend, "__name__", "callable")
    return FEMUpscalingResult(results=results, backend=backend_name)


def upscale_principal_permeabilities_fem(
    problem: FEMMapProblem,
    *,
    backend: FEMBackend = "taylor_hood_brinkman",
    axes: tuple[str, ...] | None = None,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    options: FEniCSSolverOptions | None = None,
    backend_kwargs: dict[str, object] | None = None,
) -> dict[str, float]:
    """Return only the principal FEM permeability values."""

    return upscale_permeability_fem(
        problem,
        backend=backend,
        axes=axes,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        options=options,
        backend_kwargs=backend_kwargs,
    ).permeability


__all__ = [
    "FEMUpscalingResult",
    "upscale_permeability_fem",
    "upscale_principal_permeabilities_fem",
]
