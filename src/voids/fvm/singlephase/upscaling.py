from __future__ import annotations

from dataclasses import dataclass

from voids.fvm.singlephase.tpfa import TPFAResult, solve_tpfa
from voids.image.porosity import PermeabilityMap
from voids.linalg.solve import SolverParameters


@dataclass(slots=True)
class TPFAUpscalingResult:
    """Principal-direction TPFA permeability upscaling result."""

    results: dict[str, TPFAResult]

    @property
    def permeability(self) -> dict[str, float]:
        """Return effective permeability by principal axis."""

        return {axis: result.permeability for axis, result in self.results.items()}

    @property
    def mass_balance_error(self) -> dict[str, float]:
        """Return normalized TPFA mass-balance errors by principal axis."""

        return {axis: result.mass_balance_error for axis, result in self.results.items()}

    @property
    def solve_seconds(self) -> dict[str, float]:
        """Return TPFA linear solve time by principal axis."""

        return {axis: result.solve_seconds for axis, result in self.results.items()}


def _default_axes(ndim: int) -> tuple[str, ...]:
    if ndim == 2:
        return ("x", "y")
    if ndim == 3:
        return ("x", "y", "z")
    raise ValueError("permeability maps must be 2D or 3D")


def upscale_permeability_tpfa(
    permeability_map: PermeabilityMap,
    *,
    axes: tuple[str, ...] | None = None,
    viscosity: float = 1.0,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    solver_method: str = "direct",
    solver_parameters: SolverParameters | None = None,
) -> TPFAUpscalingResult:
    """Compute principal-direction TPFA permeability estimates.

    Parameters
    ----------
    permeability_map :
        Scalar cell-wise permeability map.
    axes :
        Principal axes to solve. By default all axes supported by the map
        dimensionality are solved.
    viscosity, pressure_inlet, pressure_outlet :
        Physical coefficients passed to :func:`voids.fvm.singlephase.tpfa.solve_tpfa`.
    solver_method, solver_parameters :
        Sparse linear solver controls passed through to
        :func:`voids.fvm.singlephase.tpfa.solve_tpfa`.
    """

    solve_axes = axes or _default_axes(permeability_map.ndim)
    return TPFAUpscalingResult(
        {
            axis: solve_tpfa(
                permeability_map,
                flow_axis=axis,
                viscosity=viscosity,
                pressure_inlet=pressure_inlet,
                pressure_outlet=pressure_outlet,
                solver_method=solver_method,
                solver_parameters=solver_parameters,
            )
            for axis in solve_axes
        }
    )


def upscale_principal_permeabilities_tpfa(
    permeability_map: PermeabilityMap,
    *,
    axes: tuple[str, ...] | None = None,
    viscosity: float = 1.0,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    solver_method: str = "direct",
    solver_parameters: SolverParameters | None = None,
) -> dict[str, float]:
    """Return only the principal TPFA permeability values."""

    return upscale_permeability_tpfa(
        permeability_map,
        axes=axes,
        viscosity=viscosity,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        solver_method=solver_method,
        solver_parameters=solver_parameters,
    ).permeability


__all__ = [
    "TPFAUpscalingResult",
    "upscale_permeability_tpfa",
    "upscale_principal_permeabilities_tpfa",
]
