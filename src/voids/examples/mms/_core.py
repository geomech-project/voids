from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


MMSMethod = Literal["taylor_hood", "usfem_p1dg0", "usfem_p1dg1"]
MMSFacetLaw = Literal[
    "auto",
    "classic",
    "reaction_diffusion",
    "shifted",
    "face3d",
]
MMSFacetSizeMode = Literal["cell_diameter", "facet_diameter", "representative"]
MMSExactSolutionFactory = Callable[[Any, Any], tuple[Any, Any]]
MMSPointEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

_ERROR_NAMES = (
    "velocity_l2",
    "velocity_h1",
    "pressure_l2",
    "divergence_l2",
)


@dataclass(frozen=True, slots=True)
class BrinkmanMMSCase:
    r"""Exact solution data for a manufactured Brinkman problem.

    Parameters
    ----------
    name :
        Stable case identifier.
    dimension :
        Spatial dimension, either 2 or 3.
    viscosity :
        Constant Brinkman diffusion coefficient :math:`\nu`.
    reaction :
        Constant Darcy reaction coefficient :math:`\gamma`.
    exact_solution_factory :
        Callable receiving the imported ``ufl`` module and a DOLFINx mesh. It
        must return ``(velocity, pressure)`` as UFL expressions. The forcing is
        manufactured automatically as
        :math:`-\nu\Delta u+\gamma u+\nabla p`.
    point_evaluator :
        Optional NumPy evaluator for plotting. It receives coordinates with
        shape ``(dimension, npoints)`` and returns velocity with shape
        ``(dimension, npoints)`` and pressure with shape ``(npoints,)``.
    description, reference :
        Concise provenance suitable for notebook and metadata reporting.

    Notes
    -----
    The convergence runner imposes the exact velocity on the complete boundary
    and compares pressure modulo an additive constant.
    """

    name: str
    dimension: Literal[2, 3]
    viscosity: float
    reaction: float
    exact_solution_factory: MMSExactSolutionFactory = field(repr=False)
    point_evaluator: MMSPointEvaluator | None = field(default=None, repr=False)
    description: str = ""
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.dimension not in {2, 3}:
            raise ValueError("dimension must be either 2 or 3")
        if self.viscosity <= 0.0 or not np.isfinite(self.viscosity):
            raise ValueError("viscosity must be positive and finite")
        if self.reaction < 0.0 or not np.isfinite(self.reaction):
            raise ValueError("reaction must be nonnegative and finite")
        if not callable(self.exact_solution_factory):
            raise TypeError("exact_solution_factory must be callable")
        if self.point_evaluator is not None and not callable(self.point_evaluator):
            raise TypeError("point_evaluator must be callable")

    def ufl_solution(self, ufl: Any, domain: Any) -> tuple[Any, Any]:
        """Return exact velocity and pressure UFL expressions on ``domain``."""

        velocity, pressure = self.exact_solution_factory(ufl, domain)
        return velocity, pressure

    def evaluate(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the exact fields at NumPy coordinates for plotting."""

        if self.point_evaluator is None:
            raise NotImplementedError(f"Case {self.name!r} does not define a NumPy point evaluator")
        points = np.asarray(coordinates, dtype=float)
        if points.ndim == 1:
            points = points.reshape(self.dimension, 1)
        if points.ndim != 2 or points.shape[0] < self.dimension:
            raise ValueError(
                f"coordinates must have shape (dimension, npoints); received {points.shape}"
            )
        velocity, pressure = self.point_evaluator(points[: self.dimension])
        velocity_array = np.asarray(velocity, dtype=float)
        pressure_array = np.asarray(pressure, dtype=float)
        expected_velocity_shape = (self.dimension, points.shape[1])
        if velocity_array.shape != expected_velocity_shape:
            raise ValueError(
                "point_evaluator returned velocity with shape "
                f"{velocity_array.shape}; expected {expected_velocity_shape}"
            )
        if pressure_array.shape != (points.shape[1],):
            raise ValueError(
                "point_evaluator returned pressure with shape "
                f"{pressure_array.shape}; expected {(points.shape[1],)}"
            )
        return velocity_array, pressure_array


@dataclass(frozen=True, slots=True)
class ConvergenceExpectation:
    """Nominal smooth-solution orders used by an MMS method check."""

    velocity_l2: float
    velocity_h1: float
    pressure_l2: float

    def as_dict(self) -> dict[str, float]:
        """Return expected orders keyed by error metric."""

        return {
            "velocity_l2": self.velocity_l2,
            "velocity_h1": self.velocity_h1,
            "pressure_l2": self.pressure_l2,
        }


@dataclass(frozen=True, slots=True)
class MMSConvergenceLevel:
    """Errors and observed pairwise rates for one structured mesh level."""

    resolution: int
    h: float
    num_cells: int
    num_dofs: int
    solve_seconds: float
    velocity_l2_error: float
    velocity_h1_error: float
    pressure_l2_error: float
    divergence_l2: float
    rates: dict[str, float] = field(default_factory=dict)

    def errors(self) -> dict[str, float]:
        """Return the four reported errors as a metric mapping."""

        return {
            "velocity_l2": self.velocity_l2_error,
            "velocity_h1": self.velocity_h1_error,
            "pressure_l2": self.pressure_l2_error,
            "divergence_l2": self.divergence_l2,
        }

    def as_dict(self) -> dict[str, int | float | None]:
        """Return a row suitable for a table or CSV writer."""

        row: dict[str, int | float | None] = {
            "resolution": self.resolution,
            "h": self.h,
            "num_cells": self.num_cells,
            "num_dofs": self.num_dofs,
            "solve_seconds": self.solve_seconds,
            "velocity_l2_error": self.velocity_l2_error,
            "velocity_h1_error": self.velocity_h1_error,
            "pressure_l2_error": self.pressure_l2_error,
            "divergence_l2": self.divergence_l2,
        }
        for name in _ERROR_NAMES:
            row[f"{name}_rate"] = self.rates.get(name)
        return row


@dataclass(slots=True)
class MMSDiscreteSolution:
    """Finest-mesh DOLFINx fields retained by a convergence run."""

    mesh: Any
    velocity: Any
    pressure: Any


@dataclass(slots=True)
class MMSConvergenceResult:
    """Complete manufactured-solution refinement study."""

    case: BrinkmanMMSCase
    method: MMSMethod
    levels: tuple[MMSConvergenceLevel, ...]
    expected_rates: ConvergenceExpectation
    finest_solution: MMSDiscreteSolution | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_rates(self) -> dict[str, float]:
        """Return rates between the two finest mesh levels."""

        return dict(self.levels[-1].rates)

    def as_dicts(self) -> list[dict[str, int | float | None]]:
        """Return refinement rows without requiring pandas."""

        return [level.as_dict() for level in self.levels]

    def assert_expected_rates(self, *, absolute_tolerance: float = 0.35) -> None:
        """Raise if a finest-pair rate is below its nominal smooth rate.

        This is an asymptotic diagnostic, not a proof of convergence. Coarse
        meshes, boundary layers, algebraic solver error, or insufficient
        quadrature can all make a correct discretization fail this check.
        """

        if absolute_tolerance < 0.0 or not np.isfinite(absolute_tolerance):
            raise ValueError("absolute_tolerance must be nonnegative and finite")
        failures: list[str] = []
        for name, expected in self.expected_rates.as_dict().items():
            observed = self.last_rates.get(name, float("nan"))
            threshold = expected - absolute_tolerance
            if not np.isfinite(observed) or observed < threshold:
                failures.append(
                    f"{name}: observed {observed:.3f}, expected at least {threshold:.3f}"
                )
        if failures:
            details = "; ".join(failures)
            raise AssertionError(
                f"{self.case.name}/{self.method} did not meet the expected finest-pair "
                f"rates: {details}"
            )


__all__ = [
    "BrinkmanMMSCase",
    "ConvergenceExpectation",
    "MMSConvergenceLevel",
    "MMSConvergenceResult",
    "MMSDiscreteSolution",
    "MMSExactSolutionFactory",
    "MMSFacetLaw",
    "MMSFacetSizeMode",
    "MMSMethod",
    "MMSPointEvaluator",
]
