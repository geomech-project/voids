from __future__ import annotations

import numpy as np

from voids.examples.mms._core import BrinkmanMMSCase


def boundary_layer_case_2d(
    *,
    viscosity: float = 1.0e-2,
    reaction: float = 1.0,
) -> BrinkmanMMSCase:
    r"""Return the two-dimensional Brinkman boundary-layer MMS case.

    The unit-square exact solution is

    .. math::

       u_1 &= y -
       \\frac{\\exp((y-1)/\\nu)-\\exp(-1/\\nu)}
            {1-\\exp(-1/\\nu)},\\\\
       u_2 &= x -
       \\frac{\\exp((x-1)/\\nu)-\\exp(-1/\\nu)}
            {1-\\exp(-1/\\nu)},\\\\
       p &= x-y.

    It is exactly divergence-free and develops boundary layers at the top and
    right boundaries as ``viscosity`` decreases.
    """

    nu = float(viscosity)
    gamma = float(reaction)

    def exact_solution_factory(ufl, domain):
        x = ufl.SpatialCoordinate(domain)
        exponential_floor = ufl.exp(-1.0 / nu)
        denominator = 1.0 - exponential_floor
        u_1 = x[1] - (ufl.exp((x[1] - 1.0) / nu) - exponential_floor) / denominator
        u_2 = x[0] - (ufl.exp((x[0] - 1.0) / nu) - exponential_floor) / denominator
        return ufl.as_vector((u_1, u_2)), x[0] - x[1]

    def point_evaluator(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = points[0]
        y = points[1]
        exponential_floor = np.exp(-1.0 / nu)
        denominator = -np.expm1(-1.0 / nu)
        u_1 = y - (np.exp((y - 1.0) / nu) - exponential_floor) / denominator
        u_2 = x - (np.exp((x - 1.0) / nu) - exponential_floor) / denominator
        return np.vstack((u_1, u_2)), x - y

    return BrinkmanMMSCase(
        name="brinkman_boundary_layer_2d",
        dimension=2,
        viscosity=nu,
        reaction=gamma,
        exact_solution_factory=exact_solution_factory,
        point_evaluator=point_evaluator,
        description=(
            "Divergence-free unit-square Brinkman solution with exponential layers at y=1 and x=1."
        ),
        reference=(
            "Barrenechea and Valentin (2002), Numerische Mathematik 92, "
            "653-677, doi:10.1007/s002110100371."
        ),
    )


__all__ = ["boundary_layer_case_2d"]
