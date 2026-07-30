from __future__ import annotations

import numpy as np

from voids.examples.mms._core import BrinkmanMMSCase


def bubble_case_3d(
    *,
    viscosity: float = 1.0e-2,
    reaction: float = 1.0,
) -> BrinkmanMMSCase:
    """Return a smooth divergence-free three-dimensional Brinkman MMS case.

    The velocity is constructed from mixed first derivatives of a polynomial
    boundary bubble. It therefore vanishes on the unit-cube boundary and is
    divergence-free by cancellation of mixed derivatives. The pressure is
    ``sin(2*pi*x) * sin(pi*y) * sin(pi*z)``.
    """

    nu = float(viscosity)
    gamma = float(reaction)

    def exact_solution_factory(ufl, domain):
        x = ufl.SpatialCoordinate(domain)
        bubble = (
            x[0] ** 2
            * (1.0 - x[0]) ** 2
            * x[1] ** 2
            * (1.0 - x[1]) ** 2
            * x[2] ** 2
            * (1.0 - x[2]) ** 2
        )
        phi = 32.0 * bubble * (1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2])
        phi_gradient = ufl.grad(phi)
        phi_x = phi_gradient[0]
        phi_y = phi_gradient[1]
        phi_z = phi_gradient[2]
        velocity = ufl.as_vector(
            (
                3.0 * phi_y - 2.0 * phi_z,
                phi_z - 3.0 * phi_x,
                2.0 * phi_x - phi_y,
            )
        )
        pressure = ufl.sin(2.0 * np.pi * x[0]) * ufl.sin(np.pi * x[1]) * ufl.sin(np.pi * x[2])
        return velocity, pressure

    def point_evaluator(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = points

        def bubble_1d(value: np.ndarray) -> np.ndarray:
            return value**2 * (1.0 - value) ** 2

        def bubble_1d_derivative(value: np.ndarray) -> np.ndarray:
            return 2.0 * value * (1.0 - value) * (1.0 - 2.0 * value)

        b_x, b_y, b_z = bubble_1d(x), bubble_1d(y), bubble_1d(z)
        db_x = bubble_1d_derivative(x)
        db_y = bubble_1d_derivative(y)
        db_z = bubble_1d_derivative(z)
        bubble = b_x * b_y * b_z
        linear = 1.0 + x + 2.0 * y + 3.0 * z
        phi_x = 32.0 * (db_x * b_y * b_z * linear + bubble)
        phi_y = 32.0 * (b_x * db_y * b_z * linear + 2.0 * bubble)
        phi_z = 32.0 * (b_x * b_y * db_z * linear + 3.0 * bubble)
        velocity = np.vstack(
            (
                3.0 * phi_y - 2.0 * phi_z,
                phi_z - 3.0 * phi_x,
                2.0 * phi_x - phi_y,
            )
        )
        pressure = np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        return velocity, pressure

    return BrinkmanMMSCase(
        name="brinkman_polynomial_bubble_3d",
        dimension=3,
        viscosity=nu,
        reaction=gamma,
        exact_solution_factory=exact_solution_factory,
        point_evaluator=point_evaluator,
        description=(
            "Smooth divergence-free unit-cube Brinkman solution generated from "
            "a polynomial boundary bubble."
        ),
        reference=(
            "Manufactured case used in the voids 3D USFEM verification study; "
            "forcing is generated from the documented strong residual."
        ),
    )


__all__ = ["bubble_case_3d"]
