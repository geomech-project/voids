from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from voids.fem.singlephase import (  # noqa: E402
    FEMMapProblem,
    solve_brinkman_taylor_hood,
    solve_brinkman_usfem,
    solve_darcy_taylor_hood,
    upscale_permeability_fem,
    upscale_principal_permeabilities_fem,
)
from voids.image.porosity import PermeabilityMap, PorosityMap  # noqa: E402


def _constant_problem(shape: tuple[int, ...], permeability: float = 2.0) -> FEMMapProblem:
    return FEMMapProblem(
        permeability_map=PermeabilityMap(np.full(shape, permeability), cell_size=1.0),
        porosity_map=PorosityMap(np.ones(shape), cell_size=1.0),
        viscosity=1.0,
    )


@pytest.mark.parametrize(
    "solver",
    [
        solve_darcy_taylor_hood,
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
def test_fem_backends_recover_constant_2d_permeability(
    solver: Callable[..., Any],
) -> None:
    result = solver(_constant_problem((3, 3), permeability=2.0), flow_axis="x")

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.flow_rate > 0.0
    assert result.solve_seconds >= 0.0
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


def test_fem_taylor_hood_brinkman_supports_3d_constant_map() -> None:
    result = solve_brinkman_taylor_hood(
        _constant_problem((2, 2, 2), permeability=1.5),
        flow_axis="z",
    )

    assert result.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert result.flow_axis == "z"


def test_fem_upscaling_dispatches_backends() -> None:
    problem = _constant_problem((3, 3), permeability=3.0)

    result = upscale_permeability_fem(
        problem,
        backend="taylor_hood_darcy",
        axes=("x", "y"),
    )

    assert result.backend == "taylor_hood_darcy"
    assert set(result.results) == {"x", "y"}
    assert result.permeability["x"] == pytest.approx(3.0, rel=5.0e-4)
    assert result.permeability["y"] == pytest.approx(3.0, rel=5.0e-4)
    assert upscale_principal_permeabilities_fem(
        problem,
        backend="usfem_brinkman",
        axes=("x",),
    ) == {"x": pytest.approx(3.0, rel=5.0e-4)}


def test_fem_problem_validates_map_compatibility() -> None:
    with pytest.raises(ValueError, match="same shape"):
        FEMMapProblem(
            PermeabilityMap(np.ones((2, 2)), cell_size=1.0),
            PorosityMap(np.ones((2, 3)), cell_size=1.0),
        )
