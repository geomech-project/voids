from __future__ import annotations

import numpy as np
import pytest

from voids.fvm.singlephase import (
    solve_tpfa,
    upscale_permeability_tpfa,
    upscale_principal_permeabilities_tpfa,
)
from voids.image.porosity import PermeabilityMap


def test_tpfa_constant_2d_map_recovers_input_permeability() -> None:
    permeability = PermeabilityMap(
        np.full((4, 3), 2.5),
        cell_size=(0.5, 0.25),
        metadata={"case": "constant"},
    )

    result = solve_tpfa(permeability, flow_axis="x", viscosity=3.0)

    assert result.permeability == pytest.approx(2.5)
    assert result.flow_rate > 0.0
    assert result.mass_balance_error < 1.0e-12
    assert result.cell_size == (0.5, 0.25)
    assert result.metadata["case"] == "constant"


def test_tpfa_constant_3d_array_accepts_sequence_cell_size() -> None:
    result = solve_tpfa(
        np.full((3, 2, 4), 1.7),
        flow_axis="z",
        viscosity=2.0,
        cell_size=[0.2, 0.3, 0.4],
    )

    assert result.permeability == pytest.approx(1.7)
    assert result.mass_balance_error < 1.0e-12


def test_tpfa_upscaling_solves_requested_axes() -> None:
    permeability = PermeabilityMap(np.full((3, 4, 2), 4.2), cell_size=1.0)

    result = upscale_permeability_tpfa(permeability, axes=("x", "y"))

    assert set(result.results) == {"x", "y"}
    assert result.permeability == {"x": pytest.approx(4.2), "y": pytest.approx(4.2)}
    assert upscale_principal_permeabilities_tpfa(permeability, axes=("z",)) == {
        "z": pytest.approx(4.2)
    }


def test_tpfa_rejects_nonpositive_pressure_drop() -> None:
    with pytest.raises(ValueError, match="pressure_inlet must be greater"):
        solve_tpfa(np.ones((2, 2)), pressure_inlet=0.0, pressure_outlet=1.0)
