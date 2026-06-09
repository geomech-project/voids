from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voids.image.porosity import PermeabilityMap, PorosityMap
from voids.visualization.fields import (
    plot_scalar_midplanes,
    plot_vector_midplanes,
    reference_pressure_to_outlet,
    reconstruct_tpfa_cell_velocity,
    vector_magnitude,
    write_structured_vector_field,
)


def test_reference_pressure_to_outlet_shifts_only_the_pressure_gauge() -> None:
    pressure = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=float)

    referenced = reference_pressure_to_outlet(
        pressure,
        flow_axis="x",
        reference_pressure=1.0e5,
        pressure_outlet=0.0,
    )

    assert np.allclose(referenced, [[100001.0, 100001.0], [100000.0, 100000.0]])
    assert np.allclose(np.diff(referenced, axis=0), np.diff(pressure, axis=0))

    with pytest.raises(ValueError, match="finite"):
        reference_pressure_to_outlet(pressure, reference_pressure=np.nan)


def test_reconstruct_tpfa_cell_velocity_matches_linear_pressure_drop() -> None:
    pressure = np.array([[0.75, 0.75], [0.25, 0.25]], dtype=float)
    permeability = PermeabilityMap(
        values=np.full((2, 2), 2.0),
        cell_size=(1.0, 1.0),
    )

    velocity = reconstruct_tpfa_cell_velocity(
        pressure,
        permeability,
        flow_axis="x",
        viscosity=1.0,
        pressure_inlet=1.0,
        pressure_outlet=0.0,
    )

    assert velocity.shape == (2, 2, 2)
    assert np.allclose(velocity[0], 1.0)
    assert np.allclose(velocity[1], 0.0)


def test_vector_magnitude_and_validation() -> None:
    vector = np.array(
        [
            [[3.0, 0.0], [0.0, 0.0]],
            [[4.0, 5.0], [0.0, 12.0]],
        ]
    )

    assert np.allclose(vector_magnitude(vector), [[5.0, 5.0], [0.0, 12.0]])

    with pytest.raises(ValueError, match="dim-first"):
        vector_magnitude(np.ones((2, 2)))


def test_write_structured_vector_field_preserves_vector_cell_data(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    grid = PorosityMap(values=np.ones((2, 1)), cell_size=(2.0, 3.0))
    vector = np.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )
    path = tmp_path / "velocity.vtu"

    written = write_structured_vector_field(
        vector,
        grid,
        path,
        extra_cell_data={"pressure": np.array([[10.0], [20.0]])},
    )

    loaded = meshio.read(written)
    assert "velocity" in loaded.cell_data_dict
    assert "pressure" in loaded.cell_data_dict
    assert np.allclose(
        loaded.cell_data_dict["velocity"]["quad"], [[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]]
    )
    assert np.allclose(loaded.cell_data_dict["pressure"]["quad"], [10.0, 20.0])


def test_midplane_plots_save_scalar_and_vector_figures(tmp_path: Path) -> None:
    scalar_path = tmp_path / "pressure.png"
    vector_path = tmp_path / "velocity.png"
    scalar = np.arange(27, dtype=float).reshape(3, 3, 3)
    vector = np.stack(
        [
            np.ones((3, 3, 3)),
            np.zeros((3, 3, 3)),
            np.ones((3, 3, 3)),
        ]
    )

    scalar_fig = plot_scalar_midplanes(
        scalar, title="pressure", path=scalar_path, vmin=0.0, vmax=1.0
    )
    vector_fig = plot_vector_midplanes(
        vector,
        title="velocity",
        path=vector_path,
        quiver_stride=2,
        vmin=0.0,
        vmax=1.0,
    )

    assert scalar_path.exists()
    assert vector_path.exists()
    assert len(scalar_fig.axes) >= 3
    assert len(vector_fig.axes) >= 3
