from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import voids.visualization.fields as fields_mod
from voids.image.porosity import PermeabilityMap, PorosityMap
from voids.visualization.fields import (
    plot_scalar_midplanes,
    plot_vector_midplanes,
    reference_pressure_to_outlet,
    reconstruct_tpfa_cell_velocity,
    sample_dolfinx_function_on_grid,
    vector_magnitude,
    write_dolfinx_function_xdmf,
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

    raw_velocity = reconstruct_tpfa_cell_velocity(
        pressure,
        np.full((2, 2), 2.0),
        flow_axis="x",
        viscosity=1.0,
        pressure_inlet=1.0,
        pressure_outlet=0.0,
        cell_size=(1.0, 1.0),
    )
    assert np.allclose(raw_velocity, velocity)


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


def test_field_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="pressure must be a 2D or 3D"):
        reconstruct_tpfa_cell_velocity(np.ones((2,)), np.ones((2,)))
    with pytest.raises(ValueError, match="pressure must contain only finite"):
        reconstruct_tpfa_cell_velocity(np.array([[1.0, np.nan], [0.0, 0.0]]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="viscosity must be positive"):
        reconstruct_tpfa_cell_velocity(np.ones((2, 2)), np.ones((2, 2)), viscosity=0.0)
    with pytest.raises(ValueError, match="axis must be one of"):
        reference_pressure_to_outlet(np.ones((2, 2)), flow_axis="z")
    with pytest.raises(ValueError, match="permeability must have shape"):
        reconstruct_tpfa_cell_velocity(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite non-negative"):
        reconstruct_tpfa_cell_velocity(np.ones((2, 2)), np.array([[1.0, -1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="cell_size dimensionality"):
        reconstruct_tpfa_cell_velocity(np.ones((2, 2)), np.ones((2, 2)), cell_size=(1.0,))
    with pytest.raises(ValueError, match="cell_size values"):
        reconstruct_tpfa_cell_velocity(np.ones((2, 2)), np.ones((2, 2)), cell_size=0.0)


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


def test_write_structured_vector_field_validates_and_accepts_last_axis_vectors(
    tmp_path: Path,
) -> None:
    pytest.importorskip("meshio")
    grid = PorosityMap(values=np.ones((2, 1)), cell_size=(2.0, 3.0))
    last_axis_vector = np.array([[[1.0, 3.0]], [[2.0, 4.0]]])

    written = write_structured_vector_field(last_axis_vector, grid, tmp_path / "velocity.vtu")

    assert written.exists()
    with pytest.raises(ValueError, match="non-empty"):
        write_structured_vector_field(last_axis_vector, grid, tmp_path / "bad.vtu", name="")
    with pytest.raises(ValueError, match="contain only finite"):
        write_structured_vector_field(
            np.array([[[np.nan], [2.0]], [[3.0], [4.0]]]),
            grid,
            tmp_path / "nan.vtu",
        )
    with pytest.raises(ValueError, match="must have shape"):
        write_structured_vector_field(np.ones((2, 2, 2)), grid, tmp_path / "shape.vtu")


def test_midplane_plots_save_scalar_and_vector_figures(tmp_path: Path) -> None:
    scalar_path = tmp_path / "pressure.png"
    vector_path = tmp_path / "velocity.png"
    scalar_2d_path = tmp_path / "pressure_2d.png"
    scalar = np.arange(27, dtype=float).reshape(3, 3, 3)
    scalar_2d = np.arange(4, dtype=float).reshape(2, 2)
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
    scalar_2d_fig = plot_scalar_midplanes(
        scalar_2d,
        title="pressure 2d",
        path=scalar_2d_path,
        colorbar_use_offset=False,
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
    assert scalar_2d_path.exists()
    assert vector_path.exists()
    assert len(scalar_fig.axes) >= 3
    assert len(scalar_2d_fig.axes) >= 1
    assert len(vector_fig.axes) >= 3

    with pytest.raises(ValueError, match="scalar field must be 2D or 3D"):
        plot_scalar_midplanes(np.ones((2,)), title="bad pressure")
    with pytest.raises(ValueError, match="scalar field must contain only finite"):
        plot_scalar_midplanes(np.array([[1.0, np.nan], [2.0, 3.0]]), title="bad pressure")
    with pytest.raises(ValueError, match="quiver_stride must be positive"):
        plot_vector_midplanes(vector, title="velocity", quiver_stride=0)
    with pytest.raises(ValueError, match="at least one finite"):
        plot_vector_midplanes(np.full((2, 2, 2), np.nan), title="bad velocity")
    with pytest.raises(ValueError, match="vector dimensionality"):
        vector_magnitude(np.ones((2, 2, 2, 2)))


def test_write_dolfinx_function_xdmf_uses_linear_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture: dict[str, Any] = {}

    fake_dolfinx = ModuleType("dolfinx")
    fake_dolfinx.__path__ = []
    fake_io = ModuleType("dolfinx.io")

    class FakeXDMFFile:
        def __init__(self, comm: object, path: str, mode: str) -> None:
            capture["xdmf_init"] = (comm, path, mode)

        def __enter__(self) -> "FakeXDMFFile":
            return self

        def __exit__(self, *args: object) -> None:
            capture["closed"] = True

        def write_mesh(self, mesh: object) -> None:
            capture["mesh"] = mesh

        def write_function(self, function: object) -> None:
            capture["function"] = function

    fake_io.XDMFFile = FakeXDMFFile
    monkeypatch.setitem(sys.modules, "dolfinx", fake_dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.io", fake_io)

    fake_mesh = SimpleNamespace(comm="COMM")
    fake_export = SimpleNamespace(function_space=SimpleNamespace(mesh=fake_mesh))
    monkeypatch.setattr(
        fields_mod,
        "_linear_dolfinx_export_function",
        lambda function, *, name=None: fake_export,
    )

    path = tmp_path / "field.xdmf"
    written = write_dolfinx_function_xdmf(object(), path, name="pressure")

    assert written == path
    assert capture["xdmf_init"] == ("COMM", str(path), "w")
    assert capture["mesh"] is fake_mesh
    assert capture["function"] is fake_export


def test_linear_dolfinx_export_function_builds_scalar_and_vector_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    fake_dolfinx = ModuleType("dolfinx")
    fake_dolfinx.__path__ = []
    fake_fem = ModuleType("dolfinx.fem")
    fake_basix = ModuleType("basix")
    fake_basix.__path__ = []
    fake_basix_ufl = ModuleType("basix.ufl")

    class FakeMesh:
        def basix_cell(self) -> str:
            return "triangle"

    fake_mesh = FakeMesh()

    def fake_element(family: str, cell: str, degree: int, **kwargs: object) -> tuple[str, Any]:
        calls.append(("element", (family, cell, degree, kwargs)))
        return family, kwargs

    def fake_functionspace(mesh: object, element: object) -> SimpleNamespace:
        calls.append(("functionspace", element))
        return SimpleNamespace(mesh=mesh, element=element)

    class FakeFunction:
        def __init__(self, space: object) -> None:
            self.function_space = space
            self.name = ""
            self.interpolated_from: object | None = None

        def interpolate(self, function: object) -> None:
            self.interpolated_from = function

    fake_basix_ufl.element = fake_element
    fake_fem.functionspace = fake_functionspace
    fake_fem.Function = FakeFunction
    fake_dolfinx.fem = fake_fem
    monkeypatch.setitem(sys.modules, "dolfinx", fake_dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.fem", fake_fem)
    monkeypatch.setitem(sys.modules, "basix", fake_basix)
    monkeypatch.setitem(sys.modules, "basix.ufl", fake_basix_ufl)

    scalar = SimpleNamespace(
        function_space=SimpleNamespace(mesh=fake_mesh),
        name="pressure",
        ufl_shape=(),
    )
    vector = SimpleNamespace(
        function_space=SimpleNamespace(mesh=fake_mesh),
        name="velocity",
        ufl_shape=(2,),
    )

    scalar_export = fields_mod._linear_dolfinx_export_function(scalar, name=None)
    vector_export = fields_mod._linear_dolfinx_export_function(vector, name="u")

    assert scalar_export.name == "pressure"
    assert scalar_export.interpolated_from is scalar
    assert vector_export.name == "u"
    assert vector_export.interpolated_from is vector
    assert any(call[0] == "element" and call[1][3] == {"shape": (2,)} for call in calls)


def test_sample_dolfinx_function_on_grid_with_fake_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dolfinx = ModuleType("dolfinx")
    fake_dolfinx.__path__ = []
    fake_geometry = ModuleType("dolfinx.geometry")

    class FakeTopology:
        dim = 2

    class FakeMesh:
        topology = FakeTopology()

    class FakeFunction:
        function_space = SimpleNamespace(mesh=FakeMesh())

        def __init__(self, *, vector: bool = False) -> None:
            self.vector = vector

        def eval(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
            assert np.all(cells == 0)
            if self.vector:
                return points[:, :2]
            return points[:, 0] + points[:, 1]

    class FakeCollidingCells:
        def __init__(self, *, valid: bool = True) -> None:
            self.valid = valid

        def links(self, point_index: np.int32) -> list[int]:
            return [0] if self.valid else []

    fake_geometry.bb_tree = lambda mesh, dim: ("tree", mesh, dim)
    fake_geometry.compute_collisions_points = lambda tree, points: ("candidates", tree, points)
    fake_geometry.compute_colliding_cells = lambda mesh, candidates, points: FakeCollidingCells()
    monkeypatch.setitem(sys.modules, "dolfinx", fake_dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.geometry", fake_geometry)

    sampled_scalar = sample_dolfinx_function_on_grid(
        FakeFunction(),
        shape=(2, 2),
        cell_size=(1.0, 2.0),
        origin=(10.0, 20.0),
    )
    sampled_vector = sample_dolfinx_function_on_grid(
        FakeFunction(vector=True),
        shape=(2, 2),
        cell_size=1.0,
    )

    assert sampled_scalar.shape == (2, 2)
    assert sampled_scalar[0, 0] == pytest.approx(31.5)
    assert sampled_vector.shape == (2, 2, 2)
    assert sampled_vector[0, 0, 0] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="shape must describe"):
        sample_dolfinx_function_on_grid(FakeFunction(), shape=(2,), cell_size=1.0)
    with pytest.raises(ValueError, match="origin dimensionality"):
        sample_dolfinx_function_on_grid(FakeFunction(), shape=(2, 2), cell_size=1.0, origin=(0.0,))

    fake_geometry.compute_colliding_cells = lambda mesh, candidates, points: FakeCollidingCells(
        valid=False
    )
    with pytest.raises(RuntimeError, match="no grid sample points"):
        sample_dolfinx_function_on_grid(FakeFunction(), shape=(2, 2), cell_size=1.0)
