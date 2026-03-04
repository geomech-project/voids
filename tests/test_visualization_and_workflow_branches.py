from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from voids.visualization.plotly import _rgb_with_opacity, plot_network_plotly
from voids.visualization.pyvista import (
    _line_cells_from_conns,
    network_to_pyvista_polydata,
    plot_network_pyvista,
)
from voids.workflows.run_singlephase import main


def test_plotly_validates_scalar_inputs_and_sampling(line_network) -> None:
    """Test Plotly scalar validation and throat downsampling behavior."""

    with pytest.raises(KeyError, match="Missing pore field"):
        plot_network_plotly(line_network, point_scalars="missing")
    with pytest.raises(ValueError, match="pore scalar array must have shape"):
        plot_network_plotly(line_network, point_scalars=np.ones(2))
    with pytest.raises(ValueError, match="throat scalar array must have shape"):
        plot_network_plotly(line_network, cell_scalars=np.ones(1))

    fig = plot_network_plotly(
        line_network,
        max_throats=1,
        title=None,
        layout_kwargs={"height": 333},
    )

    assert fig.layout.title.text == "Pore network (showing 1 of 2 throats)"
    assert fig.layout.height == 333
    assert fig.data[0].marker.color == "royalblue"
    assert fig.data[1].line.color == "rgba(100,100,100,0.4)"


def test_plotly_supports_explicit_cell_scalars(line_network) -> None:
    """Test explicit throat scalar arrays in the Plotly backend."""

    fig = plot_network_plotly(line_network, cell_scalars=np.array([10.0, 20.0]))

    hover_text = [trace.text for trace in fig.data[1:]]
    assert "throat.scalar=1.000e+01" in hover_text[0]
    assert "throat.scalar=2.000e+01" in hover_text[1]


def test_plotly_supports_named_scalar_fields(line_network) -> None:
    """Test named pore and throat scalar fields in the Plotly backend."""

    fig = plot_network_plotly(line_network, point_scalars="volume", cell_scalars="length")

    assert fig.data[0].marker.colorbar.title.text == "pore.volume"
    assert "throat.length=1.000e+00" in fig.data[1].text


def test_rgb_with_opacity_leaves_non_rgb_colors_untouched() -> None:
    """Test that non-RGB colors are returned unchanged when adding opacity."""

    assert _rgb_with_opacity("blue", 0.3) == "blue"


class _FakePolyData:
    """Minimal fake ``pyvista.PolyData`` implementation for unit tests."""

    def __init__(self, points, lines=None):
        """Store points, optional lines, and attached scalar data."""

        self.points = np.asarray(points, dtype=float)
        self.lines = np.asarray(lines) if lines is not None else None
        self.point_data: dict[str, np.ndarray] = {}
        self.cell_data: dict[str, np.ndarray] = {}
        self.active_scalars: tuple[str, str] | None = None

    def set_active_scalars(self, name: str, preference: str = "point") -> None:
        """Record the active scalar selection."""

        self.active_scalars = (name, preference)

    def tube(self, **_kwargs):
        """Raise to exercise tube-rendering fallback logic."""

        raise RuntimeError("tube unavailable")


class _FakePlotter:
    """Minimal fake ``pyvista.Plotter`` implementation for unit tests."""

    def __init__(self, *, off_screen: bool, notebook: bool):
        """Store rendering flags and record mesh/screenshot calls."""

        self.off_screen = off_screen
        self.notebook = notebook
        self.meshes: list[tuple[object, dict[str, object]]] = []
        self.axes_added = False
        self.show_calls: list[bool] = []
        self.screenshots: list[str] = []

    def add_mesh(self, mesh, **kwargs):
        """Record a mesh-addition call."""

        self.meshes.append((mesh, kwargs))

    def add_axes(self):
        """Record that orientation axes were requested."""

        self.axes_added = True

    def show(self, auto_close: bool = False):
        """Record a render/show call."""

        self.show_calls.append(auto_close)

    def screenshot(self, path: str):
        """Record a screenshot output path."""

        self.screenshots.append(path)


class _FakePV:
    """Namespace-style fake PyVista module for unit tests."""

    PolyData = _FakePolyData
    Plotter = _FakePlotter


def test_line_cells_from_conns_requires_two_column_connectivity() -> None:
    """Test line-cell construction input validation."""

    with pytest.raises(ValueError, match="shape \\(Nt, 2\\)"):
        _line_cells_from_conns(np.array([0, 1, 2]))


def test_network_to_pyvista_polydata_supports_all_numeric_fields_and_validates_scalars(
    monkeypatch, line_network
) -> None:
    """Test PolyData conversion, scalar attachment, and scalar validation."""

    monkeypatch.setattr("voids.visualization.pyvista._require_pyvista", lambda: _FakePV)

    poly = network_to_pyvista_polydata(
        line_network,
        point_scalars="volume",
        cell_scalars="length",
        include_all_numeric_fields=True,
    )

    assert np.array_equal(poly.point_data["pore.id"], np.array([0, 1, 2]))
    assert np.array_equal(poly.cell_data["throat.id"], np.array([0, 1]))
    assert "pore.volume" in poly.point_data
    assert "throat.length" in poly.cell_data
    assert poly.active_scalars == ("pore.scalar", "point")

    with pytest.raises(KeyError, match="Missing pore field"):
        network_to_pyvista_polydata(line_network, point_scalars="missing")
    with pytest.raises(KeyError, match="Missing throat field"):
        network_to_pyvista_polydata(line_network, cell_scalars="missing")
    with pytest.raises(ValueError, match="point_scalars array must have shape"):
        network_to_pyvista_polydata(line_network, point_scalars=np.ones(2))
    with pytest.raises(ValueError, match="cell_scalars array must have shape"):
        network_to_pyvista_polydata(line_network, cell_scalars=np.ones(1))

    poly_with_array = network_to_pyvista_polydata(line_network, cell_scalars=np.array([2.0, 3.0]))
    assert np.array_equal(poly_with_array.cell_data["throat.scalar"], np.array([2.0, 3.0]))


def test_plot_network_pyvista_falls_back_from_tubes_and_saves_screenshot(
    monkeypatch, line_network, tmp_path: Path
) -> None:
    """Test PyVista plotting fallback from tubes and screenshot capture."""

    monkeypatch.setattr("voids.visualization.pyvista._require_pyvista", lambda: _FakePV)

    screenshot = tmp_path / "mesh.png"
    plotter, poly = plot_network_pyvista(
        line_network,
        point_scalars=np.array([1.0, 0.5, 0.0]),
        render_tubes=True,
        tube_radius=0.2,
        off_screen=True,
        screenshot=str(screenshot),
    )

    assert isinstance(poly, _FakePolyData)
    assert plotter.off_screen is True
    assert plotter.notebook is False
    assert plotter.axes_added is True
    assert plotter.show_calls == [False]
    assert plotter.screenshots == [str(screenshot)]
    assert plotter.meshes[0][1]["scalars"] == "pore.scalar"
    assert plotter.meshes[1][1]["render_points_as_spheres"] is True


def test_run_singlephase_main_regression(capsys, data_regression) -> None:
    """Test the workflow CLI JSON payload against a stored regression baseline."""

    main()

    out = json.loads(capsys.readouterr().out)
    data_regression.check(out)


def test_run_singlephase_module_entrypoint(capsys) -> None:
    """Test execution of the workflow module as a script entry point."""

    runpy.run_path(str(Path(main.__code__.co_filename)), run_name="__main__")

    out = json.loads(capsys.readouterr().out)
    assert out["Q"] == pytest.approx(0.5)
