from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from voids.visualization._sizing import resolve_size_values, scale_sizes_to_pixels
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

    tube_should_raise = True

    def __init__(self, points, lines=None):
        """Store points, optional lines, and attached scalar data."""

        self.points = np.asarray(points, dtype=float)
        self.lines = np.asarray(lines) if lines is not None else None
        self.point_data: dict[str, np.ndarray] = {}
        self.cell_data: dict[str, np.ndarray] = {}
        self.active_scalars: tuple[str, str] | None = None
        self.tube_kwargs: dict[str, object] | None = None
        self.glyph_kwargs: dict[str, object] | None = None

    def set_active_scalars(self, name: str, preference: str = "point") -> None:
        """Record the active scalar selection."""

        self.active_scalars = (name, preference)

    def tube(self, **kwargs):
        """Either raise or return a copied tube mesh."""

        if type(self).tube_should_raise:
            raise RuntimeError("tube unavailable")
        out = _FakePolyData(self.points, lines=self.lines)
        out.point_data = dict(self.point_data)
        out.cell_data = dict(self.cell_data)
        out.tube_kwargs = kwargs
        return out

    def glyph(self, **kwargs):
        """Return a copied glyph mesh."""

        out = _FakePolyData(self.points, lines=self.lines)
        out.point_data = dict(self.point_data)
        out.cell_data = dict(self.cell_data)
        out.glyph_kwargs = kwargs
        return out


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

    @staticmethod
    def Sphere(radius: float = 0.5):
        """Return a lightweight sphere descriptor for glyph tests."""

        return {"kind": "sphere", "radius": radius}


def test_line_cells_from_conns_requires_two_column_connectivity() -> None:
    """Test line-cell construction input validation."""

    with pytest.raises(ValueError, match="shape \\(Nt, 2\\)"):
        _line_cells_from_conns(np.array([0, 1, 2]))


def test_network_to_pyvista_polydata_supports_all_numeric_fields_and_validates_scalars(
    monkeypatch, line_network
) -> None:
    """Test PolyData conversion, scalar attachment, and scalar validation."""

    monkeypatch.setattr("voids.visualization.pyvista.pv", _FakePV)

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

    monkeypatch.setattr("voids.visualization.pyvista.pv", _FakePV)

    screenshot = tmp_path / "mesh.png"
    with pytest.warns(UserWarning, match="tube filter failed.*RuntimeError"):
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
    line_mesh_kwargs = plotter.meshes[0][1]
    assert line_mesh_kwargs["scalars"] == "pore.scalar"
    # When tube filter fails, render_lines_as_tubes must be True (line-tube approximation).
    assert line_mesh_kwargs["render_lines_as_tubes"] is True
    assert plotter.meshes[1][1]["render_points_as_spheres"] is True


def test_plot_network_pyvista_falls_back_from_variable_throat_tubes(
    monkeypatch, line_network
) -> None:
    """Test that fallback for variable throat sizes warns about accuracy loss."""

    monkeypatch.setattr("voids.visualization.pyvista.pv", _FakePV)
    line_network.throat["diameter_equivalent"] = np.array([0.5, 1.5])

    # tube_should_raise=True by default; variable throat sizes trigger tube rendering.
    with pytest.warns(UserWarning, match="RuntimeError.*Variable throat radii"):
        plotter, _ = plot_network_pyvista(line_network)

    line_mesh_kwargs = plotter.meshes[0][1]
    # Fallback must use render_lines_as_tubes=True to approximate tube appearance.
    assert line_mesh_kwargs["render_lines_as_tubes"] is True


def test_plotly_auto_sizes_markers_and_throats_from_characteristic_diameters(line_network) -> None:
    """Test automatic Plotly pore/throat size rendering from characteristic diameters."""

    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 4.0])
    line_network.throat["diameter_equivalent"] = np.array([0.5, 1.5])

    fig = plot_network_plotly(line_network, point_scalars="volume")

    marker_sizes = np.asarray(fig.data[0].marker.size, dtype=float)
    assert np.allclose(marker_sizes, np.array([3.0, 6.0, 12.0]))
    assert "pore.diameter_equivalent=1.000e+00" in fig.data[0].text[0]
    assert fig.data[1].line.width == pytest.approx(1.0)
    assert fig.data[2].line.width == pytest.approx(3.0)
    assert "throat.diameter_equivalent=5.000e-01" in fig.data[1].text


def test_plotly_false_sizes_disable_auto_size_fields(line_network) -> None:
    """Test that point_sizes=False/throat_sizes=False disables size-driven rendering."""

    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 4.0])
    line_network.throat["diameter_equivalent"] = np.array([0.5, 1.5])

    fig = plot_network_plotly(
        line_network,
        point_sizes=False,
        throat_sizes=False,
        point_size=10.0,
        line_width=4.0,
    )

    assert fig.data[0].marker.size == pytest.approx(10.0)
    assert fig.data[1].line.width == pytest.approx(4.0)
    assert fig.data[2].line.width == pytest.approx(4.0)


def test_plotly_point_size_acts_as_reference_with_auto_size_fields(line_network) -> None:
    """Test that point_size/line_width act as reference when auto size fields are present."""

    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 4.0])
    line_network.throat["diameter_equivalent"] = np.array([0.5, 1.5])

    fig = plot_network_plotly(line_network, point_size=6.0, line_width=2.0)

    # size-driven rendering should still be active; marker sizes must be an array
    marker_sizes = np.asarray(fig.data[0].marker.size, dtype=float)
    assert marker_sizes.ndim == 1
    assert not np.all(marker_sizes == marker_sizes[0])


def test_plotly_size_limits_none_none_disables_default_clipping(line_network) -> None:
    """Test optional disabling of Plotly default size clipping."""

    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 100.0])
    line_network.throat["diameter_equivalent"] = np.array([0.1, 1.0])

    fig = plot_network_plotly(
        line_network,
        point_size=6.0,
        line_width=2.0,
        point_size_limits=(None, None),
        throat_size_limits=(None, None),
    )

    marker_sizes = np.asarray(fig.data[0].marker.size, dtype=float)
    # With clipping disabled, the largest marker should exceed the default cap of 24 px.
    assert float(marker_sizes.max()) > 24.0
    # With clipping disabled, the smallest throat can fall below default min of 0.75 px.
    throat_widths = np.array([float(trace.line.width) for trace in fig.data[1:]], dtype=float)
    assert float(throat_widths.min()) < 0.75


def test_size_resolution_helpers_cover_auto_named_and_explicit_modes(monkeypatch) -> None:
    """Test size helper branches used by Plotly and PyVista visualizations."""

    store = {
        "radius_inscribed": np.array([1.0, 2.0, 3.0]),
        "area": np.array([np.pi, 4.0 * np.pi, 9.0 * np.pi]),
    }

    assert resolve_size_values(False, store=store, expected_shape=(3,), prefix="pore") == (
        None,
        None,
    )
    assert resolve_size_values(None, store={}, expected_shape=(3,), prefix="pore") == (None, None)

    named_values, named_label = resolve_size_values(
        "radius_inscribed", store=store, expected_shape=(3,), prefix="pore"
    )
    assert named_label == "pore.radius_inscribed"
    assert np.array_equal(named_values, np.array([2.0, 4.0, 6.0]))

    explicit_values, explicit_label = resolve_size_values(
        np.array([3.0, 4.0, 5.0]), store=store, expected_shape=(3,), prefix="throat"
    )
    assert explicit_label == "throat.size"
    assert np.array_equal(explicit_values, np.array([3.0, 4.0, 5.0]))

    with pytest.raises(KeyError, match="Missing pore field 'diameter_equivalent'"):
        resolve_size_values("diameter_equivalent", store=store, expected_shape=(3,), prefix="pore")
    with pytest.raises(ValueError, match="pore size field 'area' must have shape"):
        resolve_size_values("area", store={"area": np.ones(2)}, expected_shape=(3,), prefix="pore")
    with pytest.raises(ValueError, match="throat size array must have shape"):
        resolve_size_values(np.ones(2), store=store, expected_shape=(3,), prefix="throat")

    no_valid = scale_sizes_to_pixels(np.array([0.0, -1.0, np.nan]), reference=5.0)
    assert np.array_equal(no_valid, np.full(3, 5.0))

    monkeypatch.setattr("voids.visualization._sizing.np.median", lambda arr: np.nan)
    baseline_invalid = scale_sizes_to_pixels(np.array([1.0, 2.0, 3.0]), reference=5.0)
    assert np.array_equal(baseline_invalid, np.full(3, 5.0))


def test_plotly_auto_sized_markers_keep_diameter_mode_without_point_scalars(line_network) -> None:
    """Test Plotly diameter-mode markers for size-based rendering without pore scalars."""

    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 4.0])

    fig = plot_network_plotly(line_network)

    assert fig.data[0].marker.sizemode == "diameter"
    assert np.allclose(np.asarray(fig.data[0].marker.size, dtype=float), np.array([3.0, 6.0, 12.0]))


def test_plot_network_pyvista_auto_sizes_points_and_throats(monkeypatch, line_network) -> None:
    """Test automatic PyVista sphere and tube sizing from characteristic diameters."""

    monkeypatch.setattr("voids.visualization.pyvista.pv", _FakePV)
    _FakePolyData.tube_should_raise = False
    line_network.pore["diameter_equivalent"] = np.array([1.0, 2.0, 4.0])
    line_network.throat["diameter_equivalent"] = np.array([0.5, 1.5])

    try:
        plotter, poly = plot_network_pyvista(line_network, point_scalars="volume")
    finally:
        _FakePolyData.tube_should_raise = True

    assert np.array_equal(poly.point_data["pore.render_diameter"], np.array([1.0, 2.0, 4.0]))
    assert np.array_equal(poly.cell_data["throat.render_radius"], np.array([0.25, 0.75]))
    assert plotter.meshes[0][0].tube_kwargs == {
        "scalars": "throat.render_radius",
        "absolute": True,
        "preference": "cell",
    }
    assert plotter.meshes[1][0].glyph_kwargs["scale"] == "pore.render_diameter"


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
