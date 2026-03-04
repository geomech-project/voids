from __future__ import annotations

import numpy as np
import pytest

from voids.visualization.plotly import plot_network_plotly
from voids.visualization.pyvista import network_to_pyvista_polydata, plot_network_pyvista


def test_pyvista_visualization_api_clean_import_error(line_network):
    """Test that PyVista conversion either works or fails with a clean import error."""

    try:
        network_to_pyvista_polydata(line_network)
    except ImportError:
        return
    except Exception as exc:  # pragma: no cover - only if pyvista installed unexpectedly
        pytest.fail(f"Unexpected exception when pyvista is installed: {exc}")


def test_pyvista_plot_api_clean_import_error(line_network):
    """Test that PyVista plotting either works or fails with a clean import error."""

    try:
        plot_network_pyvista(line_network, off_screen=True)
    except ImportError:
        return
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Unexpected exception when pyvista is installed: {exc}")


def test_plotly_plot_api_clean_import_error(line_network):
    """Test that Plotly rendering either works or fails with a clean import error."""

    try:
        fig = plot_network_plotly(line_network, point_scalars=line_network.pore["volume"])
    except ImportError:
        return
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Unexpected exception when plotly is installed: {exc}")
    assert len(fig.data) >= 1


def test_plotly_throat_colors_follow_mean_point_scalars_on_point_range(line_network):
    """Test throat coloring from mean pore scalars on the pore-scalar color range."""

    try:
        from plotly.colors import sample_colorscale
    except Exception:
        return

    point_values = np.array([0.0, 1.0, 3.0], dtype=float)
    fig = plot_network_plotly(line_network, point_scalars=point_values, line_opacity=0.4)

    assert fig.data[0].marker.cmin == pytest.approx(0.0)
    assert fig.data[0].marker.cmax == pytest.approx(3.0)

    expected_means = [0.5, 2.0]
    expected_colors = []
    for value in expected_means:
        norm = value / 3.0
        rgb = sample_colorscale("Viridis", [norm])[0]
        expected_colors.append("rgba(" + rgb[4:-1] + ",0.4)")

    throat_colors = [trace.line.color for trace in fig.data[1:]]
    assert throat_colors == expected_colors
