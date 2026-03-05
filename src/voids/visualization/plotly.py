from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from voids.core.network import Network
from voids.visualization._sizing import resolve_size_values, scale_sizes_to_pixels


def _require_plotly():
    """Import Plotly lazily.

    Returns
    -------
    tuple
        ``(plotly.graph_objects, plotly.colors.sample_colorscale)``.

    Raises
    ------
    ImportError
        If Plotly is not installed.
    """

    try:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is not installed. Use an environment with plotly available to create interactive figures."
        ) from exc
    return go, sample_colorscale


def _resolve_scalars(
    values: str | np.ndarray | None,
    *,
    store: dict[str, np.ndarray],
    expected_shape: tuple[int, ...],
    prefix: str,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve scalar input from a field name or explicit array.

    Parameters
    ----------
    values :
        Field name, explicit array, or ``None``.
    store :
        Property dictionary used when ``values`` is a field name.
    expected_shape :
        Expected array shape.
    prefix :
        Name prefix used in error messages and labels.

    Returns
    -------
    tuple
        Pair ``(array, label)``. Both entries are ``None`` when no scalars are requested.

    Raises
    ------
    KeyError
        If a requested field name is missing.
    ValueError
        If an explicit array has the wrong shape.
    """

    if values is None:
        return None, None
    if isinstance(values, str):
        if values not in store:
            raise KeyError(f"Missing {prefix} field '{values}'")
        return np.asarray(store[values], dtype=float), f"{prefix}.{values}"
    arr = np.asarray(values, dtype=float)
    if arr.shape != expected_shape:
        raise ValueError(f"{prefix} scalar array must have shape {expected_shape}")
    return arr, f"{prefix}.scalar"


def _sample_indices(count: int, max_count: int | None) -> np.ndarray:
    """Return evenly spaced sample indices.

    Parameters
    ----------
    count :
        Total number of items.
    max_count :
        Maximum number of sampled items. If ``None``, keep all.

    Returns
    -------
    numpy.ndarray
        Integer indices. When downsampling is needed, the spacing is approximately
        ``count / max_count``.
    """

    if max_count is None or count <= max_count:
        return np.arange(count, dtype=np.int64)
    step = max(1, ceil(count / max_count))
    return np.arange(0, count, step, dtype=np.int64)


def _rgb_with_opacity(color: str, opacity: float) -> str:
    """Attach opacity to an RGB color string.

    Parameters
    ----------
    color :
        Plotly-style ``rgb(r,g,b)`` string.
    opacity :
        Desired opacity.

    Returns
    -------
    str
        ``rgba(r,g,b,a)`` string when possible, otherwise the original color.
    """

    if color.startswith("rgb("):
        return "rgba(" + color[4:-1] + f",{opacity})"
    return color


def _scalar_bounds(values: np.ndarray | None) -> tuple[float | None, float | None]:
    """Return scalar bounds for color normalization.

    Parameters
    ----------
    values :
        Scalar array or ``None``.

    Returns
    -------
    tuple
        ``(vmin, vmax)`` or ``(None, None)`` when no valid values are present.
    """

    if values is None or values.size == 0:
        return None, None
    return float(np.min(values)), float(np.max(values))


def plot_network_plotly(
    net: Network,
    *,
    point_scalars: str | np.ndarray | None = None,
    cell_scalars: str | np.ndarray | None = None,
    point_sizes: str | np.ndarray | bool | None = None,
    throat_sizes: str | np.ndarray | bool | None = None,
    point_size: float | None = None,
    line_width: float | None = None,
    line_opacity: float = 0.4,
    size_scale: float = 1.0,
    max_throats: int | None = 1000,
    title: str | None = None,
    show_colorbar: bool = True,
    layout_kwargs: dict[str, Any] | None = None,
):
    """Create an interactive Plotly visualization of a pore-throat network.

    Parameters
    ----------
    net :
        Network to render.
    point_scalars :
        Pore field name or explicit pore-valued array with shape ``(Np,)``.
    cell_scalars :
        Throat field name or explicit throat-valued array with shape ``(Nt,)``.
    point_sizes, throat_sizes :
        Pore/throat characteristic size field name, explicit size array, ``None`` for
        automatic size-field detection, or ``False`` to disable size-driven rendering.
        Automatically detected size fields follow the priority
        ``diameter_equivalent -> diameter_inscribed -> radius_inscribed -> area``.
    point_size :
        Constant marker size for pores when explicit size rendering is disabled. When
        size-driven rendering is active, this acts as the reference marker size for
        median-sized pores.
    line_width :
        Constant line width for throats when explicit size rendering is disabled. When
        size-driven rendering is active, this acts as the reference width for
        median-sized throats.
    line_opacity :
        Opacity applied to throat lines.
    size_scale :
        Multiplicative factor applied to size-driven pore markers and throat widths.
    max_throats :
        Maximum number of throats to draw. Large networks are downsampled for responsiveness.
    title :
        Figure title.
    show_colorbar :
        If ``True``, display a colorbar for pore scalars.
    layout_kwargs :
        Optional Plotly layout overrides.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure.

    Notes
    -----
    If only pore scalars are given, each throat is colored by the arithmetic mean of
    its endpoint pore values:

    ``s_throat = 0.5 * (s_i + s_j)``

    The throat colormap is normalized with the same scalar bounds as the pore markers
    so that equal numerical values map to equal colors across pores and throats.
    """

    go, sample_colorscale = _require_plotly()
    point_values, point_label = _resolve_scalars(
        point_scalars, store=net.pore, expected_shape=(net.Np,), prefix="pore"
    )
    cell_values, cell_label = _resolve_scalars(
        cell_scalars, store=net.throat, expected_shape=(net.Nt,), prefix="throat"
    )
    point_size_values, point_size_label = resolve_size_values(
        point_sizes, store=net.pore, expected_shape=(net.Np,), prefix="pore"
    )
    throat_size_values, throat_size_label = resolve_size_values(
        throat_sizes, store=net.throat, expected_shape=(net.Nt,), prefix="throat"
    )

    coords = np.asarray(net.pore_coords, dtype=float)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    sampled = _sample_indices(net.Nt, max_throats)
    point_vmin, point_vmax = _scalar_bounds(point_values)
    cell_vmin, cell_vmax = _scalar_bounds(cell_values)
    point_size_ref = float(
        point_size if point_size is not None else (6.0 if net.Np <= 2000 else 4.0)
    )
    line_width_ref = float(2.0 if line_width is None else line_width)
    use_variable_point_sizes = point_size_values is not None
    use_variable_throat_sizes = throat_size_values is not None

    if use_variable_point_sizes:
        marker_size: float | np.ndarray = scale_sizes_to_pixels(
            point_size_values,
            reference=point_size_ref,
            scale=size_scale,
            min_size=max(2.0, 0.5 * point_size_ref),
            max_size=max(18.0, 4.0 * point_size_ref),
        )
    else:
        marker_size = point_size_ref

    if use_variable_throat_sizes:
        sampled_line_widths = scale_sizes_to_pixels(
            throat_size_values[sampled],
            reference=line_width_ref,
            scale=size_scale,
            min_size=0.75,
            max_size=max(10.0, 4.0 * line_width_ref),
        )
    else:
        sampled_line_widths = np.full(sampled.shape, line_width_ref, dtype=float)

    if point_values is not None:
        marker: dict[str, Any] = {
            "size": marker_size,
            "color": point_values,
            "colorscale": "Viridis",
            "showscale": show_colorbar,
        }
        if use_variable_point_sizes:
            marker["sizemode"] = "diameter"
        if point_vmin is not None and point_vmax is not None:
            marker["cmin"] = point_vmin
            marker["cmax"] = point_vmax
        if show_colorbar:
            marker["colorbar"] = {"title": point_label or "pore scalar"}
    else:
        marker = {
            "size": marker_size,
            "color": "royalblue",
            "showscale": False,
        }
        if use_variable_point_sizes:
            marker["sizemode"] = "diameter"

    pore_text = []
    for idx in range(net.Np):
        hover_lines = [f"Pore {idx}"]
        if point_values is not None:
            hover_lines.append(f"{point_label or 'value'}={point_values[idx]:.3e}")
        if use_variable_point_sizes and point_size_label is not None:
            hover_lines.append(f"{point_size_label}={point_size_values[idx]:.3e}")
        pore_text.append("<br>".join(hover_lines))

    traces: list[Any] = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=marker,
            name="Pores",
            text=pore_text,
            hoverinfo="text",
        )
    ]

    if cell_values is not None:
        throat_values = np.asarray(cell_values[sampled], dtype=float)
        throat_label = cell_label or "throat scalar"
        color_vmin, color_vmax = cell_vmin, cell_vmax
    elif point_values is not None:
        conns = net.throat_conns[sampled]
        throat_values = 0.5 * (point_values[conns[:, 0]] + point_values[conns[:, 1]])
        throat_label = f"avg({point_label or 'pore scalar'})"
        color_vmin, color_vmax = point_vmin, point_vmax
    else:
        throat_values = None
        throat_label = None
        color_vmin, color_vmax = None, None

    for local_idx, throat_idx in enumerate(sampled):
        i, j = net.throat_conns[throat_idx]
        hover_lines = [f"Throat {int(throat_idx)}"]
        if throat_values is None:
            color = _rgb_with_opacity("rgb(100,100,100)", line_opacity)
        else:
            if color_vmin is not None and color_vmax is not None and color_vmax > color_vmin:
                norm = float((throat_values[local_idx] - color_vmin) / (color_vmax - color_vmin))
            else:
                norm = 0.5
            color = _rgb_with_opacity(sample_colorscale("Viridis", [norm])[0], line_opacity)
            hover_lines.append(f"{throat_label}={float(throat_values[local_idx]):.3e}")
        if use_variable_throat_sizes and throat_size_label is not None:
            hover_lines.append(f"{throat_size_label}={float(throat_size_values[throat_idx]):.3e}")
        traces.append(
            go.Scatter3d(
                x=[x[i], x[j]],
                y=[y[i], y[j]],
                z=[z[i], z[j]],
                mode="lines",
                line={"color": color, "width": float(sampled_line_widths[local_idx])},
                name="Throats",
                showlegend=False,
                text="<br>".join(hover_lines),
                hoverinfo="text",
            )
        )

    if title is None:
        title = "Pore network"
        if sampled.size != net.Nt:
            title += f" (showing {sampled.size} of {net.Nt} throats)"

    figure = go.Figure(data=traces)
    layout: dict[str, Any] = {
        "title": title,
        "scene": {
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        "width": 900,
        "height": 700,
        "hovermode": "closest",
    }
    if layout_kwargs:
        layout.update(layout_kwargs)
    figure.update_layout(**layout)
    return figure
