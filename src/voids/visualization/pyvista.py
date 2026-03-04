from __future__ import annotations

from typing import Any

import numpy as np

from voids.core.network import Network


def _require_pyvista():
    """Import PyVista lazily.

    Returns
    -------
    module
        Imported :mod:`pyvista` module.

    Raises
    ------
    ImportError
        If PyVista is not installed.
    """

    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyVista is not installed. Use the 'default' or 'test' pixi environment or install pyvista."
        ) from exc
    return pv


def _line_cells_from_conns(conns: np.ndarray) -> np.ndarray:
    """Convert throat connections to VTK polyline cell encoding.

    Parameters
    ----------
    conns :
        Integer array with shape ``(Nt, 2)``.

    Returns
    -------
    numpy.ndarray
        Flat cell array suitable for ``pyvista.PolyData(lines=...)``.

    Raises
    ------
    ValueError
        If ``conns`` does not have shape ``(Nt, 2)``.

    Notes
    -----
    Each throat line is stored as ``[2, i, j]``, where the leading ``2`` indicates
    the number of points in the polyline cell.
    """

    conns = np.asarray(conns, dtype=np.int64)
    if conns.ndim != 2 or conns.shape[1] != 2:
        raise ValueError("throat connections must have shape (Nt, 2)")
    cells = np.empty((conns.shape[0], 3), dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1:] = conns
    return cells.ravel()


def network_to_pyvista_polydata(
    net: Network,
    *,
    point_scalars: str | np.ndarray | None = None,
    cell_scalars: str | np.ndarray | None = None,
    include_all_numeric_fields: bool = False,
):
    """Convert a network to ``pyvista.PolyData``.

    Parameters
    ----------
    net :
        Network to convert.
    point_scalars, cell_scalars :
        Pore/throat scalar field name or explicit array.
    include_all_numeric_fields :
        If ``True``, attach every 1D numeric pore/throat array whose length matches
        ``Np`` or ``Nt``.

    Returns
    -------
    pyvista.PolyData
        PolyData with pores as points and throats as line cells.

    Raises
    ------
    KeyError
        If a requested scalar field name is missing.
    ValueError
        If an explicit scalar array has the wrong shape.
    """

    pv = _require_pyvista()

    points = np.asarray(net.pore_coords, dtype=float)
    line_cells = _line_cells_from_conns(net.throat_conns)
    poly = pv.PolyData(points, lines=line_cells)

    poly.point_data["pore.id"] = np.arange(net.Np, dtype=np.int64)
    poly.cell_data["throat.id"] = np.arange(net.Nt, dtype=np.int64)

    if include_all_numeric_fields:
        for k, v in net.pore.items():
            a = np.asarray(v)
            if a.ndim == 1 and a.shape[0] == net.Np and np.issubdtype(a.dtype, np.number):
                poly.point_data[f"pore.{k}"] = a
        for k, v in net.throat.items():
            a = np.asarray(v)
            if a.ndim == 1 and a.shape[0] == net.Nt and np.issubdtype(a.dtype, np.number):
                poly.cell_data[f"throat.{k}"] = a

    if isinstance(point_scalars, str):
        if point_scalars not in net.pore:
            raise KeyError(f"Missing pore field '{point_scalars}'")
        poly.point_data["pore.scalar"] = np.asarray(net.pore[point_scalars])
        poly.set_active_scalars("pore.scalar", preference="point")
    elif point_scalars is not None:
        arr = np.asarray(point_scalars)
        if arr.shape != (net.Np,):
            raise ValueError("point_scalars array must have shape (Np,)")
        poly.point_data["pore.scalar"] = arr
        poly.set_active_scalars("pore.scalar", preference="point")

    if isinstance(cell_scalars, str):
        if cell_scalars not in net.throat:
            raise KeyError(f"Missing throat field '{cell_scalars}'")
        poly.cell_data["throat.scalar"] = np.asarray(net.throat[cell_scalars])
    elif cell_scalars is not None:
        arr = np.asarray(cell_scalars)
        if arr.shape != (net.Nt,):
            raise ValueError("cell_scalars array must have shape (Nt,)")
        poly.cell_data["throat.scalar"] = arr

    return poly


def plot_network_pyvista(
    net: Network,
    *,
    point_scalars: str | np.ndarray | None = None,
    cell_scalars: str | np.ndarray | None = None,
    show_points: bool = True,
    show_lines: bool = True,
    line_width: float = 3.0,
    point_size: float = 9.0,
    render_tubes: bool = False,
    tube_radius: float | None = None,
    off_screen: bool = False,
    screenshot: str | None = None,
    show_axes: bool = True,
    notebook: bool | None = None,
    **add_mesh_kwargs: Any,
):
    """Render a pore network with PyVista.

    Parameters
    ----------
    net :
        Network to render.
    point_scalars, cell_scalars :
        Pore/throat scalar field name or explicit array.
    show_points, show_lines :
        Toggle pore and throat rendering.
    line_width :
        Width used for line rendering.
    point_size :
        Marker size used for pores.
    render_tubes :
        If ``True``, convert throats from lines to tubes.
    tube_radius :
        Optional tube radius when ``render_tubes`` is enabled.
    off_screen :
        If ``True``, create an off-screen plotter for headless rendering.
    screenshot :
        Optional screenshot output path. When provided, the plot is rendered and saved.
    show_axes :
        If ``True``, display orientation axes.
    notebook :
        Optional PyVista notebook flag. Defaults to ``False`` when omitted.
    **add_mesh_kwargs :
        Additional keyword arguments forwarded to :meth:`pyvista.Plotter.add_mesh`.

    Returns
    -------
    tuple
        Pair ``(plotter, polydata)``.

    Notes
    -----
    For throat rendering, scalar selection follows this priority:

    1. explicit throat/cell scalar data
    2. pore/point scalar data, reused on the line representation

    This allows pore-defined pressure fields to color both pores and throats in a
    consistent network visualization.
    """

    pv = _require_pyvista()
    poly = network_to_pyvista_polydata(
        net,
        point_scalars=point_scalars,
        cell_scalars=cell_scalars,
        include_all_numeric_fields=True,
    )

    if notebook is None:
        notebook = False
    pl = pv.Plotter(off_screen=off_screen, notebook=notebook)

    line_scalars_name = "throat.scalar" if "throat.scalar" in poly.cell_data else None
    point_scalars_name = "pore.scalar" if "pore.scalar" in poly.point_data else None
    if line_scalars_name is None and point_scalars_name is not None:
        line_scalars_name = point_scalars_name

    if show_lines and net.Nt > 0:
        line_mesh = poly
        if render_tubes:
            kwargs = {}
            if tube_radius is not None:
                kwargs["radius"] = float(tube_radius)
            try:
                line_mesh = poly.tube(**kwargs)
            except Exception:
                line_mesh = poly
        pl.add_mesh(
            line_mesh,
            scalars=line_scalars_name,
            line_width=line_width,
            render_lines_as_tubes=not render_tubes,
            show_scalar_bar=(line_scalars_name is not None),
            **add_mesh_kwargs,
        )

    if show_points and net.Np > 0:
        pl.add_mesh(
            poly,
            style="points",
            point_size=point_size,
            render_points_as_spheres=True,
            scalars=point_scalars_name,
            show_scalar_bar=(point_scalars_name is not None and not show_lines),
        )

    if show_axes:
        pl.add_axes()

    if screenshot is not None:
        pl.show(auto_close=False)
        pl.screenshot(screenshot)
    return pl, poly
