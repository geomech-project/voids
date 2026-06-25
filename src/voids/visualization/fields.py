from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from voids.image.porosity import PermeabilityMap, PorosityMap
from voids.mesh.structured import structured_map_mesh

_AXIS_NAMES = ("x", "y", "z")


def vector_magnitude(vector: np.ndarray) -> np.ndarray:
    """Return the Euclidean magnitude of a dim-first vector field."""

    vec = _as_vector_field(vector)
    return np.asarray(np.linalg.norm(vec, axis=0), dtype=float)


def reference_pressure_to_outlet(
    pressure: np.ndarray,
    *,
    flow_axis: str = "x",
    reference_pressure: float = 1.0e5,
    pressure_outlet: float = 0.0,
) -> np.ndarray:
    """Shift a pressure field so the outlet layer has a reference pressure.

    In incompressible Darcy, Brinkman, and Stokes solves, pressure is only
    determined up to an additive constant. This helper preserves the computed
    pressure differences and adds the constant needed to make the mean outlet
    layer pressure equal to ``reference_pressure + pressure_outlet``.

    The outlet layer is assumed to be the maximum-index face along
    ``flow_axis``, matching the pressure convention used by the map-based
    validation notebooks.
    """

    values = _as_scalar_field(pressure)
    target_outlet_pressure = float(reference_pressure) + float(pressure_outlet)
    if not np.isfinite(target_outlet_pressure):
        raise ValueError("reference and outlet pressures must be finite")
    axis_index = _axis_index(flow_axis, values.ndim)
    outlet_selector: list[slice | int] = [slice(None)] * values.ndim
    outlet_selector[axis_index] = values.shape[axis_index] - 1
    outlet_reference = float(np.mean(values[tuple(outlet_selector)]))
    return values + target_outlet_pressure - outlet_reference


def reconstruct_tpfa_cell_velocity(
    pressure: np.ndarray,
    permeability: PermeabilityMap | np.ndarray,
    *,
    flow_axis: str = "x",
    viscosity: float = 1.0,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    cell_size: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Reconstruct a cell-centered Darcy velocity field from a TPFA pressure solve.

    The reconstruction uses the same two-point face fluxes as the TPFA balance:
    harmonic permeability on interior faces, half-cell Dirichlet transmissibility
    on inlet/outlet faces, and zero transverse boundary flux. The returned field
    is a dim-first array with shape ``(ndim, *pressure.shape)``.
    """

    pressure_values = np.asarray(pressure, dtype=float)
    if pressure_values.ndim not in {2, 3}:
        raise ValueError("pressure must be a 2D or 3D cell-centered field")
    if not np.all(np.isfinite(pressure_values)):
        raise ValueError("pressure must contain only finite values")
    permeability_values, size = _permeability_values_and_cell_size(
        permeability,
        pressure_values.shape,
        cell_size=cell_size,
    )
    if viscosity <= 0.0 or not np.isfinite(viscosity):
        raise ValueError("viscosity must be positive and finite")
    flow_axis_index = _axis_index(flow_axis, pressure_values.ndim)

    velocity = np.zeros((pressure_values.ndim, *pressure_values.shape), dtype=float)
    for direction in range(pressure_values.ndim):
        low_face = np.zeros_like(pressure_values, dtype=float)
        high_face = np.zeros_like(pressure_values, dtype=float)
        spacing = float(size[direction])

        low_selector: list[slice | int] = [slice(None)] * pressure_values.ndim
        high_selector: list[slice | int] = [slice(None)] * pressure_values.ndim
        low_selector[direction] = slice(0, -1)
        high_selector[direction] = slice(1, None)
        low_index = tuple(low_selector)
        high_index = tuple(high_selector)

        k_left = permeability_values[low_index]
        k_right = permeability_values[high_index]
        k_face = _harmonic_permeability(k_left, k_right)
        face_velocity = k_face * (pressure_values[low_index] - pressure_values[high_index])
        face_velocity /= float(viscosity) * spacing
        high_face[low_index] = face_velocity
        low_face[high_index] = face_velocity

        if direction == flow_axis_index:
            inlet_selector: list[slice | int] = [slice(None)] * pressure_values.ndim
            outlet_selector: list[slice | int] = [slice(None)] * pressure_values.ndim
            inlet_selector[direction] = 0
            outlet_selector[direction] = pressure_values.shape[direction] - 1
            inlet_index = tuple(inlet_selector)
            outlet_index = tuple(outlet_selector)
            low_face[inlet_index] = (
                permeability_values[inlet_index]
                * (float(pressure_inlet) - pressure_values[inlet_index])
                / (float(viscosity) * (spacing / 2.0))
            )
            high_face[outlet_index] = (
                permeability_values[outlet_index]
                * (pressure_values[outlet_index] - float(pressure_outlet))
                / (float(viscosity) * (spacing / 2.0))
            )

        velocity[direction] = 0.5 * (low_face + high_face)

    return velocity


def write_structured_vector_field(
    vector: np.ndarray,
    grid: PorosityMap,
    path: str | Path,
    *,
    name: str = "velocity",
    extra_cell_data: Mapping[str, np.ndarray] | None = None,
    file_format: str | None = None,
) -> Path:
    """Write a regular-grid vector field to a ParaView-readable mesh file.

    The output uses the structured map mesh representation in `voids.mesh` and
    stores the vector as cell data. Use a `.vtu` or `.vtk` suffix for common
    ParaView workflows.
    """

    if not name:
        raise ValueError("name must be a non-empty string")
    vec = _as_vector_field(vector, shape=grid.shape)
    if not np.all(np.isfinite(vec)):
        raise ValueError("vector field must contain only finite values")

    mesh = structured_map_mesh(
        grid,
        extra_cell_data=extra_cell_data,
        require_finite_cell_data=True,
    )
    vector_cell_data = np.moveaxis(vec, 0, -1).reshape((-1, vec.shape[0]), order="C")
    if vector_cell_data.shape[1] == 2:
        vector_cell_data = np.column_stack(
            [vector_cell_data, np.zeros(vector_cell_data.shape[0], dtype=vector_cell_data.dtype)]
        )
    mesh.cell_data[str(name)] = [vector_cell_data]
    return mesh.write(path, file_format=file_format)


def write_dolfinx_function_xdmf(
    function: Any, path: str | Path, *, name: str | None = None
) -> Path:
    """Write a DOLFINx function to an XDMF/HDF5 file readable by ParaView.

    DOLFINx XDMF output expects a function layout compatible with the mesh
    geometry. To make high-order and discontinuous fields robustly viewable, the
    function is first interpolated to a first-order Lagrange visualization space.
    """

    from dolfinx.io import XDMFFile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_function = _linear_dolfinx_export_function(function, name=name)
    mesh = export_function.function_space.mesh
    with XDMFFile(mesh.comm, str(destination), "w") as handle:
        handle.write_mesh(mesh)
        handle.write_function(export_function)
    return destination


def sample_dolfinx_function_on_grid(
    function: Any,
    *,
    shape: Sequence[int],
    cell_size: float | Sequence[float],
    origin: Sequence[float] | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Sample a DOLFINx scalar or vector function at regular cell centers."""

    from dolfinx import geometry

    grid_shape = tuple(int(value) for value in shape)
    if len(grid_shape) not in {2, 3}:
        raise ValueError("shape must describe a 2D or 3D grid")
    size = _cell_size_tuple(cell_size, len(grid_shape))
    start = (0.0,) * len(grid_shape) if origin is None else tuple(float(value) for value in origin)
    if len(start) != len(grid_shape):
        raise ValueError("origin dimensionality must match shape")

    axes = [
        start[axis] + (np.arange(grid_shape[axis], dtype=float) + 0.5) * size[axis]
        for axis in range(len(grid_shape))
    ]
    meshgrid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([component.reshape(-1, order="C") for component in meshgrid])
    if len(grid_shape) == 2:
        points = np.column_stack([points, np.zeros(points.shape[0], dtype=float)])

    mesh = function.function_space.mesh
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, candidate_cells, points)

    valid_indices: list[int] = []
    cells: list[int] = []
    for point_index in range(points.shape[0]):
        links = colliding_cells.links(np.int32(point_index))
        if len(links) > 0:
            valid_indices.append(point_index)
            cells.append(int(links[0]))
    if not valid_indices:
        raise RuntimeError("no grid sample points were found inside the DOLFINx mesh")

    valid_points = points[np.asarray(valid_indices, dtype=np.int64)]
    values = np.asarray(function.eval(valid_points, np.asarray(cells, dtype=np.int32)), dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]

    flat = np.full((points.shape[0], values.shape[1]), float(fill_value), dtype=float)
    flat[np.asarray(valid_indices, dtype=np.int64)] = values
    if values.shape[1] == 1:
        return flat[:, 0].reshape(grid_shape, order="C")
    return np.moveaxis(flat.reshape((*grid_shape, values.shape[1]), order="C"), -1, 0)


def plot_scalar_midplanes(
    scalar: np.ndarray,
    *,
    title: str,
    path: str | Path | None = None,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_use_offset: bool = True,
) -> Any:
    """Plot scalar-field mid-slices and optionally save the figure."""

    import matplotlib.pyplot as plt

    values = _as_scalar_field(scalar)
    image_vmin = float(np.min(values)) if vmin is None else vmin
    image_vmax = float(np.max(values)) if vmax is None else vmax
    specs = _midplane_specs(values.shape)
    fig, axes = plt.subplots(
        1, len(specs), figsize=(4.2 * len(specs), 3.8), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    for ax, spec in zip(axes_array, specs, strict=True):
        image = _scalar_slice(values, spec.plane_axis).T
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=image_vmin, vmax=image_vmax)
        ax.set_title(spec.label)
        ax.set_xlabel(_AXIS_NAMES[spec.in_plane_axes[0]])
        ax.set_ylabel(_AXIS_NAMES[spec.in_plane_axes[1]])
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        if not colorbar_use_offset:
            from matplotlib.ticker import ScalarFormatter

            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            colorbar.formatter = formatter
            colorbar.update_ticks()
    fig.suptitle(title)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
    return fig


def plot_vector_midplanes(
    vector: np.ndarray,
    *,
    title: str,
    path: str | Path | None = None,
    quiver_stride: int = 3,
    cmap: str = "magma",
    colorbar_label: str = "velocity magnitude",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Plot vector-field magnitude mid-slices with in-plane quiver arrows."""

    import matplotlib.pyplot as plt

    if quiver_stride <= 0:
        raise ValueError("quiver_stride must be positive")
    vec = _as_vector_field(vector)
    magnitude = vector_magnitude(vec)
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    if not finite_magnitude.size:
        raise ValueError("vector field must contain at least one finite magnitude")
    image_vmin = float(np.min(finite_magnitude)) if vmin is None else vmin
    image_vmax = float(np.max(finite_magnitude)) if vmax is None else vmax
    specs = _midplane_specs(magnitude.shape)
    fig, axes = plt.subplots(
        1, len(specs), figsize=(4.5 * len(specs), 3.9), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    for ax, spec in zip(axes_array, specs, strict=True):
        magnitude_slice = _scalar_slice(magnitude, spec.plane_axis)
        im = ax.imshow(
            magnitude_slice.T, origin="lower", cmap=cmap, vmin=image_vmin, vmax=image_vmax
        )
        comp_x = _scalar_slice(vec[spec.in_plane_axes[0]], spec.plane_axis)
        comp_y = _scalar_slice(vec[spec.in_plane_axes[1]], spec.plane_axis)
        x = np.arange(magnitude_slice.shape[0])
        y = np.arange(magnitude_slice.shape[1])
        grid_x, grid_y = np.meshgrid(x, y)
        stride = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
        u = comp_x.T[stride]
        v = comp_y.T[stride]
        in_plane_magnitude = np.hypot(u, v)
        finite_magnitude = in_plane_magnitude[np.isfinite(in_plane_magnitude)]
        max_in_plane_magnitude = float(np.max(finite_magnitude)) if finite_magnitude.size else 0.0
        if max_in_plane_magnitude > 0.0:
            max_arrow_length = max(1.0, 0.65 * float(quiver_stride))
            ax.quiver(
                grid_x[stride],
                grid_y[stride],
                u,
                v,
                color="white",
                angles="xy",
                scale_units="xy",
                scale=max_in_plane_magnitude / max_arrow_length,
                width=0.003,
            )
        ax.set_title(spec.label)
        ax.set_xlabel(_AXIS_NAMES[spec.in_plane_axes[0]])
        ax.set_ylabel(_AXIS_NAMES[spec.in_plane_axes[1]])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.suptitle(title)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
    return fig


class _SliceSpec:
    def __init__(self, *, plane_axis: int, in_plane_axes: tuple[int, int], label: str) -> None:
        self.plane_axis = plane_axis
        self.in_plane_axes = in_plane_axes
        self.label = label


def _as_scalar_field(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim not in {2, 3}:
        raise ValueError("scalar field must be 2D or 3D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scalar field must contain only finite values")
    return arr


def _as_vector_field(vector: np.ndarray, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    if shape is not None:
        if arr.shape == (len(shape), *shape):
            return arr
        if arr.shape == (*shape, len(shape)):
            return np.moveaxis(arr, -1, 0)
        raise ValueError(
            "vector field must have shape (ndim, *shape) or (*shape, ndim); "
            f"got {arr.shape} for shape {shape}"
        )
    if arr.ndim < 3 or arr.shape[0] not in {2, 3}:
        raise ValueError("vector field must be dim-first with shape (ndim, *field_shape)")
    if arr.ndim != arr.shape[0] + 1:
        raise ValueError("vector dimensionality must match field dimensionality")
    return arr


def _permeability_values_and_cell_size(
    permeability: PermeabilityMap | np.ndarray,
    shape: tuple[int, ...],
    *,
    cell_size: float | Sequence[float] | None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    if isinstance(permeability, PermeabilityMap):
        values = np.asarray(permeability.values, dtype=float)
        size = _cell_size_tuple(permeability.cell_size, len(shape))
    else:
        values = np.asarray(permeability, dtype=float)
        if cell_size is None:
            size = (1.0,) * len(shape)
        else:
            size = _cell_size_tuple(cell_size, len(shape))
    if values.shape != shape:
        raise ValueError(f"permeability must have shape {shape}, got {values.shape}")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("permeability must contain finite non-negative values")
    return values, size


def _cell_size_tuple(cell_size: float | Sequence[float], ndim: int) -> tuple[float, ...]:
    if isinstance(cell_size, Sequence) and not isinstance(cell_size, (str, bytes)):
        size = tuple(float(value) for value in cell_size)
    else:
        size = (float(cell_size),) * ndim
    if len(size) != ndim:
        raise ValueError("cell_size dimensionality must match field dimensionality")
    if any(value <= 0.0 or not np.isfinite(value) for value in size):
        raise ValueError("cell_size values must be positive and finite")
    return size


def _axis_index(axis: str, ndim: int) -> int:
    if axis not in _AXIS_NAMES[:ndim]:
        raise ValueError(f"axis must be one of {_AXIS_NAMES[:ndim]}, got {axis!r}")
    return _AXIS_NAMES.index(axis)


def _harmonic_permeability(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros_like(left, dtype=float)
    mask = (left > 0.0) & (right > 0.0)
    out[mask] = 2.0 * left[mask] * right[mask] / (left[mask] + right[mask])
    return out


def _midplane_specs(shape: tuple[int, ...]) -> list[_SliceSpec]:
    if len(shape) == 2:
        return [_SliceSpec(plane_axis=-1, in_plane_axes=(0, 1), label="midplane")]
    specs: list[_SliceSpec] = []
    for plane_axis in range(3):
        in_plane = tuple(axis for axis in range(3) if axis != plane_axis)
        specs.append(
            _SliceSpec(
                plane_axis=plane_axis,
                in_plane_axes=(in_plane[0], in_plane[1]),
                label=f"{_AXIS_NAMES[plane_axis]}-mid",
            )
        )
    return specs


def _scalar_slice(values: np.ndarray, plane_axis: int) -> np.ndarray:
    if values.ndim == 2:
        return values
    selector: list[slice | int] = [slice(None)] * values.ndim
    selector[plane_axis] = values.shape[plane_axis] // 2
    return np.asarray(values[tuple(selector)], dtype=float)


def _linear_dolfinx_export_function(function: Any, *, name: str | None) -> Any:
    from dolfinx import fem
    import basix.ufl as basix_ufl

    mesh = function.function_space.mesh
    value_shape = tuple(int(value) for value in getattr(function, "ufl_shape", ()))
    if value_shape:
        element = basix_ufl.element("Lagrange", mesh.basix_cell(), 1, shape=value_shape)
    else:
        element = basix_ufl.element("Lagrange", mesh.basix_cell(), 1)
    export_space = fem.functionspace(mesh, element)
    export_function = fem.Function(export_space)
    export_function.interpolate(function)
    export_function.name = (
        str(name) if name is not None else str(getattr(function, "name", "field"))
    )
    return export_function


__all__ = [
    "plot_scalar_midplanes",
    "plot_vector_midplanes",
    "reference_pressure_to_outlet",
    "reconstruct_tpfa_cell_velocity",
    "sample_dolfinx_function_on_grid",
    "vector_magnitude",
    "write_dolfinx_function_xdmf",
    "write_structured_vector_field",
]
