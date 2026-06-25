from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from voids.image.porosity import PermeabilityMap, PorosityMap

MapMeshElement = Literal["auto", "quad", "triangle", "hexahedron", "tetra", "tetrahedron"]

_FORMAT_EXTENSIONS = {
    "gmsh": ".msh",
    "msh": ".msh",
    "vtk": ".vtk",
    "vtu": ".vtu",
    "netgen": ".vol",
    "vol": ".vol",
}


@dataclass(slots=True)
class StructuredMapMesh:
    """Structured mesh representation for regular porosity/permeability maps.

    Attributes
    ----------
    points :
        Mesh point coordinates. Points are always stored as three-dimensional
        coordinates so 2-D maps can be written by formats that expect 3-D
        coordinates with ``z=0``.
    cells :
        A single meshio-compatible cell block. 2-D maps use ``"quad"`` cells by
        default or ``"triangle"`` cells when requested. 3-D maps use
        ``"hexahedron"`` cells by default or ``"tetra"`` cells when requested.
    cell_data :
        Cell-wise data arrays whose flattened order follows
        ``numpy.ravel(order="C")`` of the original map arrays.
    """

    points: np.ndarray
    cells: tuple[tuple[str, np.ndarray], ...]
    cell_data: dict[str, list[np.ndarray]]

    @property
    def cell_type(self) -> str:
        """Return the meshio cell type used by the structured map mesh."""

        return self.cells[0][0]

    @property
    def cell_count(self) -> int:
        """Return the number of cells in the structured map mesh."""

        return int(self.cells[0][1].shape[0])

    def to_meshio(self, *, include_cell_data: bool = True) -> Any:
        """Return a ``meshio.Mesh`` object.

        Parameters
        ----------
        include_cell_data :
            If ``False``, only geometry is included. This is useful for formats
            that do not preserve arbitrary floating-point cell data.
        """

        meshio = _import_meshio()
        return meshio.Mesh(
            points=self.points,
            cells=list(self.cells),
            cell_data=self.cell_data if include_cell_data else {},
        )

    def write(self, path: str | Path, *, file_format: str | None = None) -> Path:
        """Write the mesh with meshio and return the destination path."""

        return _write_mesh(self, path, file_format=file_format)


def mesh_format_extension(format_name: str) -> str:
    """Return the default filename extension for a supported mesh export format."""

    key = str(format_name).strip().lower().lstrip(".")
    try:
        return _FORMAT_EXTENSIONS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_FORMAT_EXTENSIONS))
        raise ValueError(
            f"Unsupported mesh format {format_name!r}; expected one of {supported}"
        ) from exc


def structured_map_mesh(
    porosity_map: PorosityMap,
    *,
    permeability_map: PermeabilityMap | None = None,
    extra_cell_data: Mapping[str, np.ndarray] | None = None,
    element_type: MapMeshElement = "auto",
    include_cell_index: bool = True,
    require_finite_cell_data: bool = True,
) -> StructuredMapMesh:
    """Generate a structured mesh from a regular porosity map.

    Parameters
    ----------
    porosity_map :
        Cell-average porosity map. Its grid defines the mesh topology and
        physical coordinates.
    permeability_map :
        Optional permeability map on the same grid. When supplied, it is written
        as the ``"permeability"`` cell-data field.
    extra_cell_data :
        Optional additional cell-data arrays. Each array must have the same shape
        as ``porosity_map.values``.
    element_type :
        Mesh cell type. ``"auto"`` uses quadrilaterals for 2-D maps and
        hexahedra for 3-D maps. ``"triangle"`` is valid only for 2-D maps and
        splits each quadrilateral into two triangles. ``"tetra"`` and
        ``"tetrahedron"`` are valid only for 3-D maps and split each
        hexahedron into six tetrahedra.
    include_cell_index :
        If ``True``, include a zero-based ``"cell_index"`` integer field that
        maps each mesh cell back to its parent map cell. For simplex exports,
        the child triangles or tetrahedra from one map cell share the same
        value.
    require_finite_cell_data :
        If ``True``, reject NaN or infinite cell-data values before export.

    Returns
    -------
    StructuredMapMesh
        Meshio-compatible points, cell connectivity, and cell-data arrays.

    Notes
    -----
    The mesh is a representation of a regular map grid, not an image-boundary
    conforming segmentation mesh. Cell ``n`` corresponds to
    ``porosity_map.values.ravel(order="C")[n]``.
    """

    _validate_map_grid(porosity_map)
    points, cells = _structured_points_and_cells(
        shape=porosity_map.shape,
        cell_size=_map_cell_size(porosity_map),
        origin=_map_origin(porosity_map),
        element_type=element_type,
    )
    base_cell_count = int(np.prod(porosity_map.shape))
    mesh_cell_count = int(cells[1].shape[0])

    cell_data: dict[str, list[np.ndarray]] = {
        "porosity": [
            _expand_cell_data_to_mesh_cells(
                _flatten_cell_data(
                    porosity_map.values,
                    shape=porosity_map.shape,
                    name="porosity",
                    require_finite=require_finite_cell_data,
                ),
                base_cell_count=base_cell_count,
                mesh_cell_count=mesh_cell_count,
            )
        ]
    }

    if permeability_map is not None:
        _validate_matching_map_grid(porosity_map, permeability_map)
        cell_data["permeability"] = [
            _expand_cell_data_to_mesh_cells(
                _flatten_cell_data(
                    permeability_map.values,
                    shape=porosity_map.shape,
                    name="permeability",
                    require_finite=require_finite_cell_data,
                ),
                base_cell_count=base_cell_count,
                mesh_cell_count=mesh_cell_count,
            )
        ]

    if include_cell_index:
        cell_data["cell_index"] = [
            _expand_cell_data_to_mesh_cells(
                np.arange(base_cell_count, dtype=np.int64),
                base_cell_count=base_cell_count,
                mesh_cell_count=mesh_cell_count,
            )
        ]

    if extra_cell_data:
        for raw_name, raw_values in extra_cell_data.items():
            name = str(raw_name)
            if not name:
                raise ValueError("extra_cell_data names must be non-empty strings")
            if name in cell_data:
                raise ValueError(f"Duplicate cell-data name {name!r}")
            cell_data[name] = [
                _expand_cell_data_to_mesh_cells(
                    _flatten_cell_data(
                        raw_values,
                        shape=porosity_map.shape,
                        name=name,
                        require_finite=require_finite_cell_data,
                    ),
                    base_cell_count=base_cell_count,
                    mesh_cell_count=mesh_cell_count,
                )
            ]

    return StructuredMapMesh(points=points, cells=(cells,), cell_data=cell_data)


def write_structured_map_mesh(
    porosity_map: PorosityMap,
    path: str | Path,
    *,
    permeability_map: PermeabilityMap | None = None,
    extra_cell_data: Mapping[str, np.ndarray] | None = None,
    element_type: MapMeshElement = "auto",
    file_format: str | None = None,
    require_finite_cell_data: bool = True,
) -> Path:
    """Write a structured map mesh with porosity/permeability cell data.

    Parameters
    ----------
    porosity_map, permeability_map, extra_cell_data, element_type,
    require_finite_cell_data :
        Passed to :func:`structured_map_mesh`.
    path :
        Destination mesh path. The file extension normally determines the meshio
        writer.
    file_format :
        Optional meshio file-format override.

    Notes
    -----
    Netgen ``.vol`` export is geometry-oriented in meshio. The writer does not
    preserve arbitrary floating-point cell-data arrays, so keep the HDF5 map
    export as the authoritative porosity/permeability field when using Netgen.
    """

    mesh = structured_map_mesh(
        porosity_map,
        permeability_map=permeability_map,
        extra_cell_data=extra_cell_data,
        element_type=element_type,
        require_finite_cell_data=require_finite_cell_data,
    )
    return mesh.write(path, file_format=file_format)


def write_structured_map_meshes(
    porosity_map: PorosityMap,
    output_dir: str | Path,
    *,
    stem: str,
    permeability_map: PermeabilityMap | None = None,
    formats: Sequence[str] = ("gmsh", "vtk", "vtu", "netgen"),
    extra_cell_data: Mapping[str, np.ndarray] | None = None,
    element_type: MapMeshElement = "auto",
    require_finite_cell_data: bool = True,
) -> dict[str, Path]:
    """Write the same structured map mesh to several mesh formats.

    Parameters
    ----------
    porosity_map, permeability_map, extra_cell_data, element_type,
    require_finite_cell_data :
        Passed to :func:`structured_map_mesh`.
    output_dir :
        Directory where mesh files will be written.
    stem :
        Filename stem used for every exported file.
    formats :
        Format labels. Supported labels are ``"gmsh"``, ``"vtk"``, ``"vtu"``,
        and ``"netgen"``. Aliases ``"msh"`` and ``"vol"`` are also accepted.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from the requested format label to the written mesh path.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized_stem = str(stem).strip()
    if not normalized_stem:
        raise ValueError("stem must be a non-empty string")

    mesh = structured_map_mesh(
        porosity_map,
        permeability_map=permeability_map,
        extra_cell_data=extra_cell_data,
        element_type=element_type,
        require_finite_cell_data=require_finite_cell_data,
    )

    paths: dict[str, Path] = {}
    for fmt in formats:
        key = str(fmt).strip().lower().lstrip(".")
        extension = mesh_format_extension(key)
        path = out / f"{normalized_stem}{extension}"
        paths[key] = mesh.write(path)

    return paths


def _import_meshio() -> Any:
    try:
        import meshio  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise ImportError(
            "Structured mesh export requires meshio. Install it with "
            "`pip install meshio` or use the Pixi environment."
        ) from exc
    return meshio


def _validate_map_grid(map_obj: PorosityMap | PermeabilityMap) -> None:
    if map_obj.ndim not in (2, 3):
        raise ValueError("structured mesh export supports only 2D or 3D maps")
    if len(_map_cell_size(map_obj)) != map_obj.ndim:
        raise ValueError("cell_size length must match map dimensionality")
    if len(_map_origin(map_obj)) != map_obj.ndim:
        raise ValueError("origin length must match map dimensionality")


def _validate_matching_map_grid(
    porosity_map: PorosityMap,
    permeability_map: PermeabilityMap,
) -> None:
    _validate_map_grid(permeability_map)
    if permeability_map.shape != porosity_map.shape:
        raise ValueError("permeability_map must have the same shape as porosity_map")
    if not np.allclose(_map_cell_size(permeability_map), _map_cell_size(porosity_map), rtol=0.0):
        raise ValueError("permeability_map must have the same cell_size as porosity_map")
    if not np.allclose(_map_origin(permeability_map), _map_origin(porosity_map), rtol=0.0):
        raise ValueError("permeability_map must have the same origin as porosity_map")


def _map_cell_size(map_obj: PorosityMap | PermeabilityMap) -> tuple[float, ...]:
    raw = map_obj.cell_size
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(float(value) for value in raw)
    return (float(raw),) * map_obj.ndim


def _map_origin(map_obj: PorosityMap | PermeabilityMap) -> tuple[float, ...]:
    raw = map_obj.origin
    if raw is None:
        return (0.0,) * map_obj.ndim
    return tuple(float(value) for value in raw)


def _flatten_cell_data(
    values: np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
    require_finite: bool,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape != shape:
        raise ValueError(f"{name} cell data must have shape {shape}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} cell data must be numeric")
    if require_finite and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} cell data must be finite for mesh export")
    return np.asarray(arr).reshape(-1, order="C")


def _expand_cell_data_to_mesh_cells(
    values: np.ndarray,
    *,
    base_cell_count: int,
    mesh_cell_count: int,
) -> np.ndarray:
    if values.shape != (base_cell_count,):
        raise ValueError("flattened cell data length must match the map cell count")
    if mesh_cell_count == base_cell_count:
        return values
    if mesh_cell_count == 2 * base_cell_count:
        return np.repeat(values, 2)
    if mesh_cell_count == 6 * base_cell_count:
        return np.repeat(values, 6)
    raise RuntimeError("unsupported structured map mesh cell expansion")


def _structured_points_and_cells(
    *,
    shape: tuple[int, ...],
    cell_size: Sequence[float],
    origin: Sequence[float],
    element_type: MapMeshElement = "auto",
) -> tuple[np.ndarray, tuple[str, np.ndarray]]:
    normalized_element = _normalize_element_type(element_type)
    if len(shape) == 2:
        points, quad_cells = _structured_quad_mesh(shape=shape, cell_size=cell_size, origin=origin)
        if normalized_element == "auto" or normalized_element == "quad":
            return points, quad_cells
        if normalized_element == "triangle":
            return points, ("triangle", _split_quads_to_triangles(quad_cells[1]))
        raise ValueError("2D structured map meshes support element_type='quad' or 'triangle'")
    if len(shape) == 3:
        points, hexahedron_cells = _structured_hexahedron_mesh(
            shape=shape,
            cell_size=cell_size,
            origin=origin,
        )
        if normalized_element == "auto" or normalized_element == "hexahedron":
            return points, hexahedron_cells
        if normalized_element == "tetra":
            return points, ("tetra", _split_hexahedra_to_tetrahedra(hexahedron_cells[1]))
        raise ValueError("3D structured map meshes support element_type='hexahedron' or 'tetra'")
    raise ValueError("structured mesh export supports only 2D or 3D maps")


def _normalize_element_type(element_type: str) -> str:
    normalized = str(element_type).strip().lower()
    if normalized == "tetrahedron":
        return "tetra"
    if normalized not in {"auto", "quad", "triangle", "hexahedron", "tetra"}:
        raise ValueError(
            "element_type must be one of 'auto', 'quad', 'triangle', "
            "'hexahedron', 'tetra', or 'tetrahedron'"
        )
    return normalized


def _structured_quad_mesh(
    *,
    shape: tuple[int, int],
    cell_size: Sequence[float],
    origin: Sequence[float],
) -> tuple[np.ndarray, tuple[str, np.ndarray]]:
    nx, ny = shape
    dx, dy = (float(v) for v in cell_size)
    ox, oy = (float(v) for v in origin)

    x = ox + dx * np.arange(nx + 1, dtype=float)
    y = oy + dy * np.arange(ny + 1, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    point_ids = np.arange((nx + 1) * (ny + 1), dtype=np.int64).reshape((nx + 1, ny + 1))
    points = np.column_stack(
        [xx.reshape(-1, order="C"), yy.reshape(-1, order="C"), np.zeros(xx.size)]
    )

    cells = np.empty((nx * ny, 4), dtype=np.int64)
    for row, (i, j) in enumerate(np.ndindex(shape)):
        cells[row] = [
            point_ids[i, j],
            point_ids[i + 1, j],
            point_ids[i + 1, j + 1],
            point_ids[i, j + 1],
        ]

    return points, ("quad", cells)


def _split_quads_to_triangles(quad_cells: np.ndarray) -> np.ndarray:
    quads = np.asarray(quad_cells, dtype=np.int64)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError("quad_cells must have shape (n, 4)")

    triangles = np.empty((2 * quads.shape[0], 3), dtype=np.int64)
    triangles[0::2] = quads[:, [0, 1, 2]]
    triangles[1::2] = quads[:, [0, 2, 3]]
    return triangles


def _split_hexahedra_to_tetrahedra(hexahedron_cells: np.ndarray) -> np.ndarray:
    hexahedra = np.asarray(hexahedron_cells, dtype=np.int64)
    if hexahedra.ndim != 2 or hexahedra.shape[1] != 8:
        raise ValueError("hexahedron_cells must have shape (n, 8)")

    local_tetrahedra = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    tetrahedra = hexahedra[:, local_tetrahedra].reshape(-1, 4)
    return tetrahedra


def _structured_hexahedron_mesh(
    *,
    shape: tuple[int, int, int],
    cell_size: Sequence[float],
    origin: Sequence[float],
) -> tuple[np.ndarray, tuple[str, np.ndarray]]:
    nx, ny, nz = shape
    dx, dy, dz = (float(v) for v in cell_size)
    ox, oy, oz = (float(v) for v in origin)

    x = ox + dx * np.arange(nx + 1, dtype=float)
    y = oy + dy * np.arange(ny + 1, dtype=float)
    z = oz + dz * np.arange(nz + 1, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    point_ids = np.arange((nx + 1) * (ny + 1) * (nz + 1), dtype=np.int64).reshape(
        (nx + 1, ny + 1, nz + 1)
    )
    points = np.column_stack(
        [
            xx.reshape(-1, order="C"),
            yy.reshape(-1, order="C"),
            zz.reshape(-1, order="C"),
        ]
    )

    cells = np.empty((nx * ny * nz, 8), dtype=np.int64)
    for row, (i, j, k) in enumerate(np.ndindex(shape)):
        cells[row] = [
            point_ids[i, j, k],
            point_ids[i + 1, j, k],
            point_ids[i + 1, j + 1, k],
            point_ids[i, j + 1, k],
            point_ids[i, j, k + 1],
            point_ids[i + 1, j, k + 1],
            point_ids[i + 1, j + 1, k + 1],
            point_ids[i, j + 1, k + 1],
        ]

    return points, ("hexahedron", cells)


def _write_mesh(
    mesh: StructuredMapMesh,
    path: str | Path,
    *,
    file_format: str | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    meshio = _import_meshio()
    normalized_format = None if file_format is None else str(file_format).strip().lower()
    if normalized_format in {"netgen", "vol"} or destination.suffix.lower() == ".vol":
        meshio_mesh = _as_netgen_geometry_mesh(mesh)
        meshio.write(destination, meshio_mesh, file_format="netgen")
    elif normalized_format is None and destination.suffix.lower() == ".msh":
        meshio.write(destination, mesh.to_meshio(), file_format="gmsh")
    else:
        meshio.write(destination, mesh.to_meshio(), file_format=file_format)
    return destination


def _as_netgen_geometry_mesh(mesh: StructuredMapMesh) -> Any:
    """Return a Netgen-safe meshio mesh with geometry and one integer cell tag."""

    meshio = _import_meshio()
    one_based_cell_index = np.arange(1, mesh.cell_count + 1, dtype=np.int64)
    return meshio.Mesh(
        points=mesh.points,
        cells=list(mesh.cells),
        cell_data={"netgen:index": [one_based_cell_index]},
    )
