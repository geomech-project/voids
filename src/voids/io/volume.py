from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import tifffile
from scipy.io import netcdf_file
from skimage import measure

from voids.image._utils import normalize_shape

_VOLUME_SCHEMA_VERSION = "voids.volume.v1"
_SURFACE_SCHEMA_VERSION = "voids.surface_mesh.v1"

_FORMAT_ALIASES = {
    "hdf5": "h5",
    "h5": "h5",
    "netcdf": "nc",
    "nc": "nc",
    "numpy": "npy",
    "npy": "npy",
    "raw": "raw",
    "tif": "tiff",
    "tiff": "tiff",
    "stl": "stl",
    "obj": "obj",
}

_FORMAT_EXTENSIONS = {
    "h5": ".h5",
    "nc": ".nc",
    "npy": ".npy",
    "raw": ".raw",
    "tiff": ".tiff",
    "stl": ".stl",
    "obj": ".obj",
}


@dataclass(slots=True)
class SurfaceMesh:
    """Triangular surface mesh used for STL/OBJ interchange.

    Attributes
    ----------
    vertices :
        Floating-point vertex coordinates with shape ``(n_vertices, 3)``.
    faces :
        Integer triangular face connectivity with shape ``(n_faces, 3)``.
    metadata :
        JSON-serializable provenance metadata.
    """

    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize mesh arrays."""

        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (n_faces, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if np.any(faces < 0):
            raise ValueError("faces must be nonnegative")
        if faces.size and int(np.max(faces)) >= len(vertices):
            raise ValueError("faces reference a vertex outside vertices")
        self.vertices = vertices
        self.faces = faces
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class VolumeData:
    """Voxel image together with physical spacing and provenance metadata.

    Attributes
    ----------
    values :
        Two- or three-dimensional image array.
    voxel_size :
        Physical spacing along each image axis. A scalar means isotropic
        spacing; a sequence must have one entry per array dimension.
    units :
        Unit metadata for the spacing. By convention, ``{"length": "voxel"}``
        means dimensionless voxel units.
    metadata :
        JSON-serializable provenance metadata.
    """

    values: np.ndarray
    voxel_size: float | Sequence[float] = 1.0
    units: dict[str, str] = field(default_factory=lambda: {"length": "voxel"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the array and normalize physical spacing metadata."""

        values = np.asarray(self.values)
        normalize_shape(values.shape, allowed_ndim=(2, 3))
        self.values = values
        self.voxel_size = _normalize_voxel_size(self.voxel_size, ndim=values.ndim)
        self.units = dict(self.units)
        self.metadata = dict(self.metadata)

    @property
    def ndim(self) -> int:
        """Number of image dimensions."""

        return int(self.values.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """Image shape in NumPy axis order."""

        return tuple(int(v) for v in self.values.shape)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _normalize_format(path: str | Path, file_format: str | None) -> str:
    if file_format is None:
        suffix = Path(path).suffix.lower().lstrip(".")
        if not suffix:
            raise ValueError("file format could not be inferred from path")
        candidate = suffix
    else:
        candidate = str(file_format).strip().lower().lstrip(".")
    try:
        return _FORMAT_ALIASES[candidate]
    except KeyError as exc:
        raise ValueError(f"Unsupported volume or mesh format: {file_format or candidate}") from exc


def _normalize_voxel_size(
    voxel_size: float | Sequence[float],
    *,
    ndim: int = 3,
) -> tuple[float, ...]:
    if ndim not in (2, 3):
        raise ValueError("voxel_size can only be normalized for 2D or 3D volumes")
    if isinstance(voxel_size, Sequence) and not isinstance(voxel_size, (str, bytes)):
        values = tuple(float(v) for v in voxel_size)
    else:
        values = (float(voxel_size),) * ndim
    if len(values) != ndim:
        raise ValueError(f"voxel_size must be a scalar or have length {ndim}")
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError("voxel_size entries must be finite and positive")
    return values


def _binary_volume_mask(
    volume: np.ndarray,
    *,
    allowed_ndim: tuple[int, ...] = (3,),
) -> np.ndarray:
    arr = np.asarray(volume)
    normalize_shape(arr.shape, allowed_ndim=allowed_ndim)
    if arr.dtype == np.dtype(bool):
        return arr
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError("binary volume must have dtype bool or contain only numeric 0/1 values")
    if not bool(np.all((arr == 0) | (arr == 1))):
        raise ValueError("binary volume must have dtype bool or contain only numeric 0/1 values")
    return arr.astype(bool, copy=False)


def _ensure_integer_dtype_restore(values: np.ndarray, target_dtype: np.dtype[Any]) -> None:
    supported_source = (
        values.dtype == np.dtype(bool)
        or np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    )
    if not supported_source:
        raise ValueError(f"stored volume values cannot be restored as dtype {target_dtype}")
    if np.issubdtype(values.dtype, np.floating):
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"stored volume values cannot be restored as dtype {target_dtype}")
        if not bool(np.all(values == np.trunc(values))):
            raise ValueError(f"stored volume values cannot be restored as dtype {target_dtype}")
    info = np.iinfo(target_dtype)
    if values.size and (bool(np.min(values) < info.min) or bool(np.max(values) > info.max)):
        raise ValueError(f"stored volume values cannot be restored as dtype {target_dtype}")


def _restore_metadata_dtype(
    values: np.ndarray,
    stored_metadata: dict[str, Any],
    *,
    dtype_was_explicit: bool,
) -> np.ndarray:
    if dtype_was_explicit:
        return values
    dtype_name = stored_metadata.get("dtype")
    if dtype_name is None:
        return values
    target_dtype = np.dtype(dtype_name)
    if values.dtype == target_dtype:
        return values
    if target_dtype == np.dtype(bool):
        return _binary_volume_mask(values, allowed_ndim=(2, 3))
    if np.issubdtype(target_dtype, np.integer):
        _ensure_integer_dtype_restore(values, target_dtype)
        return values.astype(target_dtype, copy=False)
    if np.can_cast(values.dtype, target_dtype, casting="same_kind"):
        return values.astype(target_dtype, copy=False)
    raise ValueError(f"stored volume values cannot be restored as dtype {target_dtype}")


def _volume_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _raw_sidecar_path(path: Path) -> Path:
    return _volume_sidecar_path(path)


def _metadata_from_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.shape:
            raise ValueError("volume metadata attribute must be scalar JSON")
        value = value.item()
    if isinstance(value, bytes | np.bytes_):
        text = value.decode("utf-8")
    else:
        text = str(value)
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("volume metadata must be a JSON object")
    return cast(dict[str, Any], loaded)


def _load_volume_metadata(
    path: Path,
    fmt: str,
    *,
    hdf5_dataset: str = "volume",
    netcdf_variable: str = "volume",
) -> dict[str, Any]:
    if fmt in {"raw", "npy", "tiff"}:
        sidecar = _volume_sidecar_path(path)
        if sidecar.exists():
            return _metadata_from_json(sidecar.read_text(encoding="utf-8"))
        return {}
    if fmt == "h5":
        with h5py.File(path, "r") as f:
            if hdf5_dataset not in f:
                raise KeyError(f"HDF5 dataset {hdf5_dataset!r} not found")
            return _metadata_from_json(f.attrs.get("metadata"))
    if fmt == "nc":
        with netcdf_file(path, "r", mmap=False) as f:
            if netcdf_variable not in f.variables:
                raise KeyError(f"netCDF variable {netcdf_variable!r} not found")
            return _metadata_from_json(getattr(f, "metadata", None))
    return {}


def _volume_metadata(
    volume: np.ndarray,
    *,
    metadata: dict[str, Any] | None,
    voxel_size: float | Sequence[float],
    units: dict[str, str],
    stored_dtype: np.dtype[Any] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(volume)
    spacing = _normalize_voxel_size(voxel_size, ndim=arr.ndim)
    return {
        "schema_version": _VOLUME_SCHEMA_VERSION,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "stored_dtype": str(stored_dtype or arr.dtype),
        "ndim": int(arr.ndim),
        "voxel_size": spacing,
        "units": dict(units),
        "metadata": dict(metadata or {}),
    }


def _coerce_volume_input(
    volume: VolumeData | np.ndarray,
    *,
    metadata: dict[str, Any] | None,
    voxel_size: float | Sequence[float] | None,
    units: dict[str, str] | None,
) -> tuple[np.ndarray, tuple[float, ...], dict[str, str], dict[str, Any]]:
    if isinstance(volume, VolumeData):
        arr = np.asarray(volume.values)
        resolved_metadata = dict(volume.metadata)
        resolved_spacing = _normalize_voxel_size(
            volume.voxel_size if voxel_size is None else voxel_size,
            ndim=arr.ndim,
        )
        resolved_units = dict(volume.units if units is None else units)
    else:
        arr = np.asarray(volume)
        normalize_shape(arr.shape, allowed_ndim=(2, 3))
        resolved_metadata = {}
        resolved_spacing = _normalize_voxel_size(
            1.0 if voxel_size is None else voxel_size,
            ndim=arr.ndim,
        )
        resolved_units = dict(units or {"length": "voxel"})
    if metadata:
        resolved_metadata.update(metadata)
    return arr, resolved_spacing, resolved_units, resolved_metadata


def save_volume(
    volume: VolumeData | np.ndarray,
    path: str | Path,
    *,
    file_format: str | None = None,
    metadata: dict[str, Any] | None = None,
    raw_dtype: str | np.dtype[Any] | None = None,
    hdf5_dataset: str = "volume",
    netcdf_variable: str = "volume",
    voxel_size: float | Sequence[float] | None = None,
    units: dict[str, str] | None = None,
) -> Path:
    """Save a 2D/3D synthetic image volume or surface mesh.

    Parameters
    ----------
    volume :
        Two- or three-dimensional image, or :class:`VolumeData` with physical
        spacing metadata. Boolean arrays are interpreted as ``True=void`` for
        surface-mesh export.
    path :
        Destination path. Supported suffixes are ``.raw``, ``.npy``, ``.h5``,
        ``.nc``, ``.tif``/``.tiff``, ``.stl``, and ``.obj``.
    file_format :
        Optional explicit format when the suffix is ambiguous.
    metadata :
        JSON-serializable provenance metadata stored by metadata-capable formats.
    raw_dtype :
        Storage dtype for raw binary export. Defaults to ``uint8`` for boolean
        images and to the input dtype otherwise.
    hdf5_dataset, netcdf_variable :
        Dataset/variable names for HDF5 and netCDF.
    voxel_size :
        Physical voxel spacing. A scalar means isotropic spacing; a sequence
        must have one entry per image axis. The value is stored in metadata for
        voxel formats and used as marching-cubes spacing for STL/OBJ surfaces.
    units :
        Unit metadata for ``voxel_size``. Defaults to ``{"length": "voxel"}``
        for plain arrays, or to the units stored on :class:`VolumeData`.

    Returns
    -------
    pathlib.Path
        Path written.

    Notes
    -----
    Raw, NumPy, and TIFF files do not reliably carry the physical voxel spacing
    needed by downstream solvers, so a JSON sidecar with suffix ``.<ext>.json``
    is written next to those files.
    """

    arr, spacing, resolved_units, resolved_metadata = _coerce_volume_input(
        volume,
        metadata=metadata,
        voxel_size=voxel_size,
        units=units,
    )
    fmt = _normalize_format(path, file_format)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt in {"stl", "obj"}:
        mesh = surface_mesh_from_binary_volume(arr, voxel_size=spacing)
        mesh.metadata["units"] = dict(resolved_units)
        mesh.metadata.update(resolved_metadata)
        return save_surface_mesh(mesh, out, file_format=fmt)

    if fmt == "raw":
        dtype = np.dtype(
            np.uint8 if arr.dtype == bool and raw_dtype is None else raw_dtype or arr.dtype
        )
        arr.astype(dtype, copy=False).tofile(out)
        sidecar = _raw_sidecar_path(out)
        sidecar.write_text(
            _json_dumps(
                _volume_metadata(
                    arr,
                    metadata=resolved_metadata,
                    voxel_size=spacing,
                    units=resolved_units,
                    stored_dtype=dtype,
                )
            ),
            encoding="utf-8",
        )
        return out

    if fmt == "npy":
        np.save(out, arr)
        _volume_sidecar_path(out).write_text(
            _json_dumps(
                _volume_metadata(
                    arr,
                    metadata=resolved_metadata,
                    voxel_size=spacing,
                    units=resolved_units,
                )
            ),
            encoding="utf-8",
        )
        return out

    if fmt == "h5":
        with h5py.File(out, "w") as f:
            f.attrs["schema_version"] = _VOLUME_SCHEMA_VERSION
            f.attrs["metadata"] = _json_dumps(
                _volume_metadata(
                    arr,
                    metadata=resolved_metadata,
                    voxel_size=spacing,
                    units=resolved_units,
                )
            )
            f.create_dataset(hdf5_dataset, data=arr)
        return out

    if fmt == "nc":
        needs_signed_storage = arr.dtype == np.dtype(bool) or arr.dtype == np.dtype("uint8")
        stored = arr.astype(np.int16, copy=False) if needs_signed_storage else arr
        with netcdf_file(out, "w") as f:
            for axis, size in enumerate(stored.shape):
                f.createDimension(f"dim_{axis}", int(size))
            var = f.createVariable(
                netcdf_variable, stored.dtype, tuple(f"dim_{i}" for i in range(stored.ndim))
            )
            var[:] = stored
            f.schema_version = _VOLUME_SCHEMA_VERSION
            f.metadata = _json_dumps(
                _volume_metadata(
                    arr,
                    metadata=resolved_metadata,
                    voxel_size=spacing,
                    units=resolved_units,
                    stored_dtype=stored.dtype,
                )
            )
        return out

    if fmt == "tiff":
        stored = arr.astype(np.uint8, copy=False) if arr.dtype == bool else arr
        tifffile.imwrite(out, stored, photometric="minisblack")
        _volume_sidecar_path(out).write_text(
            _json_dumps(
                _volume_metadata(
                    arr,
                    metadata=resolved_metadata,
                    voxel_size=spacing,
                    units=resolved_units,
                    stored_dtype=stored.dtype,
                )
            ),
            encoding="utf-8",
        )
        return out

    raise ValueError(f"Unsupported volume format: {fmt}")  # pragma: no cover - guarded above


def load_volume(
    path: str | Path,
    *,
    file_format: str | None = None,
    shape: Sequence[int] | None = None,
    dtype: str | np.dtype[Any] | None = None,
    hdf5_dataset: str = "volume",
    netcdf_variable: str = "volume",
) -> np.ndarray:
    """Load a 2D/3D image volume from raw, NumPy, HDF5, netCDF, or TIFF."""

    source = Path(path)
    fmt = _normalize_format(source, file_format)

    if fmt == "raw":
        sidecar_metadata = _load_volume_metadata(source, fmt)
        resolved_shape = tuple(int(v) for v in (shape or sidecar_metadata.get("shape", ())))
        if not resolved_shape:
            raise ValueError("shape is required to load raw volume without sidecar metadata")
        normalize_shape(resolved_shape, allowed_ndim=(2, 3))
        resolved_dtype = np.dtype(dtype or sidecar_metadata.get("stored_dtype", np.uint8))
        data = np.fromfile(source, dtype=resolved_dtype)
        expected = int(np.prod(np.asarray(resolved_shape, dtype=np.int64)))
        if data.size != expected:
            raise ValueError(f"raw file has {data.size} entries but shape requires {expected}")
        return data.reshape(resolved_shape)

    if fmt == "npy":
        arr = np.load(source)
        normalize_shape(arr.shape, allowed_ndim=(2, 3))
        return np.asarray(arr)

    if fmt == "h5":
        with h5py.File(source, "r") as f:
            arr = f[hdf5_dataset][()]
        normalize_shape(arr.shape, allowed_ndim=(2, 3))
        return np.asarray(arr)

    if fmt == "nc":
        with netcdf_file(source, "r", mmap=False) as f:
            arr = np.asarray(f.variables[netcdf_variable][()]).copy()
        normalize_shape(arr.shape, allowed_ndim=(2, 3))
        return arr

    if fmt == "tiff":
        arr = np.asarray(tifffile.imread(source))
        normalize_shape(arr.shape, allowed_ndim=(2, 3))
        return arr

    raise ValueError(f"Use load_surface_mesh for {fmt!r} files")


def load_volume_data(
    path: str | Path,
    *,
    file_format: str | None = None,
    shape: Sequence[int] | None = None,
    dtype: str | np.dtype[Any] | None = None,
    hdf5_dataset: str = "volume",
    netcdf_variable: str = "volume",
    voxel_size: float | Sequence[float] | None = None,
    units: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> VolumeData:
    """Load a volume together with voxel spacing, units, and metadata.

    ``load_volume`` intentionally returns only the image array. Use this helper
    whenever physical voxel resolution matters, especially for external TIFF,
    NumPy, or raw files that may not have a reliable sidecar.
    """

    source = Path(path)
    fmt = _normalize_format(source, file_format)
    values = load_volume(
        source,
        file_format=fmt,
        shape=shape,
        dtype=dtype,
        hdf5_dataset=hdf5_dataset,
        netcdf_variable=netcdf_variable,
    )
    stored_metadata = _load_volume_metadata(
        source,
        fmt,
        hdf5_dataset=hdf5_dataset,
        netcdf_variable=netcdf_variable,
    )
    values = _restore_metadata_dtype(
        values,
        stored_metadata,
        dtype_was_explicit=dtype is not None,
    )
    resolved_metadata = dict(stored_metadata.get("metadata", {}))
    if metadata:
        resolved_metadata.update(metadata)
    resolved_units = dict(
        units if units is not None else stored_metadata.get("units", {"length": "voxel"})
    )
    resolved_voxel_size = (
        voxel_size if voxel_size is not None else stored_metadata.get("voxel_size", 1.0)
    )
    return VolumeData(
        values=values,
        voxel_size=resolved_voxel_size,
        units=resolved_units,
        metadata=resolved_metadata,
    )


def surface_mesh_from_binary_volume(
    volume: np.ndarray,
    *,
    voxel_size: float | Sequence[float] = 1.0,
    level: float = 0.5,
) -> SurfaceMesh:
    """Extract a triangular surface mesh from a 3D binary void image.

    The surface is the interface between ``True`` void voxels and ``False`` solid
    voxels, computed with marching cubes. Coordinates are scaled by
    ``voxel_size``.
    """

    mask = _binary_volume_mask(volume)
    if not (np.any(mask) and np.any(~mask)):
        raise ValueError("volume must contain both void and solid voxels for surface extraction")
    spacing = _normalize_voxel_size(voxel_size)
    vertices, faces, _normals, _values = measure.marching_cubes(
        mask.astype(float),
        level=float(level),
        spacing=spacing,
    )
    return SurfaceMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        metadata={
            "schema_version": _SURFACE_SCHEMA_VERSION,
            "source_kind": "binary_volume_marching_cubes",
            "voxel_size": spacing,
        },
    )


def save_surface_mesh(
    mesh: SurfaceMesh | np.ndarray,
    path: str | Path,
    *,
    file_format: str | None = None,
    voxel_size: float | Sequence[float] = 1.0,
) -> Path:
    """Save a triangular surface mesh as ASCII STL or OBJ.

    ``mesh`` can be a :class:`SurfaceMesh` or a 3D binary volume, in which case
    marching cubes is applied first.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_format(out, file_format)
    if not isinstance(mesh, SurfaceMesh):
        mesh = surface_mesh_from_binary_volume(mesh, voxel_size=voxel_size)

    if fmt == "obj":
        _write_obj(mesh, out)
        return out
    if fmt == "stl":
        _write_ascii_stl(mesh, out)
        return out
    raise ValueError("surface meshes can only be saved as STL or OBJ")


def load_surface_mesh(path: str | Path, *, file_format: str | None = None) -> SurfaceMesh:
    """Load an STL or OBJ triangular surface mesh."""

    source = Path(path)
    fmt = _normalize_format(source, file_format)
    if fmt == "obj":
        return _read_obj(source)
    if fmt == "stl":
        return _read_stl(source)
    raise ValueError("surface meshes can only be loaded from STL or OBJ")


def save_volume_bundle(
    volume: VolumeData | np.ndarray,
    directory: str | Path,
    *,
    stem: str = "synthetic_case",
    formats: Sequence[str] = ("raw", "npy", "h5", "stl", "obj"),
    metadata: dict[str, Any] | None = None,
    voxel_size: float | Sequence[float] | None = None,
    units: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Export one synthetic case to several interchange formats.

    Parameters
    ----------
    volume :
        2D or 3D image. STL/OBJ formats require a 3D binary volume.
    directory :
        Destination directory.
    stem :
        Base filename without suffix.
    formats :
        Iterable of format labels such as ``("raw", "npy", "h5", "stl", "obj")``.
    metadata, voxel_size, units :
        Forwarded to :func:`save_volume`.
    """

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for requested in formats:
        fmt = _normalize_format(f"{stem}.{requested}", requested)
        suffix = _FORMAT_EXTENSIONS[fmt]
        path = out_dir / f"{stem}{suffix}"
        written[fmt] = save_volume(
            volume,
            path,
            file_format=fmt,
            metadata=metadata,
            voxel_size=voxel_size,
            units=units,
        )
    return written


def _write_obj(mesh: SurfaceMesh, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# voids triangular surface mesh\n")
        if mesh.metadata:
            f.write(f"# metadata: {_json_dumps(mesh.metadata)}\n")
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in mesh.faces:
            i, j, k = (int(v) + 1 for v in face)
            f.write(f"f {i} {j} {k}\n")


def _face_normal(triangle: np.ndarray) -> np.ndarray:
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    normal = np.cross(edge1, edge2)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.zeros(3, dtype=float)
    return np.asarray(normal / norm, dtype=float)


def _write_ascii_stl(mesh: SurfaceMesh, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("solid voids_surface\n")
        for face in mesh.faces:
            triangle = mesh.vertices[np.asarray(face, dtype=np.int64)]
            normal = _face_normal(triangle)
            f.write(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n")
            f.write("    outer loop\n")
            for vertex in triangle:
                f.write(f"      vertex {vertex[0]:.9e} {vertex[1]:.9e} {vertex[2]:.9e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid voids_surface\n")


def _read_obj(path: Path) -> SurfaceMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f" and len(parts) >= 4:
            face = (
                int(parts[1].split("/")[0]) - 1,
                int(parts[2].split("/")[0]) - 1,
                int(parts[3].split("/")[0]) - 1,
            )
            faces.append(face)
    return SurfaceMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        metadata={"schema_version": _SURFACE_SCHEMA_VERSION, "source_format": "obj"},
    )


def _read_stl(path: Path) -> SurfaceMesh:
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:512]:
        return _read_ascii_stl(data)
    return _read_binary_stl(data)


def _read_ascii_stl(data: bytes) -> SurfaceMesh:
    vertices: list[tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    faces = [(i, i + 1, i + 2) for i in range(0, len(vertices), 3)]
    return SurfaceMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        metadata={"schema_version": _SURFACE_SCHEMA_VERSION, "source_format": "ascii_stl"},
    )


def _read_binary_stl(data: bytes) -> SurfaceMesh:
    if len(data) < 84:
        raise ValueError("binary STL is too small to contain a header and triangle count")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + int(triangle_count) * 50
    if len(data) < expected:
        raise ValueError("binary STL is truncated")
    vertices: list[tuple[float, float, float]] = []
    for triangle_index in range(int(triangle_count)):
        offset = 84 + triangle_index * 50 + 12
        coords = struct.unpack_from("<9f", data, offset)
        vertices.extend(
            [
                (float(coords[0]), float(coords[1]), float(coords[2])),
                (float(coords[3]), float(coords[4]), float(coords[5])),
                (float(coords[6]), float(coords[7]), float(coords[8])),
            ]
        )
    faces = [(i, i + 1, i + 2) for i in range(0, len(vertices), 3)]
    return SurfaceMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        metadata={"schema_version": _SURFACE_SCHEMA_VERSION, "source_format": "binary_stl"},
    )


__all__ = [
    "SurfaceMesh",
    "VolumeData",
    "load_surface_mesh",
    "load_volume",
    "load_volume_data",
    "save_surface_mesh",
    "save_volume",
    "save_volume_bundle",
    "surface_mesh_from_binary_volume",
]
