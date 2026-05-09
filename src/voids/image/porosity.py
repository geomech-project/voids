from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from voids.image._utils import normalize_shape

_SCHEMA_VERSION = "voids.porosity_map.v1"


def _json_default(value: Any) -> Any:
    """Convert common NumPy metadata values to JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_json_attr(obj: h5py.Group | h5py.File, name: str, value: Any) -> None:
    """Write JSON metadata into an HDF5 attribute."""

    obj.attrs[name] = json.dumps(value, default=_json_default)


def _read_json_attr(obj: h5py.Group | h5py.File, name: str, default: Any = None) -> Any:
    """Read JSON metadata from an HDF5 attribute."""

    if name not in obj.attrs:
        return default
    raw = obj.attrs[name]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _normalize_positive_tuple(
    values: float | Sequence[float],
    *,
    ndim: int,
    name: str,
) -> tuple[float, ...]:
    """Normalize scalar-or-vector physical metadata."""

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = tuple(float(v) for v in values)
    else:
        result = (float(values),) * ndim
    if len(result) != ndim:
        raise ValueError(f"{name} must be a scalar or have length {ndim}")
    if any(not np.isfinite(v) or v <= 0.0 for v in result):
        raise ValueError(f"{name} entries must be finite and positive")
    return result


def _normalize_origin(origin: Sequence[float] | None, *, ndim: int) -> tuple[float, ...]:
    """Normalize a physical origin vector."""

    if origin is None:
        return (0.0,) * ndim
    result = tuple(float(v) for v in origin)
    if len(result) != ndim:
        raise ValueError(f"origin must have length {ndim}")
    if any(not np.isfinite(v) for v in result):
        raise ValueError("origin entries must be finite")
    return result


def _validate_porosity_values(values: np.ndarray) -> np.ndarray:
    """Return a validated 2D or 3D porosity field."""

    arr = np.asarray(values, dtype=float)
    normalize_shape(arr.shape, allowed_ndim=(2, 3))
    if not np.all(np.isfinite(arr)):
        raise ValueError("porosity values must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("porosity values must lie in [0, 1]")
    return arr


def _normalize_block_shape(block_shape: Sequence[int] | None, *, ndim: int) -> tuple[int, ...]:
    """Normalize an optional coarse-cell block shape."""

    if block_shape is None:
        return (1,) * ndim
    return normalize_shape(block_shape, allowed_ndim=(ndim,))


def _block_mean(
    values: np.ndarray,
    *,
    block_shape: tuple[int, ...],
    strict: bool,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Average a fine-grid field over rectangular coarse-cell blocks."""

    arr = np.asarray(values, dtype=float)
    ndim = arr.ndim
    if len(block_shape) != ndim:
        raise ValueError(f"block_shape must have length {ndim}")
    if any(b > s for b, s in zip(block_shape, arr.shape, strict=True)):
        raise ValueError("block_shape entries must not exceed the image shape")

    coarse_shape = tuple(s // b for s, b in zip(arr.shape, block_shape, strict=True))
    trimmed_shape = tuple(c * b for c, b in zip(coarse_shape, block_shape, strict=True))
    if strict and trimmed_shape != arr.shape:
        raise ValueError("image shape must be exactly divisible by block_shape when strict=True")

    slices = tuple(slice(0, n) for n in trimmed_shape)
    trimmed = arr[slices]
    reshape_shape = tuple(
        value for pair in zip(coarse_shape, block_shape, strict=True) for value in pair
    )
    block_axes = tuple(range(1, 2 * ndim, 2))
    return trimmed.reshape(reshape_shape).mean(axis=block_axes), trimmed_shape


def _as_binary_mask(image: np.ndarray) -> np.ndarray:
    """Validate a segmented image and return a boolean mask."""

    arr = np.asarray(image)
    normalize_shape(arr.shape, allowed_ndim=(2, 3))
    if arr.dtype == bool:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("binary image values must be finite")
    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError("binary image must contain only 0/1 or boolean values")
    return arr.astype(bool)


@dataclass(slots=True)
class PorosityMap:
    """Cell-wise porosity field for continuum or external-solver workflows.

    Parameters
    ----------
    values :
        Two- or three-dimensional porosity field with entries in ``[0, 1]``.
        Values are interpreted as cell averages.
    cell_size :
        Scalar or per-axis cell side length in physical units.
    origin :
        Physical coordinate of the lower corner of the first cell.
    units :
        Unit metadata for reporting and serialization.
    metadata :
        Additional JSON-serializable provenance and calibration metadata.
    """

    values: np.ndarray
    cell_size: float | Sequence[float] = 1.0
    origin: Sequence[float] | None = None
    units: dict[str, str] = field(default_factory=lambda: {"length": "m"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate porosity-map fields."""

        values = _validate_porosity_values(self.values)
        self.values = values
        self.cell_size = _normalize_positive_tuple(
            self.cell_size,
            ndim=values.ndim,
            name="cell_size",
        )
        self.origin = _normalize_origin(self.origin, ndim=values.ndim)
        self.units = {str(k): str(v) for k, v in self.units.items()}
        self.metadata = dict(self.metadata)

    @property
    def ndim(self) -> int:
        """Return the spatial dimensionality of the porosity map."""

        return int(self.values.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the cell-count shape of the porosity map."""

        return tuple(int(v) for v in self.values.shape)

    @property
    def mean_porosity(self) -> float:
        """Return the arithmetic mean porosity over all cells."""

        return float(np.mean(self.values))

    @property
    def cell_volume(self) -> float:
        """Return the physical volume represented by one porosity-map cell."""

        return float(np.prod(np.asarray(self.cell_size, dtype=float)))

    @property
    def bulk_volume(self) -> float:
        """Return the physical bulk volume represented by the map."""

        return float(np.prod(np.asarray(self.shape, dtype=float)) * self.cell_volume)

    @property
    def void_volume(self) -> float:
        """Return the pore volume implied by the cell-average porosity field."""

        return float(np.sum(self.values) * self.cell_volume)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize map metadata without the porosity array."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "shape": self.shape,
            "cell_size": self.cell_size,
            "origin": self.origin,
            "units": self.units,
            "metadata": self.metadata,
        }

    @classmethod
    def from_metadata(cls, values: np.ndarray, metadata: dict[str, Any]) -> "PorosityMap":
        """Reconstruct a porosity map from values and serialized metadata."""

        return cls(
            values=values,
            cell_size=metadata.get("cell_size", 1.0),
            origin=metadata.get("origin"),
            units={str(k): str(v) for k, v in (metadata.get("units") or {}).items()},
            metadata=dict(metadata.get("metadata") or {}),
        )


def calibrated_porosity_from_grayscale(
    grayscale: np.ndarray,
    *,
    solid_gray: float,
    pore_gray: float,
    background_porosity: float = 0.0,
    clip: bool = True,
) -> np.ndarray:
    """Map grayscale intensity to porosity with a two-point linear calibration.

    Parameters
    ----------
    grayscale :
        Two- or three-dimensional grayscale image.
    solid_gray :
        Grayscale value assigned to the background or unresolved-microporosity
        porosity.
    pore_gray :
        Grayscale value assigned to porosity equal to one.
    background_porosity :
        Porosity assigned at ``solid_gray``. This represents unresolved
        microporosity when nonzero.
    clip :
        Whether to clip extrapolated values to
        ``[background_porosity, 1]``.

    Returns
    -------
    numpy.ndarray
        Voxel-wise porosity field.
    """

    arr = np.asarray(grayscale, dtype=float)
    normalize_shape(arr.shape, allowed_ndim=(2, 3))
    if not np.all(np.isfinite(arr)):
        raise ValueError("grayscale values must be finite")

    solid = float(solid_gray)
    pore = float(pore_gray)
    if not np.isfinite(solid) or not np.isfinite(pore):
        raise ValueError("solid_gray and pore_gray must be finite")
    if solid == pore:
        raise ValueError("solid_gray and pore_gray must differ")

    background = float(background_porosity)
    if not np.isfinite(background) or background < 0.0 or background > 1.0:
        raise ValueError("background_porosity must lie in [0, 1]")

    phi = background + (1.0 - background) * (arr - solid) / (pore - solid)
    if clip:
        phi = np.clip(phi, background, 1.0)
    return np.asarray(phi, dtype=float)


def porosity_map_from_binary(
    image: np.ndarray,
    *,
    block_shape: Sequence[int] | None = None,
    voxel_size: float | Sequence[float] = 1.0,
    image_is_void: bool = True,
    strict: bool = True,
    origin: Sequence[float] | None = None,
    units: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PorosityMap:
    """Compute a local porosity map from a segmented binary image.

    Parameters
    ----------
    image :
        Binary 2D or 3D image. By default, ``True`` or ``1`` denotes void.
    block_shape :
        Number of fine image voxels in each porosity-map cell. When omitted,
        each image voxel becomes one porosity-map cell.
    voxel_size :
        Scalar or per-axis fine-image voxel spacing in physical units.
    image_is_void :
        If ``False``, nonzero image values are interpreted as solid and are
        inverted before averaging.
    strict :
        If ``True``, image shape must be exactly divisible by ``block_shape``.
        If ``False``, trailing partial blocks are trimmed.
    origin, units, metadata :
        Optional physical and provenance metadata.

    Returns
    -------
    PorosityMap
        Cell-average local porosity map.
    """

    void_mask = _as_binary_mask(image)
    if not image_is_void:
        void_mask = ~void_mask
    block = _normalize_block_shape(block_shape, ndim=void_mask.ndim)
    values, trimmed_shape = _block_mean(void_mask.astype(float), block_shape=block, strict=strict)
    voxel = _normalize_positive_tuple(voxel_size, ndim=void_mask.ndim, name="voxel_size")
    cell_size = tuple(v * b for v, b in zip(voxel, block, strict=True))

    map_metadata: dict[str, Any] = {
        "source_kind": "binary_void_fraction",
        "fine_shape": tuple(int(v) for v in void_mask.shape),
        "trimmed_shape": trimmed_shape,
        "block_shape": block,
        "image_is_void": bool(image_is_void),
        "strict": bool(strict),
    }
    if metadata:
        map_metadata.update(metadata)

    return PorosityMap(
        values=values,
        cell_size=cell_size,
        origin=origin,
        units=units or {"length": "m"},
        metadata=map_metadata,
    )


def porosity_map_from_grayscale(
    grayscale: np.ndarray,
    *,
    solid_gray: float,
    pore_gray: float,
    background_porosity: float = 0.0,
    block_shape: Sequence[int] | None = None,
    voxel_size: float | Sequence[float] = 1.0,
    clip: bool = True,
    strict: bool = True,
    origin: Sequence[float] | None = None,
    units: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PorosityMap:
    """Compute a local porosity map from a calibrated grayscale image.

    Parameters
    ----------
    grayscale :
        Two- or three-dimensional grayscale image.
    solid_gray, pore_gray, background_porosity, clip :
        Calibration parameters passed to
        :func:`calibrated_porosity_from_grayscale`.
    block_shape :
        Number of fine image voxels in each porosity-map cell. When omitted,
        each image voxel becomes one porosity-map cell.
    voxel_size :
        Scalar or per-axis fine-image voxel spacing in physical units.
    strict :
        If ``True``, image shape must be exactly divisible by ``block_shape``.
        If ``False``, trailing partial blocks are trimmed.
    origin, units, metadata :
        Optional physical and provenance metadata.

    Returns
    -------
    PorosityMap
        Cell-average local porosity map.
    """

    voxel_porosity = calibrated_porosity_from_grayscale(
        grayscale,
        solid_gray=solid_gray,
        pore_gray=pore_gray,
        background_porosity=background_porosity,
        clip=clip,
    )
    block = _normalize_block_shape(block_shape, ndim=voxel_porosity.ndim)
    values, trimmed_shape = _block_mean(voxel_porosity, block_shape=block, strict=strict)
    voxel = _normalize_positive_tuple(voxel_size, ndim=voxel_porosity.ndim, name="voxel_size")
    cell_size = tuple(v * b for v, b in zip(voxel, block, strict=True))

    map_metadata: dict[str, Any] = {
        "source_kind": "grayscale_linear_calibration",
        "fine_shape": tuple(int(v) for v in voxel_porosity.shape),
        "trimmed_shape": trimmed_shape,
        "block_shape": block,
        "solid_gray": float(solid_gray),
        "pore_gray": float(pore_gray),
        "background_porosity": float(background_porosity),
        "clip": bool(clip),
        "strict": bool(strict),
    }
    if metadata:
        map_metadata.update(metadata)

    return PorosityMap(
        values=values,
        cell_size=cell_size,
        origin=origin,
        units=units or {"length": "m"},
        metadata=map_metadata,
    )


def save_porosity_map_hdf5(porosity_map: PorosityMap, path: str | Path) -> None:
    """Write a porosity map to a compact HDF5 interchange file.

    Parameters
    ----------
    porosity_map :
        Porosity map to serialize.
    path :
        Destination HDF5 file. Parent directories must already exist.

    Notes
    -----
    The stored array is named ``/porosity`` and contains cell-average porosity
    values. Root attributes store schema and calibration metadata as JSON.
    """

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = _SCHEMA_VERSION
        f.create_dataset("porosity", data=porosity_map.values)
        _write_json_attr(f, "metadata", porosity_map.to_metadata())


def load_porosity_map_hdf5(path: str | Path) -> PorosityMap:
    """Load a porosity map written by :func:`save_porosity_map_hdf5`."""

    path = Path(path)
    with h5py.File(path, "r") as f:
        schema_version = f.attrs.get("schema_version")
        if isinstance(schema_version, bytes):
            schema_version = schema_version.decode("utf-8")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported porosity-map schema version {schema_version!r}")
        values = f["porosity"][()]
        metadata = _read_json_attr(f, "metadata", {})
    return PorosityMap.from_metadata(values, metadata)


__all__ = [
    "PorosityMap",
    "calibrated_porosity_from_grayscale",
    "porosity_map_from_binary",
    "porosity_map_from_grayscale",
    "save_porosity_map_hdf5",
    "load_porosity_map_hdf5",
]
