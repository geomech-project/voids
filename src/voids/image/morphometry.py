from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from voids.image._utils import normalize_shape


@dataclass(slots=True)
class LocalThicknessSummary:
    """Summary statistics for a local-thickness diameter map.

    Attributes
    ----------
    label :
        Human-readable phase label, for example ``"bone"`` or ``"marrow"``.
    voxel_count :
        Number of phase voxels used for the statistics.
    mean, std, p10, p50, p90, max :
        Summary statistics of the local-thickness diameter over phase voxels.
    units :
        Physical units of the returned diameter values. Use ``"voxel"`` when
        ``voxel_size=1`` is a dimensionless grid spacing.
    method :
        PoreSpy local-thickness method used to generate the map.
    voxel_size :
        Isotropic voxel edge length in ``units``.

    Notes
    -----
    BoneJ-style trabecular thickness and separation are diameter quantities:
    each phase voxel is assigned the diameter of the largest sphere that fits
    inside the phase and contains that voxel. PoreSpy reports the corresponding
    local radius field, so `voids` multiplies by two and by ``voxel_size``.
    """

    label: str
    voxel_count: int
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    max: float
    units: str
    method: str
    voxel_size: float

    def as_dict(self) -> dict[str, int | float | str]:
        """Return the summary as a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class LocalThicknessResult:
    """Local-thickness diameter map and summary for one binary phase."""

    thickness_map: np.ndarray
    summary: LocalThicknessSummary


def _binary_phase_mask(phase_mask: np.ndarray) -> np.ndarray:
    """Normalize a binary 2D/3D phase mask to boolean values."""

    arr = np.asarray(phase_mask)
    normalize_shape(arr.shape, allowed_ndim=(2, 3))
    if arr.dtype == np.dtype(bool):
        return arr
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError("phase_mask must be boolean or contain only numeric 0/1 values")
    if not bool(np.all((arr == 0) | (arr == 1))):
        raise ValueError("phase_mask must be boolean or contain only numeric 0/1 values")
    return arr.astype(bool, copy=False)


def _isotropic_voxel_size(
    voxel_size: float | Sequence[float],
    *,
    ndim: int,
) -> float:
    """Normalize an isotropic voxel size for sphere-based morphometry."""

    if isinstance(voxel_size, Sequence) and not isinstance(voxel_size, (str, bytes)):
        values = tuple(float(v) for v in voxel_size)
    else:
        values = (float(voxel_size),) * ndim
    if len(values) != ndim:
        raise ValueError(f"voxel_size must be a scalar or have length {ndim}")
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError("voxel_size entries must be finite and positive")
    if not np.allclose(values, values[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("local thickness currently requires isotropic voxel_size")
    return float(values[0])


def _validate_distance_map(distance_map: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    """Validate a precomputed Euclidean distance map in voxel units."""

    distance = np.asarray(distance_map, dtype=float)
    if distance.shape != shape:
        raise ValueError("distance_map must have the same shape as phase_mask")
    if not bool(np.all(np.isfinite(distance))):
        raise ValueError("distance_map entries must be finite")
    if bool(np.any(distance < 0.0)):
        raise ValueError("distance_map entries must be nonnegative")
    return distance


def _porespy_local_thickness(*args: Any, **kwargs: Any) -> np.ndarray:
    """Run PoreSpy local thickness with a lazy dependency import."""

    import porespy as ps

    return np.asarray(ps.filters.local_thickness(*args, **kwargs), dtype=float)


def local_thickness_map(
    phase_mask: np.ndarray,
    *,
    voxel_size: float | Sequence[float] = 1.0,
    method: str = "dt",
    sizes: int | Sequence[float] | None = 64,
    smooth: bool = True,
    approx: bool = False,
    distance_map: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a BoneJ-style local-thickness diameter map.

    Parameters
    ----------
    phase_mask :
        Boolean or binary 2D/3D image where ``True`` marks the phase of
        interest.
    voxel_size :
        Isotropic voxel edge length. A scalar is preferred; a sequence is
        accepted only when all entries are equal.
    method :
        PoreSpy local-thickness backend. Common choices are ``"dt"`` for a
        practical distance-transform/opening approach and ``"imj"`` for a
        closer ImageJ-style sphere-insertion workflow.
    sizes :
        Radius sampling control forwarded to PoreSpy for ``"dt"`` and
        ``"conv"`` methods. ``None`` uses all unique distance-map values and can
        be expensive for large volumes.
    smooth, approx :
        Controls forwarded to PoreSpy. ``approx`` is only used by ``"imj"``.
    distance_map :
        Optional Euclidean distance map in voxel units for ``phase_mask``.

    Returns
    -------
    numpy.ndarray
        Local-thickness diameter map in units of ``voxel_size``. Voxels outside
        ``phase_mask`` are zero.

    Notes
    -----
    The public BoneJ description defines local thickness as the diameter of the
    largest sphere contained in the object and containing the point. PoreSpy's
    local-thickness filters return the corresponding local radius field; this
    function converts that radius to a diameter in physical units.
    """

    mask = _binary_phase_mask(phase_mask)
    spacing = _isotropic_voxel_size(voxel_size, ndim=mask.ndim)
    normalized_method = str(method).strip().lower()

    if not bool(np.any(mask)):
        return np.zeros(mask.shape, dtype=float)

    dt = None if distance_map is None else _validate_distance_map(distance_map, shape=mask.shape)
    radius_map = _porespy_local_thickness(
        mask,
        dt=dt,
        method=normalized_method,
        smooth=bool(smooth),
        approx=bool(approx),
        sizes=sizes,
    )
    thickness = 2.0 * spacing * np.asarray(radius_map, dtype=float)
    return np.where(mask, thickness, 0.0)


def summarize_local_thickness_map(
    thickness_map: np.ndarray,
    phase_mask: np.ndarray,
    *,
    label: str = "phase",
    units: str = "voxel",
    method: str = "unknown",
    voxel_size: float = 1.0,
) -> LocalThicknessSummary:
    """Summarize local-thickness values over phase voxels."""

    mask = _binary_phase_mask(phase_mask)
    thickness = np.asarray(thickness_map, dtype=float)
    if thickness.shape != mask.shape:
        raise ValueError("thickness_map must have the same shape as phase_mask")
    if not bool(np.all(np.isfinite(thickness))):
        raise ValueError("thickness_map entries must be finite")
    if bool(np.any(thickness < 0.0)):
        raise ValueError("thickness_map entries must be nonnegative")

    values = np.asarray(thickness[mask], dtype=float)
    if values.size == 0:
        nan = float("nan")
        return LocalThicknessSummary(
            label=str(label),
            voxel_count=0,
            mean=nan,
            std=nan,
            p10=nan,
            p50=nan,
            p90=nan,
            max=nan,
            units=str(units),
            method=str(method),
            voxel_size=float(voxel_size),
        )

    return LocalThicknessSummary(
        label=str(label),
        voxel_count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        p10=float(np.percentile(values, 10.0)),
        p50=float(np.percentile(values, 50.0)),
        p90=float(np.percentile(values, 90.0)),
        max=float(np.max(values)),
        units=str(units),
        method=str(method),
        voxel_size=float(voxel_size),
    )


def local_thickness_analysis(
    phase_mask: np.ndarray,
    *,
    voxel_size: float | Sequence[float] = 1.0,
    units: str = "voxel",
    label: str = "phase",
    method: str = "dt",
    sizes: int | Sequence[float] | None = 64,
    smooth: bool = True,
    approx: bool = False,
    distance_map: np.ndarray | None = None,
) -> LocalThicknessResult:
    """Compute a local-thickness diameter map and summary for one phase."""

    mask = _binary_phase_mask(phase_mask)
    spacing = _isotropic_voxel_size(voxel_size, ndim=mask.ndim)
    thickness = local_thickness_map(
        mask,
        voxel_size=spacing,
        method=method,
        sizes=sizes,
        smooth=smooth,
        approx=approx,
        distance_map=distance_map,
    )
    summary = summarize_local_thickness_map(
        thickness,
        mask,
        label=label,
        units=units,
        method=method,
        voxel_size=spacing,
    )
    return LocalThicknessResult(thickness_map=thickness, summary=summary)


__all__ = [
    "LocalThicknessResult",
    "LocalThicknessSummary",
    "local_thickness_analysis",
    "local_thickness_map",
    "summarize_local_thickness_map",
]
