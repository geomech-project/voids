from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import porespy as ps
from numba import njit  # type: ignore[import-untyped]
from scipy import ndimage as ndi

from voids.image.maximal_ball import compute_void_distance_map


@dataclass(slots=True)
class PregoSettings:
    """Controls for seed-based Pore Region Growing segmentation.

    The defaults mirror the PREGO paper's seed generation settings: SNOW peak
    filtering with ``r_max=4`` and Gaussian smoothing ``sigma=0.4``. PREGO is
    currently implemented for a single active pore phase, with nonzero input
    voxels treated as void.
    """

    r_max: int = 4
    sigma: float = 0.4
    distance_map_backend: str = "auto"
    edt_parallel_threads: int | None = None
    cleanup_unassigned: bool = True


@dataclass(slots=True)
class PregoSegmentationResult:
    """Intermediate PREGO segmentation data before network construction."""

    im: np.ndarray
    distance_map: np.ndarray
    peaks: np.ndarray
    regions: np.ndarray
    seed_indices: np.ndarray
    seed_radii_voxels: np.ndarray
    seed_activation_levels: np.ndarray
    settings: PregoSettings


@dataclass(slots=True)
class PregoNetworkDictResult:
    """PoreSpy-style network mapping assembled from PREGO regions."""

    network_dict: dict[str, object]
    segmentation: PregoSegmentationResult


def _connectivity_structure(ndim: int) -> np.ndarray:
    """Return cubic marker connectivity, matching PoreSpy's SNOW markers."""

    return np.ones((3,) * ndim, dtype=bool)


def _reduce_peak_labels_to_seed_points(
    peak_labels: np.ndarray,
    distance_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce possibly multi-voxel peak markers to one seed point per label."""

    labels = np.asarray(peak_labels, dtype=np.int64)
    dt = np.asarray(distance_map, dtype=float)
    if labels.shape != dt.shape:
        raise ValueError("peak labels and distance_map must have the same shape")

    raw_labels = np.unique(labels[labels > 0])
    seed_indices: list[np.ndarray] = []
    for raw_label in raw_labels:
        marker_indices = np.argwhere(labels == raw_label)
        if marker_indices.size == 0:  # pragma: no cover - guarded by raw_labels construction
            continue
        marker_radii = dt[tuple(marker_indices.T)]
        # Stable tie-break: largest distance first, then lexicographic index.
        best_radius = float(np.max(marker_radii))
        best_candidates = marker_indices[np.isclose(marker_radii, best_radius)]
        best_index = sorted(tuple(int(value) for value in row) for row in best_candidates)[0]
        seed_indices.append(np.asarray(best_index, dtype=np.int64))

    if not seed_indices:
        return np.zeros((0, dt.ndim), dtype=np.int64), np.zeros(dt.shape, dtype=np.int64)

    seed_array = np.vstack(seed_indices).astype(np.int64, copy=False)
    seed_radii = dt[tuple(seed_array.T)]
    order = sorted(
        range(seed_array.shape[0]),
        key=lambda i: (-float(seed_radii[i]), *[int(v) for v in seed_array[i]]),
    )
    seed_array = seed_array[np.asarray(order, dtype=np.int64)]
    reduced_labels = np.zeros(dt.shape, dtype=np.int64)
    for label, seed_index in enumerate(seed_array, start=1):
        reduced_labels[tuple(int(value) for value in seed_index)] = label
    return seed_array, reduced_labels


def snow_seed_points(
    void_phase_mask: np.ndarray,
    *,
    distance_map: np.ndarray | None = None,
    r_max: int = 4,
    sigma: float = 0.4,
    peaks: np.ndarray | None = None,
    distance_map_backend: str = "auto",
    edt_parallel_threads: int | None = None,
    porespy_module: Any = ps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find PREGO seed points using the peak-filtering stages of SNOW.

    Returns
    -------
    tuple
        ``(seed_indices, seed_labels, distance_map)`` where ``seed_labels`` has
        one labeled voxel per seed and labels are ordered by descending seed
        radius.
    """

    mask = np.asarray(void_phase_mask, dtype=bool)
    if mask.ndim not in {2, 3}:
        raise ValueError("void_phase_mask must be a 2D or 3D array")
    if distance_map is None:
        dt = compute_void_distance_map(
            mask,
            backend=distance_map_backend,
            edt_parallel_threads=edt_parallel_threads,
        )
    else:
        dt = np.asarray(distance_map, dtype=float)
        if dt.shape != mask.shape:
            raise ValueError("distance_map must match void_phase_mask")

    if peaks is None:
        if sigma > 0:
            peak_distance_map = ndi.gaussian_filter(input=dt, sigma=float(sigma)) * mask
        else:
            peak_distance_map = dt.copy()
        peak_mask = porespy_module.filters.find_peaks(dt=peak_distance_map, r_max=int(r_max))
        peak_mask = porespy_module.filters.trim_saddle_points(peaks=peak_mask, dt=dt)
        peak_mask = porespy_module.filters.trim_nearby_peaks(peaks=peak_mask, dt=dt)
        peak_labels, _ = ndi.label(peak_mask > 0, structure=_connectivity_structure(mask.ndim))
    else:
        supplied_peaks = np.asarray(peaks)
        if supplied_peaks.shape != mask.shape:
            raise ValueError("peaks must match void_phase_mask")
        if supplied_peaks.dtype == bool:
            peak_labels, _ = ndi.label(
                supplied_peaks & mask,
                structure=_connectivity_structure(mask.ndim),
            )
        else:
            peak_labels = np.asarray(supplied_peaks, dtype=np.int64) * mask

    seed_indices, seed_labels = _reduce_peak_labels_to_seed_points(peak_labels, dt)
    if seed_indices.size == 0 and np.any(mask):
        fallback_index = np.asarray(
            np.unravel_index(int(np.argmax(dt)), dt.shape),
            dtype=np.int64,
        )
        seed_indices = fallback_index.reshape(1, mask.ndim)
        seed_labels = np.zeros(mask.shape, dtype=np.int64)
        seed_labels[tuple(int(value) for value in fallback_index)] = 1
    return seed_indices, seed_labels, dt


@njit(cache=True)
def _stamp_seed_spheres_2d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
    seed_indices: np.ndarray,
    seed_radii: np.ndarray,
    tolerance: float,
) -> None:
    shape0, shape1 = mask.shape
    for seed_index in range(seed_indices.shape[0]):
        label = seed_index + 1
        c0 = seed_indices[seed_index, 0]
        c1 = seed_indices[seed_index, 1]
        radius = seed_radii[seed_index]
        lower0 = max(0, int(np.floor(c0 - radius - tolerance)))
        upper0 = min(shape0 - 1, int(np.ceil(c0 + radius + tolerance)))
        lower1 = max(0, int(np.floor(c1 - radius - tolerance)))
        upper1 = min(shape1 - 1, int(np.ceil(c1 + radius + tolerance)))
        radius_squared = (radius + tolerance) * (radius + tolerance)
        for i in range(lower0, upper0 + 1):
            di = float(i - c0)
            for j in range(lower1, upper1 + 1):
                if not mask[i, j] or labels[i, j] != 0:
                    continue
                protected_seed_label = seed_label_map[i, j]
                if protected_seed_label != 0 and protected_seed_label != label:
                    continue
                dj = float(j - c1)
                if di * di + dj * dj <= radius_squared:
                    labels[i, j] = label


@njit(cache=True)
def _stamp_seed_spheres_3d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
    seed_indices: np.ndarray,
    seed_radii: np.ndarray,
    tolerance: float,
) -> None:
    shape0, shape1, shape2 = mask.shape
    for seed_index in range(seed_indices.shape[0]):
        label = seed_index + 1
        c0 = seed_indices[seed_index, 0]
        c1 = seed_indices[seed_index, 1]
        c2 = seed_indices[seed_index, 2]
        radius = seed_radii[seed_index]
        lower0 = max(0, int(np.floor(c0 - radius - tolerance)))
        upper0 = min(shape0 - 1, int(np.ceil(c0 + radius + tolerance)))
        lower1 = max(0, int(np.floor(c1 - radius - tolerance)))
        upper1 = min(shape1 - 1, int(np.ceil(c1 + radius + tolerance)))
        lower2 = max(0, int(np.floor(c2 - radius - tolerance)))
        upper2 = min(shape2 - 1, int(np.ceil(c2 + radius + tolerance)))
        radius_squared = (radius + tolerance) * (radius + tolerance)
        for i in range(lower0, upper0 + 1):
            di = float(i - c0)
            for j in range(lower1, upper1 + 1):
                dj = float(j - c1)
                for k in range(lower2, upper2 + 1):
                    if not mask[i, j, k] or labels[i, j, k] != 0:
                        continue
                    protected_seed_label = seed_label_map[i, j, k]
                    if protected_seed_label != 0 and protected_seed_label != label:
                        continue
                    dk = float(k - c2)
                    if di * di + dj * dj + dk * dk <= radius_squared:
                        labels[i, j, k] = label


@njit(cache=True)
def _fifo_fill_regions_2d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
) -> None:
    shape0, shape1 = mask.shape
    max_size = labels.size
    queue0 = np.empty(max_size, dtype=np.int64)
    queue1 = np.empty(max_size, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(shape0):
        for j in range(shape1):
            if mask[i, j] and labels[i, j] > 0:
                queue0[tail] = i
                queue1[tail] = j
                tail += 1

    while head < tail:
        i = queue0[head]
        j = queue1[head]
        head += 1
        label = labels[i, j]
        for axis_step in range(4):
            ni = i
            nj = j
            if axis_step == 0:
                ni = i - 1
            elif axis_step == 1:
                ni = i + 1
            elif axis_step == 2:
                nj = j - 1
            else:
                nj = j + 1
            if ni < 0 or ni >= shape0 or nj < 0 or nj >= shape1:
                continue
            if not mask[ni, nj] or labels[ni, nj] != 0:
                continue
            protected_seed_label = seed_label_map[ni, nj]
            if protected_seed_label != 0 and protected_seed_label != label:
                continue
            labels[ni, nj] = label
            queue0[tail] = ni
            queue1[tail] = nj
            tail += 1


@njit(cache=True)
def _fifo_fill_regions_3d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
) -> None:
    shape0, shape1, shape2 = mask.shape
    max_size = labels.size
    queue0 = np.empty(max_size, dtype=np.int64)
    queue1 = np.empty(max_size, dtype=np.int64)
    queue2 = np.empty(max_size, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(shape0):
        for j in range(shape1):
            for k in range(shape2):
                if mask[i, j, k] and labels[i, j, k] > 0:
                    queue0[tail] = i
                    queue1[tail] = j
                    queue2[tail] = k
                    tail += 1

    while head < tail:
        i = queue0[head]
        j = queue1[head]
        k = queue2[head]
        head += 1
        label = labels[i, j, k]
        for axis_step in range(6):
            ni = i
            nj = j
            nk = k
            if axis_step == 0:
                ni = i - 1
            elif axis_step == 1:
                ni = i + 1
            elif axis_step == 2:
                nj = j - 1
            elif axis_step == 3:
                nj = j + 1
            elif axis_step == 4:
                nk = k - 1
            else:
                nk = k + 1
            if ni < 0 or ni >= shape0 or nj < 0 or nj >= shape1 or nk < 0 or nk >= shape2:
                continue
            if not mask[ni, nj, nk] or labels[ni, nj, nk] != 0:
                continue
            protected_seed_label = seed_label_map[ni, nj, nk]
            if protected_seed_label != 0 and protected_seed_label != label:
                continue
            labels[ni, nj, nk] = label
            queue0[tail] = ni
            queue1[tail] = nj
            queue2[tail] = nk
            tail += 1


def _seed_activation_levels(seed_radii: np.ndarray) -> np.ndarray:
    """Return documented PREGO seed activation levels for diagnostics."""

    if seed_radii.size == 0:
        return np.zeros(0, dtype=np.int64)
    max_radius = float(np.max(seed_radii))
    return np.asarray(np.ceil(np.maximum(0.0, max_radius - seed_radii)), dtype=np.int64)


def prego_partitioning(
    im: np.ndarray,
    *,
    settings: PregoSettings | None = None,
    distance_map: np.ndarray | None = None,
    peaks: np.ndarray | None = None,
    porespy_module: Any = ps,
) -> PregoSegmentationResult:
    """Partition a binary pore image with PREGO-style region growing.

    Notes
    -----
    The PREGO paper specifies descending-radius seed activation and FIFO
    region growth, but leaves several floating-point tie cases underspecified.
    This implementation resolves those cases deterministically by first
    stamping non-overlapping seed spheres in descending radius order, protecting
    all seed voxels from being claimed by other regions, then filling remaining
    foreground voxels by a face-connected FIFO queue. The resulting labels are
    suitable for PoreSpy's ``regions_to_network``.
    """

    resolved_settings = settings or PregoSettings()
    mask = np.asarray(im, dtype=bool)
    if mask.ndim not in {2, 3}:
        raise ValueError("im must be a 2D or 3D binary image")

    seed_indices, seed_label_map, dt = snow_seed_points(
        mask,
        distance_map=distance_map,
        r_max=resolved_settings.r_max,
        sigma=resolved_settings.sigma,
        peaks=peaks,
        distance_map_backend=resolved_settings.distance_map_backend,
        edt_parallel_threads=resolved_settings.edt_parallel_threads,
        porespy_module=porespy_module,
    )
    seed_radii = (
        dt[tuple(seed_indices.T)].astype(float, copy=False)
        if seed_indices.size
        else np.zeros(0, dtype=float)
    )
    labels = np.zeros(mask.shape, dtype=np.int64)
    if seed_indices.size:
        for label, seed_index in enumerate(seed_indices, start=1):
            labels[tuple(int(value) for value in seed_index)] = label
        if mask.ndim == 2:
            _stamp_seed_spheres_2d(mask, labels, seed_label_map, seed_indices, seed_radii, 1e-12)
            _fifo_fill_regions_2d(mask, labels, seed_label_map)
        else:
            _stamp_seed_spheres_3d(mask, labels, seed_label_map, seed_indices, seed_radii, 1e-12)
            _fifo_fill_regions_3d(mask, labels, seed_label_map)

    if resolved_settings.cleanup_unassigned:
        labels = labels * mask
    return PregoSegmentationResult(
        im=mask,
        distance_map=dt,
        peaks=seed_label_map,
        regions=labels.astype(np.int64, copy=False),
        seed_indices=seed_indices,
        seed_radii_voxels=seed_radii,
        seed_activation_levels=_seed_activation_levels(seed_radii),
        settings=resolved_settings,
    )


def _regions_have_interfaces(regions: np.ndarray) -> bool:
    """Return whether positive neighboring labels touch across a face."""

    labels = np.asarray(regions, dtype=np.int64)
    for axis in range(labels.ndim):
        lower_slices = [slice(None)] * labels.ndim
        upper_slices = [slice(None)] * labels.ndim
        lower_slices[axis] = slice(0, -1)
        upper_slices[axis] = slice(1, None)
        lower = labels[tuple(lower_slices)]
        upper = labels[tuple(upper_slices)]
        if np.any((lower > 0) & (upper > 0) & (lower != upper)):
            return True
    return False


def _network_dict_without_interfaces(
    regions: np.ndarray,
    distance_map: np.ndarray,
) -> dict[str, object]:
    """Build a minimal PoreSpy-style network dict for isolated pore regions."""

    labels = np.asarray(regions, dtype=np.int64)
    dt = np.asarray(distance_map, dtype=float)
    region_labels = np.unique(labels[labels > 0])
    pore_count = int(region_labels.size)
    coords = np.zeros((pore_count, labels.ndim), dtype=float)
    region_volume = np.zeros(pore_count, dtype=float)
    inscribed_diameter = np.zeros(pore_count, dtype=float)
    for pore_index, region_label in enumerate(region_labels):
        region_mask = labels == int(region_label)
        indices = np.argwhere(region_mask)
        coords[pore_index] = np.mean(indices, axis=0)
        region_volume[pore_index] = float(indices.shape[0])
        inscribed_diameter[pore_index] = 2.0 * float(np.max(dt[region_mask]))
    if labels.ndim == 2:
        equivalent_diameter = 2.0 * np.sqrt(region_volume / np.pi)
    else:
        equivalent_diameter = 2.0 * np.cbrt(3.0 * region_volume / (4.0 * np.pi))
    return {
        "pore.coords": coords,
        "throat.conns": np.zeros((0, 2), dtype=np.int64),
        "pore.all": np.ones(pore_count, dtype=bool),
        "throat.all": np.zeros(0, dtype=bool),
        "pore.region_label": region_labels.astype(np.int64, copy=False),
        "pore.region_volume": region_volume,
        "pore.inscribed_diameter": inscribed_diameter,
        "pore.equivalent_diameter": equivalent_diameter,
        "throat.volume": np.zeros(0, dtype=float),
        "throat.total_length": np.zeros(0, dtype=float),
        "throat.inscribed_diameter": np.zeros(0, dtype=float),
        "throat.equivalent_diameter": np.zeros(0, dtype=float),
    }


def extract_prego_network_dict(
    im: np.ndarray,
    *,
    settings: PregoSettings | None = None,
    distance_map: np.ndarray | None = None,
    peaks: np.ndarray | None = None,
    porespy_module: Any = ps,
    regions_to_network_kwargs: dict[str, object] | None = None,
) -> PregoNetworkDictResult:
    """Run PREGO segmentation and convert regions to a PoreSpy network dict."""

    segmentation = prego_partitioning(
        im,
        settings=settings,
        distance_map=distance_map,
        peaks=peaks,
        porespy_module=porespy_module,
    )
    kwargs = dict(regions_to_network_kwargs or {})
    if _regions_have_interfaces(segmentation.regions):
        network_dict = dict(
            porespy_module.networks.regions_to_network(segmentation.regions, **kwargs)
        )
    else:
        network_dict = _network_dict_without_interfaces(
            segmentation.regions,
            segmentation.distance_map,
        )
    return PregoNetworkDictResult(network_dict=network_dict, segmentation=segmentation)


__all__ = [
    "PregoNetworkDictResult",
    "PregoSegmentationResult",
    "PregoSettings",
    "extract_prego_network_dict",
    "prego_partitioning",
    "snow_seed_points",
]
