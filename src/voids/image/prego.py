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

    The defaults use ``r_max=4`` and Gaussian smoothing ``sigma=0.4``. PREGO is
    currently implemented for a single active pore phase, with nonzero input
    voxels treated as void.
    """

    r_max: int = 4
    sigma: float = 0.4
    peak_footprint: str = "sphere"
    distance_map_backend: str = "auto"
    edt_parallel_threads: int | None = None
    cleanup_unassigned: bool = True
    growth_mode: str = "level_queue"


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


def _smallest_signed_integer_dtype(max_value: int) -> type[np.signedinteger[Any]]:
    """Return the smallest signed NumPy integer dtype that can store ``max_value``."""

    if max_value <= np.iinfo(np.int16).max:
        return np.int16
    if max_value <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _prego_label_dtype(
    *,
    max_label: int,
    shape: tuple[int, ...],
) -> type[np.signedinteger[Any]]:
    """Return a safe compact dtype for labels and FIFO coordinate queues."""

    max_coordinate = max((int(size) - 1 for size in shape), default=0)
    return _smallest_signed_integer_dtype(max(max_label, max_coordinate, 0))


def _reduce_peak_labels_to_seed_points(
    peak_labels: np.ndarray,
    distance_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce possibly multi-voxel peak markers to one seed point per label."""

    labels = np.asarray(peak_labels)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64, copy=False)
    dt = np.asarray(distance_map, dtype=float)
    if labels.shape != dt.shape:
        raise ValueError("peak labels and distance_map must have the same shape")

    marker_indices = np.argwhere(labels > 0)
    if marker_indices.size == 0:
        return np.zeros((0, dt.ndim), dtype=np.int64), np.zeros(
            dt.shape,
            dtype=_prego_label_dtype(max_label=0, shape=dt.shape),
        )

    marker_labels = labels[tuple(marker_indices.T)]
    order_by_label = np.argsort(marker_labels, kind="stable")
    marker_indices = marker_indices[order_by_label]
    marker_labels = marker_labels[order_by_label]

    seed_indices: list[np.ndarray] = []
    start = 0
    while start < marker_labels.size:
        stop = start + 1
        while stop < marker_labels.size and marker_labels[stop] == marker_labels[start]:
            stop += 1
        label_indices = marker_indices[start:stop]
        marker_radii = dt[tuple(label_indices.T)]
        # Stable tie-break: largest distance first, then lexicographic index.
        best_radius = float(np.max(marker_radii))
        best_candidates = label_indices[np.isclose(marker_radii, best_radius)]
        best_index = sorted(tuple(int(value) for value in row) for row in best_candidates)[0]
        seed_indices.append(np.asarray(best_index, dtype=np.int64))
        start = stop

    seed_array = np.vstack(seed_indices).astype(np.int64, copy=False)
    seed_radii = dt[tuple(seed_array.T)]
    order = sorted(
        range(seed_array.shape[0]),
        key=lambda i: (-float(seed_radii[i]), *[int(v) for v in seed_array[i]]),
    )
    seed_array = seed_array[np.asarray(order, dtype=np.int64)]
    label_dtype = _prego_label_dtype(max_label=seed_array.shape[0], shape=dt.shape)
    reduced_labels = np.zeros(dt.shape, dtype=label_dtype)
    for label, seed_index in enumerate(seed_array, start=1):
        reduced_labels[tuple(int(value) for value in seed_index)] = label
    return seed_array, reduced_labels


def snow_seed_points(
    void_phase_mask: np.ndarray,
    *,
    distance_map: np.ndarray | None = None,
    r_max: int = 4,
    sigma: float = 0.4,
    peak_footprint: str = "sphere",
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
        normalized_footprint = str(peak_footprint).strip().lower()
        if normalized_footprint == "sphere":
            peak_mask = porespy_module.filters.find_peaks(dt=peak_distance_map, r_max=int(r_max))
        elif normalized_footprint in {"cube", "box"}:
            peak_distance_for_filter = peak_distance_map + 2.0 * (~mask)
            peak_max = ndi.maximum_filter(
                peak_distance_for_filter,
                size=(2 * int(r_max) + 1,) * mask.ndim,
            )
            peak_mask = (peak_distance_map == peak_max) & mask & (peak_distance_map > 0)
        else:
            raise ValueError("peak_footprint must be 'cube' or 'sphere'")
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
            supplied_labels = np.asarray(supplied_peaks)
            max_label = int(np.max(supplied_labels)) if supplied_labels.size else 0
            peak_dtype = _prego_label_dtype(max_label=max_label, shape=mask.shape)
            peak_labels = supplied_labels.astype(peak_dtype, copy=False) * mask

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
    queue0 = np.empty(max_size, dtype=labels.dtype)
    queue1 = np.empty(max_size, dtype=labels.dtype)
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
    queue0 = np.empty(max_size, dtype=labels.dtype)
    queue1 = np.empty(max_size, dtype=labels.dtype)
    queue2 = np.empty(max_size, dtype=labels.dtype)
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


def _normalize_growth_mode(growth_mode: str) -> str:
    """Normalize PREGO region-growth mode aliases."""

    normalized = str(growth_mode).strip().lower()
    aliases = {
        "fast": "fast",
        "stamp": "fast",
        "stamp_then_fill": "fast",
        "level_queue": "level_queue",
        "paper": "level_queue",
        "paper_like": "level_queue",
    }
    if normalized not in aliases:
        raise ValueError(
            "growth_mode must be one of {'fast', 'stamp_then_fill', "
            "'level_queue', 'paper', 'paper_like'}"
        )
    return aliases[normalized]


@njit(cache=True)
def _level_queue_grow_regions_2d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
    seed_indices: np.ndarray,
    seed_radii: np.ndarray,
    activation_levels: np.ndarray,
    tolerance: float,
) -> None:
    shape0, shape1 = mask.shape
    max_size = labels.size
    current0 = np.empty(max_size, dtype=labels.dtype)
    current1 = np.empty(max_size, dtype=labels.dtype)
    next0 = np.empty(max_size, dtype=labels.dtype)
    next1 = np.empty(max_size, dtype=labels.dtype)
    current_tail = 0
    next_seed = 0
    level = 0

    while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
        c0 = seed_indices[next_seed, 0]
        c1 = seed_indices[next_seed, 1]
        current0[current_tail] = c0
        current1[current_tail] = c1
        current_tail += 1
        next_seed += 1

    while current_tail > 0 or next_seed < seed_indices.shape[0]:
        if current_tail == 0:
            level = max(level, activation_levels[next_seed])
            while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
                c0 = seed_indices[next_seed, 0]
                c1 = seed_indices[next_seed, 1]
                current0[current_tail] = c0
                current1[current_tail] = c1
                current_tail += 1
                next_seed += 1

        level += 1
        current_head = 0
        next_tail = 0
        while current_head < current_tail:
            i = current0[current_head]
            j = current1[current_head]
            current_head += 1
            label = labels[i, j]
            if label <= 0:
                continue
            seed_index = int(label - 1)
            c0 = seed_indices[seed_index, 0]
            c1 = seed_indices[seed_index, 1]
            radius = seed_radii[seed_index]
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
                d0 = float(ni - c0)
                d1 = float(nj - c1)
                distance = np.sqrt(d0 * d0 + d1 * d1)
                if distance <= radius + tolerance:
                    labels[ni, nj] = label
                    if distance < radius - tolerance:
                        current0[current_tail] = ni
                        current1[current_tail] = nj
                        current_tail += 1
                    else:
                        next0[next_tail] = ni
                        next1[next_tail] = nj
                        next_tail += 1

        while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
            c0 = seed_indices[next_seed, 0]
            c1 = seed_indices[next_seed, 1]
            next0[next_tail] = c0
            next1[next_tail] = c1
            next_tail += 1
            next_seed += 1

        tmp0 = current0
        tmp1 = current1
        current0 = next0
        current1 = next1
        next0 = tmp0
        next1 = tmp1
        current_tail = next_tail


@njit(cache=True)
def _level_queue_grow_regions_3d(
    mask: np.ndarray,
    labels: np.ndarray,
    seed_label_map: np.ndarray,
    seed_indices: np.ndarray,
    seed_radii: np.ndarray,
    activation_levels: np.ndarray,
    tolerance: float,
) -> None:
    shape0, shape1, shape2 = mask.shape
    max_size = labels.size
    current0 = np.empty(max_size, dtype=labels.dtype)
    current1 = np.empty(max_size, dtype=labels.dtype)
    current2 = np.empty(max_size, dtype=labels.dtype)
    next0 = np.empty(max_size, dtype=labels.dtype)
    next1 = np.empty(max_size, dtype=labels.dtype)
    next2 = np.empty(max_size, dtype=labels.dtype)
    current_tail = 0
    next_seed = 0
    level = 0

    while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
        c0 = seed_indices[next_seed, 0]
        c1 = seed_indices[next_seed, 1]
        c2 = seed_indices[next_seed, 2]
        current0[current_tail] = c0
        current1[current_tail] = c1
        current2[current_tail] = c2
        current_tail += 1
        next_seed += 1

    while current_tail > 0 or next_seed < seed_indices.shape[0]:
        if current_tail == 0:
            level = max(level, activation_levels[next_seed])
            while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
                c0 = seed_indices[next_seed, 0]
                c1 = seed_indices[next_seed, 1]
                c2 = seed_indices[next_seed, 2]
                current0[current_tail] = c0
                current1[current_tail] = c1
                current2[current_tail] = c2
                current_tail += 1
                next_seed += 1

        level += 1
        current_head = 0
        next_tail = 0
        while current_head < current_tail:
            i = current0[current_head]
            j = current1[current_head]
            k = current2[current_head]
            current_head += 1
            label = labels[i, j, k]
            if label <= 0:
                continue
            seed_index = int(label - 1)
            c0 = seed_indices[seed_index, 0]
            c1 = seed_indices[seed_index, 1]
            c2 = seed_indices[seed_index, 2]
            radius = seed_radii[seed_index]
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
                d0 = float(ni - c0)
                d1 = float(nj - c1)
                d2 = float(nk - c2)
                distance = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                if distance <= radius + tolerance:
                    labels[ni, nj, nk] = label
                    if distance < radius - tolerance:
                        current0[current_tail] = ni
                        current1[current_tail] = nj
                        current2[current_tail] = nk
                        current_tail += 1
                    else:
                        next0[next_tail] = ni
                        next1[next_tail] = nj
                        next2[next_tail] = nk
                        next_tail += 1

        while next_seed < seed_indices.shape[0] and activation_levels[next_seed] <= level:
            c0 = seed_indices[next_seed, 0]
            c1 = seed_indices[next_seed, 1]
            c2 = seed_indices[next_seed, 2]
            next0[next_tail] = c0
            next1[next_tail] = c1
            next2[next_tail] = c2
            next_tail += 1
            next_seed += 1

        tmp0 = current0
        tmp1 = current1
        tmp2 = current2
        current0 = next0
        current1 = next1
        current2 = next2
        next0 = tmp0
        next1 = tmp1
        next2 = tmp2
        current_tail = next_tail


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
    The default ``growth_mode="level_queue"`` follows the paper's delayed seed
    activation and level-by-level FIFO queue before the final expansion of
    unassigned foreground voxels. ``growth_mode="fast"`` remains available as
    a faster approximation that stamps non-overlapping seed spheres in
    descending radius order before the same final FIFO fill.
    """

    resolved_settings = settings or PregoSettings()
    mask = np.asarray(im, dtype=bool)
    if mask.ndim not in {2, 3}:
        raise ValueError("im must be a 2D or 3D binary image")
    growth_mode = _normalize_growth_mode(resolved_settings.growth_mode)

    seed_indices, seed_label_map, dt = snow_seed_points(
        mask,
        distance_map=distance_map,
        r_max=resolved_settings.r_max,
        sigma=resolved_settings.sigma,
        peak_footprint=resolved_settings.peak_footprint,
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
    activation_levels = _seed_activation_levels(seed_radii)
    labels = np.zeros(
        mask.shape,
        dtype=_prego_label_dtype(max_label=seed_indices.shape[0], shape=mask.shape),
    )
    if seed_indices.size:
        for label, seed_index in enumerate(seed_indices, start=1):
            labels[tuple(int(value) for value in seed_index)] = label
        if growth_mode == "fast":
            if mask.ndim == 2:
                _stamp_seed_spheres_2d(
                    mask, labels, seed_label_map, seed_indices, seed_radii, 1e-12
                )
            else:
                _stamp_seed_spheres_3d(
                    mask, labels, seed_label_map, seed_indices, seed_radii, 1e-12
                )
        elif mask.ndim == 2:
            _level_queue_grow_regions_2d(
                mask,
                labels,
                seed_label_map,
                seed_indices,
                seed_radii,
                activation_levels,
                1e-12,
            )
        else:
            _level_queue_grow_regions_3d(
                mask,
                labels,
                seed_label_map,
                seed_indices,
                seed_radii,
                activation_levels,
                1e-12,
            )
        if mask.ndim == 2:
            _fifo_fill_regions_2d(mask, labels, seed_label_map)
        else:
            _fifo_fill_regions_3d(mask, labels, seed_label_map)

    if resolved_settings.cleanup_unassigned:
        labels = labels * mask
    return PregoSegmentationResult(
        im=mask,
        distance_map=dt,
        peaks=seed_label_map,
        regions=labels,
        seed_indices=seed_indices,
        seed_radii_voxels=seed_radii,
        seed_activation_levels=activation_levels,
        settings=resolved_settings,
    )


def _regions_have_interfaces(regions: np.ndarray) -> bool:
    """Return whether positive neighboring labels touch across a face."""

    labels = np.asarray(regions)
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

    labels = np.asarray(regions)
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
