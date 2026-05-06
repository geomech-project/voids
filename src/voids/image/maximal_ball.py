from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import edt as fast_edt
except ImportError:  # pragma: no cover - optional acceleration dependency
    fast_edt = None


@dataclass(slots=True)
class MaximalBallSettings:
    """User-facing controls for the native maximal-ball extraction stages.

    These settings mirror the main Imperial `pnextract` controls closely enough
    for staged verification work, while keeping Python names explicit and
    readable. The current implementation covers the maximal-ball candidate stage
    and the initial overlap suppression stage. Hierarchy construction and voxel
    growth are planned follow-on steps.
    """

    minimal_pore_radius_voxels: float | None = None
    clip_radius_fraction_streamwise: float = 0.05
    clip_radius_fraction_transverse: float = 0.98
    medial_surface_mid_radius_fraction: float = 0.7
    medial_surface_noise_voxels: float | None = None
    hierarchy_length_factor: float = 0.6
    hierarchy_radius_factor: float = 1.1
    radius_smoothing_iterations: int = 3
    retention_radius_factor: float = 0.15
    retention_radius_offset_voxels: float | None = None


@dataclass(slots=True)
class ResolvedMaximalBallSettings:
    """Concrete maximal-ball settings after Imperial-style default resolution."""

    minimal_pore_radius_voxels: float
    clip_radius_fraction_streamwise: float
    clip_radius_fraction_transverse: float
    medial_surface_mid_radius_fraction: float
    medial_surface_noise_voxels: float
    hierarchy_length_factor: float
    hierarchy_radius_factor: float
    radius_smoothing_iterations: int
    retention_radius_factor: float
    retention_radius_offset_voxels: float


@dataclass(slots=True)
class MaximalBallCandidates:
    """Candidate and retained maximal-ball data on voxel centers."""

    center_indices: np.ndarray
    radii_voxels: np.ndarray
    candidate_mask: np.ndarray
    retained_mask: np.ndarray
    distance_map: np.ndarray
    settings: ResolvedMaximalBallSettings

    @property
    def retained_center_indices(self) -> np.ndarray:
        """Return retained maximal-ball centers in descending-radius order."""

        return self.center_indices[self.retained_mask]

    @property
    def retained_radii_voxels(self) -> np.ndarray:
        """Return retained maximal-ball radii in descending-radius order."""

        return self.radii_voxels[self.retained_mask]


@dataclass(slots=True)
class MaximalBallHierarchy:
    """Parent-child hierarchy over retained maximal-ball candidates.

    The hierarchy is stored on the retained-ball order of
    :class:`MaximalBallCandidates`, which is already sorted by descending
    radius. Each ball points either to itself, if it is a root/master ball, or
    to the index of its parent ball in the same retained ordering.
    """

    center_indices: np.ndarray
    radii_voxels: np.ndarray
    parent_indices: np.ndarray
    master_indices: np.ndarray
    hierarchy_levels: np.ndarray
    distance_map: np.ndarray
    settings: ResolvedMaximalBallSettings

    @property
    def root_mask(self) -> np.ndarray:
        """Return a boolean mask of root/master balls."""

        return self.parent_indices == np.arange(self.parent_indices.size, dtype=np.int64)


def compute_void_distance_map(
    void_phase_mask: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Compute the void-space Euclidean distance map.

    Parameters
    ----------
    void_phase_mask :
        Boolean array where ``True`` marks void voxels.
    backend :
        Distance-transform backend. ``"auto"`` prefers the optional `edt`
        package for 3D arrays when available, otherwise falls back to SciPy.
        Explicit options are ``"scipy"`` and ``"edt"``.
    """

    mask = np.asarray(void_phase_mask, dtype=bool)
    if mask.ndim not in {2, 3}:
        raise ValueError("void_phase_mask must be a 2D or 3D boolean array")

    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"auto", "scipy", "edt"}:
        raise ValueError("backend must be one of {'auto', 'scipy', 'edt'}")

    use_fast_edt = normalized_backend == "edt" or (
        normalized_backend == "auto" and mask.ndim == 3 and fast_edt is not None
    )
    if use_fast_edt:
        if fast_edt is None:
            raise ImportError(
                "backend='edt' requested, but the optional 'edt' package is unavailable"
            )
        return np.asarray(fast_edt.edt(mask, black_border=True, parallel=1), dtype=float)
    return np.asarray(ndi.distance_transform_edt(mask), dtype=float)


def resolve_maximal_ball_settings(
    distance_map: np.ndarray,
    settings: MaximalBallSettings | None = None,
) -> ResolvedMaximalBallSettings:
    """Resolve Imperial-style default settings from a distance map.

    The Imperial code derives several defaults from the average void-space
    radius. We mirror that default logic here so staged comparisons use the
    same parameter semantics even before the full extractor is implemented.
    """

    raw_settings = settings or MaximalBallSettings()
    positive_radii = np.asarray(distance_map, dtype=float)
    positive_radii = positive_radii[positive_radii > 0.0]
    average_radius = float(positive_radii.mean()) if positive_radii.size else 0.0

    default_minimal_radius = min(1.25, 0.25 * average_radius) + 0.5
    minimal_pore_radius_voxels = (
        default_minimal_radius
        if raw_settings.minimal_pore_radius_voxels is None
        else float(raw_settings.minimal_pore_radius_voxels)
    )
    if minimal_pore_radius_voxels <= 0.0:
        raise ValueError("minimal_pore_radius_voxels must be positive")

    medial_surface_noise_voxels = (
        abs(minimal_pore_radius_voxels) + 1.0
        if raw_settings.medial_surface_noise_voxels is None
        else float(raw_settings.medial_surface_noise_voxels)
    )
    retention_radius_offset_voxels = (
        abs(minimal_pore_radius_voxels)
        if raw_settings.retention_radius_offset_voxels is None
        else float(raw_settings.retention_radius_offset_voxels)
    )
    if medial_surface_noise_voxels <= 0.0:
        raise ValueError("medial_surface_noise_voxels must be positive")
    if retention_radius_offset_voxels <= 0.0:
        raise ValueError("retention_radius_offset_voxels must be positive")
    if raw_settings.radius_smoothing_iterations < 0:
        raise ValueError("radius_smoothing_iterations must be nonnegative")

    return ResolvedMaximalBallSettings(
        minimal_pore_radius_voxels=float(minimal_pore_radius_voxels),
        clip_radius_fraction_streamwise=float(raw_settings.clip_radius_fraction_streamwise),
        clip_radius_fraction_transverse=float(raw_settings.clip_radius_fraction_transverse),
        medial_surface_mid_radius_fraction=float(raw_settings.medial_surface_mid_radius_fraction),
        medial_surface_noise_voxels=float(medial_surface_noise_voxels),
        hierarchy_length_factor=float(raw_settings.hierarchy_length_factor),
        hierarchy_radius_factor=float(raw_settings.hierarchy_radius_factor),
        radius_smoothing_iterations=int(raw_settings.radius_smoothing_iterations),
        retention_radius_factor=float(raw_settings.retention_radius_factor),
        retention_radius_offset_voxels=float(retention_radius_offset_voxels),
    )


def clip_distance_map_to_domain_boundaries(
    distance_map: np.ndarray,
    *,
    settings: ResolvedMaximalBallSettings,
) -> np.ndarray:
    """Apply the Imperial-style boundary clipping heuristic to a distance map."""

    clipped_distance_map = np.asarray(distance_map, dtype=float).copy()
    if clipped_distance_map.ndim not in {2, 3}:
        raise ValueError("distance_map must be a 2D or 3D array")

    for axis_index, axis_size in enumerate(clipped_distance_map.shape):
        voxel_coordinates = np.arange(axis_size, dtype=float)
        boundary_distance = np.minimum(
            voxel_coordinates + 2.0,
            axis_size - voxel_coordinates + 1.0,
        )
        broadcast_shape = [1] * clipped_distance_map.ndim
        broadcast_shape[axis_index] = axis_size
        boundary_distance = boundary_distance.reshape(broadcast_shape)

        if axis_index == 0:
            clip_fraction = settings.clip_radius_fraction_streamwise
            radius_floor = 0.1
        else:
            clip_fraction = settings.clip_radius_fraction_transverse
            radius_floor = 0.01

        needs_clipping = boundary_distance < clipped_distance_map
        blended_radius = (
            1.0 - clip_fraction
        ) * clipped_distance_map + clip_fraction * boundary_distance
        clipped_distance_map = np.where(
            needs_clipping,
            np.maximum(blended_radius, radius_floor),
            clipped_distance_map,
        )
    return clipped_distance_map


def find_maximal_ball_candidates(
    distance_map: np.ndarray,
    *,
    minimal_radius_voxels: float,
    footprint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find local-maximal ball candidates from a void-space distance map."""

    if minimal_radius_voxels <= 0.0:
        raise ValueError("minimal_radius_voxels must be positive")

    working_distance_map = np.asarray(distance_map, dtype=float)
    if working_distance_map.ndim not in {2, 3}:
        raise ValueError("distance_map must be a 2D or 3D array")

    if footprint is None:
        footprint = np.ones((3,) * working_distance_map.ndim, dtype=bool)
    local_maxima = ndi.maximum_filter(
        working_distance_map,
        footprint=footprint,
        mode="nearest",
    )
    candidate_mask = (
        (working_distance_map > 0.0)
        & (working_distance_map >= minimal_radius_voxels)
        & np.isclose(working_distance_map, local_maxima)
    )
    center_indices = np.argwhere(candidate_mask)
    candidate_radii = working_distance_map[candidate_mask]
    if center_indices.size == 0:
        empty_centers = np.zeros((0, working_distance_map.ndim), dtype=np.int64)
        empty_radii = np.zeros(0, dtype=float)
        return empty_centers, empty_radii, candidate_mask

    descending_order = np.argsort(-candidate_radii, kind="stable")
    return (
        center_indices[descending_order].astype(np.int64, copy=False),
        candidate_radii[descending_order].astype(float, copy=False),
        candidate_mask,
    )


def suppress_overlapping_maximal_balls(
    center_indices: np.ndarray,
    radii_voxels: np.ndarray,
    *,
    settings: ResolvedMaximalBallSettings,
) -> np.ndarray:
    """Retain descending-radius maximal-ball candidates after overlap suppression."""

    centers = np.asarray(center_indices, dtype=np.int64)
    radii = np.asarray(radii_voxels, dtype=float)
    if centers.ndim != 2:
        raise ValueError("center_indices must have shape (count, ndim)")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("radii_voxels must have shape (count,) matching center_indices")

    retained_mask = np.zeros(radii.shape[0], dtype=bool)
    retained_centers: list[np.ndarray] = []
    retained_radii: list[float] = []

    for candidate_index in range(radii.shape[0]):
        candidate_center = centers[candidate_index].astype(float, copy=False)
        candidate_radius = float(radii[candidate_index])
        if not retained_centers:
            retained_mask[candidate_index] = True
            retained_centers.append(candidate_center)
            retained_radii.append(candidate_radius)
            continue

        retained_center_array = np.vstack(retained_centers)
        retained_radius_array = np.asarray(retained_radii, dtype=float)
        center_offsets = retained_center_array - candidate_center
        center_distances = np.sqrt(np.sum(center_offsets * center_offsets, axis=1))
        too_close_to_retained = center_distances < (
            settings.retention_radius_factor * retained_radius_array
            + settings.retention_radius_offset_voxels
        )
        fully_covered_by_retained = center_distances + candidate_radius < (
            retained_radius_array + settings.medial_surface_noise_voxels
        )
        if np.any(too_close_to_retained | fully_covered_by_retained):
            continue

        retained_mask[candidate_index] = True
        retained_centers.append(candidate_center)
        retained_radii.append(candidate_radius)
    return retained_mask


def _find_root_index(parent_indices: np.ndarray, ball_index: int) -> int:
    """Return the root/master index of one retained ball with path compression."""

    path_indices: list[int] = []
    current_index = int(ball_index)
    while parent_indices[current_index] != current_index:
        path_indices.append(current_index)
        current_index = int(parent_indices[current_index])
    for path_index in path_indices:
        parent_indices[path_index] = current_index
    return current_index


def _is_ancestor_index(parent_indices: np.ndarray, ancestor_index: int, child_index: int) -> bool:
    """Return whether ``ancestor_index`` is an ancestor of ``child_index``."""

    current_index = int(child_index)
    visited_indices: set[int] = set()
    while True:
        if current_index == ancestor_index:
            return True
        parent_index = int(parent_indices[current_index])
        if parent_index == current_index or current_index in visited_indices:
            return False
        visited_indices.add(current_index)
        current_index = parent_index


def _weighted_midpoint_index(
    first_center_index: np.ndarray,
    first_radius_voxels: float,
    second_center_index: np.ndarray,
    second_radius_voxels: float,
    *,
    image_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the Imperial-style radius-weighted midpoint voxel index."""

    first_radius_squared = float(first_radius_voxels) ** 2
    second_radius_squared = float(second_radius_voxels) ** 2
    weight_sum = first_radius_squared + second_radius_squared
    weighted_midpoint = (
        first_center_index.astype(float) * second_radius_squared
        + second_center_index.astype(float) * first_radius_squared
    ) / max(weight_sum, 1.0e-30)
    clipped_midpoint = np.clip(
        np.rint(weighted_midpoint).astype(np.int64),
        0,
        np.asarray(image_shape, dtype=np.int64) - 1,
    )
    return tuple(int(value) for value in clipped_midpoint)


def _pair_has_supported_midpoint(
    first_center_index: np.ndarray,
    first_radius_voxels: float,
    second_center_index: np.ndarray,
    second_radius_voxels: float,
    *,
    distance_map: np.ndarray,
    settings: ResolvedMaximalBallSettings,
) -> bool:
    """Return whether two balls satisfy the Imperial midpoint support test."""

    midpoint_index = _weighted_midpoint_index(
        first_center_index,
        first_radius_voxels,
        second_center_index,
        second_radius_voxels,
        image_shape=distance_map.shape,
    )
    midpoint_radius_voxels = float(distance_map[midpoint_index])
    smaller_radius_voxels = min(first_radius_voxels, second_radius_voxels)
    center_distance_voxels = float(
        np.linalg.norm(first_center_index.astype(float) - second_center_index.astype(float))
    )
    midpoint_supported = midpoint_radius_voxels > (
        settings.medial_surface_mid_radius_fraction * smaller_radius_voxels - 0.5
    )
    pair_is_close = 1.01 * center_distance_voxels < (
        first_radius_voxels + second_radius_voxels + 1.0 + settings.medial_surface_noise_voxels
    )
    return midpoint_supported and pair_is_close


def _assign_parent_if_allowed(
    parent_indices: np.ndarray,
    child_index: int,
    parent_index: int,
    *,
    radii_voxels: np.ndarray,
) -> None:
    """Assign a parent if the assignment is acyclic and radius-consistent."""

    if child_index == parent_index:
        return
    if _is_ancestor_index(parent_indices, child_index, parent_index):
        return
    current_parent_index = int(parent_indices[child_index])
    if (
        current_parent_index == child_index
        or radii_voxels[parent_index] >= radii_voxels[current_parent_index]
    ):
        parent_indices[child_index] = parent_index


def build_maximal_ball_hierarchy(
    maximal_ball_candidates: MaximalBallCandidates,
) -> MaximalBallHierarchy:
    """Build an Imperial-inspired hierarchy over retained maximal balls.

    Notes
    -----
    This stage mirrors the main geometric ideas in the Imperial parent
    competition logic:

    - only retained maximal balls participate
    - nearby balls interact only when their midpoint is supported by the void
      distance map
    - smaller balls preferentially attach to larger nearby balls
    - nearby master balls can also merge into a higher-level hierarchy

    This is still a staged native implementation. The downstream voxel-growth
    and throat-construction stages are not yet included here.
    """

    retained_center_indices = np.asarray(
        maximal_ball_candidates.retained_center_indices,
        dtype=np.int64,
    )
    retained_radii_voxels = np.asarray(
        maximal_ball_candidates.retained_radii_voxels,
        dtype=float,
    )
    retained_count = retained_radii_voxels.size
    parent_indices = np.arange(retained_count, dtype=np.int64)
    if retained_count == 0:
        return MaximalBallHierarchy(
            center_indices=retained_center_indices,
            radii_voxels=retained_radii_voxels,
            parent_indices=parent_indices,
            master_indices=parent_indices.copy(),
            hierarchy_levels=np.zeros(0, dtype=np.int64),
            distance_map=np.asarray(maximal_ball_candidates.distance_map, dtype=float),
            settings=maximal_ball_candidates.settings,
        )

    settings = maximal_ball_candidates.settings
    center_tree = cKDTree(retained_center_indices.astype(float))
    distance_map = np.asarray(maximal_ball_candidates.distance_map, dtype=float)

    for first_ball_index in range(retained_count):
        first_center_index = retained_center_indices[first_ball_index]
        first_radius_voxels = float(retained_radii_voxels[first_ball_index])
        neighbor_search_radius_voxels = (
            settings.hierarchy_length_factor * first_radius_voxels
            + 2.0 * settings.medial_surface_noise_voxels
            + 2.0
        )
        nearby_ball_indices = center_tree.query_ball_point(
            first_center_index.astype(float),
            r=neighbor_search_radius_voxels,
        )
        for second_ball_index in nearby_ball_indices:
            if second_ball_index <= first_ball_index:
                continue
            second_center_index = retained_center_indices[second_ball_index]
            second_radius_voxels = float(retained_radii_voxels[second_ball_index])
            if not _pair_has_supported_midpoint(
                first_center_index,
                first_radius_voxels,
                second_center_index,
                second_radius_voxels,
                distance_map=distance_map,
                settings=settings,
            ):
                continue

            larger_ball_index = (
                first_ball_index
                if first_radius_voxels >= second_radius_voxels
                else second_ball_index
            )
            smaller_ball_index = (
                second_ball_index if larger_ball_index == first_ball_index else first_ball_index
            )
            _assign_parent_if_allowed(
                parent_indices,
                smaller_ball_index,
                larger_ball_index,
                radii_voxels=retained_radii_voxels,
            )

            first_root_index = _find_root_index(parent_indices, first_ball_index)
            second_root_index = _find_root_index(parent_indices, second_ball_index)
            if first_root_index == second_root_index:
                continue

            first_root_radius_voxels = float(retained_radii_voxels[first_root_index])
            second_root_radius_voxels = float(retained_radii_voxels[second_root_index])
            first_root_center_index = retained_center_indices[first_root_index]
            second_root_center_index = retained_center_indices[second_root_index]
            root_distance_voxels = float(
                np.linalg.norm(
                    first_root_center_index.astype(float) - second_root_center_index.astype(float)
                )
            )
            average_root_radius_voxels = 0.5 * (
                first_root_radius_voxels + second_root_radius_voxels
            )
            merge_threshold_voxels = np.sqrt(settings.hierarchy_length_factor) * (
                average_root_radius_voxels + 2.0 * settings.medial_surface_noise_voxels
            )
            if root_distance_voxels > merge_threshold_voxels:
                continue

            larger_root_index = (
                first_root_index
                if first_root_radius_voxels >= second_root_radius_voxels
                else second_root_index
            )
            smaller_root_index = (
                second_root_index if larger_root_index == first_root_index else first_root_index
            )
            if retained_radii_voxels[smaller_root_index] < (
                settings.hierarchy_radius_factor * retained_radii_voxels[smaller_ball_index]
                + settings.medial_surface_noise_voxels
            ):
                _assign_parent_if_allowed(
                    parent_indices,
                    smaller_root_index,
                    larger_root_index,
                    radii_voxels=retained_radii_voxels,
                )

    master_indices = np.array(
        [_find_root_index(parent_indices, ball_index) for ball_index in range(retained_count)],
        dtype=np.int64,
    )
    hierarchy_levels = np.zeros(retained_count, dtype=np.int64)
    for ball_index in range(retained_count):
        current_index = ball_index
        while parent_indices[current_index] != current_index:
            hierarchy_levels[ball_index] += 1
            current_index = int(parent_indices[current_index])

    return MaximalBallHierarchy(
        center_indices=retained_center_indices,
        radii_voxels=retained_radii_voxels,
        parent_indices=parent_indices,
        master_indices=master_indices,
        hierarchy_levels=hierarchy_levels,
        distance_map=distance_map,
        settings=settings,
    )


def extract_maximal_ball_candidates(
    void_phase_mask: np.ndarray,
    *,
    distance_map_backend: str = "auto",
    settings: MaximalBallSettings | None = None,
    apply_boundary_clipping: bool = True,
) -> MaximalBallCandidates:
    """Compute and suppress maximal-ball candidates for a void-phase image."""

    distance_map = compute_void_distance_map(void_phase_mask, backend=distance_map_backend)
    resolved_settings = resolve_maximal_ball_settings(distance_map, settings)
    working_distance_map = (
        clip_distance_map_to_domain_boundaries(distance_map, settings=resolved_settings)
        if apply_boundary_clipping
        else np.asarray(distance_map, dtype=float)
    )
    center_indices, radii_voxels, candidate_mask = find_maximal_ball_candidates(
        working_distance_map,
        minimal_radius_voxels=resolved_settings.minimal_pore_radius_voxels,
    )
    retained_mask = suppress_overlapping_maximal_balls(
        center_indices,
        radii_voxels,
        settings=resolved_settings,
    )
    return MaximalBallCandidates(
        center_indices=center_indices,
        radii_voxels=radii_voxels,
        candidate_mask=candidate_mask,
        retained_mask=retained_mask,
        distance_map=working_distance_map,
        settings=resolved_settings,
    )


__all__ = [
    "MaximalBallCandidates",
    "MaximalBallHierarchy",
    "MaximalBallSettings",
    "ResolvedMaximalBallSettings",
    "build_maximal_ball_hierarchy",
    "clip_distance_map_to_domain_boundaries",
    "compute_void_distance_map",
    "extract_maximal_ball_candidates",
    "find_maximal_ball_candidates",
    "resolve_maximal_ball_settings",
    "suppress_overlapping_maximal_balls",
]
