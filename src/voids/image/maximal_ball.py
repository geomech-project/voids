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


@dataclass(slots=True)
class MaximalBallVoxelRegions:
    """Voxel ownership assignment grown from maximal-ball hierarchy roots."""

    label_image: np.ndarray
    root_ball_indices: np.ndarray
    root_labels: np.ndarray
    root_center_indices: np.ndarray
    root_radii_voxels: np.ndarray
    root_of_ball_index: np.ndarray
    unassigned_label: int

    @property
    def assigned_void_mask(self) -> np.ndarray:
        """Return a mask of voxels assigned to some pore/root region."""

        return self.label_image >= 0


@dataclass(slots=True)
class MaximalBallRegionAdjacency:
    """Region-wise geometric summaries derived from voxel ownership labels.

    Notes
    -----
    The fields here are deliberately close to the intermediate quantities that
    the Imperial extractor builds before CNM export:

    - per-region occupied voxel counts
    - per-region exposed face counts
    - region-to-region interface face counts
    - interface centroids in voxel-index coordinates
    - boundary-face contact counts on each sample side

    This is still an intermediate voxel-geometry product, not yet a final
    ``voids.Network``.
    """

    region_labels: np.ndarray
    region_volume_voxels: np.ndarray
    region_surface_face_counts: np.ndarray
    throat_region_pairs: np.ndarray
    throat_face_counts: np.ndarray
    throat_axis_face_balance: np.ndarray
    throat_centroid_indices: np.ndarray
    boundary_face_counts: np.ndarray


@dataclass(slots=True)
class MaximalBallExtractionResult:
    """Staged native maximal-ball extraction outputs before CNM assembly."""

    candidates: MaximalBallCandidates
    hierarchy: MaximalBallHierarchy
    voxel_regions: MaximalBallVoxelRegions
    region_adjacency: MaximalBallRegionAdjacency


@dataclass(slots=True)
class MaximalBallNetworkDictResult:
    """PoreSpy-style network mapping assembled from native maximal-ball regions."""

    network_dict: dict[str, np.ndarray]
    extraction: MaximalBallExtractionResult


_CIRCULAR_SHAPE_FACTOR = 1.0 / (4.0 * np.pi)


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


def initialize_root_region_labels(
    void_phase_mask: np.ndarray,
    maximal_ball_hierarchy: MaximalBallHierarchy,
    *,
    unassigned_label: int = -1,
) -> MaximalBallVoxelRegions:
    """Seed voxel-region labels from hierarchy root balls.

    Notes
    -----
    This stage mirrors the first pore-element seeding in the Imperial code:
    each root/master maximal ball defines an initial pore region, and each
    retained non-root ball maps to the region of its hierarchy root.
    """

    mask = np.asarray(void_phase_mask, dtype=bool)
    if mask.shape != maximal_ball_hierarchy.distance_map.shape:
        raise ValueError("void_phase_mask must match the hierarchy distance-map shape")

    root_ball_indices = np.flatnonzero(maximal_ball_hierarchy.root_mask).astype(np.int64)
    root_labels = np.arange(root_ball_indices.size, dtype=np.int64)
    label_image = np.full(mask.shape, unassigned_label, dtype=np.int64)
    if root_ball_indices.size == 0:
        return MaximalBallVoxelRegions(
            label_image=label_image,
            root_ball_indices=root_ball_indices,
            root_labels=root_labels,
            root_center_indices=np.zeros((0, mask.ndim), dtype=np.int64),
            root_radii_voxels=np.zeros(0, dtype=float),
            root_of_ball_index=np.zeros(0, dtype=np.int64),
            unassigned_label=int(unassigned_label),
        )

    root_center_indices = maximal_ball_hierarchy.center_indices[root_ball_indices]
    root_radii_voxels = maximal_ball_hierarchy.radii_voxels[root_ball_indices]
    root_lookup = {
        int(ball_index): int(label) for label, ball_index in enumerate(root_ball_indices)
    }
    root_of_ball_index = np.array(
        [root_lookup[int(root_index)] for root_index in maximal_ball_hierarchy.master_indices],
        dtype=np.int64,
    )

    for root_label, root_center_index in zip(root_labels, root_center_indices, strict=False):
        label_image[tuple(int(value) for value in root_center_index)] = int(root_label)

    return MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=root_ball_indices,
        root_labels=root_labels,
        root_center_indices=root_center_indices,
        root_radii_voxels=root_radii_voxels,
        root_of_ball_index=root_of_ball_index,
        unassigned_label=int(unassigned_label),
    )


def seed_root_region_ball_interiors(
    void_phase_mask: np.ndarray,
    maximal_ball_hierarchy: MaximalBallHierarchy,
    voxel_regions: MaximalBallVoxelRegions,
) -> MaximalBallVoxelRegions:
    """Assign small interior neighborhoods around retained balls to their root regions."""

    mask = np.asarray(void_phase_mask, dtype=bool)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    retained_center_indices = maximal_ball_hierarchy.center_indices
    retained_radii_voxels = maximal_ball_hierarchy.radii_voxels

    for ball_index, center_index in enumerate(retained_center_indices):
        root_label = int(voxel_regions.root_of_ball_index[ball_index])
        radius_voxels = float(retained_radii_voxels[ball_index])
        seeding_radius_voxels = max(radius_voxels * 0.25 - 1.0, 1.001)
        radius_squared = seeding_radius_voxels * seeding_radius_voxels

        lower_bounds = np.maximum(
            np.floor(center_index - seeding_radius_voxels).astype(np.int64),
            0,
        )
        upper_bounds = np.minimum(
            np.ceil(center_index + seeding_radius_voxels).astype(np.int64) + 1,
            np.asarray(mask.shape, dtype=np.int64),
        )
        index_slices = tuple(
            slice(int(lower), int(upper)) for lower, upper in zip(lower_bounds, upper_bounds)
        )
        candidate_offsets = np.indices(
            tuple(int(upper - lower) for lower, upper in zip(lower_bounds, upper_bounds))
        )
        for axis_index, lower_bound in enumerate(lower_bounds):
            candidate_offsets[axis_index] = (
                candidate_offsets[axis_index] + int(lower_bound) - int(center_index[axis_index])
            )
        candidate_distance_squared = np.sum(candidate_offsets.astype(float) ** 2, axis=0)
        local_mask = candidate_distance_squared <= radius_squared
        local_void_mask = mask[index_slices]
        assignable_mask = local_mask & local_void_mask
        local_labels = label_image[index_slices]
        local_labels[assignable_mask] = root_label
        label_image[index_slices] = local_labels

    return MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=voxel_regions.root_ball_indices,
        root_labels=voxel_regions.root_labels,
        root_center_indices=voxel_regions.root_center_indices,
        root_radii_voxels=voxel_regions.root_radii_voxels,
        root_of_ball_index=voxel_regions.root_of_ball_index,
        unassigned_label=voxel_regions.unassigned_label,
    )


def _neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    """Return 6-connectivity offsets in 3D or 4-connectivity offsets in 2D."""

    offsets: list[tuple[int, ...]] = []
    for axis_index in range(ndim):
        negative_offset = [0] * ndim
        positive_offset = [0] * ndim
        negative_offset[axis_index] = -1
        positive_offset[axis_index] = 1
        offsets.append(tuple(negative_offset))
        offsets.append(tuple(positive_offset))
    return offsets


def grow_root_regions_by_radius(
    void_phase_mask: np.ndarray,
    distance_map: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
    *,
    minimum_supporting_neighbors: int,
    require_strictly_larger_radius: bool,
    iterations: int = 1,
) -> MaximalBallVoxelRegions:
    """Grow root regions across unassigned void voxels using local radius rules."""

    if minimum_supporting_neighbors < 1:
        raise ValueError("minimum_supporting_neighbors must be at least 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    mask = np.asarray(void_phase_mask, dtype=bool)
    working_distance_map = np.asarray(distance_map, dtype=float)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    if mask.shape != label_image.shape or mask.shape != working_distance_map.shape:
        raise ValueError("void_phase_mask, distance_map, and voxel_regions.label_image must match")

    neighbor_offsets = _neighbor_offsets(mask.ndim)
    image_shape = np.asarray(mask.shape, dtype=np.int64)
    for _ in range(iterations):
        previous_labels = label_image.copy()
        changed_any_voxel = False
        unassigned_indices = np.argwhere(mask & (previous_labels == voxel_regions.unassigned_label))
        for voxel_index in unassigned_indices:
            voxel_index_tuple = tuple(int(value) for value in voxel_index)
            voxel_radius = float(working_distance_map[voxel_index_tuple])
            supporting_label_counts: dict[int, int] = {}
            for neighbor_offset in neighbor_offsets:
                neighbor_index = voxel_index + np.asarray(neighbor_offset, dtype=np.int64)
                if np.any(neighbor_index < 0) or np.any(neighbor_index >= image_shape):
                    continue
                neighbor_index_tuple = tuple(int(value) for value in neighbor_index)
                neighbor_label = int(previous_labels[neighbor_index_tuple])
                if neighbor_label < 0:
                    continue
                neighbor_radius = float(working_distance_map[neighbor_index_tuple])
                supports_voxel = (
                    neighbor_radius > voxel_radius
                    if require_strictly_larger_radius
                    else neighbor_radius >= voxel_radius
                )
                if not supports_voxel:
                    continue
                supporting_label_counts[neighbor_label] = (
                    supporting_label_counts.get(neighbor_label, 0) + 1
                )

            if not supporting_label_counts:
                continue
            best_label, best_support = max(
                supporting_label_counts.items(),
                key=lambda item: (item[1], -item[0]),
            )
            if best_support >= minimum_supporting_neighbors:
                label_image[voxel_index_tuple] = int(best_label)
                changed_any_voxel = True
        if not changed_any_voxel:
            break

    return MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=voxel_regions.root_ball_indices,
        root_labels=voxel_regions.root_labels,
        root_center_indices=voxel_regions.root_center_indices,
        root_radii_voxels=voxel_regions.root_radii_voxels,
        root_of_ball_index=voxel_regions.root_of_ball_index,
        unassigned_label=voxel_regions.unassigned_label,
    )


def assign_voxel_regions_from_hierarchy(
    void_phase_mask: np.ndarray,
    maximal_ball_hierarchy: MaximalBallHierarchy,
) -> MaximalBallVoxelRegions:
    """Assign voxel ownership from a maximal-ball hierarchy using staged growth passes."""

    voxel_regions = initialize_root_region_labels(
        void_phase_mask,
        maximal_ball_hierarchy,
    )
    voxel_regions = seed_root_region_ball_interiors(
        void_phase_mask,
        maximal_ball_hierarchy,
        voxel_regions,
    )
    growth_schedule = [
        (3, True, 4),
        (2, True, 6),
        (2, False, 8),
        (1, False, 8),
    ]
    for minimum_supporting_neighbors, require_strictly_larger_radius, iterations in growth_schedule:
        voxel_regions = grow_root_regions_by_radius(
            void_phase_mask,
            maximal_ball_hierarchy.distance_map,
            voxel_regions,
            minimum_supporting_neighbors=minimum_supporting_neighbors,
            require_strictly_larger_radius=require_strictly_larger_radius,
            iterations=iterations,
        )
    return voxel_regions


def measure_region_adjacency(
    void_phase_mask: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
) -> MaximalBallRegionAdjacency:
    """Measure pore-region volumes, interfaces, and boundary contacts.

    Parameters
    ----------
    void_phase_mask :
        Boolean void-domain mask used for extraction.
    voxel_regions :
        Labeled pore/root ownership image.

    Returns
    -------
    MaximalBallRegionAdjacency
        Region-wise voxel volumes and region-pair interface measurements.

    Notes
    -----
    This stage converts the voxel partition into the basic discrete geometry we
    need for a native `pnextract`-like network assembly:

    - region voxel counts become pore-region volumes
    - region-pair contact faces become throat candidates
    - boundary-face contacts expose inlet/outlet touching regions

    The centroid coordinates are reported in voxel-index units, using face
    midpoint locations such as ``i + 0.5`` along the axis normal to the
    interface.
    """

    mask = np.asarray(void_phase_mask, dtype=bool)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64)
    if mask.shape != label_image.shape:
        raise ValueError("void_phase_mask and voxel_regions.label_image must match")

    region_labels = np.asarray(voxel_regions.root_labels, dtype=np.int64)
    region_count = int(region_labels.size)
    region_volume_voxels = np.zeros(region_count, dtype=np.int64)
    region_surface_face_counts = np.zeros(region_count, dtype=np.int64)
    boundary_face_counts = np.zeros((region_count, 2 * mask.ndim), dtype=np.int64)

    assigned_void_mask = mask & (label_image >= 0)
    if np.any(label_image[assigned_void_mask] >= region_count):
        raise ValueError("voxel region labels must be contiguous root labels starting at zero")

    if np.any(assigned_void_mask):
        region_volume_voxels += np.bincount(
            label_image[assigned_void_mask],
            minlength=region_count,
        ).astype(np.int64, copy=False)

    pair_face_counts: dict[tuple[int, int], int] = {}
    pair_axis_face_balance: dict[tuple[int, int], np.ndarray] = {}
    pair_centroid_sums: dict[tuple[int, int], np.ndarray] = {}

    image_shape = np.asarray(mask.shape, dtype=np.int64)
    for axis_index in range(mask.ndim):
        lower_boundary_selector = [slice(None)] * mask.ndim
        lower_boundary_selector[axis_index] = 0
        lower_boundary_void_mask = assigned_void_mask[tuple(lower_boundary_selector)]
        lower_boundary_labels = label_image[tuple(lower_boundary_selector)]
        if np.any(lower_boundary_void_mask):
            lower_counts = np.bincount(
                lower_boundary_labels[lower_boundary_void_mask],
                minlength=region_count,
            )
            boundary_face_counts[:, 2 * axis_index] += lower_counts.astype(np.int64, copy=False)
            region_surface_face_counts += lower_counts.astype(np.int64, copy=False)

        upper_boundary_selector = [slice(None)] * mask.ndim
        upper_boundary_selector[axis_index] = int(image_shape[axis_index] - 1)
        upper_boundary_void_mask = assigned_void_mask[tuple(upper_boundary_selector)]
        upper_boundary_labels = label_image[tuple(upper_boundary_selector)]
        if np.any(upper_boundary_void_mask):
            upper_counts = np.bincount(
                upper_boundary_labels[upper_boundary_void_mask],
                minlength=region_count,
            )
            boundary_face_counts[:, 2 * axis_index + 1] += upper_counts.astype(
                np.int64,
                copy=False,
            )
            region_surface_face_counts += upper_counts.astype(np.int64, copy=False)

        lower_slices = [slice(None)] * mask.ndim
        upper_slices = [slice(None)] * mask.ndim
        lower_slices[axis_index] = slice(0, -1)
        upper_slices[axis_index] = slice(1, None)
        lower_slices_tuple = tuple(lower_slices)
        upper_slices_tuple = tuple(upper_slices)

        lower_assigned_mask = assigned_void_mask[lower_slices_tuple]
        upper_assigned_mask = assigned_void_mask[upper_slices_tuple]
        lower_labels = label_image[lower_slices_tuple]
        upper_labels = label_image[upper_slices_tuple]

        differing_assignment_mask = lower_assigned_mask & (
            (~upper_assigned_mask) | (lower_labels != upper_labels)
        )
        if np.any(differing_assignment_mask):
            lower_surface_counts = np.bincount(
                lower_labels[differing_assignment_mask],
                minlength=region_count,
            )
            region_surface_face_counts += lower_surface_counts.astype(np.int64, copy=False)

        differing_assignment_mask = upper_assigned_mask & (
            (~lower_assigned_mask) | (lower_labels != upper_labels)
        )
        if np.any(differing_assignment_mask):
            upper_surface_counts = np.bincount(
                upper_labels[differing_assignment_mask],
                minlength=region_count,
            )
            region_surface_face_counts += upper_surface_counts.astype(np.int64, copy=False)

        shared_interface_mask = (
            lower_assigned_mask
            & upper_assigned_mask
            & (lower_labels >= 0)
            & (upper_labels >= 0)
            & (lower_labels != upper_labels)
        )
        if not np.any(shared_interface_mask):
            continue

        lower_interface_labels = lower_labels[shared_interface_mask]
        upper_interface_labels = upper_labels[shared_interface_mask]
        interface_indices = np.argwhere(shared_interface_mask).astype(float, copy=False)
        face_midpoint_indices = interface_indices.copy()
        face_midpoint_indices[:, axis_index] += 0.5

        for face_index in range(interface_indices.shape[0]):
            first_label = int(lower_interface_labels[face_index])
            second_label = int(upper_interface_labels[face_index])
            if first_label < second_label:
                ordered_pair = (first_label, second_label)
                orientation_sign = 1.0
            else:
                ordered_pair = (second_label, first_label)
                orientation_sign = -1.0

            pair_face_counts[ordered_pair] = pair_face_counts.get(ordered_pair, 0) + 1
            if ordered_pair not in pair_axis_face_balance:
                pair_axis_face_balance[ordered_pair] = np.zeros(mask.ndim, dtype=float)
            if ordered_pair not in pair_centroid_sums:
                pair_centroid_sums[ordered_pair] = np.zeros(mask.ndim, dtype=float)
            pair_axis_face_balance[ordered_pair][axis_index] += orientation_sign
            pair_centroid_sums[ordered_pair] += face_midpoint_indices[face_index]

    ordered_pairs = sorted(pair_face_counts)
    throat_region_pairs = np.asarray(ordered_pairs, dtype=np.int64)
    throat_count = len(ordered_pairs)
    throat_face_counts = np.zeros(throat_count, dtype=np.int64)
    throat_axis_face_balance = np.zeros((throat_count, mask.ndim), dtype=float)
    throat_centroid_indices = np.zeros((throat_count, mask.ndim), dtype=float)
    for throat_index, ordered_pair in enumerate(ordered_pairs):
        face_count = int(pair_face_counts[ordered_pair])
        throat_face_counts[throat_index] = face_count
        throat_axis_face_balance[throat_index] = pair_axis_face_balance[ordered_pair]
        throat_centroid_indices[throat_index] = pair_centroid_sums[ordered_pair] / max(
            face_count, 1
        )

    return MaximalBallRegionAdjacency(
        region_labels=region_labels,
        region_volume_voxels=region_volume_voxels,
        region_surface_face_counts=region_surface_face_counts,
        throat_region_pairs=throat_region_pairs,
        throat_face_counts=throat_face_counts,
        throat_axis_face_balance=throat_axis_face_balance,
        throat_centroid_indices=throat_centroid_indices,
        boundary_face_counts=boundary_face_counts,
    )


def extract_maximal_ball_regions(
    void_phase_mask: np.ndarray,
    *,
    distance_map_backend: str = "auto",
    settings: MaximalBallSettings | None = None,
    apply_boundary_clipping: bool = True,
) -> MaximalBallExtractionResult:
    """Run the staged native maximal-ball pipeline up to region adjacency.

    This is the current highest-level native extraction entry point that stays
    independent of PoreSpy network generation. It stops at voxel-region and
    interface geometry because the final pore/throat-to-`Network` assembly is
    still under active implementation.
    """

    candidates = extract_maximal_ball_candidates(
        void_phase_mask,
        distance_map_backend=distance_map_backend,
        settings=settings,
        apply_boundary_clipping=apply_boundary_clipping,
    )
    hierarchy = build_maximal_ball_hierarchy(candidates)
    voxel_regions = assign_voxel_regions_from_hierarchy(void_phase_mask, hierarchy)
    region_adjacency = measure_region_adjacency(void_phase_mask, voxel_regions)
    return MaximalBallExtractionResult(
        candidates=candidates,
        hierarchy=hierarchy,
        voxel_regions=voxel_regions,
        region_adjacency=region_adjacency,
    )


def build_network_dict_from_maximal_ball_regions(
    extraction_result: MaximalBallExtractionResult,
    *,
    voxel_size: float,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> dict[str, np.ndarray]:
    """Assemble a PoreSpy-style network mapping from maximal-ball regions.

    Parameters
    ----------
    extraction_result :
        Native maximal-ball extraction outputs through the region-adjacency
        stage.
    voxel_size :
        Physical edge length of one voxel.
    axis_names :
        Axis labels associated with the image dimensions. Only the first
        ``ndim`` entries are used.

    Notes
    -----
    This builder intentionally uses explicit, readable geometric rules rather
    than hidden heuristics:

    - pore coordinates are the root maximal-ball centers
    - pore volumes are the labeled region voxel counts
    - throat areas are the counted interface faces
    - throat centroids are the mean interface-face midpoints
    - conduit lengths are derived from pore-center to interface-centroid
      distances with a minimum half-voxel regularization

    These rules are scientifically reasonable and easy to audit, but they are
    not yet guaranteed to match the final Imperial `pnextract` geometry. They
    are the current native baseline that subsequent parity work can refine.
    """

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    hierarchy = extraction_result.hierarchy
    voxel_regions = extraction_result.voxel_regions
    region_adjacency = extraction_result.region_adjacency
    image_ndim = (
        int(hierarchy.center_indices.shape[1])
        if hierarchy.center_indices.size
        else int(voxel_regions.label_image.ndim)
    )
    if image_ndim not in {2, 3}:
        raise ValueError("maximal-ball network assembly supports only 2D or 3D images")
    if len(axis_names) < image_ndim:
        raise ValueError("axis_names must provide at least one label per image dimension")

    active_axis_names = axis_names[:image_ndim]
    region_count = int(region_adjacency.region_labels.size)
    root_center_indices = np.asarray(voxel_regions.root_center_indices, dtype=float)
    root_radii_voxels = np.asarray(voxel_regions.root_radii_voxels, dtype=float)

    if root_center_indices.shape[0] != region_count:
        raise ValueError("root_center_indices must align with region labels")
    if root_radii_voxels.shape != (region_count,):
        raise ValueError("root_radii_voxels must align with region labels")

    pore_coords = root_center_indices * float(voxel_size)
    if pore_coords.shape[1] == 2:
        pore_coords = np.column_stack([pore_coords, np.zeros(region_count, dtype=float)])
    pore_radius = root_radii_voxels * float(voxel_size)
    pore_area = np.pi * pore_radius**2
    pore_volume = (
        np.asarray(region_adjacency.region_volume_voxels, dtype=float) * float(voxel_size) ** 3
    )
    pore_surface_area = (
        np.asarray(region_adjacency.region_surface_face_counts, dtype=float)
        * float(voxel_size) ** 2
    )

    throat_region_pairs = np.asarray(region_adjacency.throat_region_pairs, dtype=np.int64)
    if throat_region_pairs.size == 0:
        throat_region_pairs = np.zeros((0, 2), dtype=np.int64)
    throat_count = int(throat_region_pairs.shape[0])
    throat_face_counts = np.asarray(region_adjacency.throat_face_counts, dtype=float)
    throat_area = throat_face_counts * float(voxel_size) ** 2
    throat_radius = np.sqrt(np.maximum(throat_area, 0.0) / np.pi)
    throat_centroid_indices = np.asarray(region_adjacency.throat_centroid_indices, dtype=float)
    if throat_centroid_indices.shape[0] != throat_count:
        raise ValueError("throat_centroid_indices must align with throat_region_pairs")
    throat_centroid_coords = throat_centroid_indices * float(voxel_size)
    if throat_centroid_coords.shape[1] == 2:
        throat_centroid_coords = np.column_stack(
            [throat_centroid_coords, np.zeros(throat_count, dtype=float)]
        )

    minimum_segment_length = 0.5 * float(voxel_size)
    if throat_count:
        first_pore_coordinates = pore_coords[throat_region_pairs[:, 0]]
        second_pore_coordinates = pore_coords[throat_region_pairs[:, 1]]
        pore1_to_interface_length = np.linalg.norm(
            throat_centroid_coords - first_pore_coordinates, axis=1
        )
        pore2_to_interface_length = np.linalg.norm(
            second_pore_coordinates - throat_centroid_coords, axis=1
        )
        pore1_length = np.maximum(
            pore1_to_interface_length - minimum_segment_length, minimum_segment_length
        )
        pore2_length = np.maximum(
            pore2_to_interface_length - minimum_segment_length, minimum_segment_length
        )
        direct_center_distance = np.linalg.norm(
            second_pore_coordinates - first_pore_coordinates, axis=1
        )
        core_length = np.maximum(
            direct_center_distance - pore1_length - pore2_length,
            minimum_segment_length,
        )
    else:
        pore1_length = np.zeros(0, dtype=float)
        pore2_length = np.zeros(0, dtype=float)
        core_length = np.zeros(0, dtype=float)
    throat_total_length = pore1_length + core_length + pore2_length
    throat_volume = throat_area * core_length

    network_dict: dict[str, np.ndarray] = {
        "pore.coords": pore_coords,
        "throat.conns": throat_region_pairs,
        "pore.radius_inscribed": pore_radius,
        "pore.area": pore_area,
        "pore.shape_factor": np.full(region_count, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "pore.volume": pore_volume,
        "pore.region_volume": pore_volume.copy(),
        "pore.surface_area": pore_surface_area,
        "throat.radius_inscribed": throat_radius,
        "throat.cross_sectional_area": throat_area,
        "throat.shape_factor": np.full(throat_count, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "throat.volume": throat_volume,
        "throat.total_length": throat_total_length,
        "throat.conduit_lengths.pore1": pore1_length,
        "throat.conduit_lengths.throat": core_length,
        "throat.conduit_lengths.pore2": pore2_length,
        "throat.centroid": throat_centroid_coords,
        "throat.face_count": throat_face_counts.astype(np.int64, copy=False),
        "throat.axis_face_balance": np.asarray(
            region_adjacency.throat_axis_face_balance, dtype=float
        ),
        "pore.boundary_face_count": np.asarray(
            region_adjacency.boundary_face_counts.sum(axis=1),
            dtype=np.int64,
        ),
    }

    pore_boundary = np.zeros(region_count, dtype=bool)
    for axis_index, axis_name in enumerate(active_axis_names):
        lower_contact = (
            np.asarray(region_adjacency.boundary_face_counts[:, 2 * axis_index], dtype=np.int64) > 0
        )
        upper_contact = (
            np.asarray(region_adjacency.boundary_face_counts[:, 2 * axis_index + 1], dtype=np.int64)
            > 0
        )
        network_dict[f"pore.inlet_{axis_name}min"] = lower_contact
        network_dict[f"pore.outlet_{axis_name}max"] = upper_contact
        pore_boundary |= lower_contact | upper_contact
    network_dict["pore.boundary"] = pore_boundary
    return network_dict


def extract_maximal_ball_network_dict(
    void_phase_mask: np.ndarray,
    *,
    voxel_size: float,
    distance_map_backend: str = "auto",
    settings: MaximalBallSettings | None = None,
    apply_boundary_clipping: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> MaximalBallNetworkDictResult:
    """Run the staged native maximal-ball path and assemble a network mapping."""

    extraction_result = extract_maximal_ball_regions(
        void_phase_mask,
        distance_map_backend=distance_map_backend,
        settings=settings,
        apply_boundary_clipping=apply_boundary_clipping,
    )
    network_dict = build_network_dict_from_maximal_ball_regions(
        extraction_result,
        voxel_size=voxel_size,
        axis_names=axis_names,
    )
    return MaximalBallNetworkDictResult(
        network_dict=network_dict,
        extraction=extraction_result,
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
    "MaximalBallExtractionResult",
    "MaximalBallHierarchy",
    "MaximalBallNetworkDictResult",
    "MaximalBallRegionAdjacency",
    "MaximalBallSettings",
    "MaximalBallVoxelRegions",
    "ResolvedMaximalBallSettings",
    "assign_voxel_regions_from_hierarchy",
    "build_network_dict_from_maximal_ball_regions",
    "build_maximal_ball_hierarchy",
    "clip_distance_map_to_domain_boundaries",
    "compute_void_distance_map",
    "extract_maximal_ball_network_dict",
    "extract_maximal_ball_regions",
    "extract_maximal_ball_candidates",
    "find_maximal_ball_candidates",
    "grow_root_regions_by_radius",
    "initialize_root_region_labels",
    "measure_region_adjacency",
    "resolve_maximal_ball_settings",
    "seed_root_region_ball_interiors",
    "suppress_overlapping_maximal_balls",
]
