from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    import edt as fast_edt  # type: ignore[import-not-found]
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
    radius_field_mode: str = "half_voxel"
    candidate_selection_mode: str = "threshold_all"


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
    radius_field_mode: str
    candidate_selection_mode: str


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

        return cast(np.ndarray, self.center_indices[self.retained_mask])

    @property
    def retained_radii_voxels(self) -> np.ndarray:
        """Return retained maximal-ball radii in descending-radius order."""

        return cast(np.ndarray, self.radii_voxels[self.retained_mask])


@dataclass(slots=True)
class MaximalBallHierarchy:
    """Parent-child hierarchy over retained maximal-ball candidates.

    The hierarchy is stored on the retained-ball order of
    :class:`MaximalBallCandidates`, which is already sorted by descending
    radius. Each ball points either to itself, if it is a root/master ball, or
    to the index of its parent ball in the same retained ordering.
    """

    center_indices: np.ndarray
    center_coordinates: np.ndarray
    radii_voxels: np.ndarray
    parent_indices: np.ndarray
    master_indices: np.ndarray
    hierarchy_levels: np.ndarray
    distance_map: np.ndarray
    settings: ResolvedMaximalBallSettings

    @property
    def root_mask(self) -> np.ndarray:
        """Return a boolean mask of root/master balls."""

        return cast(
            np.ndarray,
            self.parent_indices == np.arange(self.parent_indices.size, dtype=np.int64),
        )


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
    throat_max_touch_radius_side1_voxels: np.ndarray
    throat_max_touch_radius_side2_voxels: np.ndarray
    throat_max_touch_index_side1: np.ndarray
    throat_max_touch_index_side2: np.ndarray
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


@dataclass(slots=True)
class MaximalBallExtractionDiagnostics:
    """Compact diagnostics for step-by-step maximal-ball extraction comparison."""

    retained_ball_count: int
    root_region_count: int
    occupied_region_count: int
    assigned_void_fraction: float
    unassigned_void_voxel_count: int
    zero_throat_region_count: int
    internal_zero_throat_region_count: int
    boundary_zero_throat_region_count: int
    throat_touch_radius_side1_mean_voxels: float
    throat_touch_radius_side2_mean_voxels: float
    throat_refined_support_radius_side1_mean_voxels: float
    throat_refined_support_radius_side2_mean_voxels: float


_CIRCULAR_SHAPE_FACTOR = 1.0 / (4.0 * np.pi)
_RADIUS_SUPPORT_MODE_ANY = 0
_RADIUS_SUPPORT_MODE_STRICTLY_LARGER = 1
_RADIUS_SUPPORT_MODE_GREATER_OR_EQUAL = 2


def _resolve_edt_parallel_threads(edt_parallel_threads: int | None) -> int:
    """Resolve the worker count for the optional `edt` distance transform."""

    if edt_parallel_threads is not None:
        resolved_threads = int(edt_parallel_threads)
        if resolved_threads < 1:
            raise ValueError("edt_parallel_threads must be a positive integer")
        return resolved_threads

    environment_value = os.getenv("VOIDS_EDT_THREADS")
    if environment_value:
        try:
            resolved_threads = int(environment_value)
        except ValueError as exc:  # pragma: no cover - invalid shell environment
            raise ValueError("VOIDS_EDT_THREADS must be a positive integer") from exc
        if resolved_threads < 1:
            raise ValueError("VOIDS_EDT_THREADS must be a positive integer")
        return resolved_threads

    return 1


def compute_void_distance_map(
    void_phase_mask: np.ndarray,
    *,
    backend: str = "auto",
    edt_parallel_threads: int | None = None,
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
    edt_parallel_threads :
        Number of worker threads to use when the optional `edt` backend is
        active. When omitted, `voids` first checks ``VOIDS_EDT_THREADS`` and
        otherwise uses one thread for stable default behavior.
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
        return np.asarray(
            fast_edt.edt(
                mask,
                black_border=True,
                parallel=_resolve_edt_parallel_threads(edt_parallel_threads),
            ),
            dtype=float,
        )
    return np.asarray(ndi.distance_transform_edt(mask), dtype=float)


def compute_maximal_ball_radius_field(
    void_phase_mask: np.ndarray,
    *,
    backend: str = "auto",
    edt_parallel_threads: int | None = None,
    mode: str = "half_voxel",
) -> np.ndarray:
    """Compute the radius field used by the maximal-ball extractor.

    Parameters
    ----------
    void_phase_mask :
        Boolean array where ``True`` marks void voxels.
    backend :
        Distance-transform backend passed through to
        :func:`compute_void_distance_map`.
    edt_parallel_threads :
        Number of worker threads to use when the optional `edt` backend is
        active.
    mode :
        Radius-field convention. ``"half_voxel"`` returns the nearest non-void
        Euclidean distance minus half a voxel. ``"edt"`` returns the plain
        Euclidean distance map unchanged. ``"imperial_pnextract"`` is accepted
        as a backward-compatible alias for ``"half_voxel"``.
    """

    normalized_mode = _normalize_radius_field_mode(mode)

    distance_map = compute_void_distance_map(
        void_phase_mask,
        backend=backend,
        edt_parallel_threads=edt_parallel_threads,
    )
    if normalized_mode == "edt":
        return distance_map
    radius_field = np.asarray(distance_map, dtype=float) - 0.5
    return np.where(radius_field > 0.0, radius_field, 0.0)


def smooth_radius_field_local_relaxation(
    radius_field: np.ndarray,
    void_phase_mask: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    """Smooth a radius field with a compact local relaxation stencil."""

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    mask = np.asarray(void_phase_mask, dtype=bool)
    smoothed_radius = np.asarray(radius_field, dtype=float).copy()
    if smoothed_radius.shape != mask.shape:
        raise ValueError("radius_field and void_phase_mask must have the same shape")
    if smoothed_radius.ndim not in {2, 3}:
        raise ValueError("radius_field must be a 2D or 3D array")
    if iterations == 0:
        return smoothed_radius

    neighborhood_kernel = np.ones((3,) * smoothed_radius.ndim, dtype=float)
    mask_float = mask.astype(float, copy=False)
    for _ in range(iterations):
        neighbor_count = ndi.convolve(
            mask_float,
            neighborhood_kernel,
            mode="constant",
            cval=0.0,
        )
        radius_sum = ndi.convolve(
            np.where(mask, smoothed_radius, 0.0),
            neighborhood_kernel,
            mode="constant",
            cval=0.0,
        )
        radius_delta = np.zeros_like(smoothed_radius, dtype=float)
        radius_delta[mask] = (
            4.0 * radius_sum[mask] / (3.0 * neighbor_count[mask] + 27.0) - smoothed_radius[mask]
        )
        radius_delta_sum = ndi.convolve(
            np.where(mask, radius_delta, 0.0),
            neighborhood_kernel,
            mode="constant",
            cval=0.0,
        )
        local_update = np.zeros_like(smoothed_radius, dtype=float)
        local_update[mask] = 0.02 * (
            radius_delta[mask] - 1.98 * radius_delta_sum[mask] / (neighbor_count[mask] + 27.0)
        )
        smoothed_radius[mask] += np.clip(local_update[mask], -0.005, 0.01)
    return smoothed_radius


def _normalize_radius_field_mode(mode: str) -> str:
    """Normalize public radius-field mode names and legacy aliases."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"half_voxel", "half-voxel", "imperial_pnextract"}:
        return "half_voxel"
    if normalized_mode == "edt":
        return "edt"
    raise ValueError("radius field mode must be one of {'half_voxel', 'edt'}")


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
    radius_field_mode = _normalize_radius_field_mode(raw_settings.radius_field_mode)
    candidate_selection_mode = str(raw_settings.candidate_selection_mode).strip().lower()
    if candidate_selection_mode not in {"threshold_all", "local_maxima"}:
        raise ValueError(
            "candidate_selection_mode must be one of {'threshold_all', 'local_maxima'}"
        )

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
        radius_field_mode=radius_field_mode,
        candidate_selection_mode=candidate_selection_mode,
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
    selection_mode: str = "local_maxima",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find maximal-ball candidates from a radius field."""

    if minimal_radius_voxels <= 0.0:
        raise ValueError("minimal_radius_voxels must be positive")

    working_distance_map = np.asarray(distance_map, dtype=float)
    if working_distance_map.ndim not in {2, 3}:
        raise ValueError("distance_map must be a 2D or 3D array")

    normalized_selection_mode = str(selection_mode).strip().lower()
    if normalized_selection_mode not in {"threshold_all", "local_maxima"}:
        raise ValueError("selection_mode must be one of {'threshold_all', 'local_maxima'}")

    candidate_mask = (working_distance_map > 0.0) & (working_distance_map >= minimal_radius_voxels)
    if normalized_selection_mode == "local_maxima":
        if footprint is None:
            footprint = np.ones((3,) * working_distance_map.ndim, dtype=bool)
        local_maxima = ndi.maximum_filter(
            working_distance_map,
            footprint=footprint,
            mode="nearest",
        )
        candidate_mask &= np.isclose(working_distance_map, local_maxima)

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
    return np.asarray(
        _suppress_overlapping_maximal_balls_compiled(
            centers=centers,
            radii=radii,
            retention_radius_factor=float(settings.retention_radius_factor),
            retention_radius_offset_voxels=float(settings.retention_radius_offset_voxels),
            medial_surface_noise_voxels=float(settings.medial_surface_noise_voxels),
        ),
        dtype=bool,
    )


@njit(cache=True)
def _suppress_overlapping_maximal_balls_compiled(
    centers: np.ndarray,
    radii: np.ndarray,
    retention_radius_factor: float,
    retention_radius_offset_voxels: float,
    medial_surface_noise_voxels: float,
) -> np.ndarray:
    """Compiled overlap suppression over descending-radius maximal-ball candidates."""

    candidate_count = radii.shape[0]
    retained_mask = np.zeros(candidate_count, dtype=np.bool_)
    retained_centers = np.zeros((candidate_count, centers.shape[1]), dtype=np.float64)
    retained_radii = np.zeros(candidate_count, dtype=np.float64)
    retained_count = 0

    for candidate_index in range(candidate_count):
        candidate_radius = radii[candidate_index]
        should_retain = True

        for retained_index in range(retained_count):
            squared_distance = 0.0
            for axis_index in range(centers.shape[1]):
                offset = (
                    float(centers[candidate_index, axis_index])
                    - retained_centers[retained_index, axis_index]
                )
                squared_distance += offset * offset
            center_distance = np.sqrt(squared_distance)

            if center_distance < (
                retention_radius_factor * retained_radii[retained_index]
                + retention_radius_offset_voxels
            ):
                should_retain = False
                break
            if center_distance + candidate_radius < (
                retained_radii[retained_index] + medial_surface_noise_voxels
            ):
                should_retain = False
                break

        if not should_retain:
            continue

        retained_mask[candidate_index] = True
        for axis_index in range(centers.shape[1]):
            retained_centers[retained_count, axis_index] = float(
                centers[candidate_index, axis_index]
            )
        retained_radii[retained_count] = candidate_radius
        retained_count += 1

    return retained_mask


def _sample_radius_at_integer_index(
    distance_map: np.ndarray,
    integer_index: np.ndarray,
) -> float:
    """Return the radius value at one integer voxel index."""

    return float(distance_map[tuple(int(value) for value in integer_index)])


def _refine_ball_center_subvoxel(
    distance_map: np.ndarray,
    integer_center_index: np.ndarray,
    *,
    displacement_limit: float,
    radius_gain_factor: float,
) -> tuple[np.ndarray, float]:
    """Apply the Imperial-style subvoxel uphill interpolation in one voxel."""

    image_shape = np.asarray(distance_map.shape, dtype=np.int64)
    base_index = np.asarray(integer_center_index, dtype=np.int64)
    base_radius = _sample_radius_at_integer_index(distance_map, base_index)
    displacement = np.zeros(base_index.size, dtype=float)

    for axis_index in range(base_index.size):
        lower_index = base_index.copy()
        lower_index[axis_index] -= 1
        upper_index = base_index.copy()
        upper_index[axis_index] += 1
        if np.any(lower_index < 0) or np.any(upper_index >= image_shape):
            continue
        lower_radius = _sample_radius_at_integer_index(distance_map, lower_index)
        upper_radius = _sample_radius_at_integer_index(distance_map, upper_index)
        gradient_plus = upper_radius - base_radius
        gradient_minus = base_radius - lower_radius
        if abs(gradient_plus - gradient_minus) <= 0.01:
            continue
        displacement[axis_index] = np.clip(
            -0.5 * (gradient_plus + gradient_minus) / (gradient_plus - gradient_minus),
            -displacement_limit,
            displacement_limit,
        )

    refined_center_coordinate = base_index.astype(float) - 0.5 + displacement
    refined_radius = base_radius + radius_gain_factor * float(np.linalg.norm(displacement))
    return refined_center_coordinate, refined_radius


def _refine_ball_center_relocation(
    distance_map: np.ndarray,
    integer_center_index: np.ndarray,
    occupied_center_lookup: set[tuple[int, ...]],
) -> tuple[np.ndarray, float]:
    """Apply the Imperial uphill relocation step to a retained ball."""

    image_shape = np.asarray(distance_map.shape, dtype=np.int64)
    base_index = np.asarray(integer_center_index, dtype=np.int64)
    base_radius = _sample_radius_at_integer_index(distance_map, base_index)
    displacement = np.zeros(base_index.size, dtype=float)
    gradient = np.zeros(base_index.size, dtype=float)

    for axis_index in range(base_index.size):
        lower_index = base_index.copy()
        lower_index[axis_index] -= 1
        upper_index = base_index.copy()
        upper_index[axis_index] += 1
        if np.any(lower_index < 0) or np.any(upper_index >= image_shape):
            continue
        lower_radius = _sample_radius_at_integer_index(distance_map, lower_index)
        upper_radius = _sample_radius_at_integer_index(distance_map, upper_index)
        gradient_plus = upper_radius - base_radius
        gradient_minus = base_radius - lower_radius
        gradient[axis_index] = 0.5 * (gradient_plus + gradient_minus)
        if abs(gradient_plus - gradient_minus) <= 0.01:
            continue
        displacement[axis_index] = np.clip(
            -0.5 * (gradient_plus + gradient_minus) / (gradient_plus - gradient_minus),
            -0.59,
            0.59,
        )

    displacement += 1.4 * gradient
    displacement_norm = float(np.linalg.norm(displacement))
    if displacement_norm <= 1.0e-12:
        return base_index, base_radius
    displacement /= 0.55 * displacement_norm + 0.05

    target_index = np.rint(base_index.astype(float) - 0.5 + displacement).astype(np.int64)
    if np.any(target_index < 0) or np.any(target_index >= image_shape):
        return base_index, base_radius
    target_index_tuple = tuple(int(value) for value in target_index)
    if target_index_tuple in occupied_center_lookup and not np.array_equal(
        target_index, base_index
    ):
        return base_index, base_radius

    target_radius = _sample_radius_at_integer_index(distance_map, target_index)
    if target_radius <= base_radius:
        return base_index, base_radius
    return target_index, target_radius


def refine_retained_ball_coordinates(
    maximal_ball_candidates: MaximalBallCandidates,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Imperial-style uphill refinements to retained maximal balls.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Refined integer voxel indices, refined floating-point center
        coordinates, and refined radii.
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
    if retained_count == 0:
        ndim = maximal_ball_candidates.center_indices.shape[1]
        return (
            np.zeros((0, ndim), dtype=np.int64),
            np.zeros((0, ndim), dtype=float),
            np.zeros(0, dtype=float),
        )

    distance_map = np.asarray(maximal_ball_candidates.distance_map, dtype=float)
    refined_integer_indices = retained_center_indices.copy()
    refined_center_coordinates = retained_center_indices.astype(float) - 0.5
    refined_radii_voxels = retained_radii_voxels.copy()

    for ball_index in range(retained_count):
        refined_center_coordinates[ball_index], refined_radii_voxels[ball_index] = (
            _refine_ball_center_subvoxel(
                distance_map,
                refined_integer_indices[ball_index],
                displacement_limit=0.49,
                radius_gain_factor=0.95,
            )
        )

    occupied_center_lookup = {
        tuple(int(value) for value in integer_index) for integer_index in refined_integer_indices
    }
    for ball_index in range(retained_count):
        original_index_tuple = tuple(int(value) for value in refined_integer_indices[ball_index])
        occupied_center_lookup.discard(original_index_tuple)
        relocated_index, relocated_radius = _refine_ball_center_relocation(
            distance_map,
            refined_integer_indices[ball_index],
            occupied_center_lookup,
        )
        refined_integer_indices[ball_index] = relocated_index
        occupied_center_lookup.add(tuple(int(value) for value in relocated_index))
        refined_radii_voxels[ball_index] = relocated_radius

    for ball_index in range(retained_count):
        refined_center_coordinates[ball_index], refined_radii_voxels[ball_index] = (
            _refine_ball_center_subvoxel(
                distance_map,
                refined_integer_indices[ball_index],
                displacement_limit=0.49,
                radius_gain_factor=0.95,
            )
        )

    return refined_integer_indices, refined_center_coordinates, refined_radii_voxels


def refine_ball_from_seed_index(
    distance_map: np.ndarray,
    seed_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine one voxel-centered ball seed with the Imperial uphill sequence."""

    working_distance_map = np.asarray(distance_map, dtype=float)
    integer_seed_index = np.asarray(seed_index, dtype=np.int64)
    refined_center_coordinate, refined_radius_voxels = _refine_ball_center_subvoxel(
        working_distance_map,
        integer_seed_index,
        displacement_limit=0.49,
        radius_gain_factor=0.95,
    )
    relocated_index, relocated_radius_voxels = _refine_ball_center_relocation(
        working_distance_map,
        integer_seed_index,
        occupied_center_lookup=set(),
    )
    refined_center_coordinate, refined_radius_voxels = _refine_ball_center_subvoxel(
        working_distance_map,
        relocated_index,
        displacement_limit=0.49,
        radius_gain_factor=0.95,
    )
    return (
        relocated_index,
        refined_center_coordinate,
        max(
            float(refined_radius_voxels),
            float(relocated_radius_voxels),
        ),
    )


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
    first_center_coordinate: np.ndarray,
    first_radius_voxels: float,
    second_center_coordinate: np.ndarray,
    second_radius_voxels: float,
    *,
    distance_map: np.ndarray,
    settings: ResolvedMaximalBallSettings,
) -> bool:
    """Return whether two balls satisfy the Imperial midpoint support test."""

    midpoint_index = _weighted_midpoint_index(
        first_center_coordinate,
        first_radius_voxels,
        second_center_coordinate,
        second_radius_voxels,
        image_shape=distance_map.shape,
    )
    midpoint_radius_voxels = float(distance_map[midpoint_index])
    smaller_radius_voxels = min(first_radius_voxels, second_radius_voxels)
    center_distance_voxels = float(
        np.linalg.norm(
            first_center_coordinate.astype(float) - second_center_coordinate.astype(float)
        )
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

    retained_center_indices, retained_center_coordinates, retained_radii_voxels = (
        refine_retained_ball_coordinates(maximal_ball_candidates)
    )
    retained_count = retained_radii_voxels.size
    parent_indices = np.arange(retained_count, dtype=np.int64)
    if retained_count == 0:
        return MaximalBallHierarchy(
            center_indices=retained_center_indices,
            center_coordinates=retained_center_coordinates,
            radii_voxels=retained_radii_voxels,
            parent_indices=parent_indices,
            master_indices=parent_indices.copy(),
            hierarchy_levels=np.zeros(0, dtype=np.int64),
            distance_map=np.asarray(maximal_ball_candidates.distance_map, dtype=float),
            settings=maximal_ball_candidates.settings,
        )

    settings = maximal_ball_candidates.settings
    center_tree = cKDTree(retained_center_coordinates.astype(float))
    distance_map = np.asarray(maximal_ball_candidates.distance_map, dtype=float)

    for first_ball_index in range(retained_count):
        first_center_coordinate = retained_center_coordinates[first_ball_index]
        first_radius_voxels = float(retained_radii_voxels[first_ball_index])
        neighbor_search_radius_voxels = (
            settings.hierarchy_length_factor * first_radius_voxels
            + 2.0 * settings.medial_surface_noise_voxels
            + 2.0
        )
        nearby_ball_indices = center_tree.query_ball_point(
            first_center_coordinate.astype(float),
            r=neighbor_search_radius_voxels,
        )
        for second_ball_index in nearby_ball_indices:
            if second_ball_index <= first_ball_index:
                continue
            second_center_coordinate = retained_center_coordinates[second_ball_index]
            second_radius_voxels = float(retained_radii_voxels[second_ball_index])
            if not _pair_has_supported_midpoint(
                first_center_coordinate,
                first_radius_voxels,
                second_center_coordinate,
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
            first_root_center_index = retained_center_coordinates[first_root_index]
            second_root_center_index = retained_center_coordinates[second_root_index]
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
        center_coordinates=retained_center_coordinates,
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


def _neighbor_offsets_with_growth_priority(ndim: int) -> list[tuple[int, ...]]:
    """Return Imperial-like axis sweep order for neighbor inspection."""

    offsets: list[tuple[int, ...]] = []
    for axis_index in range(ndim):
        positive_offset = [0] * ndim
        negative_offset = [0] * ndim
        positive_offset[axis_index] = 1
        negative_offset[axis_index] = -1
        offsets.append(tuple(positive_offset))
        offsets.append(tuple(negative_offset))
    return offsets


def _normalize_radius_support_mode(
    *,
    radius_support_mode: str | None,
    require_strictly_larger_radius: bool | None,
) -> str:
    """Normalize the local radius-support rule for voxel growth."""

    if radius_support_mode is None:
        if require_strictly_larger_radius is True:
            return "strictly_larger"
        return "greater_or_equal"

    normalized_mode = str(radius_support_mode).strip().lower()
    valid_modes = {"strictly_larger", "greater_or_equal", "any"}
    if normalized_mode not in valid_modes:
        raise ValueError(
            "radius_support_mode must be one of {'strictly_larger', 'greater_or_equal', 'any'}"
        )
    return normalized_mode


def _neighbor_satisfies_radius_support(  # pragma: no cover
    neighbor_radius: float,
    current_radius: float,
    *,
    radius_support_mode: str,
) -> bool:
    """Return whether a neighboring voxel may support assignment."""

    if radius_support_mode == "any":
        return True
    if radius_support_mode == "strictly_larger":
        return neighbor_radius > current_radius
    if radius_support_mode == "greater_or_equal":
        return neighbor_radius >= current_radius
    raise ValueError(f"Unsupported radius_support_mode={radius_support_mode!r}")


def _encode_radius_support_mode(radius_support_mode: str) -> int:
    """Encode a normalized radius-support mode for compiled kernels."""

    if radius_support_mode == "any":
        return _RADIUS_SUPPORT_MODE_ANY
    if radius_support_mode == "strictly_larger":
        return _RADIUS_SUPPORT_MODE_STRICTLY_LARGER
    if radius_support_mode == "greater_or_equal":
        return _RADIUS_SUPPORT_MODE_GREATER_OR_EQUAL
    raise ValueError(f"Unsupported radius_support_mode={radius_support_mode!r}")


@njit(cache=True)
def _compiled_neighbor_satisfies_radius_support(
    neighbor_radius: float,
    current_radius: float,
    radius_support_mode_code: int,
) -> bool:  # pragma: no cover
    """Return whether a neighboring voxel may support assignment."""

    if radius_support_mode_code == _RADIUS_SUPPORT_MODE_ANY:
        return True
    if radius_support_mode_code == _RADIUS_SUPPORT_MODE_STRICTLY_LARGER:
        return neighbor_radius > current_radius
    return neighbor_radius >= current_radius


@njit(cache=True)
def _compiled_accumulate_label_count(
    supporting_labels: np.ndarray,
    supporting_counts: np.ndarray,
    supporting_label_count: int,
    neighbor_label: int,
) -> int:  # pragma: no cover
    """Accumulate one label count into fixed-size support arrays."""

    for label_index in range(supporting_label_count):
        if supporting_labels[label_index] == neighbor_label:
            supporting_counts[label_index] += 1
            return supporting_label_count
    supporting_labels[supporting_label_count] = neighbor_label
    supporting_counts[supporting_label_count] = 1
    return supporting_label_count + 1


@njit(cache=True)
def _compiled_select_best_label(
    supporting_labels: np.ndarray,
    supporting_counts: np.ndarray,
    supporting_label_count: int,
) -> tuple[int, int]:  # pragma: no cover
    """Select the strongest label with the same tie-break as the Python path."""

    best_label = -1
    best_support = 0
    for label_index in range(supporting_label_count):
        candidate_label = int(supporting_labels[label_index])
        candidate_support = int(supporting_counts[label_index])
        if candidate_support > best_support or (
            candidate_support == best_support and (best_label < 0 or candidate_label < best_label)
        ):
            best_label = candidate_label
            best_support = candidate_support
    return best_label, best_support


@njit(cache=True)
def _grow_root_regions_by_radius_compiled_2d(
    previous_labels_flat: np.ndarray,
    label_image_flat: np.ndarray,
    working_distance_map_flat: np.ndarray,
    void_indices: np.ndarray,
    *,
    shape0: int,
    shape1: int,
    unassigned_label: int,
    minimum_supporting_neighbors: int,
    radius_support_mode_code: int,
) -> bool:  # pragma: no cover
    """Compiled 2D radius-aware voxel growth."""

    changed_any_voxel = False
    for voxel_row in range(void_indices.shape[0]):
        index0 = int(void_indices[voxel_row, 0])
        index1 = int(void_indices[voxel_row, 1])
        linear_index = index0 * shape1 + index1
        if int(previous_labels_flat[linear_index]) != unassigned_label:
            continue

        current_radius = float(working_distance_map_flat[linear_index])
        supporting_labels = np.empty(4, dtype=np.int64)
        supporting_counts = np.zeros(4, dtype=np.int64)
        supporting_label_count = 0

        for axis_index in range(2):
            for direction in (-1, 1):
                neighbor0 = index0
                neighbor1 = index1
                if axis_index == 0:
                    neighbor0 += direction
                    if neighbor0 < 0 or neighbor0 >= shape0:
                        continue
                else:
                    neighbor1 += direction
                    if neighbor1 < 0 or neighbor1 >= shape1:
                        continue

                neighbor_linear_index = neighbor0 * shape1 + neighbor1
                neighbor_label = int(previous_labels_flat[neighbor_linear_index])
                if neighbor_label < 0:
                    continue
                neighbor_radius = float(working_distance_map_flat[neighbor_linear_index])
                if not _compiled_neighbor_satisfies_radius_support(
                    neighbor_radius,
                    current_radius,
                    radius_support_mode_code,
                ):
                    continue
                supporting_label_count = _compiled_accumulate_label_count(
                    supporting_labels,
                    supporting_counts,
                    supporting_label_count,
                    neighbor_label,
                )

        if supporting_label_count == 0:
            continue

        best_label, best_support = _compiled_select_best_label(
            supporting_labels,
            supporting_counts,
            supporting_label_count,
        )
        if best_support >= minimum_supporting_neighbors:
            label_image_flat[linear_index] = best_label
            changed_any_voxel = True

    return changed_any_voxel


@njit(cache=True)
def _grow_root_regions_by_radius_compiled_3d(
    previous_labels_flat: np.ndarray,
    label_image_flat: np.ndarray,
    working_distance_map_flat: np.ndarray,
    void_indices: np.ndarray,
    *,
    shape0: int,
    shape1: int,
    shape2: int,
    unassigned_label: int,
    minimum_supporting_neighbors: int,
    radius_support_mode_code: int,
) -> bool:  # pragma: no cover
    """Compiled 3D radius-aware voxel growth."""

    changed_any_voxel = False
    plane_size = shape1 * shape2
    for voxel_row in range(void_indices.shape[0]):
        index0 = int(void_indices[voxel_row, 0])
        index1 = int(void_indices[voxel_row, 1])
        index2 = int(void_indices[voxel_row, 2])
        linear_index = index0 * plane_size + index1 * shape2 + index2
        if int(previous_labels_flat[linear_index]) != unassigned_label:
            continue

        current_radius = float(working_distance_map_flat[linear_index])
        supporting_labels = np.empty(6, dtype=np.int64)
        supporting_counts = np.zeros(6, dtype=np.int64)
        supporting_label_count = 0

        for axis_index in range(3):
            for direction in (-1, 1):
                neighbor0 = index0
                neighbor1 = index1
                neighbor2 = index2
                if axis_index == 0:
                    neighbor0 += direction
                    if neighbor0 < 0 or neighbor0 >= shape0:
                        continue
                elif axis_index == 1:
                    neighbor1 += direction
                    if neighbor1 < 0 or neighbor1 >= shape1:
                        continue
                else:
                    neighbor2 += direction
                    if neighbor2 < 0 or neighbor2 >= shape2:
                        continue

                neighbor_linear_index = neighbor0 * plane_size + neighbor1 * shape2 + neighbor2
                neighbor_label = int(previous_labels_flat[neighbor_linear_index])
                if neighbor_label < 0:
                    continue
                neighbor_radius = float(working_distance_map_flat[neighbor_linear_index])
                if not _compiled_neighbor_satisfies_radius_support(
                    neighbor_radius,
                    current_radius,
                    radius_support_mode_code,
                ):
                    continue
                supporting_label_count = _compiled_accumulate_label_count(
                    supporting_labels,
                    supporting_counts,
                    supporting_label_count,
                    neighbor_label,
                )

        if supporting_label_count == 0:
            continue

        best_label, best_support = _compiled_select_best_label(
            supporting_labels,
            supporting_counts,
            supporting_label_count,
        )
        if best_support >= minimum_supporting_neighbors:
            label_image_flat[linear_index] = best_label
            changed_any_voxel = True

    return changed_any_voxel


@njit(cache=True)
def _reassign_region_boundary_voxels_by_majority_compiled_2d(
    previous_labels_flat: np.ndarray,
    label_image_flat: np.ndarray,
    working_distance_map_flat: np.ndarray,
    void_indices: np.ndarray,
    *,
    shape0: int,
    shape1: int,
    radius_support_mode_code: int,
) -> bool:  # pragma: no cover
    """Compiled 2D weak-boundary majority reassignment."""

    changed_any_voxel = False
    for voxel_row in range(void_indices.shape[0]):
        index0 = int(void_indices[voxel_row, 0])
        index1 = int(void_indices[voxel_row, 1])
        linear_index = index0 * shape1 + index1
        current_label = int(previous_labels_flat[linear_index])
        if current_label < 0:
            continue

        current_radius = float(working_distance_map_flat[linear_index])
        same_label_neighbor_count = 0
        different_label_neighbor_count = 0
        supporting_labels = np.empty(4, dtype=np.int64)
        supporting_counts = np.zeros(4, dtype=np.int64)
        supporting_label_count = 0

        for axis_index in range(2):
            for direction in (-1, 1):
                neighbor0 = index0
                neighbor1 = index1
                if axis_index == 0:
                    neighbor0 += direction
                    if neighbor0 < 0 or neighbor0 >= shape0:
                        continue
                else:
                    neighbor1 += direction
                    if neighbor1 < 0 or neighbor1 >= shape1:
                        continue

                neighbor_linear_index = neighbor0 * shape1 + neighbor1
                neighbor_label = int(previous_labels_flat[neighbor_linear_index])
                if neighbor_label < 0:
                    continue
                if neighbor_label == current_label:
                    same_label_neighbor_count += 1
                    continue

                different_label_neighbor_count += 1
                neighbor_radius = float(working_distance_map_flat[neighbor_linear_index])
                if not _compiled_neighbor_satisfies_radius_support(
                    neighbor_radius,
                    current_radius,
                    radius_support_mode_code,
                ):
                    continue
                supporting_label_count = _compiled_accumulate_label_count(
                    supporting_labels,
                    supporting_counts,
                    supporting_label_count,
                    neighbor_label,
                )

        if different_label_neighbor_count <= same_label_neighbor_count:
            continue
        if supporting_label_count == 0:
            continue

        best_label, best_support = _compiled_select_best_label(
            supporting_labels,
            supporting_counts,
            supporting_label_count,
        )
        if best_support > same_label_neighbor_count:
            label_image_flat[linear_index] = best_label
            changed_any_voxel = True

    return changed_any_voxel


@njit(cache=True)
def _reassign_region_boundary_voxels_by_majority_compiled_3d(
    previous_labels_flat: np.ndarray,
    label_image_flat: np.ndarray,
    working_distance_map_flat: np.ndarray,
    void_indices: np.ndarray,
    *,
    shape0: int,
    shape1: int,
    shape2: int,
    radius_support_mode_code: int,
) -> bool:  # pragma: no cover
    """Compiled 3D weak-boundary majority reassignment."""

    changed_any_voxel = False
    plane_size = shape1 * shape2
    for voxel_row in range(void_indices.shape[0]):
        index0 = int(void_indices[voxel_row, 0])
        index1 = int(void_indices[voxel_row, 1])
        index2 = int(void_indices[voxel_row, 2])
        linear_index = index0 * plane_size + index1 * shape2 + index2
        current_label = int(previous_labels_flat[linear_index])
        if current_label < 0:
            continue

        current_radius = float(working_distance_map_flat[linear_index])
        same_label_neighbor_count = 0
        different_label_neighbor_count = 0
        supporting_labels = np.empty(6, dtype=np.int64)
        supporting_counts = np.zeros(6, dtype=np.int64)
        supporting_label_count = 0

        for axis_index in range(3):
            for direction in (-1, 1):
                neighbor0 = index0
                neighbor1 = index1
                neighbor2 = index2
                if axis_index == 0:
                    neighbor0 += direction
                    if neighbor0 < 0 or neighbor0 >= shape0:
                        continue
                elif axis_index == 1:
                    neighbor1 += direction
                    if neighbor1 < 0 or neighbor1 >= shape1:
                        continue
                else:
                    neighbor2 += direction
                    if neighbor2 < 0 or neighbor2 >= shape2:
                        continue

                neighbor_linear_index = neighbor0 * plane_size + neighbor1 * shape2 + neighbor2
                neighbor_label = int(previous_labels_flat[neighbor_linear_index])
                if neighbor_label < 0:
                    continue
                if neighbor_label == current_label:
                    same_label_neighbor_count += 1
                    continue

                different_label_neighbor_count += 1
                neighbor_radius = float(working_distance_map_flat[neighbor_linear_index])
                if not _compiled_neighbor_satisfies_radius_support(
                    neighbor_radius,
                    current_radius,
                    radius_support_mode_code,
                ):
                    continue
                supporting_label_count = _compiled_accumulate_label_count(
                    supporting_labels,
                    supporting_counts,
                    supporting_label_count,
                    neighbor_label,
                )

        if different_label_neighbor_count <= same_label_neighbor_count:
            continue
        if supporting_label_count == 0:
            continue

        best_label, best_support = _compiled_select_best_label(
            supporting_labels,
            supporting_counts,
            supporting_label_count,
        )
        if best_support > same_label_neighbor_count:
            label_image_flat[linear_index] = best_label
            changed_any_voxel = True

    return changed_any_voxel


def _count_supporting_neighbor_labels(  # pragma: no cover
    previous_labels: np.ndarray,
    working_distance_map: np.ndarray,
    voxel_index: np.ndarray,
    *,
    image_shape: np.ndarray,
    neighbor_offsets: list[tuple[int, ...]],
    current_label: int | None,
    current_radius: float,
    radius_support_mode: str,
) -> dict[int, int]:
    """Count neighboring region labels that support a local reassignment."""

    supporting_label_counts: dict[int, int] = {}
    for neighbor_offset in neighbor_offsets:
        neighbor_index = voxel_index + np.asarray(neighbor_offset, dtype=np.int64)
        if np.any(neighbor_index < 0) or np.any(neighbor_index >= image_shape):
            continue
        neighbor_index_tuple = tuple(int(value) for value in neighbor_index)
        neighbor_label = int(previous_labels[neighbor_index_tuple])
        if neighbor_label < 0 or neighbor_label == current_label:
            continue
        neighbor_radius = float(working_distance_map[neighbor_index_tuple])
        if not _neighbor_satisfies_radius_support(
            neighbor_radius,
            current_radius,
            radius_support_mode=radius_support_mode,
        ):
            continue
        supporting_label_counts[neighbor_label] = supporting_label_counts.get(neighbor_label, 0) + 1
    return supporting_label_counts


def grow_root_regions_by_radius(
    void_phase_mask: np.ndarray,
    distance_map: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
    *,
    minimum_supporting_neighbors: int,
    radius_support_mode: str | None = None,
    require_strictly_larger_radius: bool | None = None,
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

    normalized_radius_support_mode = _normalize_radius_support_mode(
        radius_support_mode=radius_support_mode,
        require_strictly_larger_radius=require_strictly_larger_radius,
    )
    void_indices = np.argwhere(mask).astype(np.int64, copy=False)
    image_shape = np.asarray(mask.shape, dtype=np.int64)
    radius_support_mode_code = _encode_radius_support_mode(normalized_radius_support_mode)

    for _ in range(iterations):
        previous_labels = label_image.copy()
        previous_labels_flat = previous_labels.reshape(-1)
        label_image_flat = label_image.reshape(-1)
        working_distance_map_flat = working_distance_map.reshape(-1)
        if mask.ndim == 2:
            changed_any_voxel = _grow_root_regions_by_radius_compiled_2d(
                previous_labels_flat,
                label_image_flat,
                working_distance_map_flat,
                void_indices,
                shape0=int(image_shape[0]),
                shape1=int(image_shape[1]),
                unassigned_label=int(voxel_regions.unassigned_label),
                minimum_supporting_neighbors=int(minimum_supporting_neighbors),
                radius_support_mode_code=int(radius_support_mode_code),
            )
        else:
            changed_any_voxel = _grow_root_regions_by_radius_compiled_3d(
                previous_labels_flat,
                label_image_flat,
                working_distance_map_flat,
                void_indices,
                shape0=int(image_shape[0]),
                shape1=int(image_shape[1]),
                shape2=int(image_shape[2]),
                unassigned_label=int(voxel_regions.unassigned_label),
                minimum_supporting_neighbors=int(minimum_supporting_neighbors),
                radius_support_mode_code=int(radius_support_mode_code),
            )
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


def reassign_region_boundary_voxels_by_majority(
    void_phase_mask: np.ndarray,
    distance_map: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
    *,
    radius_support_mode: str = "any",
    iterations: int = 1,
) -> MaximalBallVoxelRegions:
    """Reassign weakly supported labeled voxels using a neighbor majority rule.

    This mirrors the Imperial `medianElem` stage conceptually: if a labeled
    voxel is more exposed to different neighboring pore labels than to its own
    label, it may be reassigned to the strongest competing neighbor label.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    mask = np.asarray(void_phase_mask, dtype=bool)
    working_distance_map = np.asarray(distance_map, dtype=float)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    if mask.shape != label_image.shape or mask.shape != working_distance_map.shape:
        raise ValueError("void_phase_mask, distance_map, and voxel_regions.label_image must match")

    normalized_radius_support_mode = _normalize_radius_support_mode(
        radius_support_mode=radius_support_mode,
        require_strictly_larger_radius=None,
    )
    void_indices = np.argwhere(mask).astype(np.int64, copy=False)
    image_shape = np.asarray(mask.shape, dtype=np.int64)
    radius_support_mode_code = _encode_radius_support_mode(normalized_radius_support_mode)

    for _ in range(iterations):
        previous_labels = label_image.copy()
        previous_labels_flat = previous_labels.reshape(-1)
        label_image_flat = label_image.reshape(-1)
        working_distance_map_flat = working_distance_map.reshape(-1)
        if mask.ndim == 2:
            changed_any_voxel = _reassign_region_boundary_voxels_by_majority_compiled_2d(
                previous_labels_flat,
                label_image_flat,
                working_distance_map_flat,
                void_indices,
                shape0=int(image_shape[0]),
                shape1=int(image_shape[1]),
                radius_support_mode_code=int(radius_support_mode_code),
            )
        else:
            changed_any_voxel = _reassign_region_boundary_voxels_by_majority_compiled_3d(
                previous_labels_flat,
                label_image_flat,
                working_distance_map_flat,
                void_indices,
                shape0=int(image_shape[0]),
                shape1=int(image_shape[1]),
                shape2=int(image_shape[2]),
                radius_support_mode_code=int(radius_support_mode_code),
            )
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


def retreat_mixed_region_boundary_voxels(
    void_phase_mask: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
) -> MaximalBallVoxelRegions:
    """Retreat mixed boundary voxels back to the unassigned state.

    This mirrors the Imperial `retreatPoresMedian` stage: labeled voxels that
    touch both same-label and different-label neighbors are temporarily removed
    so later growth passes can rebuild cleaner interfaces.
    """

    mask = np.asarray(void_phase_mask, dtype=bool)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    if mask.shape != label_image.shape:
        raise ValueError("void_phase_mask and voxel_regions.label_image must match")

    neighbor_offsets = _neighbor_offsets(mask.ndim)
    image_shape = np.asarray(mask.shape, dtype=np.int64)
    previous_labels = label_image.copy()
    assigned_indices = np.argwhere(mask & (previous_labels >= 0))
    for voxel_index in assigned_indices:
        voxel_index_tuple = tuple(int(value) for value in voxel_index)
        current_label = int(previous_labels[voxel_index_tuple])
        same_label_neighbor_count = 0
        different_label_neighbor_count = 0
        for neighbor_offset in neighbor_offsets:
            neighbor_index = voxel_index + np.asarray(neighbor_offset, dtype=np.int64)
            if np.any(neighbor_index < 0) or np.any(neighbor_index >= image_shape):
                continue
            neighbor_index_tuple = tuple(int(value) for value in neighbor_index)
            neighbor_label = int(previous_labels[neighbor_index_tuple])
            if neighbor_label < 0:
                continue
            if neighbor_label == current_label:
                same_label_neighbor_count += 1
            else:
                different_label_neighbor_count += 1
        if same_label_neighbor_count > 0 and different_label_neighbor_count > 0:
            label_image[voxel_index_tuple] = voxel_regions.unassigned_label

    return MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=voxel_regions.root_ball_indices,
        root_labels=voxel_regions.root_labels,
        root_center_indices=voxel_regions.root_center_indices,
        root_radii_voxels=voxel_regions.root_radii_voxels,
        root_of_ball_index=voxel_regions.root_of_ball_index,
        unassigned_label=voxel_regions.unassigned_label,
    )


def stamp_retained_ball_centers_to_root_labels(
    voxel_regions: MaximalBallVoxelRegions,
    maximal_ball_hierarchy: MaximalBallHierarchy,
) -> MaximalBallVoxelRegions:
    """Restore retained-ball centers to their hierarchy-root region labels."""

    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    for ball_index, center_index in enumerate(maximal_ball_hierarchy.center_indices):
        root_label = int(voxel_regions.root_of_ball_index[ball_index])
        center_index_tuple = tuple(int(value) for value in center_index)
        label_image[center_index_tuple] = root_label

    return MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=voxel_regions.root_ball_indices,
        root_labels=voxel_regions.root_labels,
        root_center_indices=voxel_regions.root_center_indices,
        root_radii_voxels=voxel_regions.root_radii_voxels,
        root_of_ball_index=voxel_regions.root_of_ball_index,
        unassigned_label=voxel_regions.unassigned_label,
    )


def grow_root_regions_by_neighbor_priority(
    void_phase_mask: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
    *,
    iterations: int = 1,
) -> MaximalBallVoxelRegions:
    """Grow unassigned voxels by direct neighbor propagation in sweep order.

    This mirrors the late Imperial `growPores` / `growPores_X2` stages more
    closely than the earlier radius-aware majority passes.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    mask = np.asarray(void_phase_mask, dtype=bool)
    label_image = np.asarray(voxel_regions.label_image, dtype=np.int64).copy()
    if mask.shape != label_image.shape:
        raise ValueError("void_phase_mask and voxel_regions.label_image must match")

    image_shape = np.asarray(mask.shape, dtype=np.int64)
    neighbor_offsets = _neighbor_offsets_with_growth_priority(mask.ndim)
    forward_indices = np.argwhere(mask)
    backward_indices = forward_indices[::-1].copy()

    for _ in range(iterations):
        changed_any_voxel = False
        for traversal_indices in (forward_indices, backward_indices):
            for voxel_index in traversal_indices:
                voxel_index_tuple = tuple(int(value) for value in voxel_index)
                if label_image[voxel_index_tuple] != voxel_regions.unassigned_label:
                    continue
                for neighbor_offset in neighbor_offsets:
                    neighbor_index = voxel_index + np.asarray(neighbor_offset, dtype=np.int64)
                    if np.any(neighbor_index < 0) or np.any(neighbor_index >= image_shape):
                        continue
                    neighbor_index_tuple = tuple(int(value) for value in neighbor_index)
                    neighbor_label = int(label_image[neighbor_index_tuple])
                    if neighbor_label < 0:
                        continue
                    label_image[voxel_index_tuple] = neighbor_label
                    changed_any_voxel = True
                    break
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
    """Assign voxel ownership using an Imperial-inspired staged growth schedule."""

    voxel_regions = initialize_root_region_labels(
        void_phase_mask,
        maximal_ball_hierarchy,
    )
    voxel_regions = seed_root_region_ball_interiors(
        void_phase_mask,
        maximal_ball_hierarchy,
        voxel_regions,
    )
    initial_growth_schedule = [
        (3, "strictly_larger", 3),
        (2, "strictly_larger", 3),
        (2, "greater_or_equal", 4),
    ]
    for minimum_supporting_neighbors, radius_support_mode, iterations in initial_growth_schedule:
        voxel_regions = grow_root_regions_by_radius(
            void_phase_mask,
            maximal_ball_hierarchy.distance_map,
            voxel_regions,
            minimum_supporting_neighbors=minimum_supporting_neighbors,
            radius_support_mode=radius_support_mode,
            iterations=iterations,
        )

    voxel_regions = reassign_region_boundary_voxels_by_majority(
        void_phase_mask,
        maximal_ball_hierarchy.distance_map,
        voxel_regions,
        radius_support_mode="any",
        iterations=2,
    )
    voxel_regions = retreat_mixed_region_boundary_voxels(
        void_phase_mask,
        voxel_regions,
    )
    voxel_regions = stamp_retained_ball_centers_to_root_labels(
        voxel_regions,
        maximal_ball_hierarchy,
    )

    late_growth_schedule = [
        (2, "greater_or_equal", 6),
        (2, "any", 4),
        (1, "any", 4),
    ]
    for minimum_supporting_neighbors, radius_support_mode, iterations in late_growth_schedule:
        voxel_regions = grow_root_regions_by_radius(
            void_phase_mask,
            maximal_ball_hierarchy.distance_map,
            voxel_regions,
            minimum_supporting_neighbors=minimum_supporting_neighbors,
            radius_support_mode=radius_support_mode,
            iterations=iterations,
        )

    voxel_regions = reassign_region_boundary_voxels_by_majority(
        void_phase_mask,
        maximal_ball_hierarchy.distance_map,
        voxel_regions,
        radius_support_mode="any",
        iterations=2,
    )
    voxel_regions = grow_root_regions_by_neighbor_priority(
        void_phase_mask,
        voxel_regions,
        iterations=max(16, 2 * max(void_phase_mask.shape)),
    )
    return voxel_regions


def measure_region_adjacency(
    void_phase_mask: np.ndarray,
    voxel_regions: MaximalBallVoxelRegions,
    *,
    distance_map: np.ndarray | None = None,
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
    working_distance_map: np.ndarray | None = None
    if distance_map is not None:
        working_distance_map = np.asarray(distance_map, dtype=float)
        if working_distance_map.shape != mask.shape:
            raise ValueError(
                "distance_map must match void_phase_mask and voxel_regions.label_image"
            )

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
    pair_max_touch_radius_side1: dict[tuple[int, int], float] = {}
    pair_max_touch_radius_side2: dict[tuple[int, int], float] = {}
    pair_max_touch_index_side1: dict[tuple[int, int], np.ndarray] = {}
    pair_max_touch_index_side2: dict[tuple[int, int], np.ndarray] = {}

    image_shape = np.asarray(mask.shape, dtype=np.int64)
    for axis_index in range(mask.ndim):
        lower_boundary_selector: list[int | slice] = [slice(None)] * mask.ndim
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

        upper_boundary_selector: list[int | slice] = [slice(None)] * mask.ndim
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
        lower_touch_radii = (
            working_distance_map[lower_slices_tuple][shared_interface_mask]
            if working_distance_map is not None
            else None
        )
        upper_touch_radii = (
            working_distance_map[upper_slices_tuple][shared_interface_mask]
            if working_distance_map is not None
            else None
        )

        for face_index in range(interface_indices.shape[0]):
            first_label = int(lower_interface_labels[face_index])
            second_label = int(upper_interface_labels[face_index])
            lower_voxel_index = interface_indices[face_index].astype(np.int64, copy=False)
            upper_voxel_index = lower_voxel_index.copy()
            upper_voxel_index[axis_index] += 1
            if first_label < second_label:
                ordered_pair = (first_label, second_label)
                orientation_sign = 1.0
                side1_touch_radius = (
                    float(lower_touch_radii[face_index])
                    if lower_touch_radii is not None
                    else float("nan")
                )
                side2_touch_radius = (
                    float(upper_touch_radii[face_index])
                    if upper_touch_radii is not None
                    else float("nan")
                )
                side1_touch_index = lower_voxel_index
                side2_touch_index = upper_voxel_index
            else:
                ordered_pair = (second_label, first_label)
                orientation_sign = -1.0
                side1_touch_radius = (
                    float(upper_touch_radii[face_index])
                    if upper_touch_radii is not None
                    else float("nan")
                )
                side2_touch_radius = (
                    float(lower_touch_radii[face_index])
                    if lower_touch_radii is not None
                    else float("nan")
                )
                side1_touch_index = upper_voxel_index
                side2_touch_index = lower_voxel_index

            pair_face_counts[ordered_pair] = pair_face_counts.get(ordered_pair, 0) + 1
            if ordered_pair not in pair_axis_face_balance:
                pair_axis_face_balance[ordered_pair] = np.zeros(mask.ndim, dtype=float)
            if ordered_pair not in pair_centroid_sums:
                pair_centroid_sums[ordered_pair] = np.zeros(mask.ndim, dtype=float)
            pair_axis_face_balance[ordered_pair][axis_index] += orientation_sign
            pair_centroid_sums[ordered_pair] += face_midpoint_indices[face_index]
            if np.isfinite(side1_touch_radius):
                previous_side1_radius = pair_max_touch_radius_side1.get(ordered_pair, -np.inf)
                if side1_touch_radius > previous_side1_radius:
                    pair_max_touch_radius_side1[ordered_pair] = side1_touch_radius
                    pair_max_touch_index_side1[ordered_pair] = side1_touch_index.copy()
            if np.isfinite(side2_touch_radius):
                previous_side2_radius = pair_max_touch_radius_side2.get(ordered_pair, -np.inf)
                if side2_touch_radius > previous_side2_radius:
                    pair_max_touch_radius_side2[ordered_pair] = side2_touch_radius
                    pair_max_touch_index_side2[ordered_pair] = side2_touch_index.copy()

    occupied_region_mask = region_volume_voxels > 0
    occupied_region_indices = np.flatnonzero(occupied_region_mask).astype(np.int64)
    if occupied_region_indices.size != region_count:
        compact_label_of_region = np.full(region_count, -1, dtype=np.int64)
        compact_label_of_region[occupied_region_indices] = np.arange(
            occupied_region_indices.size,
            dtype=np.int64,
        )
        region_labels = region_labels[occupied_region_indices]
        region_volume_voxels = region_volume_voxels[occupied_region_indices]
        region_surface_face_counts = region_surface_face_counts[occupied_region_indices]
        boundary_face_counts = boundary_face_counts[occupied_region_indices]
        remapped_pair_face_counts: dict[tuple[int, int], int] = {}
        remapped_pair_axis_face_balance: dict[tuple[int, int], np.ndarray] = {}
        remapped_pair_centroid_sums: dict[tuple[int, int], np.ndarray] = {}
        remapped_pair_max_touch_radius_side1: dict[tuple[int, int], float] = {}
        remapped_pair_max_touch_radius_side2: dict[tuple[int, int], float] = {}
        remapped_pair_max_touch_index_side1: dict[tuple[int, int], np.ndarray] = {}
        remapped_pair_max_touch_index_side2: dict[tuple[int, int], np.ndarray] = {}
        for ordered_pair, face_count in pair_face_counts.items():
            first_region = int(compact_label_of_region[ordered_pair[0]])
            second_region = int(compact_label_of_region[ordered_pair[1]])
            if first_region < 0 or second_region < 0:
                continue
            remapped_pair = (first_region, second_region)
            remapped_pair_face_counts[remapped_pair] = face_count
            remapped_pair_axis_face_balance[remapped_pair] = pair_axis_face_balance[ordered_pair]
            remapped_pair_centroid_sums[remapped_pair] = pair_centroid_sums[ordered_pair]
            if ordered_pair in pair_max_touch_radius_side1:
                remapped_pair_max_touch_radius_side1[remapped_pair] = pair_max_touch_radius_side1[
                    ordered_pair
                ]
                remapped_pair_max_touch_index_side1[remapped_pair] = pair_max_touch_index_side1[
                    ordered_pair
                ].copy()
            if ordered_pair in pair_max_touch_radius_side2:
                remapped_pair_max_touch_radius_side2[remapped_pair] = pair_max_touch_radius_side2[
                    ordered_pair
                ]
                remapped_pair_max_touch_index_side2[remapped_pair] = pair_max_touch_index_side2[
                    ordered_pair
                ].copy()
        pair_face_counts = remapped_pair_face_counts
        pair_axis_face_balance = remapped_pair_axis_face_balance
        pair_centroid_sums = remapped_pair_centroid_sums
        pair_max_touch_radius_side1 = remapped_pair_max_touch_radius_side1
        pair_max_touch_radius_side2 = remapped_pair_max_touch_radius_side2
        pair_max_touch_index_side1 = remapped_pair_max_touch_index_side1
        pair_max_touch_index_side2 = remapped_pair_max_touch_index_side2

    ordered_pairs = sorted(pair_face_counts)
    throat_region_pairs = np.asarray(ordered_pairs, dtype=np.int64)
    throat_count = len(ordered_pairs)
    throat_face_counts = np.zeros(throat_count, dtype=np.int64)
    throat_axis_face_balance = np.zeros((throat_count, mask.ndim), dtype=float)
    throat_centroid_indices = np.zeros((throat_count, mask.ndim), dtype=float)
    throat_max_touch_radius_side1_voxels = np.full(throat_count, np.nan, dtype=float)
    throat_max_touch_radius_side2_voxels = np.full(throat_count, np.nan, dtype=float)
    throat_max_touch_index_side1 = np.full((throat_count, mask.ndim), -1, dtype=np.int64)
    throat_max_touch_index_side2 = np.full((throat_count, mask.ndim), -1, dtype=np.int64)
    for throat_index, ordered_pair in enumerate(ordered_pairs):
        face_count = int(pair_face_counts[ordered_pair])
        throat_face_counts[throat_index] = face_count
        throat_axis_face_balance[throat_index] = pair_axis_face_balance[ordered_pair]
        throat_centroid_indices[throat_index] = pair_centroid_sums[ordered_pair] / max(
            face_count, 1
        )
        if ordered_pair in pair_max_touch_radius_side1:
            throat_max_touch_radius_side1_voxels[throat_index] = pair_max_touch_radius_side1[
                ordered_pair
            ]
            throat_max_touch_index_side1[throat_index] = pair_max_touch_index_side1[ordered_pair]
        if ordered_pair in pair_max_touch_radius_side2:
            throat_max_touch_radius_side2_voxels[throat_index] = pair_max_touch_radius_side2[
                ordered_pair
            ]
            throat_max_touch_index_side2[throat_index] = pair_max_touch_index_side2[ordered_pair]

    return MaximalBallRegionAdjacency(
        region_labels=region_labels,
        region_volume_voxels=region_volume_voxels,
        region_surface_face_counts=region_surface_face_counts,
        throat_region_pairs=throat_region_pairs,
        throat_face_counts=throat_face_counts,
        throat_axis_face_balance=throat_axis_face_balance,
        throat_centroid_indices=throat_centroid_indices,
        throat_max_touch_radius_side1_voxels=throat_max_touch_radius_side1_voxels,
        throat_max_touch_radius_side2_voxels=throat_max_touch_radius_side2_voxels,
        throat_max_touch_index_side1=throat_max_touch_index_side1,
        throat_max_touch_index_side2=throat_max_touch_index_side2,
        boundary_face_counts=boundary_face_counts,
    )


def extract_maximal_ball_regions(
    void_phase_mask: np.ndarray,
    *,
    distance_map_backend: str = "auto",
    edt_parallel_threads: int | None = None,
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
        edt_parallel_threads=edt_parallel_threads,
        settings=settings,
        apply_boundary_clipping=apply_boundary_clipping,
    )
    hierarchy = build_maximal_ball_hierarchy(candidates)
    voxel_regions = assign_voxel_regions_from_hierarchy(void_phase_mask, hierarchy)
    region_adjacency = measure_region_adjacency(
        void_phase_mask,
        voxel_regions,
        distance_map=hierarchy.distance_map,
    )
    return MaximalBallExtractionResult(
        candidates=candidates,
        hierarchy=hierarchy,
        voxel_regions=voxel_regions,
        region_adjacency=region_adjacency,
    )


def summarize_maximal_ball_extraction_diagnostics(
    void_phase_mask: np.ndarray,
    extraction_result: MaximalBallExtractionResult,
) -> MaximalBallExtractionDiagnostics:
    """Summarize intermediate maximal-ball extraction behavior for comparison work."""

    mask = np.asarray(void_phase_mask, dtype=bool)
    label_image = np.asarray(extraction_result.voxel_regions.label_image, dtype=np.int64)
    if mask.shape != label_image.shape:
        raise ValueError("void_phase_mask must match extraction_result voxel-region labels")

    assigned_void_mask = mask & (label_image >= 0)
    unassigned_void_voxel_count = int(np.count_nonzero(mask & (label_image < 0)))
    region_adjacency = extraction_result.region_adjacency
    (
        refined_support_radius_side1_voxels,
        refined_support_radius_side2_voxels,
    ) = _select_interface_supporting_ball_radii(extraction_result)
    occupied_region_count = int(region_adjacency.region_volume_voxels.size)
    neighbor_counts = np.zeros(occupied_region_count, dtype=np.int64)
    throat_region_pairs = np.asarray(region_adjacency.throat_region_pairs, dtype=np.int64)
    if throat_region_pairs.size:
        np.add.at(neighbor_counts, throat_region_pairs[:, 0], 1)
        np.add.at(neighbor_counts, throat_region_pairs[:, 1], 1)
    zero_throat_region_mask = neighbor_counts == 0
    boundary_face_counts = np.asarray(region_adjacency.boundary_face_counts, dtype=np.int64)
    boundary_touch_mask = np.sum(boundary_face_counts, axis=1) > 0

    return MaximalBallExtractionDiagnostics(
        retained_ball_count=int(extraction_result.candidates.retained_center_indices.shape[0]),
        root_region_count=int(extraction_result.voxel_regions.root_labels.size),
        occupied_region_count=occupied_region_count,
        assigned_void_fraction=float(
            np.count_nonzero(assigned_void_mask) / max(np.count_nonzero(mask), 1)
        ),
        unassigned_void_voxel_count=unassigned_void_voxel_count,
        zero_throat_region_count=int(np.count_nonzero(zero_throat_region_mask)),
        internal_zero_throat_region_count=int(
            np.count_nonzero(zero_throat_region_mask & ~boundary_touch_mask)
        ),
        boundary_zero_throat_region_count=int(
            np.count_nonzero(zero_throat_region_mask & boundary_touch_mask)
        ),
        throat_touch_radius_side1_mean_voxels=float(
            np.nanmean(region_adjacency.throat_max_touch_radius_side1_voxels)
            if region_adjacency.throat_max_touch_radius_side1_voxels.size
            else np.nan
        ),
        throat_touch_radius_side2_mean_voxels=float(
            np.nanmean(region_adjacency.throat_max_touch_radius_side2_voxels)
            if region_adjacency.throat_max_touch_radius_side2_voxels.size
            else np.nan
        ),
        throat_refined_support_radius_side1_mean_voxels=float(
            np.nanmean(refined_support_radius_side1_voxels)
            if refined_support_radius_side1_voxels.size
            else np.nan
        ),
        throat_refined_support_radius_side2_mean_voxels=float(
            np.nanmean(refined_support_radius_side2_voxels)
            if refined_support_radius_side2_voxels.size
            else np.nan
        ),
    )


def _select_interface_supporting_ball_data(
    extraction_result: MaximalBallExtractionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return side-specific throat-supporting ball centers and radii in voxel units."""

    hierarchy = extraction_result.hierarchy
    voxel_regions = extraction_result.voxel_regions
    region_adjacency = extraction_result.region_adjacency

    throat_region_pairs = np.asarray(region_adjacency.throat_region_pairs, dtype=np.int64)
    throat_centroid_indices = np.asarray(region_adjacency.throat_centroid_indices, dtype=float)
    touch_radius_side1_voxels = np.asarray(
        region_adjacency.throat_max_touch_radius_side1_voxels,
        dtype=float,
    )
    touch_radius_side2_voxels = np.asarray(
        region_adjacency.throat_max_touch_radius_side2_voxels,
        dtype=float,
    )
    touch_index_side1 = np.asarray(region_adjacency.throat_max_touch_index_side1, dtype=np.int64)
    touch_index_side2 = np.asarray(region_adjacency.throat_max_touch_index_side2, dtype=np.int64)
    image_ndim = (
        int(throat_centroid_indices.shape[1])
        if throat_centroid_indices.size
        else int(voxel_regions.label_image.ndim)
    )

    throat_count = int(throat_region_pairs.shape[0])
    first_side_radii = np.asarray(touch_radius_side1_voxels, dtype=float).copy()
    second_side_radii = np.asarray(touch_radius_side2_voxels, dtype=float).copy()
    first_side_center_coordinates = np.full((throat_count, image_ndim), np.nan, dtype=float)
    second_side_center_coordinates = np.full((throat_count, image_ndim), np.nan, dtype=float)
    if throat_count == 0:
        return (
            first_side_center_coordinates,
            first_side_radii,
            second_side_center_coordinates,
            second_side_radii,
        )

    refined_touch_seed_cache: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}

    def _lookup_refined_touch_seed(seed_index: np.ndarray) -> tuple[np.ndarray, float]:
        seed_index_tuple = tuple(int(value) for value in seed_index)
        if any(value < 0 for value in seed_index_tuple):
            return np.full(image_ndim, np.nan, dtype=float), float("nan")
        if seed_index_tuple not in refined_touch_seed_cache:
            refined_center_coordinate, refined_radius_voxels = _refine_ball_center_subvoxel(
                hierarchy.distance_map,
                np.asarray(seed_index, dtype=np.int64),
                displacement_limit=0.49,
                radius_gain_factor=0.95,
            )
            refined_touch_seed_cache[seed_index_tuple] = (
                np.asarray(refined_center_coordinate, dtype=float),
                float(refined_radius_voxels),
            )
        return refined_touch_seed_cache[seed_index_tuple]

    for throat_index, region_pair in enumerate(throat_region_pairs):
        side1_seed_center_coordinate, side1_seed_radius = _lookup_refined_touch_seed(
            touch_index_side1[throat_index]
        )
        side2_seed_center_coordinate, side2_seed_radius = _lookup_refined_touch_seed(
            touch_index_side2[throat_index]
        )
        if np.isfinite(side1_seed_radius):
            first_side_center_coordinates[throat_index] = side1_seed_center_coordinate
            if not np.isfinite(first_side_radii[throat_index]):
                first_side_radii[throat_index] = side1_seed_radius
        if np.isfinite(side2_seed_radius):
            second_side_center_coordinates[throat_index] = side2_seed_center_coordinate
            if not np.isfinite(second_side_radii[throat_index]):
                second_side_radii[throat_index] = side2_seed_radius

    return (
        first_side_center_coordinates,
        first_side_radii,
        second_side_center_coordinates,
        second_side_radii,
    )


def _select_interface_supporting_ball_radii(
    extraction_result: MaximalBallExtractionResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return side-specific throat-supporting maximal-ball radii in voxel units."""

    _, first_side_radii, _, second_side_radii = _select_interface_supporting_ball_data(
        extraction_result
    )
    return first_side_radii, second_side_radii


def _redistribute_region_volumes_like_imperial_export(
    pore_data: dict[str, np.ndarray],
    throat_data: dict[str, np.ndarray],
    throat_conns: np.ndarray,
) -> None:
    """Redistribute region volume between pores and throats like the Imperial exporter."""

    pore_region_volume = np.asarray(pore_data["region_volume"], dtype=float)
    pore_area = np.asarray(pore_data["area"], dtype=float)
    throat_area = np.asarray(throat_data["area"], dtype=float)
    redistributed_pore_volume = np.zeros_like(pore_region_volume)
    redistributed_throat_volume = np.asarray(
        throat_data.get("volume", np.zeros(throat_conns.shape[0])), dtype=float
    )
    redistributed_throat_volume.fill(0.0)

    for pore_index in range(pore_region_volume.size):
        connected_throat_mask = (throat_conns[:, 0] == pore_index) | (
            throat_conns[:, 1] == pore_index
        )
        total_connected_throat_area = float(throat_area[connected_throat_mask].sum())
        pore_cross_section_area = float(pore_area[pore_index])
        normalization_area = pore_cross_section_area + total_connected_throat_area + 1.0e-36
        raw_region_volume = float(pore_region_volume[pore_index])
        redistributed_pore_volume[pore_index] = (
            raw_region_volume * pore_cross_section_area / normalization_area
        )
        if np.any(connected_throat_mask):
            redistributed_throat_volume[connected_throat_mask] += (
                raw_region_volume * throat_area[connected_throat_mask] / normalization_area
            )

    pore_data["volume"] = redistributed_pore_volume
    throat_data["volume"] = redistributed_throat_volume


def _resolve_axis_boundary_label_overlap(
    lower_contact: np.ndarray,
    upper_contact: np.ndarray,
    *,
    lower_face_count: np.ndarray,
    upper_face_count: np.ndarray,
    pore_axis_coordinates: np.ndarray,
    sample_axis_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Make per-axis inlet/outlet pore labels mutually exclusive."""

    lower_contact = np.asarray(lower_contact, dtype=bool).copy()
    upper_contact = np.asarray(upper_contact, dtype=bool).copy()
    overlap_mask = lower_contact & upper_contact
    if not np.any(overlap_mask):
        return lower_contact, upper_contact

    lower_face_count = np.asarray(lower_face_count, dtype=np.int64)
    upper_face_count = np.asarray(upper_face_count, dtype=np.int64)
    pore_axis_coordinates = np.asarray(pore_axis_coordinates, dtype=float)
    sample_midpoint = 0.5 * float(sample_axis_length)
    overlap_indices = np.flatnonzero(overlap_mask)
    for pore_index in overlap_indices:
        lower_faces = int(lower_face_count[pore_index])
        upper_faces = int(upper_face_count[pore_index])
        if lower_faces > upper_faces:
            upper_contact[pore_index] = False
        elif upper_faces > lower_faces:
            lower_contact[pore_index] = False
        elif pore_axis_coordinates[pore_index] <= sample_midpoint:
            upper_contact[pore_index] = False
        else:
            lower_contact[pore_index] = False
    return lower_contact, upper_contact


def _resolve_flow_boundary_mode(flow_boundary_mode: str) -> str:
    """Normalize the network boundary treatment used for flow solves."""

    normalized_mode = str(flow_boundary_mode).strip().lower()
    if normalized_mode not in {"direct", "external_reservoir"}:
        raise ValueError("flow_boundary_mode must be one of {'direct', 'external_reservoir'}")
    return normalized_mode


def _resolve_throat_area_mode(throat_area_mode: str) -> str:
    """Normalize the throat cross-section area convention."""

    normalized_mode = str(throat_area_mode).strip().lower()
    aliases = {
        "face_count": "face_count",
        "interface_face_count": "face_count",
        "vector_magnitude": "vector_magnitude",
        "cross_area_magnitude": "vector_magnitude",
    }
    if normalized_mode not in aliases:
        raise ValueError("throat_area_mode must be one of {'face_count', 'vector_magnitude'}")
    return aliases[normalized_mode]


def _resolve_throat_shape_factor_radius_mode(throat_shape_factor_radius_mode: str) -> str:
    """Normalize the radius convention used when deriving throat shape factors."""

    normalized_mode = str(throat_shape_factor_radius_mode).strip().lower()
    aliases = {
        "inscribed": "inscribed",
        "exported": "inscribed",
        "radius_inscribed": "inscribed",
        "surface_ball": "surface_ball",
        "interface_support": "surface_ball",
    }
    if normalized_mode not in aliases:
        raise ValueError(
            "throat_shape_factor_radius_mode must be one of {'inscribed', 'surface_ball'}"
        )
    return aliases[normalized_mode]


def _resolve_throat_anchor_mode(throat_anchor_mode: str) -> str:
    """Normalize the throat-center anchor convention used for conduit lengths."""

    normalized_mode = str(throat_anchor_mode).strip().lower()
    aliases = {
        "largest_support": "largest_support",
        "largest_radius": "largest_support",
        "second_side": "second_side",
        "higher_label_side": "second_side",
        "region_pair_second": "second_side",
    }
    if normalized_mode not in aliases:
        raise ValueError("throat_anchor_mode must be one of {'largest_support', 'second_side'}")
    return aliases[normalized_mode]


def _max_boundary_touch_radii_by_side(
    label_image: np.ndarray,
    region_labels: np.ndarray,
    distance_map: np.ndarray,
) -> np.ndarray:
    """Return max boundary-contact radius for each compact region and side."""

    labels = np.asarray(label_image, dtype=np.int64)
    radii = np.asarray(distance_map, dtype=float)
    if labels.shape != radii.shape:
        raise ValueError("label_image and distance_map must have the same shape")

    retained_region_labels = np.asarray(region_labels, dtype=np.int64)
    boundary_touch_radii = np.full(
        (retained_region_labels.size, 2 * labels.ndim),
        np.nan,
        dtype=float,
    )
    for compact_region_index, original_region_label in enumerate(retained_region_labels):
        for axis_index in range(labels.ndim):
            for side_offset, side_index in enumerate((0, labels.shape[axis_index] - 1)):
                boundary_selector = [slice(None)] * labels.ndim
                boundary_selector[axis_index] = side_index
                selector_tuple = tuple(boundary_selector)
                side_region_mask = labels[selector_tuple] == int(original_region_label)
                if np.any(side_region_mask):
                    boundary_touch_radii[compact_region_index, 2 * axis_index + side_offset] = (
                        float(np.max(radii[selector_tuple][side_region_mask]))
                    )
    return boundary_touch_radii


def build_network_dict_from_maximal_ball_regions(
    extraction_result: MaximalBallExtractionResult,
    *,
    voxel_size: float,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    flow_boundary_mode: str = "direct",
    boundary_axis: str | None = None,
    boundary_length_epsilon: float = 1.0e-300,
    boundary_radius_scale: float = 1.1,
    throat_area_mode: str = "face_count",
    throat_shape_factor_radius_mode: str = "inscribed",
    throat_anchor_mode: str = "second_side",
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

    The current implementation now follows the Imperial export logic more
    closely in three places:

    - throat radii are taken from interface-supporting maximal balls
    - throat and pore shape factors follow the Imperial export repair and
      throat-area-weighted pore averaging logic
    - region volumes are redistributed between pores and throats using the same
      area-partition rule used by the Imperial CNM writer

    This is still not full `pnextract` parity, because the upstream voxel
    ownership and throat-surface ball construction remain a native
    approximation.
    """

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    normalized_flow_boundary_mode = _resolve_flow_boundary_mode(flow_boundary_mode)
    normalized_throat_area_mode = _resolve_throat_area_mode(throat_area_mode)
    normalized_throat_shape_factor_radius_mode = _resolve_throat_shape_factor_radius_mode(
        throat_shape_factor_radius_mode
    )
    normalized_throat_anchor_mode = _resolve_throat_anchor_mode(throat_anchor_mode)
    if boundary_length_epsilon <= 0.0:
        raise ValueError("boundary_length_epsilon must be positive")
    if boundary_radius_scale <= 0.0:
        raise ValueError("boundary_radius_scale must be positive")

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
    if boundary_axis is not None and boundary_axis not in active_axis_names:
        raise ValueError("boundary_axis must be one of the active axis names")
    if normalized_flow_boundary_mode == "external_reservoir" and boundary_axis is None:
        boundary_axis = active_axis_names[0]
    retained_region_labels = np.asarray(region_adjacency.region_labels, dtype=np.int64)
    region_count = int(retained_region_labels.size)
    all_root_center_indices = np.asarray(voxel_regions.root_center_indices, dtype=float)
    all_root_radii_voxels = np.asarray(voxel_regions.root_radii_voxels, dtype=float)

    if all_root_center_indices.ndim != 2 or all_root_center_indices.shape[1] != image_ndim:
        raise ValueError("root_center_indices must have shape (n_roots, image_ndim)")
    if all_root_radii_voxels.shape != (all_root_center_indices.shape[0],):
        raise ValueError("root_radii_voxels must align with root_center_indices")
    if np.any(retained_region_labels < 0) or np.any(
        retained_region_labels >= all_root_center_indices.shape[0]
    ):
        raise ValueError("region_adjacency.region_labels must index the root-region arrays")

    all_root_center_coordinates = np.asarray(
        hierarchy.center_coordinates[voxel_regions.root_ball_indices],
        dtype=float,
    )
    if all_root_center_coordinates.ndim != 2 or all_root_center_coordinates.shape[1] != image_ndim:
        raise ValueError("root center coordinates must have shape (n_roots, image_ndim)")

    root_center_coordinates = all_root_center_coordinates[retained_region_labels]
    root_radii_voxels = all_root_radii_voxels[retained_region_labels]

    pore_coords = root_center_coordinates * float(voxel_size)
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
    axis_face_balance = np.asarray(region_adjacency.throat_axis_face_balance, dtype=float)
    if normalized_throat_area_mode == "vector_magnitude":
        throat_cross_area_face_counts = np.linalg.norm(axis_face_balance, axis=1)
        throat_cross_area_face_counts = np.where(
            throat_cross_area_face_counts > 0.0,
            throat_cross_area_face_counts,
            np.maximum(throat_face_counts, 0.1),
        )
    else:
        throat_cross_area_face_counts = throat_face_counts
    throat_area = throat_cross_area_face_counts * float(voxel_size) ** 2
    throat_centroid_indices = np.asarray(region_adjacency.throat_centroid_indices, dtype=float)
    if throat_centroid_indices.shape[0] != throat_count:
        raise ValueError("throat_centroid_indices must align with throat_region_pairs")
    throat_centroid_coords = throat_centroid_indices * float(voxel_size)
    if throat_centroid_coords.shape[1] == 2:
        throat_centroid_coords = np.column_stack(
            [throat_centroid_coords, np.zeros(throat_count, dtype=float)]
        )

    (
        first_side_center_indices,
        first_side_radius_voxels,
        second_side_center_indices,
        second_side_radius_voxels,
    ) = _select_interface_supporting_ball_data(extraction_result)
    pore_radius_by_region = root_radii_voxels
    equivalent_interface_radius = np.sqrt(np.maximum(throat_area, 0.0) / np.pi) / float(voxel_size)
    throat_support_radius_voxels = np.empty(throat_count, dtype=float)
    throat_shape_factor_radius_voxels = np.empty(throat_count, dtype=float)
    for throat_index, region_pair in enumerate(throat_region_pairs):
        first_region_label = int(region_pair[0])
        second_region_label = int(region_pair[1])
        first_radius_voxels = float(first_side_radius_voxels[throat_index])
        second_radius_voxels = float(second_side_radius_voxels[throat_index])
        if not np.isfinite(first_radius_voxels):
            first_radius_voxels = min(
                float(pore_radius_by_region[first_region_label]),
                float(equivalent_interface_radius[throat_index]),
            )
        if not np.isfinite(second_radius_voxels):
            second_radius_voxels = min(
                float(pore_radius_by_region[second_region_label]),
                float(equivalent_interface_radius[throat_index]),
            )
        throat_support_radius_voxels[throat_index] = min(
            0.5 * (first_radius_voxels + second_radius_voxels),
            float(pore_radius_by_region[first_region_label]),
            float(pore_radius_by_region[second_region_label]),
        )
        surface_ball_radius_voxels = min(
            max(second_radius_voxels, 0.5),
            float(pore_radius_by_region[first_region_label]),
            float(pore_radius_by_region[second_region_label]),
        )
        if normalized_throat_shape_factor_radius_mode == "surface_ball":
            throat_shape_factor_radius_voxels[throat_index] = surface_ball_radius_voxels
        else:
            throat_shape_factor_radius_voxels[throat_index] = throat_support_radius_voxels[
                throat_index
            ]
    throat_radius = throat_support_radius_voxels * float(voxel_size)
    throat_shape_factor_radius = throat_shape_factor_radius_voxels * float(voxel_size)
    first_side_radius_physical = np.where(
        np.isfinite(first_side_radius_voxels),
        first_side_radius_voxels * float(voxel_size),
        throat_radius,
    )
    second_side_radius_physical = np.where(
        np.isfinite(second_side_radius_voxels),
        second_side_radius_voxels * float(voxel_size),
        throat_radius,
    )

    minimum_pore_segment_length = 0.5 * float(voxel_size)
    minimum_total_length = 3.01 * float(voxel_size)
    minimum_throat_core_length = 1.0 * float(voxel_size)
    if throat_count:
        first_pore_coordinates = pore_coords[throat_region_pairs[:, 0]]
        second_pore_coordinates = pore_coords[throat_region_pairs[:, 1]]
        if normalized_throat_anchor_mode == "second_side":
            throat_anchor_center_indices = second_side_center_indices.copy()
        else:
            throat_anchor_center_indices = np.where(
                (first_side_radius_voxels >= second_side_radius_voxels)[:, np.newaxis],
                first_side_center_indices,
                second_side_center_indices,
            )
        invalid_anchor_mask = ~np.isfinite(throat_anchor_center_indices).all(axis=1)
        throat_anchor_center_indices[invalid_anchor_mask] = throat_centroid_indices[
            invalid_anchor_mask
        ]
        throat_anchor_coordinates = throat_anchor_center_indices * float(voxel_size)
        if throat_anchor_coordinates.shape[1] == 2:
            throat_anchor_coordinates = np.column_stack(
                [throat_anchor_coordinates, np.zeros(throat_count, dtype=float)]
            )
        pore1_to_anchor_length = np.linalg.norm(
            throat_anchor_coordinates - first_pore_coordinates,
            axis=1,
        )
        pore2_to_anchor_length = np.linalg.norm(
            second_pore_coordinates - throat_anchor_coordinates,
            axis=1,
        )
        throat_total_length = np.maximum(
            pore1_to_anchor_length + pore2_to_anchor_length,
            minimum_total_length,
        )
        pore1_length = np.maximum(0.67 * pore1_to_anchor_length, minimum_pore_segment_length)
        pore2_length = np.maximum(0.67 * pore2_to_anchor_length, minimum_pore_segment_length)
        core_length = np.maximum(
            throat_total_length - pore1_length - pore2_length,
            minimum_throat_core_length,
        )
        throat_total_length = pore1_length + core_length + pore2_length
    else:
        pore1_length = np.zeros(0, dtype=float)
        pore2_length = np.zeros(0, dtype=float)
        core_length = np.zeros(0, dtype=float)
        throat_total_length = np.zeros(0, dtype=float)

    direct_boundary_label_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    connected_boundary_label_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    boundary_touch_radii_voxels = _max_boundary_touch_radii_by_side(
        voxel_regions.label_image,
        retained_region_labels,
        hierarchy.distance_map,
    )
    pore_boundary = np.zeros(region_count, dtype=bool)
    for axis_index, axis_name in enumerate(active_axis_names):
        lower_face_count = np.asarray(
            region_adjacency.boundary_face_counts[:, 2 * axis_index],
            dtype=np.int64,
        )
        upper_face_count = np.asarray(
            region_adjacency.boundary_face_counts[:, 2 * axis_index + 1],
            dtype=np.int64,
        )
        lower_contact = lower_face_count > 0
        upper_contact = upper_face_count > 0
        lower_contact, upper_contact = _resolve_axis_boundary_label_overlap(
            lower_contact,
            upper_contact,
            lower_face_count=lower_face_count,
            upper_face_count=upper_face_count,
            pore_axis_coordinates=pore_coords[:, axis_index],
            sample_axis_length=voxel_regions.label_image.shape[axis_index] * float(voxel_size),
        )
        direct_boundary_label_masks[axis_name] = (lower_contact, upper_contact)
        connected_boundary_label_masks[axis_name] = (lower_contact.copy(), upper_contact.copy())
        pore_boundary |= lower_contact | upper_contact

    if normalized_flow_boundary_mode == "external_reservoir" and boundary_axis is not None:
        boundary_axis_index = active_axis_names.index(boundary_axis)
        lower_contact, upper_contact = direct_boundary_label_masks[boundary_axis]
        lower_face_count = np.asarray(
            region_adjacency.boundary_face_counts[:, 2 * boundary_axis_index],
            dtype=np.int64,
        )
        upper_face_count = np.asarray(
            region_adjacency.boundary_face_counts[:, 2 * boundary_axis_index + 1],
            dtype=np.int64,
        )
        sample_axis_length = voxel_regions.label_image.shape[boundary_axis_index] * float(
            voxel_size
        )

        helper_coordinates: list[np.ndarray] = []
        helper_radii: list[float] = []
        helper_areas: list[float] = []
        helper_volumes: list[float] = []
        helper_surface_areas: list[float] = []
        boundary_throat_connections: list[tuple[int, int]] = []
        boundary_throat_area: list[float] = []
        boundary_throat_radius: list[float] = []
        boundary_throat_shape_factor_radius: list[float] = []
        boundary_throat_pore1_length: list[float] = []
        boundary_throat_core_length: list[float] = []
        boundary_throat_pore2_length: list[float] = []
        boundary_throat_centroids: list[np.ndarray] = []
        boundary_throat_face_counts: list[float] = []
        boundary_axis_face_balance: list[np.ndarray] = []
        boundary_support_radius_side1: list[float] = []
        boundary_support_radius_side2: list[float] = []
        inlet_helper_mask: list[bool] = []
        outlet_helper_mask: list[bool] = []

        def append_boundary_helper_pores(
            contact_mask: np.ndarray,
            face_counts: np.ndarray,
            side: str,
        ) -> None:
            boundary_coordinate = 0.0 if side == "lower" else sample_axis_length
            contact_indices = np.flatnonzero(contact_mask)
            for pore_index in contact_indices:
                helper_index = region_count + len(helper_coordinates)
                physical_coordinate = pore_coords[pore_index]
                helper_coordinate = physical_coordinate.copy()
                helper_coordinate[boundary_axis_index] = boundary_coordinate
                face_count = max(int(face_counts[pore_index]), 1)
                contact_area = float(face_count) * float(voxel_size) ** 2
                boundary_touch_radius_voxels = boundary_touch_radii_voxels[
                    pore_index,
                    2 * boundary_axis_index + (0 if side == "lower" else 1),
                ]
                if np.isfinite(boundary_touch_radius_voxels):
                    contact_radius = min(
                        float(pore_radius[pore_index]),
                        float(boundary_touch_radius_voxels) * float(voxel_size),
                    )
                else:
                    contact_radius = min(
                        float(pore_radius[pore_index]),
                        float(np.sqrt(contact_area / np.pi)),
                    )
                contact_radius = max(contact_radius, 0.5 * float(voxel_size))
                helper_radius = boundary_radius_scale * contact_radius
                center_to_boundary_length = abs(
                    float(physical_coordinate[boundary_axis_index] - boundary_coordinate)
                )
                total_boundary_length = max(
                    center_to_boundary_length,
                    3.01 * float(voxel_size),
                )
                internal_pore_length = max(0.67 * center_to_boundary_length, 0.0)
                boundary_core_length = max(
                    total_boundary_length - boundary_length_epsilon - internal_pore_length,
                    float(voxel_size),
                )

                helper_coordinates.append(helper_coordinate)
                helper_radii.append(helper_radius)
                helper_areas.append(np.pi * helper_radius**2)
                helper_volumes.append(0.0)
                helper_surface_areas.append(0.0)
                boundary_throat_area.append(contact_area)
                boundary_throat_radius.append(contact_radius)
                boundary_throat_shape_factor_radius.append(contact_radius)
                boundary_throat_centroids.append(helper_coordinate.copy())
                boundary_throat_face_counts.append(float(face_count))
                axis_balance = np.zeros(image_ndim, dtype=float)
                axis_balance[boundary_axis_index] = -face_count if side == "lower" else face_count
                boundary_axis_face_balance.append(axis_balance)
                if side == "lower":
                    boundary_throat_connections.append((helper_index, int(pore_index)))
                    boundary_throat_pore1_length.append(boundary_length_epsilon)
                    boundary_throat_pore2_length.append(internal_pore_length)
                    inlet_helper_mask.append(True)
                    outlet_helper_mask.append(False)
                else:
                    boundary_throat_connections.append((int(pore_index), helper_index))
                    boundary_throat_pore1_length.append(internal_pore_length)
                    boundary_throat_pore2_length.append(boundary_length_epsilon)
                    inlet_helper_mask.append(False)
                    outlet_helper_mask.append(True)
                boundary_throat_core_length.append(boundary_core_length)
                boundary_support_radius_side1.append(contact_radius)
                boundary_support_radius_side2.append(contact_radius)

        append_boundary_helper_pores(lower_contact, lower_face_count, "lower")
        append_boundary_helper_pores(upper_contact, upper_face_count, "upper")

        helper_count = len(helper_coordinates)
        if helper_count:
            pore_coords = np.vstack([pore_coords, np.asarray(helper_coordinates, dtype=float)])
            pore_radius = np.concatenate([pore_radius, np.asarray(helper_radii, dtype=float)])
            pore_area = np.concatenate([pore_area, np.asarray(helper_areas, dtype=float)])
            pore_volume = np.concatenate([pore_volume, np.asarray(helper_volumes, dtype=float)])
            pore_surface_area = np.concatenate(
                [pore_surface_area, np.asarray(helper_surface_areas, dtype=float)]
            )
            throat_region_pairs = np.vstack(
                [
                    throat_region_pairs,
                    np.asarray(boundary_throat_connections, dtype=np.int64),
                ]
            )
            throat_area = np.concatenate(
                [throat_area, np.asarray(boundary_throat_area, dtype=float)]
            )
            throat_radius = np.concatenate(
                [throat_radius, np.asarray(boundary_throat_radius, dtype=float)]
            )
            throat_shape_factor_radius = np.concatenate(
                [
                    throat_shape_factor_radius,
                    np.asarray(boundary_throat_shape_factor_radius, dtype=float),
                ]
            )
            pore1_length = np.concatenate(
                [pore1_length, np.asarray(boundary_throat_pore1_length, dtype=float)]
            )
            core_length = np.concatenate(
                [core_length, np.asarray(boundary_throat_core_length, dtype=float)]
            )
            pore2_length = np.concatenate(
                [pore2_length, np.asarray(boundary_throat_pore2_length, dtype=float)]
            )
            added_total_lengths = (
                pore1_length[-helper_count:]
                + core_length[-helper_count:]
                + pore2_length[-helper_count:]
            )
            throat_total_length = np.concatenate([throat_total_length, added_total_lengths])
            throat_centroid_coords = np.vstack(
                [throat_centroid_coords, np.asarray(boundary_throat_centroids, dtype=float)]
            )
            throat_face_counts = np.concatenate(
                [throat_face_counts, np.asarray(boundary_throat_face_counts, dtype=float)]
            )
            axis_face_balance = np.vstack(
                [
                    axis_face_balance,
                    np.asarray(boundary_axis_face_balance, dtype=float),
                ]
            )
            first_side_radius_physical = np.concatenate(
                [
                    first_side_radius_physical,
                    np.asarray(boundary_support_radius_side1, dtype=float),
                ]
            )
            second_side_radius_physical = np.concatenate(
                [
                    second_side_radius_physical,
                    np.asarray(boundary_support_radius_side2, dtype=float),
                ]
            )
            pore_boundary = np.concatenate([pore_boundary, np.ones(helper_count, dtype=bool)])
            for axis_name in active_axis_names:
                lower_mask, upper_mask = direct_boundary_label_masks[axis_name]
                direct_boundary_label_masks[axis_name] = (
                    np.concatenate([lower_mask, np.zeros(helper_count, dtype=bool)]),
                    np.concatenate([upper_mask, np.zeros(helper_count, dtype=bool)]),
                )
                connected_lower, connected_upper = connected_boundary_label_masks[axis_name]
                connected_boundary_label_masks[axis_name] = (
                    np.concatenate([connected_lower, np.zeros(helper_count, dtype=bool)]),
                    np.concatenate([connected_upper, np.zeros(helper_count, dtype=bool)]),
                )
            inlet_label = np.concatenate(
                [
                    np.zeros(region_count, dtype=bool),
                    np.asarray(inlet_helper_mask, dtype=bool),
                ]
            )
            outlet_label = np.concatenate(
                [
                    np.zeros(region_count, dtype=bool),
                    np.asarray(outlet_helper_mask, dtype=bool),
                ]
            )
            direct_boundary_label_masks[boundary_axis] = (inlet_label, outlet_label)
            region_count = int(pore_coords.shape[0])
            throat_count = int(throat_region_pairs.shape[0])

    pore_data: dict[str, np.ndarray] = {
        "radius_inscribed": pore_radius.copy(),
        "area": pore_area.copy(),
        "shape_factor": np.full(region_count, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "volume": pore_volume.copy(),
        "region_volume": pore_volume.copy(),
        "surface_area": pore_surface_area,
    }
    throat_data: dict[str, np.ndarray] = {
        "radius_inscribed": throat_radius.copy(),
        "shape_factor_radius": throat_shape_factor_radius.copy(),
        "area": throat_area.copy(),
        "shape_factor": np.full(throat_count, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "volume": np.zeros(throat_count, dtype=float),
        "total_length": throat_total_length,
        "pore1_length": pore1_length,
        "core_length": core_length,
        "pore2_length": pore2_length,
        "centroid": throat_centroid_coords,
        "face_count": throat_face_counts.astype(np.int64, copy=False),
        "axis_face_balance": axis_face_balance,
        "supporting_radius_side1": first_side_radius_physical,
        "supporting_radius_side2": second_side_radius_physical,
    }

    from voids.io.porespy import _apply_imperial_export_geometry_repairs, _derive_missing_geometry

    geometry_repair_summary = _apply_imperial_export_geometry_repairs(
        pore_data,
        throat_data,
        throat_region_pairs,
        num_pores=region_count,
        random_seed=1001,
    )
    _redistribute_region_volumes_like_imperial_export(
        pore_data,
        throat_data,
        throat_region_pairs,
    )
    _derive_missing_geometry(pore_data, throat_data)
    boundary_face_count = np.asarray(
        region_adjacency.boundary_face_counts.sum(axis=1),
        dtype=np.int64,
    )
    if boundary_face_count.size < region_count:
        boundary_face_count = np.concatenate(
            [
                boundary_face_count,
                np.zeros(region_count - boundary_face_count.size, dtype=np.int64),
            ]
        )

    network_dict: dict[str, np.ndarray] = {
        "pore.coords": pore_coords,
        "throat.conns": throat_region_pairs,
        "pore.radius_inscribed": pore_data["radius_inscribed"],
        "pore.area": pore_data["area"],
        "pore.shape_factor": pore_data["shape_factor"],
        "pore.volume": pore_data["volume"],
        "pore.region_volume": pore_data["region_volume"],
        "pore.surface_area": pore_data["surface_area"],
        "throat.radius_inscribed": throat_data["radius_inscribed"],
        "throat.shape_factor_radius": throat_data["shape_factor_radius"],
        "throat.cross_sectional_area": throat_data["area"],
        "throat.shape_factor": throat_data["shape_factor"],
        "throat.volume": throat_data["volume"],
        "throat.total_length": throat_data["length"],
        "throat.conduit_lengths.pore1": throat_data["pore1_length"],
        "throat.conduit_lengths.throat": throat_data["core_length"],
        "throat.conduit_lengths.pore2": throat_data["pore2_length"],
        "throat.centroid": throat_data["centroid"],
        "throat.face_count": throat_data["face_count"],
        "throat.axis_face_balance": throat_data["axis_face_balance"],
        "throat.supporting_radius_side1": throat_data["supporting_radius_side1"],
        "throat.supporting_radius_side2": throat_data["supporting_radius_side2"],
        "pore.boundary_face_count": boundary_face_count,
        "throat.geometry_repairs_mode": np.full(
            throat_count, geometry_repair_summary["mode"], dtype=object
        ),
    }

    for axis_name in active_axis_names:
        lower_contact, upper_contact = direct_boundary_label_masks[axis_name]
        network_dict[f"pore.inlet_{axis_name}min"] = lower_contact
        network_dict[f"pore.outlet_{axis_name}max"] = upper_contact
        connected_lower, connected_upper = connected_boundary_label_masks[axis_name]
        network_dict[f"pore.boundary_connected_inlet_{axis_name}min"] = connected_lower
        network_dict[f"pore.boundary_connected_outlet_{axis_name}max"] = connected_upper
    network_dict["pore.boundary"] = pore_boundary
    return network_dict


def extract_maximal_ball_network_dict(
    void_phase_mask: np.ndarray,
    *,
    voxel_size: float,
    distance_map_backend: str = "auto",
    edt_parallel_threads: int | None = None,
    settings: MaximalBallSettings | None = None,
    apply_boundary_clipping: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    flow_boundary_mode: str = "direct",
    boundary_axis: str | None = None,
    boundary_length_epsilon: float = 1.0e-300,
    boundary_radius_scale: float = 1.1,
    throat_area_mode: str = "face_count",
    throat_shape_factor_radius_mode: str = "inscribed",
    throat_anchor_mode: str = "second_side",
) -> MaximalBallNetworkDictResult:
    """Run the staged native maximal-ball path and assemble a network mapping."""

    extraction_result = extract_maximal_ball_regions(
        void_phase_mask,
        distance_map_backend=distance_map_backend,
        edt_parallel_threads=edt_parallel_threads,
        settings=settings,
        apply_boundary_clipping=apply_boundary_clipping,
    )
    network_dict = build_network_dict_from_maximal_ball_regions(
        extraction_result,
        voxel_size=voxel_size,
        axis_names=axis_names,
        flow_boundary_mode=flow_boundary_mode,
        boundary_axis=boundary_axis,
        boundary_length_epsilon=boundary_length_epsilon,
        boundary_radius_scale=boundary_radius_scale,
        throat_area_mode=throat_area_mode,
        throat_shape_factor_radius_mode=throat_shape_factor_radius_mode,
        throat_anchor_mode=throat_anchor_mode,
    )
    return MaximalBallNetworkDictResult(
        network_dict=network_dict,
        extraction=extraction_result,
    )


def extract_maximal_ball_candidates(
    void_phase_mask: np.ndarray,
    *,
    distance_map_backend: str = "auto",
    edt_parallel_threads: int | None = None,
    settings: MaximalBallSettings | None = None,
    apply_boundary_clipping: bool = True,
) -> MaximalBallCandidates:
    """Compute and suppress maximal-ball candidates for a void-phase image."""

    raw_settings = settings or MaximalBallSettings()
    distance_map = compute_maximal_ball_radius_field(
        void_phase_mask,
        backend=distance_map_backend,
        edt_parallel_threads=edt_parallel_threads,
        mode=raw_settings.radius_field_mode,
    )
    resolved_settings = resolve_maximal_ball_settings(distance_map, raw_settings)
    working_distance_map = (
        clip_distance_map_to_domain_boundaries(distance_map, settings=resolved_settings)
        if apply_boundary_clipping
        else np.asarray(distance_map, dtype=float)
    )
    center_indices, radii_voxels, candidate_mask = find_maximal_ball_candidates(
        working_distance_map,
        minimal_radius_voxels=resolved_settings.minimal_pore_radius_voxels,
        selection_mode=resolved_settings.candidate_selection_mode,
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
    "MaximalBallExtractionDiagnostics",
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
    "compute_maximal_ball_radius_field",
    "compute_void_distance_map",
    "extract_maximal_ball_network_dict",
    "extract_maximal_ball_regions",
    "extract_maximal_ball_candidates",
    "find_maximal_ball_candidates",
    "grow_root_regions_by_neighbor_priority",
    "grow_root_regions_by_radius",
    "initialize_root_region_labels",
    "measure_region_adjacency",
    "reassign_region_boundary_voxels_by_majority",
    "resolve_maximal_ball_settings",
    "retreat_mixed_region_boundary_voxels",
    "seed_root_region_ball_interiors",
    "smooth_radius_field_local_relaxation",
    "stamp_retained_ball_centers_to_root_labels",
    "summarize_maximal_ball_extraction_diagnostics",
    "suppress_overlapping_maximal_balls",
]
