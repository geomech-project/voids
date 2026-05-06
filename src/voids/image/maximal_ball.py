from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

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
    "MaximalBallSettings",
    "ResolvedMaximalBallSettings",
    "clip_distance_map_to_domain_boundaries",
    "compute_void_distance_map",
    "extract_maximal_ball_candidates",
    "find_maximal_ball_candidates",
    "resolve_maximal_ball_settings",
    "suppress_overlapping_maximal_balls",
]
