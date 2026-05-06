from __future__ import annotations

import numpy as np
import pytest

from voids.image.maximal_ball import (
    MaximalBallSettings,
    clip_distance_map_to_domain_boundaries,
    compute_void_distance_map,
    extract_maximal_ball_candidates,
    find_maximal_ball_candidates,
    resolve_maximal_ball_settings,
    suppress_overlapping_maximal_balls,
)


def test_compute_void_distance_map_matches_expected_center_radius() -> None:
    """A centered cubic void should yield the expected Euclidean center radius."""

    void_phase_mask = np.zeros((5, 5, 5), dtype=bool)
    void_phase_mask[1:4, 1:4, 1:4] = True

    distance_map = compute_void_distance_map(void_phase_mask, backend="scipy")

    assert distance_map.shape == void_phase_mask.shape
    assert distance_map[2, 2, 2] == pytest.approx(2.0)
    assert np.count_nonzero(distance_map) == 27


def test_resolve_maximal_ball_settings_matches_imperial_default_logic() -> None:
    """Imperial-style defaults should be resolved from the mean positive radius."""

    distance_map = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    settings = resolve_maximal_ball_settings(distance_map)

    assert settings.minimal_pore_radius_voxels == pytest.approx(0.8)
    assert settings.medial_surface_noise_voxels == pytest.approx(1.8)
    assert settings.retention_radius_offset_voxels == pytest.approx(0.8)
    assert settings.hierarchy_length_factor == pytest.approx(0.6)
    assert settings.hierarchy_radius_factor == pytest.approx(1.1)


def test_clip_distance_map_to_domain_boundaries_reduces_boundary_radii() -> None:
    """Boundary clipping should shrink overly large near-wall radii."""

    distance_map = np.full((5, 5, 5), 6.0, dtype=float)
    settings = resolve_maximal_ball_settings(
        distance_map,
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )

    clipped_distance_map = clip_distance_map_to_domain_boundaries(
        distance_map,
        settings=settings,
    )

    assert clipped_distance_map[0, 0, 0] < distance_map[0, 0, 0]
    assert clipped_distance_map[2, 2, 2] > clipped_distance_map[0, 0, 0]


def test_find_maximal_ball_candidates_detects_single_center_peak() -> None:
    """A centered cubic void should produce one maximal-ball candidate at the center."""

    void_phase_mask = np.zeros((5, 5, 5), dtype=bool)
    void_phase_mask[1:4, 1:4, 1:4] = True
    distance_map = compute_void_distance_map(void_phase_mask, backend="scipy")

    center_indices, radii_voxels, candidate_mask = find_maximal_ball_candidates(
        distance_map,
        minimal_radius_voxels=1.5,
    )

    assert candidate_mask[2, 2, 2]
    assert center_indices.shape == (1, 3)
    assert np.array_equal(center_indices[0], np.array([2, 2, 2]))
    assert radii_voxels[0] == pytest.approx(2.0)


def test_suppress_overlapping_maximal_balls_prefers_larger_candidates() -> None:
    """Overlap suppression should retain the larger ball when two candidates compete."""

    center_indices = np.array(
        [
            [5, 5, 5],
            [5, 5, 6],
            [12, 12, 12],
        ],
        dtype=np.int64,
    )
    radii_voxels = np.array([4.0, 3.5, 2.0], dtype=float)
    settings = resolve_maximal_ball_settings(
        np.array([4.0, 3.5, 2.0]),
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )

    retained_mask = suppress_overlapping_maximal_balls(
        center_indices,
        radii_voxels,
        settings=settings,
    )

    assert np.array_equal(retained_mask, np.array([True, False, True]))


def test_extract_maximal_ball_candidates_returns_retained_candidates_in_radius_order() -> None:
    """The staged maximal-ball extractor should expose retained candidates in sorted order."""

    void_phase_mask = np.zeros((7, 7, 7), dtype=bool)
    void_phase_mask[1:4, 1:4, 1:4] = True
    void_phase_mask[4:7, 4:7, 4:7] = True

    maximal_ball_data = extract_maximal_ball_candidates(
        void_phase_mask,
        distance_map_backend="scipy",
        settings=MaximalBallSettings(minimal_pore_radius_voxels=1.0),
        apply_boundary_clipping=False,
    )

    assert maximal_ball_data.center_indices.shape[1] == 3
    assert maximal_ball_data.retained_center_indices.shape[0] >= 2
    assert np.all(
        maximal_ball_data.retained_radii_voxels[:-1] >= maximal_ball_data.retained_radii_voxels[1:]
    )
