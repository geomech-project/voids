from __future__ import annotations

import numpy as np
import pytest

from voids.image.maximal_ball import (
    MaximalBallCandidates,
    MaximalBallExtractionResult,
    MaximalBallHierarchy,
    MaximalBallSettings,
    MaximalBallVoxelRegions,
    assign_voxel_regions_from_hierarchy,
    build_network_dict_from_maximal_ball_regions,
    build_maximal_ball_hierarchy,
    clip_distance_map_to_domain_boundaries,
    compute_maximal_ball_radius_field,
    compute_void_distance_map,
    extract_maximal_ball_network_dict,
    extract_maximal_ball_regions,
    extract_maximal_ball_candidates,
    find_maximal_ball_candidates,
    grow_root_regions_by_radius,
    grow_root_regions_by_neighbor_priority,
    initialize_root_region_labels,
    measure_region_adjacency,
    reassign_region_boundary_voxels_by_majority,
    resolve_maximal_ball_settings,
    retreat_mixed_region_boundary_voxels,
    seed_root_region_ball_interiors,
    stamp_retained_ball_centers_to_root_labels,
    summarize_maximal_ball_extraction_diagnostics,
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


def test_compute_maximal_ball_radius_field_matches_imperial_half_voxel_shift() -> None:
    """The Imperial radius field should equal EDT minus half a voxel in the void."""

    void_phase_mask = np.zeros((5, 5, 5), dtype=bool)
    void_phase_mask[1:4, 1:4, 1:4] = True

    radius_field = compute_maximal_ball_radius_field(
        void_phase_mask,
        backend="scipy",
        mode="imperial_pnextract",
    )

    assert radius_field.shape == void_phase_mask.shape
    assert radius_field[2, 2, 2] == pytest.approx(1.5)
    assert radius_field[1, 1, 1] == pytest.approx(0.5)
    assert np.count_nonzero(radius_field) == 27


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
        selection_mode="local_maxima",
    )

    assert candidate_mask[2, 2, 2]
    assert center_indices.shape == (1, 3)
    assert np.array_equal(center_indices[0], np.array([2, 2, 2]))
    assert radii_voxels[0] == pytest.approx(2.0)


def test_find_maximal_ball_candidates_threshold_mode_keeps_all_above_threshold_voxels() -> None:
    """Threshold-all mode should keep every above-threshold voxel candidate."""

    distance_map = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.7, 0.9, 0.7],
            [0.0, 0.5, 0.0],
        ]
    )

    center_indices, radii_voxels, candidate_mask = find_maximal_ball_candidates(
        distance_map,
        minimal_radius_voxels=0.6,
        selection_mode="threshold_all",
    )

    assert candidate_mask.sum() == 3
    assert center_indices.shape == (3, 2)
    assert radii_voxels[0] == pytest.approx(0.9)
    assert set(map(tuple, center_indices.tolist())) == {(1, 0), (1, 1), (1, 2)}


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


def test_build_maximal_ball_hierarchy_links_smaller_supported_ball_to_larger_ball() -> None:
    """A supported nearby smaller ball should attach under the larger retained ball."""

    center_indices = np.array(
        [
            [4, 4, 4],
            [5, 4, 4],
        ],
        dtype=np.int64,
    )
    radii_voxels = np.array([4.0, 2.5], dtype=float)
    candidate_mask = np.zeros((10, 10, 10), dtype=bool)
    candidate_mask[4, 4, 4] = True
    candidate_mask[5, 4, 4] = True
    retained_mask = np.array([True, True], dtype=bool)
    distance_map = np.zeros((10, 10, 10), dtype=float)
    distance_map[4, 4, 4] = 4.0
    distance_map[5, 4, 4] = 2.5
    distance_map[4, 4, 4] = 4.0
    settings = resolve_maximal_ball_settings(
        distance_map,
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )
    maximal_ball_data = MaximalBallCandidates(
        center_indices=center_indices,
        radii_voxels=radii_voxels,
        candidate_mask=candidate_mask,
        retained_mask=retained_mask,
        distance_map=np.maximum(distance_map, 3.0),
        settings=settings,
    )

    hierarchy = build_maximal_ball_hierarchy(maximal_ball_data)

    assert hierarchy.parent_indices.shape == hierarchy.radii_voxels.shape
    assert np.all(hierarchy.parent_indices <= np.arange(hierarchy.parent_indices.size))
    assert np.array_equal(hierarchy.parent_indices, np.array([0, 0], dtype=np.int64))
    assert np.array_equal(hierarchy.master_indices, np.array([0, 0], dtype=np.int64))


def test_build_maximal_ball_hierarchy_keeps_separated_balls_as_independent_roots() -> None:
    """Well-separated retained balls should remain separate hierarchy roots."""

    center_indices = np.array(
        [
            [2, 2, 2],
            [12, 12, 12],
        ],
        dtype=np.int64,
    )
    radii_voxels = np.array([3.0, 2.5], dtype=float)
    candidate_mask = np.zeros((16, 16, 16), dtype=bool)
    candidate_mask[2, 2, 2] = True
    candidate_mask[12, 12, 12] = True
    retained_mask = np.array([True, True], dtype=bool)
    distance_map = np.zeros((16, 16, 16), dtype=float)
    distance_map[2, 2, 2] = 3.0
    distance_map[12, 12, 12] = 2.5
    settings = resolve_maximal_ball_settings(
        distance_map,
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )

    maximal_ball_data = MaximalBallCandidates(
        center_indices=center_indices,
        radii_voxels=radii_voxels,
        candidate_mask=candidate_mask,
        retained_mask=retained_mask,
        distance_map=distance_map,
        settings=settings,
    )
    hierarchy = build_maximal_ball_hierarchy(maximal_ball_data)

    assert np.array_equal(hierarchy.parent_indices, np.array([0, 1], dtype=np.int64))
    assert np.array_equal(hierarchy.master_indices, np.array([0, 1], dtype=np.int64))
    assert np.array_equal(hierarchy.hierarchy_levels, np.array([0, 0], dtype=np.int64))


def test_initialize_root_region_labels_seeds_root_centers() -> None:
    """Root-region initialization should label only hierarchy-root centers initially."""

    center_indices = np.array([[2, 2, 2], [6, 6, 6]], dtype=np.int64)
    radii_voxels = np.array([3.0, 2.5], dtype=float)
    candidate_mask = np.zeros((9, 9, 9), dtype=bool)
    retained_mask = np.array([True, True], dtype=bool)
    distance_map = np.zeros((9, 9, 9), dtype=float)
    settings = resolve_maximal_ball_settings(
        np.array([3.0, 2.5]),
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )
    maximal_ball_data = MaximalBallCandidates(
        center_indices=center_indices,
        radii_voxels=radii_voxels,
        candidate_mask=candidate_mask,
        retained_mask=retained_mask,
        distance_map=distance_map,
        settings=settings,
    )
    hierarchy = build_maximal_ball_hierarchy(maximal_ball_data)
    void_phase_mask = np.ones((9, 9, 9), dtype=bool)

    voxel_regions = initialize_root_region_labels(void_phase_mask, hierarchy)

    assert voxel_regions.label_image[2, 2, 2] == 0
    assert voxel_regions.label_image[6, 6, 6] == 1
    assert np.count_nonzero(voxel_regions.assigned_void_mask) == 2


def test_seed_root_region_ball_interiors_assigns_local_ball_neighborhoods() -> None:
    """Ball-interior seeding should label a compact neighborhood around each retained ball."""

    center_indices = np.array([[3, 3, 3]], dtype=np.int64)
    radii_voxels = np.array([4.0], dtype=float)
    candidate_mask = np.zeros((9, 9, 9), dtype=bool)
    retained_mask = np.array([True], dtype=bool)
    distance_map = np.ones((9, 9, 9), dtype=float)
    settings = resolve_maximal_ball_settings(
        np.array([4.0]),
        MaximalBallSettings(minimal_pore_radius_voxels=1.75),
    )
    hierarchy = build_maximal_ball_hierarchy(
        MaximalBallCandidates(
            center_indices=center_indices,
            radii_voxels=radii_voxels,
            candidate_mask=candidate_mask,
            retained_mask=retained_mask,
            distance_map=distance_map,
            settings=settings,
        )
    )
    void_phase_mask = np.ones((9, 9, 9), dtype=bool)
    voxel_regions = initialize_root_region_labels(void_phase_mask, hierarchy)

    seeded_regions = seed_root_region_ball_interiors(void_phase_mask, hierarchy, voxel_regions)

    assert seeded_regions.label_image[3, 3, 3] == 0
    assert seeded_regions.label_image[4, 3, 3] == 0
    assert np.count_nonzero(seeded_regions.assigned_void_mask) > 1


def test_grow_root_regions_by_radius_assigns_supported_unassigned_voxel() -> None:
    """Radius-aware growth should assign an unassigned voxel with enough supporting neighbors."""

    void_phase_mask = np.ones((5, 5, 5), dtype=bool)
    distance_map = np.zeros((5, 5, 5), dtype=float)
    distance_map[2, 2, 2] = 2.0
    distance_map[1, 2, 2] = 3.0
    distance_map[3, 2, 2] = 3.0
    distance_map[2, 1, 2] = 3.0
    label_image = np.full((5, 5, 5), -1, dtype=np.int64)
    label_image[1, 2, 2] = 0
    label_image[3, 2, 2] = 0
    label_image[2, 1, 2] = 0

    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0], dtype=np.int64),
        root_labels=np.array([0], dtype=np.int64),
        root_center_indices=np.array([[1, 2, 2]], dtype=np.int64),
        root_radii_voxels=np.array([3.0], dtype=float),
        root_of_ball_index=np.array([0], dtype=np.int64),
        unassigned_label=-1,
    )
    voxel_regions = grow_root_regions_by_radius(
        void_phase_mask,
        distance_map,
        voxel_regions,
        minimum_supporting_neighbors=2,
        require_strictly_larger_radius=True,
        iterations=1,
    )

    assert voxel_regions.label_image[2, 2, 2] == 0


def test_assign_voxel_regions_from_hierarchy_expands_beyond_root_centers() -> None:
    """The staged voxel assignment should grow labels beyond the root-center seeds."""

    void_phase_mask = np.zeros((9, 9, 9), dtype=bool)
    void_phase_mask[1:8, 1:8, 1:8] = True
    maximal_ball_data = extract_maximal_ball_candidates(
        void_phase_mask,
        distance_map_backend="scipy",
        settings=MaximalBallSettings(minimal_pore_radius_voxels=1.0),
        apply_boundary_clipping=False,
    )
    hierarchy = build_maximal_ball_hierarchy(maximal_ball_data)

    voxel_regions = assign_voxel_regions_from_hierarchy(void_phase_mask, hierarchy)

    assert (
        np.count_nonzero(voxel_regions.assigned_void_mask) >= voxel_regions.root_ball_indices.size
    )
    assert np.count_nonzero(voxel_regions.assigned_void_mask) > voxel_regions.root_ball_indices.size


def test_reassign_region_boundary_voxels_by_majority_switches_weak_boundary_label() -> None:
    """A weakly supported labeled voxel should adopt the stronger competing label."""

    void_phase_mask = np.ones((5, 5, 1), dtype=bool)
    distance_map = np.ones((5, 5, 1), dtype=float)
    label_image = np.full((5, 5, 1), -1, dtype=np.int64)
    label_image[2, 2, 0] = 0
    label_image[1, 2, 0] = 1
    label_image[3, 2, 0] = 1
    label_image[2, 1, 0] = 1

    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1], dtype=np.int64),
        root_labels=np.array([0, 1], dtype=np.int64),
        root_center_indices=np.array([[2, 2, 0], [1, 2, 0]], dtype=np.int64),
        root_radii_voxels=np.array([1.0, 1.0], dtype=float),
        root_of_ball_index=np.array([0, 1], dtype=np.int64),
        unassigned_label=-1,
    )

    reassigned_regions = reassign_region_boundary_voxels_by_majority(
        void_phase_mask,
        distance_map,
        voxel_regions,
        radius_support_mode="any",
        iterations=1,
    )

    assert reassigned_regions.label_image[2, 2, 0] == 1


def test_retreat_mixed_region_boundary_voxels_unassigns_mixed_interface_voxel() -> None:
    """A voxel touching both same and different labels should retreat to unassigned."""

    void_phase_mask = np.ones((5, 5, 1), dtype=bool)
    label_image = np.full((5, 5, 1), -1, dtype=np.int64)
    label_image[2, 2, 0] = 0
    label_image[1, 2, 0] = 0
    label_image[3, 2, 0] = 1

    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1], dtype=np.int64),
        root_labels=np.array([0, 1], dtype=np.int64),
        root_center_indices=np.array([[2, 2, 0], [3, 2, 0]], dtype=np.int64),
        root_radii_voxels=np.array([1.0, 1.0], dtype=float),
        root_of_ball_index=np.array([0, 1], dtype=np.int64),
        unassigned_label=-1,
    )

    retreated_regions = retreat_mixed_region_boundary_voxels(
        void_phase_mask,
        voxel_regions,
    )

    assert retreated_regions.label_image[2, 2, 0] == -1


def test_grow_root_regions_by_neighbor_priority_propagates_without_radius_filter() -> None:
    """Late sweep growth should fill an unassigned voxel from a directly touching label."""

    void_phase_mask = np.ones((4, 4, 1), dtype=bool)
    label_image = np.full((4, 4, 1), -1, dtype=np.int64)
    label_image[1, 1, 0] = 0

    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0], dtype=np.int64),
        root_labels=np.array([0], dtype=np.int64),
        root_center_indices=np.array([[1, 1, 0]], dtype=np.int64),
        root_radii_voxels=np.array([1.0], dtype=float),
        root_of_ball_index=np.array([0], dtype=np.int64),
        unassigned_label=-1,
    )

    grown_regions = grow_root_regions_by_neighbor_priority(
        void_phase_mask,
        voxel_regions,
        iterations=2,
    )

    assert np.count_nonzero(grown_regions.assigned_void_mask) > 1


def test_stamp_retained_ball_centers_to_root_labels_restores_center_assignment() -> None:
    """Retained-ball centers should be restored after temporary retreat passes."""

    label_image = np.full((5, 5, 1), -1, dtype=np.int64)
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0], dtype=np.int64),
        root_labels=np.array([0], dtype=np.int64),
        root_center_indices=np.array([[2, 2, 0]], dtype=np.int64),
        root_radii_voxels=np.array([2.0], dtype=float),
        root_of_ball_index=np.array([0], dtype=np.int64),
        unassigned_label=-1,
    )
    hierarchy = MaximalBallHierarchy(
        center_indices=np.array([[2, 2, 0]], dtype=np.int64),
        center_coordinates=np.array([[1.5, 1.5, -0.5]], dtype=float),
        radii_voxels=np.array([2.0], dtype=float),
        parent_indices=np.array([0], dtype=np.int64),
        master_indices=np.array([0], dtype=np.int64),
        hierarchy_levels=np.array([0], dtype=np.int64),
        distance_map=np.ones((5, 5, 1), dtype=float),
        settings=resolve_maximal_ball_settings(
            np.ones((5, 5, 1), dtype=float),
            MaximalBallSettings(minimal_pore_radius_voxels=1.0),
        ),
    )

    stamped_regions = stamp_retained_ball_centers_to_root_labels(
        voxel_regions,
        hierarchy,
    )

    assert stamped_regions.label_image[2, 2, 0] == 0


def test_measure_region_adjacency_extracts_one_interface_between_two_regions() -> None:
    """Two adjacent labeled regions should yield one throat candidate with the right centroid."""

    void_phase_mask = np.ones((4, 3, 3), dtype=bool)
    label_image = np.full((4, 3, 3), -1, dtype=np.int64)
    label_image[0:2, :, :] = 0
    label_image[2:4, :, :] = 1
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1], dtype=np.int64),
        root_labels=np.array([0, 1], dtype=np.int64),
        root_center_indices=np.array([[0, 1, 1], [3, 1, 1]], dtype=np.int64),
        root_radii_voxels=np.array([2.0, 2.0], dtype=float),
        root_of_ball_index=np.array([0, 1], dtype=np.int64),
        unassigned_label=-1,
    )

    distance_map = np.ones((4, 3, 3), dtype=float)
    distance_map[1, :, :] = 2.0
    distance_map[2, :, :] = 3.0
    region_adjacency = measure_region_adjacency(
        void_phase_mask,
        voxel_regions,
        distance_map=distance_map,
    )

    assert np.array_equal(region_adjacency.region_volume_voxels, np.array([18, 18], dtype=np.int64))
    assert np.array_equal(region_adjacency.throat_region_pairs, np.array([[0, 1]], dtype=np.int64))
    assert np.array_equal(region_adjacency.throat_face_counts, np.array([9], dtype=np.int64))
    assert np.allclose(region_adjacency.throat_axis_face_balance, np.array([[9.0, 0.0, 0.0]]))
    assert np.allclose(region_adjacency.throat_centroid_indices, np.array([[1.5, 1.0, 1.0]]))
    assert np.allclose(region_adjacency.throat_max_touch_radius_side1_voxels, np.array([2.0]))
    assert np.allclose(region_adjacency.throat_max_touch_radius_side2_voxels, np.array([3.0]))
    assert np.array_equal(region_adjacency.throat_max_touch_index_side1, np.array([[1, 0, 0]]))
    assert np.array_equal(region_adjacency.throat_max_touch_index_side2, np.array([[2, 0, 0]]))


def test_measure_region_adjacency_reports_boundary_contact_faces() -> None:
    """Boundary-face accounting should identify which pore regions touch each sample side."""

    void_phase_mask = np.ones((3, 3, 1), dtype=bool)
    label_image = np.full((3, 3, 1), -1, dtype=np.int64)
    label_image[0:2, :, :] = 0
    label_image[2:3, :, :] = 1
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1], dtype=np.int64),
        root_labels=np.array([0, 1], dtype=np.int64),
        root_center_indices=np.array([[0, 1, 0], [2, 1, 0]], dtype=np.int64),
        root_radii_voxels=np.array([1.5, 1.0], dtype=float),
        root_of_ball_index=np.array([0, 1], dtype=np.int64),
        unassigned_label=-1,
    )

    region_adjacency = measure_region_adjacency(void_phase_mask, voxel_regions)

    assert np.array_equal(
        region_adjacency.boundary_face_counts,
        np.array(
            [
                [3, 0, 2, 2, 6, 6],
                [0, 3, 1, 1, 3, 3],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        region_adjacency.region_surface_face_counts,
        np.array([22, 14], dtype=np.int64),
    )


def test_build_network_dict_from_maximal_ball_regions_assembles_expected_fields() -> None:
    """Region geometry should assemble into a consistent pore-network mapping."""

    void_phase_mask = np.ones((3, 3, 1), dtype=bool)
    label_image = np.full((3, 3, 1), -1, dtype=np.int64)
    label_image[0:2, :, :] = 0
    label_image[2:3, :, :] = 1
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1], dtype=np.int64),
        root_labels=np.array([0, 1], dtype=np.int64),
        root_center_indices=np.array([[0, 1, 0], [2, 1, 0]], dtype=np.int64),
        root_radii_voxels=np.array([1.5, 1.0], dtype=float),
        root_of_ball_index=np.array([0, 1], dtype=np.int64),
        unassigned_label=-1,
    )
    settings = resolve_maximal_ball_settings(
        np.array([1.0, 1.5, 2.0], dtype=float),
        MaximalBallSettings(minimal_pore_radius_voxels=1.0),
    )
    hierarchy = MaximalBallHierarchy(
        center_indices=voxel_regions.root_center_indices.copy(),
        center_coordinates=voxel_regions.root_center_indices.astype(float) - 0.5,
        radii_voxels=voxel_regions.root_radii_voxels.copy(),
        parent_indices=np.array([0, 1], dtype=np.int64),
        master_indices=np.array([0, 1], dtype=np.int64),
        hierarchy_levels=np.array([0, 0], dtype=np.int64),
        distance_map=np.ones((3, 3, 1), dtype=float),
        settings=settings,
    )
    extraction_result = MaximalBallExtractionResult(
        candidates=MaximalBallCandidates(
            center_indices=voxel_regions.root_center_indices.copy(),
            radii_voxels=voxel_regions.root_radii_voxels.copy(),
            candidate_mask=np.zeros((3, 3, 1), dtype=bool),
            retained_mask=np.array([True, True], dtype=bool),
            distance_map=np.ones((3, 3, 1), dtype=float),
            settings=settings,
        ),
        hierarchy=hierarchy,
        voxel_regions=voxel_regions,
        region_adjacency=measure_region_adjacency(void_phase_mask, voxel_regions),
    )

    network_dict = build_network_dict_from_maximal_ball_regions(
        extraction_result,
        voxel_size=2.0,
    )

    assert network_dict["pore.coords"].shape == (2, 3)
    assert np.array_equal(network_dict["throat.conns"], np.array([[0, 1]], dtype=np.int64))
    assert np.array_equal(network_dict["pore.inlet_xmin"], np.array([True, False]))
    assert np.array_equal(network_dict["pore.outlet_xmax"], np.array([False, True]))
    assert np.all(network_dict["throat.total_length"] > 0.0)
    assert np.all(network_dict["throat.conduit_lengths.pore1"] >= 0.0)
    assert np.all(network_dict["throat.conduit_lengths.throat"] > 0.0)
    assert np.all(network_dict["throat.conduit_lengths.pore2"] >= 0.0)
    assert network_dict["pore.volume"].sum() + network_dict["throat.volume"].sum() == pytest.approx(
        72.0
    )
    assert network_dict["pore.volume"][0] == pytest.approx(33.698087479653445)
    assert network_dict["pore.volume"][1] == pytest.approx(12.276656553569734)
    assert network_dict["throat.volume"][0] == pytest.approx(26.02525596677682)
    assert network_dict["throat.cross_sectional_area"][0] == pytest.approx(12.0)
    assert network_dict["throat.radius_inscribed"][0] == pytest.approx(1.9544100476116797)
    assert network_dict["throat.shape_factor"][0] == pytest.approx(1.0 / (4.0 * np.pi))
    assert np.allclose(
        network_dict["pore.shape_factor"],
        np.array([1.0 / (4.0 * np.pi), 1.0 / (4.0 * np.pi)]),
    )

    reservoir_network_dict = build_network_dict_from_maximal_ball_regions(
        extraction_result,
        voxel_size=2.0,
        flow_boundary_mode="external_reservoir",
        boundary_axis="x",
    )

    assert reservoir_network_dict["pore.coords"].shape == (4, 3)
    assert np.array_equal(
        reservoir_network_dict["throat.conns"],
        np.array([[0, 1], [2, 0], [1, 3]], dtype=np.int64),
    )
    assert np.array_equal(
        reservoir_network_dict["pore.inlet_xmin"],
        np.array([False, False, True, False]),
    )
    assert np.array_equal(
        reservoir_network_dict["pore.outlet_xmax"],
        np.array([False, False, False, True]),
    )
    assert np.array_equal(
        reservoir_network_dict["pore.boundary_connected_inlet_xmin"],
        np.array([True, False, False, False]),
    )
    assert np.array_equal(
        reservoir_network_dict["pore.boundary_connected_outlet_xmax"],
        np.array([False, True, False, False]),
    )
    assert np.all(reservoir_network_dict["throat.conduit_lengths.pore1"] >= 0.0)
    assert np.all(reservoir_network_dict["throat.conduit_lengths.throat"] > 0.0)
    assert np.all(reservoir_network_dict["throat.conduit_lengths.pore2"] >= 0.0)


def test_build_network_dict_from_maximal_ball_regions_resolves_overlapping_boundary_labels() -> (
    None
):
    """A pore touching both sample sides should be assigned to only one BC label per axis."""

    void_phase_mask = np.ones((4, 2, 1), dtype=bool)
    label_image = np.zeros((4, 2, 1), dtype=np.int64)
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0], dtype=np.int64),
        root_labels=np.array([0], dtype=np.int64),
        root_center_indices=np.array([[0, 1, 0]], dtype=np.int64),
        root_radii_voxels=np.array([2.0], dtype=float),
        root_of_ball_index=np.array([0], dtype=np.int64),
        unassigned_label=-1,
    )
    settings = resolve_maximal_ball_settings(
        np.array([1.0, 2.0], dtype=float),
        MaximalBallSettings(minimal_pore_radius_voxels=1.0),
    )
    extraction_result = MaximalBallExtractionResult(
        candidates=MaximalBallCandidates(
            center_indices=voxel_regions.root_center_indices.copy(),
            radii_voxels=voxel_regions.root_radii_voxels.copy(),
            candidate_mask=np.zeros((4, 2, 1), dtype=bool),
            retained_mask=np.array([True], dtype=bool),
            distance_map=np.ones((4, 2, 1), dtype=float),
            settings=settings,
        ),
        hierarchy=MaximalBallHierarchy(
            center_indices=voxel_regions.root_center_indices.copy(),
            center_coordinates=voxel_regions.root_center_indices.astype(float) - 0.5,
            radii_voxels=voxel_regions.root_radii_voxels.copy(),
            parent_indices=np.array([0], dtype=np.int64),
            master_indices=np.array([0], dtype=np.int64),
            hierarchy_levels=np.array([0], dtype=np.int64),
            distance_map=np.ones((4, 2, 1), dtype=float),
            settings=settings,
        ),
        voxel_regions=voxel_regions,
        region_adjacency=measure_region_adjacency(void_phase_mask, voxel_regions),
    )

    network_dict = build_network_dict_from_maximal_ball_regions(
        extraction_result,
        voxel_size=1.0,
    )

    assert np.array_equal(network_dict["pore.inlet_xmin"], np.array([True]))
    assert np.array_equal(network_dict["pore.outlet_xmax"], np.array([False]))


def test_summarize_maximal_ball_extraction_diagnostics_reports_unassigned_and_isolated_regions() -> (
    None
):
    """Extraction diagnostics should expose unassigned voids and isolated regions."""

    void_phase_mask = np.ones((4, 3, 1), dtype=bool)
    label_image = np.full((4, 3, 1), -1, dtype=np.int64)
    label_image[0:2, :, :] = 0
    label_image[2, 0:2, :] = 1
    label_image[3, 2, :] = 2
    voxel_regions = MaximalBallVoxelRegions(
        label_image=label_image,
        root_ball_indices=np.array([0, 1, 2], dtype=np.int64),
        root_labels=np.array([0, 1, 2], dtype=np.int64),
        root_center_indices=np.array([[0, 1, 0], [2, 0, 0], [3, 2, 0]], dtype=np.int64),
        root_radii_voxels=np.array([2.0, 1.5, 1.0], dtype=float),
        root_of_ball_index=np.array([0, 1, 2], dtype=np.int64),
        unassigned_label=-1,
    )
    settings = resolve_maximal_ball_settings(
        np.ones((4, 3, 1), dtype=float),
        MaximalBallSettings(minimal_pore_radius_voxels=1.0),
    )
    extraction_result = MaximalBallExtractionResult(
        candidates=MaximalBallCandidates(
            center_indices=voxel_regions.root_center_indices.copy(),
            radii_voxels=voxel_regions.root_radii_voxels.copy(),
            candidate_mask=np.zeros((4, 3, 1), dtype=bool),
            retained_mask=np.array([True, True, True], dtype=bool),
            distance_map=np.ones((4, 3, 1), dtype=float),
            settings=settings,
        ),
        hierarchy=MaximalBallHierarchy(
            center_indices=voxel_regions.root_center_indices.copy(),
            center_coordinates=voxel_regions.root_center_indices.astype(float) - 0.5,
            radii_voxels=voxel_regions.root_radii_voxels.copy(),
            parent_indices=np.array([0, 1, 2], dtype=np.int64),
            master_indices=np.array([0, 1, 2], dtype=np.int64),
            hierarchy_levels=np.array([0, 0, 0], dtype=np.int64),
            distance_map=np.ones((4, 3, 1), dtype=float),
            settings=settings,
        ),
        voxel_regions=voxel_regions,
        region_adjacency=measure_region_adjacency(
            void_phase_mask,
            voxel_regions,
            distance_map=np.ones((4, 3, 1), dtype=float),
        ),
    )

    diagnostics = summarize_maximal_ball_extraction_diagnostics(
        void_phase_mask,
        extraction_result,
    )

    assert diagnostics.retained_ball_count == 3
    assert diagnostics.root_region_count == 3
    assert diagnostics.occupied_region_count == 3
    assert diagnostics.unassigned_void_voxel_count == 3
    assert diagnostics.zero_throat_region_count >= 1
    assert (
        diagnostics.throat_refined_support_radius_side1_mean_voxels
        >= diagnostics.throat_touch_radius_side1_mean_voxels
    )
    assert (
        diagnostics.throat_refined_support_radius_side2_mean_voxels
        >= diagnostics.throat_touch_radius_side2_mean_voxels
    )


def test_extract_maximal_ball_network_dict_wraps_extraction_and_assembly() -> None:
    """The high-level network-dict wrapper should expose both mapping and staged outputs."""

    void_phase_mask = np.zeros((7, 7, 7), dtype=bool)
    void_phase_mask[1:6, 1:6, 1:6] = True

    result = extract_maximal_ball_network_dict(
        void_phase_mask,
        voxel_size=1.0,
        distance_map_backend="scipy",
        settings=MaximalBallSettings(minimal_pore_radius_voxels=1.0),
        apply_boundary_clipping=False,
    )

    assert "pore.coords" in result.network_dict
    assert "throat.conns" in result.network_dict
    assert result.extraction.voxel_regions.label_image.shape == void_phase_mask.shape


def test_extract_maximal_ball_regions_returns_consistent_staged_outputs() -> None:
    """The staged convenience wrapper should return mutually consistent extraction layers."""

    void_phase_mask = np.zeros((9, 9, 9), dtype=bool)
    void_phase_mask[1:8, 1:8, 1:8] = True

    extraction_result = extract_maximal_ball_regions(
        void_phase_mask,
        distance_map_backend="scipy",
        settings=MaximalBallSettings(minimal_pore_radius_voxels=1.0),
        apply_boundary_clipping=False,
    )

    assert extraction_result.candidates.retained_center_indices.shape[1] == 3
    assert extraction_result.hierarchy.center_indices.shape == (
        extraction_result.candidates.retained_center_indices.shape
    )
    assert extraction_result.voxel_regions.label_image.shape == void_phase_mask.shape
    assert extraction_result.region_adjacency.region_volume_voxels.sum() == np.count_nonzero(
        extraction_result.voxel_regions.assigned_void_mask
    )


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
