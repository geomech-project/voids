from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from voids.generators import porous_image as pimg
from voids.image import connectivity as iconn
from voids.image import segmentation as iseg


def test_has_spanning_cluster_2d_and_3d() -> None:
    mask2d = np.zeros((6, 6), dtype=bool)
    mask2d[:, 2] = True
    assert iconn.has_spanning_cluster(mask2d, axis_index=0)
    assert not iconn.has_spanning_cluster(mask2d, axis_index=1)

    mask3d = np.zeros((5, 4, 3), dtype=bool)
    mask3d[2, :, 1] = True
    assert iconn.has_spanning_cluster(mask3d, axis_index=1)
    assert not iconn.has_spanning_cluster(mask3d, axis_index=0)


def test_has_spanning_cluster_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        iconn.has_spanning_cluster(np.ones(8, dtype=bool), axis_index=0)
    with pytest.raises(ValueError, match="out of bounds"):
        iconn.has_spanning_cluster(np.ones((5, 5), dtype=bool), axis_index=2)


def test_generate_spanning_blobs_matrix_uses_seed_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_blobs(*, shape, porosity, blobiness, seed):
        del porosity, blobiness
        arr = np.zeros(shape, dtype=bool)
        if seed >= 103:
            arr[:, shape[1] // 2] = True
        return arr

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(generators=SimpleNamespace(blobs=fake_blobs)),
    )
    matrix, seed = pimg.generate_spanning_blobs_matrix(
        shape=(8, 8),
        porosity=0.20,
        blobiness=1.8,
        axis_index=0,
        seed_start=101,
        max_tries=6,
    )
    assert seed == 103
    assert matrix.dtype == bool
    assert iconn.has_spanning_cluster(matrix, axis_index=0)


def test_generate_spanning_voronoi_matrix_2d_uses_best_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pimg, "estimate_voronoi_ncells_for_porosity_2d", lambda *a, **k: 80)

    def fake_voronoi_edges(*, shape, ncells, r, seed):
        del r, ncells
        void = np.zeros(shape, dtype=bool)
        if seed == 12:
            void[:, shape[1] // 2] = True
            void[:2, :2] = True
        return ~void

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(generators=SimpleNamespace(voronoi_edges=fake_voronoi_edges)),
    )

    matrix, seed = pimg.generate_spanning_voronoi_matrix_2d(
        shape=(12, 12),
        porosity=0.10,
        axis_index=0,
        seed_start=10,
        max_tries=5,
        target_tol=0.0,
        search_half_window=0,
    )
    assert seed == 12
    assert matrix.shape == (12, 12)
    assert iconn.has_spanning_cluster(matrix, axis_index=0)


def test_generate_spanning_matrix_2d_blobs_uses_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_generate(*, porosity, **kwargs):
        del kwargs
        calls.append(float(porosity))
        if porosity < 0.30:
            raise RuntimeError("not spanning")
        return np.ones((6, 6), dtype=bool), 77

    monkeypatch.setattr(pimg, "generate_spanning_blobs_matrix", fake_generate)

    matrix, seed, porosity_used = pimg.generate_spanning_matrix_2d(
        shape=(6, 6),
        porosity=0.20,
        axis_index=0,
        generator_name="blobs",
        seed_start=1,
        max_tries=2,
        blobs_fallback_porosity_levels=[0.25, 0.30, 0.35],
    )
    assert matrix.shape == (6, 6)
    assert seed == 77
    assert porosity_used == pytest.approx(0.30)
    assert calls == [0.20, 0.25, 0.30]


def test_generate_connected_matrix_compat_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pimg,
        "generate_spanning_blobs_matrix",
        lambda **kwargs: (np.ones((6, 6, 6), dtype=bool), 123),
    )
    matrix, seed = pimg.generate_connected_matrix(
        shape=(6, 6, 6),
        porosity=0.15,
        blobiness=1.8,
        axis_index=0,
        seed_start=10,
        max_tries=4,
        show_progress=True,
        progress_desc="ignored",
    )
    assert matrix.shape == (6, 6, 6)
    assert seed == 123


def test_generate_connected_matrix_2d_compat_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pimg,
        "generate_spanning_matrix_2d",
        lambda **kwargs: (np.ones((5, 5), dtype=bool), 17, 0.24),
    )
    matrix, seed, used = pimg.generate_connected_matrix_2d(
        shape=(5, 5),
        porosity=0.2,
        axis_index=0,
        generator_name="blobs",
        seed_start=1,
        max_tries=2,
        show_progress=True,
        progress_desc="ignored",
    )
    assert matrix.shape == (5, 5)
    assert seed == 17
    assert used == pytest.approx(0.24)


def test_generate_spanning_matrix_2d_rejects_invalid_generator() -> None:
    with pytest.raises(ValueError, match="Unsupported generator_name"):
        pimg.generate_spanning_matrix_2d(
            shape=(8, 8),
            porosity=0.2,
            axis_index=0,
            generator_name="invalid",
            seed_start=1,
            max_tries=5,
        )


def test_insert_vug_helpers_add_void_voxels() -> None:
    base3d = np.zeros((20, 20, 20), dtype=bool)
    out3d, mask3d = pimg.insert_ellipsoidal_vug(base3d, radii_vox=(4, 3, 2))
    assert out3d.shape == base3d.shape
    assert mask3d.any()
    assert np.count_nonzero(out3d) == np.count_nonzero(mask3d)

    out_sphere, mask_sphere = pimg.insert_spherical_vug(base3d, radius_vox=3)
    assert mask_sphere.any()
    assert np.count_nonzero(out_sphere) == np.count_nonzero(mask_sphere)

    base2d = np.zeros((30, 30), dtype=bool)
    out2d, mask2d = pimg.insert_elliptical_vug_2d(base2d, radii_vox=(6, 4))
    assert out2d.shape == base2d.shape
    assert mask2d.any()
    assert np.count_nonzero(out2d) == np.count_nonzero(mask2d)

    out_circle, mask_circle = pimg.insert_circular_vug_2d(base2d, radius_vox=5)
    assert mask_circle.any()
    assert np.count_nonzero(out_circle) == np.count_nonzero(mask_circle)


def test_make_synthetic_grayscale_is_reproducible_and_bounded() -> None:
    phase = np.zeros((8, 8), dtype=bool)
    phase[2:6, 2:6] = True
    gray_a = pimg.make_synthetic_grayscale(phase, seed=123)
    gray_b = pimg.make_synthetic_grayscale(phase, seed=123)
    assert np.allclose(gray_a, gray_b)
    assert gray_a.min() >= 0.0
    assert gray_a.max() <= 255.0
    assert gray_a.dtype == float


def test_make_synthetic_grayscale_2d_compat_wrapper() -> None:
    phase = np.zeros((6, 6), dtype=bool)
    phase[1:5, 1:5] = True
    gray = pimg.make_synthetic_grayscale_2d(phase, seed=11)
    assert gray.shape == phase.shape
    assert gray.dtype == float
    with pytest.raises(ValueError, match="must be 2D"):
        pimg.make_synthetic_grayscale_2d(np.zeros((3, 3, 3), dtype=bool), seed=1)


def test_binarize_2d_with_voids_threshold_branch() -> None:
    gray = np.array(
        [
            [50.0, 50.0, 200.0],
            [50.0, 200.0, 200.0],
            [200.0, 200.0, 200.0],
        ]
    )
    seg, thr = iseg.binarize_2d_with_voids(gray, threshold=100.0, void_phase="dark")
    expected = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )
    assert thr == pytest.approx(100.0)
    assert np.array_equal(seg, expected)


def test_binarize_2d_with_voids_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="gray2d must be a 2D array"):
        iseg.binarize_2d_with_voids(np.ones((2, 2, 2), dtype=float))


def test_has_spanning_cluster_2d_wrapper() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, 3] = True
    assert iconn.has_spanning_cluster_2d(mask, axis_index=0)
    with pytest.raises(ValueError, match="2D"):
        iconn.has_spanning_cluster_2d(np.zeros((3, 3, 3), dtype=bool), axis_index=0)


def test_porous_image_validation_and_generation_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="porosity must be in \\(0, 1\\)"):
        pimg.generate_spanning_blobs_matrix(
            shape=(8, 8),
            porosity=1.0,
            blobiness=1.2,
            axis_index=0,
            seed_start=1,
            max_tries=2,
        )
    with pytest.raises(ValueError, match="blobiness must be positive"):
        pimg.generate_spanning_blobs_matrix(
            shape=(8, 8),
            porosity=0.2,
            blobiness=0.0,
            axis_index=0,
            seed_start=1,
            max_tries=2,
        )
    with pytest.raises(ValueError, match="max_tries must be >= 1"):
        pimg.generate_spanning_blobs_matrix(
            shape=(8, 8),
            porosity=0.2,
            blobiness=1.2,
            axis_index=0,
            seed_start=1,
            max_tries=0,
        )

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(
            generators=SimpleNamespace(blobs=lambda **kwargs: np.zeros(kwargs["shape"]))
        ),
    )
    with pytest.raises(RuntimeError, match="Could not generate spanning blobs matrix"):
        pimg.generate_spanning_blobs_matrix(
            shape=(8, 8),
            porosity=0.2,
            blobiness=1.8,
            axis_index=0,
            seed_start=1,
            max_tries=2,
        )

    with pytest.raises(ValueError, match="porosity must be in \\(0, 1\\)"):
        pimg.estimate_voronoi_ncells_for_porosity_2d((10, 10), 0.0)
    with pytest.raises(ValueError, match="slope must be positive"):
        pimg.estimate_voronoi_ncells_for_porosity_2d((10, 10), 0.2, slope=0.0)
    with pytest.raises(ValueError, match="min_ncells must be >= 1"):
        pimg.estimate_voronoi_ncells_for_porosity_2d((10, 10), 0.2, min_ncells=0)
    assert pimg.estimate_voronoi_ncells_for_porosity_2d((180, 180), 0.2) >= 40


def test_voronoi_generation_branch_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="porosity must be in \\(0, 1\\)"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10), porosity=1.0, axis_index=0, seed_start=1, max_tries=1
        )
    with pytest.raises(ValueError, match="max_tries must be >= 1"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10), porosity=0.2, axis_index=0, seed_start=1, max_tries=0
        )
    with pytest.raises(ValueError, match="edge_radius_vox must be >= 0"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10),
            porosity=0.2,
            axis_index=0,
            seed_start=1,
            max_tries=1,
            edge_radius_vox=-1,
        )
    with pytest.raises(ValueError, match="target_tol must be >= 0"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10), porosity=0.2, axis_index=0, seed_start=1, max_tries=1, target_tol=-1.0
        )
    with pytest.raises(ValueError, match="ncells_step must be >= 1"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10), porosity=0.2, axis_index=0, seed_start=1, max_tries=1, ncells_step=0
        )
    with pytest.raises(ValueError, match="search_half_window must be >= 0"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10),
            porosity=0.2,
            axis_index=0,
            seed_start=1,
            max_tries=1,
            search_half_window=-1,
        )
    with pytest.raises(ValueError, match="min_ncells must be >= 1"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10), porosity=0.2, axis_index=0, seed_start=1, max_tries=1, min_ncells=0
        )

    monkeypatch.setattr(pimg, "estimate_voronoi_ncells_for_porosity_2d", lambda *a, **k: 12)

    def fake_voronoi_edges_spanning(*, shape, ncells, r, seed):
        del ncells, r, seed
        void = np.zeros(shape, dtype=bool)
        void[:, shape[1] // 2] = True  # 10 / 100 = 0.1, spanning along axis 0
        return ~void

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(generators=SimpleNamespace(voronoi_edges=fake_voronoi_edges_spanning)),
    )
    matrix, seed = pimg.generate_spanning_voronoi_matrix_2d(
        shape=(10, 10),
        porosity=0.1,
        axis_index=0,
        seed_start=5,
        max_tries=1,
        target_tol=0.0,
        search_half_window=0,
    )
    assert seed == 5
    assert np.isclose(matrix.mean(), 0.1)

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(
            generators=SimpleNamespace(
                voronoi_edges=lambda **kwargs: np.ones(kwargs["shape"], dtype=bool)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Could not generate spanning Voronoi matrix"):
        pimg.generate_spanning_voronoi_matrix_2d(
            shape=(10, 10),
            porosity=0.2,
            axis_index=0,
            seed_start=1,
            max_tries=2,
            search_half_window=0,
        )


def test_generate_spanning_matrix_2d_voronoi_and_blobs_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pimg,
        "generate_spanning_voronoi_matrix_2d",
        lambda **kwargs: (np.ones((4, 4), dtype=bool), 42),
    )
    matrix, seed, used = pimg.generate_spanning_matrix_2d(
        shape=(4, 4),
        porosity=0.2,
        axis_index=0,
        generator_name="voronoi_edges",
        seed_start=1,
        max_tries=1,
    )
    assert matrix.shape == (4, 4)
    assert seed == 42
    assert used == pytest.approx(1.0)

    with pytest.raises(ValueError, match="porosity must be in \\(0, 1\\)"):
        pimg.generate_spanning_matrix_2d(
            shape=(4, 4),
            porosity=1.1,
            axis_index=0,
            generator_name="blobs",
            seed_start=1,
            max_tries=1,
        )

    monkeypatch.setattr(
        pimg,
        "generate_spanning_blobs_matrix",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("not spanning")),
    )
    with pytest.raises(RuntimeError, match="requested or fallback porosities"):
        pimg.generate_spanning_matrix_2d(
            shape=(6, 6),
            porosity=0.2,
            axis_index=0,
            generator_name="blobs",
            seed_start=1,
            max_tries=2,
            blobs_fallback_porosity_levels=[0.3],
        )


def test_vug_image_insert_and_grayscale_validation_branches() -> None:
    out3d, mask3d = pimg.insert_ellipsoidal_vug(
        np.zeros((7, 7, 7), dtype=bool),
        radii_vox=(2, 2, 2),
        center=(3, 3, 3),
    )
    assert out3d[3, 3, 3]
    assert mask3d[3, 3, 3]

    with pytest.raises(ValueError, match="matrix_void must be a 3D array"):
        pimg.insert_ellipsoidal_vug(np.zeros((4, 4), dtype=bool), radii_vox=(1, 1, 1))
    with pytest.raises(ValueError, match="center must have length 3"):
        pimg.insert_ellipsoidal_vug(
            np.zeros((4, 4, 4), dtype=bool),
            radii_vox=(1, 1, 1),
            center=(1, 1),
        )
    with pytest.raises(ValueError, match="All ellipsoid radii must be positive"):
        pimg.insert_ellipsoidal_vug(np.zeros((4, 4, 4), dtype=bool), radii_vox=(0, 1, 1))
    with pytest.raises(ValueError, match="radius_vox must be positive"):
        pimg.insert_spherical_vug(np.zeros((4, 4, 4), dtype=bool), radius_vox=0)

    out2d, mask2d = pimg.insert_elliptical_vug_2d(
        np.zeros((9, 9), dtype=bool),
        radii_vox=(2, 2),
        center=(4, 4),
    )
    assert out2d[4, 4]
    assert mask2d[4, 4]

    with pytest.raises(ValueError, match="matrix_void must be a 2D array"):
        pimg.insert_elliptical_vug_2d(np.zeros((4, 4, 4), dtype=bool), radii_vox=(1, 1))
    with pytest.raises(ValueError, match="center must have length 2"):
        pimg.insert_elliptical_vug_2d(
            np.zeros((4, 4), dtype=bool),
            radii_vox=(1, 1),
            center=(1, 1, 1),
        )
    with pytest.raises(ValueError, match="All ellipse radii must be positive"):
        pimg.insert_elliptical_vug_2d(np.zeros((4, 4), dtype=bool), radii_vox=(0, 1))
    with pytest.raises(ValueError, match="radius_vox must be positive"):
        pimg.insert_circular_vug_2d(np.zeros((4, 4), dtype=bool), radius_vox=0)

    with pytest.raises(ValueError, match="binary_void must be 2D or 3D"):
        pimg.make_synthetic_grayscale(np.zeros((2,), dtype=bool), seed=1)
    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        pimg.make_synthetic_grayscale(np.zeros((3, 3), dtype=bool), seed=1, noise_std=-1.0)
    with pytest.raises(ValueError, match="clip_max must be larger than clip_min"):
        pimg.make_synthetic_grayscale(
            np.zeros((3, 3), dtype=bool), seed=1, clip_min=5.0, clip_max=5.0
        )
