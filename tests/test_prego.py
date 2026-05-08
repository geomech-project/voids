from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import voids.image.prego as prego_mod
from voids.image.prego import (
    PregoSettings,
    extract_prego_network_dict,
    prego_partitioning,
    snow_seed_points,
)


def test_snow_seed_points_validates_inputs() -> None:
    """Seed detection should reject shapes that would make geometry ambiguous."""

    with pytest.raises(ValueError, match="void_phase_mask must be a 2D or 3D array"):
        snow_seed_points(np.ones((2,), dtype=bool))

    with pytest.raises(ValueError, match="distance_map must match"):
        snow_seed_points(
            np.ones((2, 2), dtype=bool),
            distance_map=np.ones((3, 3), dtype=float),
        )

    with pytest.raises(ValueError, match="peaks must match"):
        snow_seed_points(
            np.ones((2, 2), dtype=bool),
            distance_map=np.ones((2, 2), dtype=float),
            peaks=np.ones((3, 3), dtype=bool),
        )

    with pytest.raises(ValueError, match="peak_footprint"):
        snow_seed_points(
            np.ones((2, 2), dtype=bool),
            distance_map=np.ones((2, 2), dtype=float),
            peak_footprint="triangle",
        )

    with pytest.raises(ValueError, match="peak labels and distance_map"):
        prego_mod._reduce_peak_labels_to_seed_points(
            np.ones((2, 2), dtype=int),
            np.ones((3, 3), dtype=float),
        )


def test_snow_seed_points_reduce_marker_regions_to_distance_maxima() -> None:
    """SNOW-style markers should become one seed point per marker label."""

    mask = np.ones((5, 5), dtype=bool)
    distance_map = np.ones((5, 5), dtype=float)
    distance_map[2, 2] = 4.0
    distance_map[1, 1] = 2.0
    distance_map[4, 4] = 3.0
    peaks = np.zeros((5, 5), dtype=int)
    peaks[1:3, 1:3] = 10
    peaks[4, 4] = 20

    seed_indices, seed_labels, returned_distance_map = snow_seed_points(
        mask,
        distance_map=distance_map,
        peaks=peaks,
    )

    assert np.array_equal(returned_distance_map, distance_map)
    assert seed_indices.tolist() == [[2, 2], [4, 4]]
    assert seed_labels[2, 2] == 1
    assert seed_labels[4, 4] == 2
    assert np.count_nonzero(seed_labels) == 2


def test_prego_uses_compact_safe_integer_dtypes() -> None:
    """PREGO labels should use int16 when safe and widen before overflow is possible."""

    assert prego_mod._prego_label_dtype(max_label=10, shape=(256, 256, 256)) == np.int16
    assert prego_mod._prego_label_dtype(max_label=40_000, shape=(256, 256, 256)) == np.int32
    assert prego_mod._prego_label_dtype(max_label=10, shape=(40_000, 2, 2)) == np.int32
    assert prego_mod._prego_label_dtype(max_label=3_000_000_000, shape=(2, 2)) == np.int64

    mask = np.ones((5, 5), dtype=bool)
    distance_map = np.ones(mask.shape, dtype=float)
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[2, 2] = 1

    result = prego_partitioning(mask, distance_map=distance_map, peaks=peaks)

    assert result.peaks.dtype == np.int16
    assert result.regions.dtype == np.int16


def test_reduce_peak_labels_accepts_boolean_marker_arrays() -> None:
    """The internal reducer should normalize non-integer marker masks."""

    distance_map = np.ones((3, 3), dtype=float)
    distance_map[1, 1] = 2.0
    peaks = np.zeros(distance_map.shape, dtype=bool)
    peaks[1, 1] = True

    seed_indices, seed_labels = prego_mod._reduce_peak_labels_to_seed_points(
        peaks,
        distance_map,
    )

    assert seed_indices.tolist() == [[1, 1]]
    assert seed_labels[1, 1] == 1


def test_snow_seed_points_accepts_boolean_peaks_and_fallback_seed() -> None:
    """Boolean markers and empty marker sets should both produce deterministic labels."""

    mask = np.ones((3, 3), dtype=bool)
    distance_map = np.ones(mask.shape, dtype=float)
    boolean_peaks = np.zeros(mask.shape, dtype=bool)
    boolean_peaks[0, 0] = True
    boolean_peaks[0, 1] = True

    seed_indices, seed_labels, _ = snow_seed_points(
        mask,
        distance_map=distance_map,
        peaks=boolean_peaks,
    )

    assert seed_indices.tolist() == [[0, 0]]
    assert seed_labels[0, 0] == 1

    fallback_indices, fallback_labels, _ = snow_seed_points(
        mask,
        distance_map_backend="scipy",
        peaks=np.zeros(mask.shape, dtype=bool),
    )

    assert fallback_indices.shape == (1, 2)
    assert np.count_nonzero(fallback_labels) == 1


def test_snow_seed_points_uses_porespy_filter_stages_when_peaks_are_not_supplied() -> None:
    """The PREGO seed path should call the SNOW peak-filtering stages."""

    calls: list[str] = []

    def find_peaks(dt, r_max):
        calls.append(f"find:{r_max}:{float(dt[1, 1]):.1f}")
        peaks = np.zeros_like(dt, dtype=bool)
        peaks[1, 1] = True
        return peaks

    def trim_saddle_points(peaks, dt):
        calls.append(f"saddle:{float(dt[1, 1]):.1f}")
        return peaks

    def trim_nearby_peaks(peaks, dt):
        calls.append(f"nearby:{float(dt[1, 1]):.1f}")
        return peaks

    fake_porespy = SimpleNamespace(
        filters=SimpleNamespace(
            find_peaks=find_peaks,
            trim_saddle_points=trim_saddle_points,
            trim_nearby_peaks=trim_nearby_peaks,
        )
    )
    distance_map = np.ones((3, 3), dtype=float)
    distance_map[1, 1] = 3.0

    seed_indices, seed_labels, _ = snow_seed_points(
        np.ones((3, 3), dtype=bool),
        distance_map=distance_map,
        peaks=None,
        sigma=0.0,
        r_max=2,
        peak_footprint="sphere",
        porespy_module=fake_porespy,
    )

    assert seed_indices.tolist() == [[1, 1]]
    assert seed_labels[1, 1] == 1
    assert calls == ["find:2:3.0", "saddle:3.0", "nearby:3.0"]

    calls.clear()
    seed_indices, seed_labels, _ = snow_seed_points(
        np.ones((3, 3), dtype=bool),
        distance_map=distance_map,
        peaks=None,
        sigma=0.5,
        r_max=2,
        peak_footprint="sphere",
        porespy_module=fake_porespy,
    )

    assert seed_indices.tolist() == [[1, 1]]
    assert seed_labels[1, 1] == 1
    assert len(calls) == 3


def test_snow_seed_points_cube_filter_avoids_porespy_peak_search() -> None:
    """The default cubic peak filter should keep PREGO seed search lightweight."""

    calls: list[str] = []

    def find_peaks(dt, r_max):  # pragma: no cover - should not be reached
        raise AssertionError("cube peak filtering should not call porespy.find_peaks")

    def trim_saddle_points(peaks, dt):
        calls.append("saddle")
        return peaks

    def trim_nearby_peaks(peaks, dt):
        calls.append("nearby")
        return peaks

    fake_porespy = SimpleNamespace(
        filters=SimpleNamespace(
            find_peaks=find_peaks,
            trim_saddle_points=trim_saddle_points,
            trim_nearby_peaks=trim_nearby_peaks,
        )
    )
    distance_map = np.ones((5, 5), dtype=float)
    distance_map[2, 2] = 4.0

    seed_indices, seed_labels, _ = snow_seed_points(
        np.ones((5, 5), dtype=bool),
        distance_map=distance_map,
        peaks=None,
        sigma=0.0,
        r_max=2,
        peak_footprint="cube",
        porespy_module=fake_porespy,
    )

    assert seed_indices.tolist() == [[2, 2]]
    assert seed_labels[2, 2] == 1
    assert calls == ["saddle", "nearby"]


def test_prego_partitioning_fills_face_connected_foreground_from_supplied_seeds() -> None:
    """PREGO regions should fill a connected foreground without entering background."""

    mask = np.zeros((5, 9), dtype=bool)
    mask[1:4, 1:8] = True
    distance_map = np.ones(mask.shape, dtype=float)
    distance_map[2, 2] = 2.0
    distance_map[2, 6] = 2.0
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[2, 2] = 1
    peaks[2, 6] = 2

    result = prego_partitioning(mask, distance_map=distance_map, peaks=peaks)

    assert result.regions.shape == mask.shape
    assert np.all(result.regions[mask] > 0)
    assert np.all(result.regions[~mask] == 0)
    assert result.regions[2, 1] == 1
    assert result.regions[2, 7] == 2
    assert result.seed_activation_levels.tolist() == [0, 0]


def test_prego_partitioning_validates_rank_and_handles_empty_foreground() -> None:
    """Invalid ranks should fail, while empty masks should return empty labels."""

    with pytest.raises(ValueError, match="im must be a 2D or 3D binary image"):
        prego_partitioning(np.ones((2,), dtype=bool))

    result = prego_partitioning(
        np.zeros((2, 2), dtype=bool),
        distance_map=np.zeros((2, 2), dtype=float),
        peaks=np.zeros((2, 2), dtype=int),
    )

    assert result.seed_indices.shape == (0, 2)
    assert result.regions.shape == (2, 2)
    assert not np.any(result.regions)


def test_prego_partitioning_uses_four_and_six_connectivity_for_growth() -> None:
    """Diagonal-only foreground voxels should remain unassigned without their own seed."""

    mask_2d = np.zeros((3, 3), dtype=bool)
    mask_2d[0, 0] = True
    mask_2d[1, 1] = True
    distance_map_2d = np.ones(mask_2d.shape, dtype=float)
    peaks_2d = np.zeros(mask_2d.shape, dtype=int)
    peaks_2d[0, 0] = 1

    result_2d = prego_partitioning(mask_2d, distance_map=distance_map_2d, peaks=peaks_2d)

    assert result_2d.regions[0, 0] == 1
    assert result_2d.regions[1, 1] == 0

    mask_3d = np.zeros((3, 3, 3), dtype=bool)
    mask_3d[0, 0, 0] = True
    mask_3d[1, 1, 1] = True
    distance_map_3d = np.ones(mask_3d.shape, dtype=float)
    peaks_3d = np.zeros(mask_3d.shape, dtype=int)
    peaks_3d[0, 0, 0] = 1

    result_3d = prego_partitioning(mask_3d, distance_map=distance_map_3d, peaks=peaks_3d)

    assert result_3d.regions[0, 0, 0] == 1
    assert result_3d.regions[1, 1, 1] == 0


def test_prego_partitioning_accepts_settings_for_seed_generation() -> None:
    """Settings should be stored with the result for provenance and diagnostics."""

    mask = np.ones((3, 3), dtype=bool)
    distance_map = np.ones(mask.shape, dtype=float)
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[1, 1] = 1
    settings = PregoSettings(r_max=2, sigma=0.0, cleanup_unassigned=False)

    result = prego_partitioning(
        mask,
        settings=settings,
        distance_map=distance_map,
        peaks=peaks,
    )

    assert result.settings is settings
    assert result.regions[1, 1] == 1


def test_prego_partitioning_accepts_level_queue_growth(monkeypatch) -> None:
    """The default level-queue mode should use delayed seed activation."""

    def fail_fast_stamp(*args, **kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("level_queue mode should not use the fast sphere stamping path")

    monkeypatch.setattr(prego_mod, "_stamp_seed_spheres_2d", fail_fast_stamp)

    mask = np.zeros((5, 9), dtype=bool)
    mask[1:4, 1:8] = True
    distance_map = np.ones(mask.shape, dtype=float)
    distance_map[2, 2] = 3.0
    distance_map[2, 6] = 1.0
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[2, 2] = 1
    peaks[2, 6] = 2
    result = prego_partitioning(
        mask,
        distance_map=distance_map,
        peaks=peaks,
    )

    assert np.all(result.regions[mask] > 0)
    assert result.seed_activation_levels.tolist() == [0, 2]
    assert result.regions[2, 2] == 1
    assert result.regions[2, 6] == 2


def test_prego_partitioning_level_queue_growth_compiles_in_three_dimensions(
    monkeypatch,
) -> None:
    """The level queue should be covered for the 3D Numba specialization."""

    def fail_fast_stamp(*args, **kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("level_queue mode should not use the fast sphere stamping path")

    monkeypatch.setattr(prego_mod, "_stamp_seed_spheres_3d", fail_fast_stamp)

    mask = np.zeros((5, 7, 7), dtype=bool)
    mask[1:4, 1:6, 1:6] = True
    distance_map = np.ones(mask.shape, dtype=float)
    distance_map[2, 2, 2] = 3.0
    distance_map[2, 4, 4] = 1.0
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[2, 2, 2] = 1
    peaks[2, 4, 4] = 2

    result = prego_partitioning(mask, distance_map=distance_map, peaks=peaks)

    assert np.all(result.regions[mask] > 0)
    assert result.seed_activation_levels.tolist() == [0, 2]
    assert result.regions[2, 2, 2] == 1
    assert result.regions[2, 4, 4] == 2


def test_prego_partitioning_fast_growth_mode_remains_available() -> None:
    """The explicit fast approximation should still work in 2D and 3D."""

    settings = PregoSettings(growth_mode="fast")

    mask_2d = np.zeros((5, 7), dtype=bool)
    mask_2d[1:4, 1:6] = True
    distance_map_2d = np.ones(mask_2d.shape, dtype=float)
    distance_map_2d[2, 2] = 2.0
    distance_map_2d[2, 4] = 1.0
    peaks_2d = np.zeros(mask_2d.shape, dtype=int)
    peaks_2d[2, 2] = 1
    peaks_2d[2, 4] = 2

    result_2d = prego_partitioning(
        mask_2d,
        settings=settings,
        distance_map=distance_map_2d,
        peaks=peaks_2d,
    )

    assert np.all(result_2d.regions[mask_2d] > 0)
    assert result_2d.settings.growth_mode == "fast"

    mask_3d = np.zeros((5, 5, 5), dtype=bool)
    mask_3d[1:4, 1:4, 1:4] = True
    distance_map_3d = np.ones(mask_3d.shape, dtype=float)
    distance_map_3d[2, 2, 2] = 2.0
    distance_map_3d[2, 3, 3] = 1.0
    peaks_3d = np.zeros(mask_3d.shape, dtype=int)
    peaks_3d[2, 2, 2] = 1
    peaks_3d[2, 3, 3] = 2

    result_3d = prego_partitioning(
        mask_3d,
        settings=settings,
        distance_map=distance_map_3d,
        peaks=peaks_3d,
    )

    assert np.all(result_3d.regions[mask_3d] > 0)
    assert result_3d.settings.growth_mode == "fast"


def test_prego_partitioning_rejects_unknown_growth_mode() -> None:
    """Unknown growth modes should fail before segmentation work starts."""

    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="growth_mode"):
        prego_partitioning(
            mask,
            settings=PregoSettings(growth_mode="surprise"),
            distance_map=np.ones(mask.shape, dtype=float),
            peaks=np.zeros(mask.shape, dtype=int),
        )


def test_extract_prego_network_dict_handles_single_region_without_throats() -> None:
    """A one-region segmentation is a valid no-throat pore network."""

    mask = np.ones((3, 3, 3), dtype=bool)
    distance_map = np.ones(mask.shape, dtype=float)
    distance_map[1, 1, 1] = 2.0
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[1, 1, 1] = 1

    result = extract_prego_network_dict(
        mask,
        distance_map=distance_map,
        peaks=peaks,
    )

    assert result.network_dict["pore.coords"].shape == (1, 3)
    assert result.network_dict["throat.conns"].shape == (0, 2)
    assert result.network_dict["pore.inscribed_diameter"][0] == 4.0


def test_extract_prego_network_dict_handles_two_dimensional_region_without_throats() -> None:
    """The no-interface fallback should use the 2D equivalent-diameter convention."""

    mask = np.ones((3, 3), dtype=bool)
    distance_map = np.ones(mask.shape, dtype=float)
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[1, 1] = 1

    result = extract_prego_network_dict(
        mask,
        distance_map=distance_map,
        peaks=peaks,
    )

    assert result.network_dict["pore.coords"].shape == (1, 2)
    assert result.network_dict["throat.conns"].shape == (0, 2)
    assert result.network_dict["pore.equivalent_diameter"][0] == pytest.approx(
        2.0 * np.sqrt(9.0 / np.pi)
    )


def test_extract_prego_network_dict_uses_regions_to_network_when_regions_touch() -> None:
    """Touching PREGO labels should be delegated to PoreSpy region geometry extraction."""

    captured: dict[str, object] = {}

    def fake_regions_to_network(regions, **kwargs):
        captured["regions"] = regions.copy()
        captured["kwargs"] = kwargs
        return {
            "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
        }

    fake_porespy = SimpleNamespace(
        networks=SimpleNamespace(regions_to_network=fake_regions_to_network)
    )
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 1:4] = True
    distance_map = np.ones(mask.shape, dtype=float)
    peaks = np.zeros(mask.shape, dtype=int)
    peaks[1, 1] = 1
    peaks[1, 3] = 2

    result = extract_prego_network_dict(
        mask,
        distance_map=distance_map,
        peaks=peaks,
        porespy_module=fake_porespy,
        regions_to_network_kwargs={"accuracy": "standard"},
    )

    assert set(result.network_dict) == {"pore.coords", "throat.conns"}
    assert captured["kwargs"] == {"accuracy": "standard"}
    assert np.any(captured["regions"] == 1)
    assert np.any(captured["regions"] == 2)
