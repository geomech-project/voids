from __future__ import annotations

import numpy as np
import pytest

from voids.geom import (
    area_equivalent_diameter,
    characteristic_size,
    normalize_characteristic_size,
)
from voids.image import segmentation as iseg
from voids.image import network_extraction as nex
from voids.image import (
    extract_spanning_pore_network,
    infer_sample_axes,
)
from voids.image import (
    binarize_grayscale_volume,
    crop_nonzero_cylindrical_volume,
    largest_true_rectangle,
    preprocess_grayscale_cylindrical_volume,
)


def test_area_equivalent_diameter_and_characteristic_size_priority() -> None:
    """Test public characteristic-size helpers used by diagnostics and plotting."""

    area = np.array([np.pi, 4.0 * np.pi])
    assert np.allclose(area_equivalent_diameter(area), np.array([2.0, 4.0]))

    store = {
        "diameter_equivalent": np.array([5.0, 6.0]),
        "diameter_inscribed": np.array([3.0, 4.0]),
        "radius_inscribed": np.array([1.0, 2.0]),
        "area": np.array([np.pi, 4.0 * np.pi]),
    }
    values, label = characteristic_size(store, expected_shape=(2,))
    assert label == "diameter_equivalent"
    assert np.array_equal(values, np.array([5.0, 6.0]))

    radius_values, radius_label = characteristic_size(
        {"radius_inscribed": np.array([1.0, 2.0])},
        expected_shape=(2,),
    )
    assert radius_label == "radius_inscribed"
    assert np.array_equal(radius_values, np.array([2.0, 4.0]))

    area_values, area_label = characteristic_size(
        {"area": np.array([np.pi, 4.0 * np.pi])},
        expected_shape=(2,),
    )
    assert area_label == "area"
    assert np.array_equal(area_values, np.array([2.0, 4.0]))

    with pytest.raises(KeyError, match="characteristic size fields"):
        characteristic_size({})
    with pytest.raises(ValueError, match="field 'diameter_equivalent' must have shape"):
        characteristic_size({"diameter_equivalent": np.ones(3)}, expected_shape=(2,))


def test_normalize_characteristic_size_branches() -> None:
    """Test all three branches of normalize_characteristic_size via voids.geom."""

    # radius_inscribed branch: values should be doubled
    radii = np.array([1.0, 2.0, 3.0])
    result = normalize_characteristic_size(radii, field_name="radius_inscribed")
    assert np.array_equal(result, np.array([2.0, 4.0, 6.0]))

    # area branch: values should be converted to area-equivalent diameters
    areas = np.array([np.pi, 4.0 * np.pi])
    result = normalize_characteristic_size(areas, field_name="area")
    assert np.allclose(result, np.array([2.0, 4.0]))

    # passthrough branch: any other field_name returns values unchanged
    diameters = np.array([5.0, 6.0])
    result = normalize_characteristic_size(diameters, field_name="diameter_equivalent")
    assert np.array_equal(result, diameters)

    result = normalize_characteristic_size(diameters, field_name=None)
    assert np.array_equal(result, diameters)


def test_largest_true_rectangle_and_crop_fill_internal_holes() -> None:
    """Test maximal rectangle detection and slice-wise support hole filling."""

    mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=bool,
    )
    assert largest_true_rectangle(mask) == (1, 4, 1, 4)

    raw = np.zeros((3, 6, 8), dtype=float)
    raw[:, 1:5, 1:7] = 10.0
    raw[:, 2:4, 3:5] = 2.0
    raw[1, 2:4, 3:5] = 0.0  # interior hole that should be filled in the specimen support

    crop = crop_nonzero_cylindrical_volume(raw)

    assert crop.crop_bounds_yx == (1, 5, 1, 7)
    assert crop.cropped.shape == (3, 4, 6)
    assert crop.specimen_mask[1, 2:4, 3:5].all()
    assert crop.common_mask[1:5, 1:7].all()


def test_workflow_preprocessing_validation_branches() -> None:
    """Test public preprocessing validation branches and unsupported inputs."""

    with pytest.raises(ValueError, match="voxel_size must be positive"):
        infer_sample_axes((4, 4, 4), voxel_size=0.0)
    with pytest.raises(ValueError, match="shape must have length 2 or 3"):
        infer_sample_axes((4,), voxel_size=1.0)
    with pytest.raises(ValueError, match="axis_names must cover every image dimension"):
        infer_sample_axes((4, 4, 4), voxel_size=1.0, axis_names=("x", "y"))

    counts, lengths, areas, flow_axis = infer_sample_axes((5, 8), voxel_size=2.0)
    assert counts == {"x": 5, "y": 8}
    assert lengths == {"x": 10.0, "y": 16.0}
    assert areas == {"x": 16.0, "y": 10.0}
    assert flow_axis == "y"

    with pytest.raises(ValueError, match="mask2d must be a 2D boolean array"):
        largest_true_rectangle(np.ones((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="does not contain any True pixels"):
        largest_true_rectangle(np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="raw must be a 3D grayscale volume"):
        crop_nonzero_cylindrical_volume(np.ones((4, 4), dtype=float))

    cropped = np.ones((2, 3, 4), dtype=float)
    with pytest.raises(ValueError, match="cropped must be a 3D grayscale volume"):
        binarize_grayscale_volume(np.ones((3, 4), dtype=float))
    with pytest.raises(ValueError, match="void_phase must be either 'dark' or 'bright'"):
        binarize_grayscale_volume(cropped, threshold=0.5, void_phase="invalid")
    with pytest.raises(ValueError, match="Unsupported threshold method 'bad'"):
        binarize_grayscale_volume(cropped, method="bad")


def test_preprocess_grayscale_cylindrical_volume_segments_dark_voids() -> None:
    """Test grayscale crop plus automatic thresholding for dark void segmentation."""

    raw = np.zeros((3, 6, 8), dtype=float)
    raw[:, 1:5, 1:7] = 10.0
    raw[:, 2:4, 3:5] = 2.0

    seg = preprocess_grayscale_cylindrical_volume(raw, threshold_method="otsu", void_phase="dark")

    assert seg.crop.crop_bounds_yx == (1, 5, 1, 7)
    assert 2.0 < seg.threshold < 10.0
    assert seg.binary.shape == (3, 4, 6)
    assert seg.binary[:, 1:3, 2:4].all()
    assert not seg.binary[:, 0, 0].any()

    bright_binary, used_threshold = binarize_grayscale_volume(
        seg.crop.cropped, threshold=6.0, void_phase="bright"
    )
    assert used_threshold == pytest.approx(6.0)
    assert bright_binary[:, 0, 0].all()
    assert not bright_binary[:, 1:3, 2:4].any()


def test_progress_iter_tqdm_wrapping() -> None:
    """Test that _progress_iter wraps with tqdm when show_progress is True."""

    items = list(range(5))

    # show_progress=False: original iterable returned unchanged
    result = iseg._progress_iter(items, show_progress=False)
    assert result is items

    # show_progress=True and tqdm available: iteration must yield correct values
    wrapped = iseg._progress_iter(items, show_progress=True, desc="test", total=5)
    assert list(wrapped) == items


def test_crop_and_preprocess_progress_hooks(monkeypatch) -> None:
    """Test progress-hook wiring for slice-wise cylindrical preprocessing."""

    raw = np.zeros((3, 6, 8), dtype=float)
    raw[:, 1:5, 1:7] = 10.0
    raw[:, 2:4, 3:5] = 2.0

    progress_calls: list[tuple[bool, str | None, int | None]] = []

    def fake_progress_iter(iterable, *, show_progress, desc=None, total=None):
        progress_calls.append((bool(show_progress), desc, total))
        return iterable

    monkeypatch.setattr(iseg, "_progress_iter", fake_progress_iter)
    seg = preprocess_grayscale_cylindrical_volume(
        raw,
        threshold_method="otsu",
        void_phase="dark",
        show_progress=True,
        progress_desc="unit-test-progress",
    )

    assert seg.binary.shape == (3, 4, 6)
    assert progress_calls
    assert progress_calls[0] == (True, "unit-test-progress", 3)


def test_snow2_network_dict_normalizes_all_supported_porespy_return_styles() -> None:
    """Test the internal snow2 result normalization across supported return shapes."""

    class _WithNetwork:
        def __init__(self):
            self.network = {
                "pore.coords": np.zeros((1, 3)),
                "throat.conns": np.zeros((0, 2), dtype=int),
            }

    class _WithRegions:
        def __init__(self):
            self.regions = "regions-object"

    class _FakeNetworks:
        def __init__(self, snow_result):
            self._snow_result = snow_result

        def snow2(self, phases, **kwargs):
            assert np.array_equal(phases, np.ones((2, 2), dtype=int))
            assert kwargs == {"sigma": 0.75}
            return self._snow_result

        def regions_to_network(self, regions):
            assert regions == "regions-object"
            return {"pore.coords": np.ones((2, 3)), "throat.conns": np.array([[0, 1]], dtype=int)}

    class _FakePoreSpy:
        def __init__(self, snow_result):
            self.networks = _FakeNetworks(snow_result)

    phases = np.ones((2, 2), dtype=int)

    from_attr = nex._snow2_network_dict(
        phases,
        porespy_module=_FakePoreSpy(_WithNetwork()),
        snow2_kwargs={"sigma": 0.75},
    )
    assert set(from_attr) == {"pore.coords", "throat.conns"}

    from_network_key = nex._snow2_network_dict(
        phases,
        porespy_module=_FakePoreSpy(
            {
                "network": {
                    "pore.coords": np.zeros((1, 3)),
                    "throat.conns": np.zeros((0, 2), dtype=int),
                }
            }
        ),
        snow2_kwargs={"sigma": 0.75},
    )
    assert set(from_network_key) == {"pore.coords", "throat.conns"}

    direct_dict = nex._snow2_network_dict(
        phases,
        porespy_module=_FakePoreSpy(
            {"pore.coords": np.zeros((1, 3)), "throat.conns": np.zeros((0, 2), dtype=int)}
        ),
        snow2_kwargs={"sigma": 0.75},
    )
    assert set(direct_dict) == {"pore.coords", "throat.conns"}

    from_regions_attr = nex._snow2_network_dict(
        phases,
        porespy_module=_FakePoreSpy(_WithRegions()),
        snow2_kwargs={"sigma": 0.75},
    )
    assert set(from_regions_attr) == {"pore.coords", "throat.conns"}

    from_regions_key = nex._snow2_network_dict(
        phases,
        porespy_module=_FakePoreSpy({"regions": "regions-object"}),
        snow2_kwargs={"sigma": 0.75},
    )
    assert set(from_regions_key) == {"pore.coords", "throat.conns"}

    with pytest.raises(RuntimeError, match="Could not find a network dict or regions"):
        nex._snow2_network_dict(
            phases,
            porespy_module=_FakePoreSpy({"unexpected": 1}),
            snow2_kwargs={"sigma": 0.75},
        )


def test_infer_axes_and_extract_spanning_pore_network() -> None:
    """Test extraction workflow metadata and imported networks."""

    _, axis_lengths, axis_areas, flow_axis = infer_sample_axes((12, 16, 16), voxel_size=1.0)
    assert flow_axis == "y"
    assert axis_lengths == {"x": 12.0, "y": 16.0, "z": 16.0}
    assert axis_areas == {"x": 256.0, "y": 192.0, "z": 192.0}

    im = np.zeros((12, 16, 16), dtype=int)
    im[:, 5:11, 5:11] = 1
    im[2:4, 1:3, 1:3] = 1

    result = extract_spanning_pore_network(
        im,
        voxel_size=1.0,
        flow_axis="x",
        length_unit="voxel",
        provenance_notes={"case": "tiny"},
    )

    assert result.flow_axis == "x"
    assert result.backend_version is not None
    assert result.provenance.user_notes["case"] == "tiny"
    assert result.sample.units["length"] == "voxel"
    assert result.net_full.Np >= result.net.Np
    assert result.net_full.Nt >= result.net.Nt
    assert np.array_equal(result.image, im)
    assert result.pore_indices.ndim == 1
    assert result.throat_mask.shape == (result.net_full.Nt,)


def test_extract_spanning_pore_network_validates_image_rank_and_flow_axis() -> None:
    """Test public extraction validation before backend extraction work starts."""

    with pytest.raises(ValueError, match="phases must be a 2D or 3D integer image"):
        extract_spanning_pore_network(np.ones((2,), dtype=int), voxel_size=1.0)

    with pytest.raises(ValueError, match="flow_axis 'q' is not compatible with shape"):
        extract_spanning_pore_network(np.ones((4, 5, 6), dtype=int), voxel_size=1.0, flow_axis="q")


def test_extract_spanning_pore_network_forwards_extraction_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_snow2(phases, *, snow2_kwargs):
        captured["phases"] = phases
        captured["kwargs"] = snow2_kwargs
        return {
            "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.xmin": np.array([True, False], dtype=bool),
            "pore.xmax": np.array([False, True], dtype=bool),
        }

    monkeypatch.setattr(nex, "_snow2_network_dict", fake_snow2)
    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        flow_axis="x",
        extraction_kwargs={"sigma": 0.5},
    )
    assert result.net.Np >= 1
    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=int))
    assert captured["kwargs"] == {"sigma": 0.5}


def test_extract_spanning_pore_network_enables_imperial_export_repairs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image extraction should apply Imperial export-style importer repairs by default."""

    def fake_snow2(_phases, *, snow2_kwargs):
        assert snow2_kwargs == {}
        return {
            "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.inscribed_diameter": np.array([2.0, 2.0]),
            "throat.inscribed_diameter": np.array([1.0]),
            "throat.cross_sectional_area": np.array([2.0]),
            "throat.total_length": np.array([1.0]),
            "pore.xmin": np.array([True, False], dtype=bool),
            "pore.xmax": np.array([False, True], dtype=bool),
        }

    monkeypatch.setattr(nex, "_snow2_network_dict", fake_snow2)
    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        flow_axis="x",
    )

    assert result.provenance.random_seed == 0
    assert result.net_full.extra["geometry_repairs"]["mode"] == "imperial_export"
    assert result.net_full.throat["shape_factor"][0] == pytest.approx(0.03125)
    assert np.allclose(result.net_full.pore["shape_factor"], np.array([0.03125, 0.03125]))


def test_extract_spanning_pore_network_accepts_legacy_geometry_repairs_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public extraction workflow should accept deprecated repair-mode aliases."""

    def fake_snow2(_phases, *, snow2_kwargs):
        assert snow2_kwargs == {}
        return {
            "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.inscribed_diameter": np.array([2.0, 2.0]),
            "throat.inscribed_diameter": np.array([1.0]),
            "throat.cross_sectional_area": np.array([2.0]),
            "throat.total_length": np.array([1.0]),
            "pore.xmin": np.array([True, False], dtype=bool),
            "pore.xmax": np.array([False, True], dtype=bool),
        }

    monkeypatch.setattr(nex, "_snow2_network_dict", fake_snow2)
    with pytest.warns(DeprecationWarning, match=r"geometry_repairs='pnextract'.*'imperial_export'"):
        result = extract_spanning_pore_network(
            np.ones((2, 2, 2), dtype=int),
            voxel_size=1.0,
            flow_axis="x",
            geometry_repairs="pnextract",
        )

    assert result.net_full.extra["geometry_repairs"]["mode"] == "imperial_export"
    assert result.net_full.throat["shape_factor"][0] == pytest.approx(0.03125)
