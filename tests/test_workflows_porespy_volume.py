from __future__ import annotations

import numpy as np
import pytest

from voids.examples.demo import make_linear_chain_network
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
from voids.io.hdf5 import load_hdf5, save_hdf5


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


def test_extract_network_dict_forwards_native_maximal_ball_threading_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native backend dispatch should forward EDT threading and normalized settings."""

    captured: dict[str, object] = {}

    class FakeExtractionResult:
        def __init__(self) -> None:
            self.network_dict = {
                "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
                "throat.conns": np.array([[0, 1]], dtype=int),
            }

    def fake_extract_maximal_ball_network_dict(
        phases,
        *,
        voxel_size,
        distance_map_backend,
        edt_parallel_threads,
        settings,
        apply_boundary_clipping,
        flow_boundary_mode,
        boundary_axis,
        boundary_length_epsilon,
        boundary_radius_scale,
        throat_area_mode,
        throat_shape_factor_radius_mode,
        throat_anchor_mode,
    ):
        captured["phases"] = phases
        captured["voxel_size"] = voxel_size
        captured["distance_map_backend"] = distance_map_backend
        captured["edt_parallel_threads"] = edt_parallel_threads
        captured["settings"] = settings
        captured["apply_boundary_clipping"] = apply_boundary_clipping
        captured["flow_boundary_mode"] = flow_boundary_mode
        captured["boundary_axis"] = boundary_axis
        captured["boundary_length_epsilon"] = boundary_length_epsilon
        captured["boundary_radius_scale"] = boundary_radius_scale
        captured["throat_area_mode"] = throat_area_mode
        captured["throat_shape_factor_radius_mode"] = throat_shape_factor_radius_mode
        captured["throat_anchor_mode"] = throat_anchor_mode
        return FakeExtractionResult()

    monkeypatch.setattr(
        nex, "extract_maximal_ball_network_dict", fake_extract_maximal_ball_network_dict
    )

    network_dict = nex._extract_network_dict(
        np.ones((2, 2, 2), dtype=int),
        backend="native_maximal_ball",
        voxel_size=1.5,
        extraction_kwargs={
            "distance_map_backend": "edt",
            "edt_parallel_threads": "7",
            "settings": {"minimal_pore_radius_voxels": 1.0},
            "apply_boundary_clipping": False,
            "flow_boundary_mode": "direct",
            "boundary_length_epsilon": "1e-12",
            "boundary_radius_scale": "1.25",
            "throat_area_mode": "face_count",
            "throat_shape_factor_radius_mode": "inscribed",
            "throat_anchor_mode": "second_side",
        },
        flow_axis="x",
    )

    assert set(network_dict) == {"pore.coords", "throat.conns"}
    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=bool))
    assert captured["voxel_size"] == pytest.approx(1.5)
    assert captured["distance_map_backend"] == "edt"
    assert captured["edt_parallel_threads"] == 7
    assert isinstance(captured["settings"], nex.MaximalBallSettings)
    assert captured["apply_boundary_clipping"] is False
    assert captured["flow_boundary_mode"] == "direct"
    assert captured["boundary_axis"] == "x"
    assert captured["boundary_length_epsilon"] == pytest.approx(1.0e-12)
    assert captured["boundary_radius_scale"] == pytest.approx(1.25)
    assert captured["throat_area_mode"] == "face_count"
    assert captured["throat_shape_factor_radius_mode"] == "inscribed"
    assert captured["throat_anchor_mode"] == "second_side"


def test_extract_network_dict_forwards_prego_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PREGO dispatch should forward deterministic segmentation controls."""

    captured: dict[str, object] = {}

    class FakeExtractionResult:
        def __init__(self) -> None:
            self.network_dict = {
                "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
                "throat.conns": np.array([[0, 1]], dtype=int),
            }

    def fake_extract_prego_network_dict(
        im,
        *,
        settings,
        distance_map,
        peaks,
        regions_to_network_kwargs,
    ):
        captured["im"] = im
        captured["settings"] = settings
        captured["distance_map"] = distance_map
        captured["peaks"] = peaks
        captured["regions_to_network_kwargs"] = regions_to_network_kwargs
        return FakeExtractionResult()

    monkeypatch.setattr(nex, "extract_prego_network_dict", fake_extract_prego_network_dict)
    distance_map = np.ones((2, 2, 2), dtype=float)
    peaks = np.zeros((2, 2, 2), dtype=int)
    peaks[0, 0, 0] = 1

    network_dict = nex._extract_network_dict(
        np.ones((2, 2, 2), dtype=int),
        backend="prego",
        voxel_size=1.5,
        extraction_kwargs={
            "settings": {"r_max": 2, "sigma": 0.0, "distance_map_backend": "scipy"},
            "distance_map": distance_map,
            "peaks": peaks,
            "regions_to_network_kwargs": {"accuracy": "standard"},
        },
        flow_axis="x",
    )

    assert set(network_dict) == {"pore.coords", "throat.conns"}
    assert np.array_equal(captured["im"], np.ones((2, 2, 2), dtype=bool))
    assert isinstance(captured["settings"], nex.PregoSettings)
    assert captured["settings"].r_max == 2
    assert captured["settings"].sigma == pytest.approx(0.0)
    assert np.array_equal(captured["distance_map"], distance_map)
    assert np.array_equal(captured["peaks"], peaks)
    assert captured["regions_to_network_kwargs"] == {"accuracy": "standard"}


def test_extract_network_dict_rejects_invalid_native_maximal_ball_kwargs() -> None:
    """Native backend dispatch should fail clearly on invalid settings or stray kwargs."""

    with pytest.raises(TypeError, match="maximal-ball extraction settings must be"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="native_maximal_ball",
            voxel_size=1.0,
            extraction_kwargs={"settings": 3.14},
            flow_axis="x",
        )

    with pytest.raises(ValueError, match="Unexpected extraction_kwargs for backend='maximal_ball'"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="native_maximal_ball",
            voxel_size=1.0,
            extraction_kwargs={"unexpected_key": True},
            flow_axis="x",
        )


def test_extract_network_dict_rejects_invalid_prego_kwargs() -> None:
    """PREGO dispatch should fail clearly on invalid settings or stray kwargs."""

    with pytest.raises(TypeError, match="PREGO extraction settings must be"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="prego",
            voxel_size=1.0,
            extraction_kwargs={"settings": 3.14},
            flow_axis="x",
        )

    with pytest.raises(TypeError, match="regions_to_network_kwargs must be"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="prego",
            voxel_size=1.0,
            extraction_kwargs={"regions_to_network_kwargs": 3.14},
            flow_axis="x",
        )

    with pytest.raises(ValueError, match="Unexpected extraction_kwargs for backend='prego'"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="prego",
            voxel_size=1.0,
            extraction_kwargs={"unexpected_key": True},
            flow_axis="x",
        )


def test_extract_network_dict_asserts_on_unhandled_normalized_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private dispatcher should guard against impossible backend normalization drift."""

    monkeypatch.setattr(nex, "_normalize_extraction_backend", lambda backend: "unexpected_backend")

    with pytest.raises(AssertionError, match="Unhandled normalized backend"):
        nex._extract_network_dict(
            np.ones((2, 2, 2), dtype=int),
            backend="porespy",
            voxel_size=1.0,
            extraction_kwargs=None,
            flow_axis="x",
        )


def test_normalize_construction_backend_rejects_unsupported_backend() -> None:
    """Construction backend normalization should reject unsupported backend names."""

    with pytest.raises(ValueError, match="Unsupported construction backend"):
        nex._normalize_construction_backend("unsupported_backend")


def test_extract_spanning_pore_network_applies_imperial_snow2_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibrated Imperial-style alias should inject benchmark-tuned defaults."""

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
        backend="porespy_imperial",
        flow_axis="x",
        extraction_kwargs={"sigma": 0.8},
    )

    assert result.backend == "porespy_snow2_imperial"
    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=int))
    assert captured["kwargs"] == {"sigma": 0.8, "r_max": 4, "boundary_width": 1}


def test_extract_spanning_pore_network_normalizes_backend_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public extraction should expose an explicit backend choice with aliases."""

    captured: dict[str, object] = {}

    def fake_extract(phases, *, backend, voxel_size, extraction_kwargs, flow_axis):
        captured["phases"] = phases
        captured["backend"] = backend
        captured["voxel_size"] = voxel_size
        captured["kwargs"] = extraction_kwargs
        captured["flow_axis"] = flow_axis
        return {
            "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.xmin": np.array([True, False], dtype=bool),
            "pore.xmax": np.array([False, True], dtype=bool),
        }

    monkeypatch.setattr(nex, "_extract_network_dict", fake_extract)
    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="snow2",
        flow_axis="x",
    )

    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=int))
    assert captured["backend"] == "porespy_snow2"
    assert captured["voxel_size"] == pytest.approx(1.0)
    assert captured["kwargs"] is None
    assert captured["flow_axis"] == "x"
    assert result.backend == "porespy_snow2"
    assert result.provenance.extraction_method == "porespy_snow2"

    captured.clear()
    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="prego",
        flow_axis="x",
    )

    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=int))
    assert captured["backend"] == "prego"
    assert captured["kwargs"] is None
    assert captured["flow_axis"] == "x"
    assert result.backend == "prego"
    assert result.provenance.extraction_method == "prego"

    captured.clear()
    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="snow2_imperial",
        flow_axis="x",
    )

    assert np.array_equal(captured["phases"], np.ones((2, 2, 2), dtype=int))
    assert captured["backend"] == "porespy_snow2_imperial"
    assert captured["kwargs"] is None
    assert captured["flow_axis"] == "x"
    assert result.backend == "porespy_snow2_imperial"
    assert result.provenance.extraction_method == "porespy_snow2_imperial"


def test_extract_spanning_pore_network_skips_second_geometry_repair_for_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native maximal-ball assembly should not reapply importer geometry repairs."""

    captured: dict[str, object] = {}
    net = make_linear_chain_network(num_pores=2)

    def fake_extract(phases, *, backend, voxel_size, extraction_kwargs, flow_axis):
        assert backend == "native_maximal_ball"
        assert voxel_size == pytest.approx(1.0)
        assert flow_axis == "x"
        return {
            "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.inlet_xmin": np.array([True, False], dtype=bool),
            "pore.outlet_xmax": np.array([False, True], dtype=bool),
            "pore.radius_inscribed": np.array([1.0, 1.0], dtype=float),
            "pore.area": np.array([1.0, 1.0], dtype=float),
            "pore.shape_factor": np.array([0.03, 0.03], dtype=float),
            "pore.volume": np.array([1.0, 1.0], dtype=float),
            "throat.radius_inscribed": np.array([0.5], dtype=float),
            "throat.cross_sectional_area": np.array([0.5], dtype=float),
            "throat.shape_factor": np.array([0.02], dtype=float),
            "throat.volume": np.array([0.1], dtype=float),
            "throat.conduit_lengths.pore1": np.array([0.2], dtype=float),
            "throat.conduit_lengths.throat": np.array([0.6], dtype=float),
            "throat.conduit_lengths.pore2": np.array([0.2], dtype=float),
        }

    def fake_from_porespy(
        network_dict,
        *,
        sample,
        provenance,
        strict,
        geometry_repairs,
        repair_seed,
    ):
        captured["geometry_repairs"] = geometry_repairs
        captured["strict"] = strict
        captured["repair_seed"] = repair_seed
        return net

    def fake_spanning_subnetwork(net_full, axis):
        assert axis == "x"
        return net_full, np.arange(net_full.Np, dtype=np.int64), np.ones(net_full.Nt, dtype=bool)

    monkeypatch.setattr(nex, "_extract_network_dict", fake_extract)
    monkeypatch.setattr(nex, "from_porespy", fake_from_porespy)
    monkeypatch.setattr(nex, "spanning_subnetwork", fake_spanning_subnetwork)

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="native_maximal_ball",
        flow_axis="x",
    )

    assert captured["geometry_repairs"] is None
    assert captured["strict"] is True
    assert captured["repair_seed"] == 0
    assert result.backend == "native_maximal_ball"


def _make_minimal_network_dict() -> dict[str, object]:
    """Return a minimal network_dict accepted by from_porespy."""
    return {
        "pore.coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.inlet_xmin": np.array([True, False], dtype=bool),
        "pore.outlet_xmax": np.array([False, True], dtype=bool),
        "pore.radius_inscribed": np.array([1.0, 1.0], dtype=float),
        "pore.area": np.array([1.0, 1.0], dtype=float),
        "pore.shape_factor": np.array([0.03, 0.03], dtype=float),
        "pore.volume": np.array([1.0, 1.0], dtype=float),
        "throat.radius_inscribed": np.array([0.5], dtype=float),
        "throat.cross_sectional_area": np.array([0.5], dtype=float),
        "throat.shape_factor": np.array([0.02], dtype=float),
        "throat.volume": np.array([0.1], dtype=float),
        "throat.conduit_lengths.pore1": np.array([0.2], dtype=float),
        "throat.conduit_lengths.throat": np.array([0.6], dtype=float),
        "throat.conduit_lengths.pore2": np.array([0.2], dtype=float),
    }


def test_porespy_style_external_reservoir_boundary_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PoreSpy-style image backends can use helper reservoir pores for flow BCs."""

    monkeypatch.setattr(nex, "_extract_network_dict", lambda *a, **kw: _make_minimal_network_dict())

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="porespy",
        flow_axis="x",
        extraction_kwargs={"flow_boundary_mode": "external_reservoir"},
        geometry_repairs=None,
    )

    assert result.net_full.Np == 4
    assert result.net_full.Nt == 3
    assert np.array_equal(
        result.net_full.throat_conns,
        np.array([[0, 1], [2, 0], [1, 3]], dtype=np.int64),
    )
    assert np.array_equal(
        result.net_full.pore_labels["inlet_xmin"],
        np.array([False, False, True, False]),
    )
    assert np.array_equal(
        result.net_full.pore_labels["outlet_xmax"],
        np.array([False, False, False, True]),
    )
    assert np.array_equal(
        result.net_full.pore_labels["boundary_connected_inlet_xmin"],
        np.array([True, False, False, False]),
    )
    assert np.array_equal(
        result.net_full.pore_labels["boundary_connected_outlet_xmax"],
        np.array([False, True, False, False]),
    )
    assert np.all(result.net_full.throat["pore1_length"] > 0.0)
    assert np.all(result.net_full.throat["core_length"] > 0.0)
    assert np.all(result.net_full.throat["pore2_length"] > 0.0)


def test_external_reservoir_helper_size_fallbacks_and_validation() -> None:
    """Boundary-helper geometry should use explicit fields before conservative fallbacks."""

    with pytest.raises(ValueError, match="flow_boundary_mode must be one of"):
        nex._resolve_flow_boundary_mode("reservoir")
    with pytest.raises(ValueError, match="transport_geometry must be None"):
        nex._resolve_transport_geometry("cylinders")

    net = make_linear_chain_network(num_pores=3)

    net.pore = {"diameter_inscribed": np.array([1.0, 0.0, 2.0])}
    radius = nex._entity_radius_from_fields(net, "pore", fallback_radius=0.2)
    assert np.allclose(radius, np.array([0.5, 0.2, 1.0]))

    net.pore = {"area": np.array([4.0 * np.pi, 0.0, 0.25 * np.pi])}
    radius = nex._entity_radius_from_fields(net, "pore", fallback_radius=0.2)
    assert np.allclose(radius, np.array([2.0, 0.2, 0.5]))

    net.pore = {}
    radius = nex._entity_radius_from_fields(net, "pore", fallback_radius=0.2)
    assert np.allclose(radius, np.array([0.2, 0.2, 0.2]))

    net.pore = {"radius_inscribed": np.array([0.5, 0.0, 2.0])}
    diameter = nex._entity_diameter_for_pyramids_and_cuboids(
        net,
        "pore",
        fallback_diameter=0.3,
    )
    assert diameter[0] == pytest.approx(1.0)
    assert diameter[1] > 0.0
    assert diameter[2] == pytest.approx(4.0)

    net.pore = {"area": np.array([4.0, 0.0, 9.0])}
    diameter = nex._entity_diameter_for_pyramids_and_cuboids(
        net,
        "pore",
        fallback_diameter=0.3,
    )
    assert diameter[0] == pytest.approx(2.0)
    assert diameter[1] > 0.0
    assert diameter[2] == pytest.approx(3.0)

    net.pore = {}
    diameter = nex._entity_diameter_for_pyramids_and_cuboids(
        net,
        "pore",
        fallback_diameter=0.3,
    )
    assert np.allclose(diameter, np.full(net.Np, 0.3))


def test_external_reservoir_field_extension_branches() -> None:
    """Helper pore and throat extension should preserve field semantics."""

    helper_coords = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    helper_radii = np.array([0.5, 0.75])
    helper_area = np.pi * helper_radii**2
    helper_source_indices = np.array([1, 0], dtype=np.int64)

    local_peak = nex._extend_pore_field(
        "local_peak",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        helper_coords=helper_coords,
        helper_radii=helper_radii,
        helper_area=helper_area,
        helper_source_indices=helper_source_indices,
    )
    assert np.allclose(local_peak[2:], helper_coords[:, :2])

    phase = nex._extend_pore_field(
        "phase",
        np.array([7, 9], dtype=np.int64),
        helper_coords=helper_coords,
        helper_radii=helper_radii,
        helper_area=helper_area,
        helper_source_indices=helper_source_indices,
    )
    assert np.array_equal(phase, np.array([7, 9, 9, 7]))

    pore_flag = nex._extend_pore_field(
        "boundary_flag",
        np.array([True, False]),
        helper_coords=helper_coords,
        helper_radii=helper_radii,
        helper_area=helper_area,
        helper_source_indices=helper_source_indices,
    )
    assert np.array_equal(pore_flag, np.array([True, False, False, False]))

    pore_unknown = nex._extend_pore_field(
        "quality_score",
        np.array([1.5, 2.5]),
        helper_coords=helper_coords,
        helper_radii=helper_radii,
        helper_area=helper_area,
        helper_source_indices=helper_source_indices,
    )
    assert np.allclose(pore_unknown, np.array([1.5, 2.5, 0.0, 0.0]))

    boundary_length = np.array([1.0, 2.0])
    boundary_area = np.array([3.0, 5.0])
    boundary_radius = np.array([0.4, 0.6])
    boundary_pore1_length = np.array([0.1, 0.2])
    boundary_core_length = np.array([0.7, 1.6])
    boundary_pore2_length = np.array([0.2, 0.2])
    boundary_centroid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    kwargs = {
        "boundary_length": boundary_length,
        "boundary_area": boundary_area,
        "boundary_radius": boundary_radius,
        "boundary_pore1_length": boundary_pore1_length,
        "boundary_core_length": boundary_core_length,
        "boundary_pore2_length": boundary_pore2_length,
        "boundary_centroid": boundary_centroid,
    }

    assert np.allclose(
        nex._extend_throat_field("pore1_length", np.array([9.0]), **kwargs)[1:],
        boundary_pore1_length,
    )
    assert np.allclose(
        nex._extend_throat_field("core_length", np.array([9.0]), **kwargs)[1:],
        boundary_core_length,
    )
    assert np.allclose(
        nex._extend_throat_field("pore2_length", np.array([9.0]), **kwargs)[1:],
        boundary_pore2_length,
    )
    assert np.allclose(
        nex._extend_throat_field("cross_sectional_area", np.array([9.0]), **kwargs)[1:],
        boundary_area,
    )
    assert np.allclose(
        nex._extend_throat_field("shape_factor_radius", np.array([9.0]), **kwargs)[1:],
        boundary_radius,
    )
    assert np.allclose(
        nex._extend_throat_field("equivalent_diameter", np.array([9.0]), **kwargs)[1:],
        2.0 * boundary_radius,
    )
    assert np.allclose(
        nex._extend_throat_field("shape_factor", np.array([9.0]), **kwargs)[1:],
        np.full(2, nex.DEFAULT_G_REF),
    )
    assert np.allclose(
        nex._extend_throat_field("volume", np.array([9.0]), **kwargs)[1:],
        boundary_area * boundary_core_length,
    )

    centroid = nex._extend_throat_field(
        "centroid",
        np.array([[8.0, 9.0]]),
        **kwargs,
    )
    assert np.allclose(centroid[1:], boundary_centroid[:, :2])

    face_count = nex._extend_throat_field("face_count", np.array([4], dtype=np.int64), **kwargs)
    assert np.array_equal(face_count, np.array([4, 1, 1]))

    active = nex._extend_throat_field("active", np.array([True]), **kwargs)
    assert np.array_equal(active, np.array([True, False, False]))

    throat_unknown = nex._extend_throat_field("roughness", np.array([1.25]), **kwargs)
    assert np.allclose(throat_unknown, np.array([1.25, 0.0, 0.0]))

    with pytest.raises(ValueError, match="precomputed throat.hydraulic_conductance"):
        nex._extend_throat_field("hydraulic_conductance", np.array([1.0]), **kwargs)


def test_external_reservoir_network_validation_branches() -> None:
    """External reservoirs should reject inconsistent boundary geometry early."""

    def without_precomputed_conductance():
        net = make_linear_chain_network(num_pores=2)
        net.throat.pop("hydraulic_conductance")
        return net

    kwargs = {
        "axis": "x",
        "axis_length": 1.0,
        "voxel_size": 1.0,
        "boundary_length_epsilon": 1.0e-9,
        "boundary_radius_scale": 1.1,
    }

    with pytest.raises(ValueError, match="boundary_axis must be one of"):
        nex._add_external_reservoirs_to_network(
            without_precomputed_conductance(), **{**kwargs, "axis": "q"}
        )
    with pytest.raises(ValueError, match="boundary_length_epsilon must be positive"):
        nex._add_external_reservoirs_to_network(
            without_precomputed_conductance(),
            **{**kwargs, "boundary_length_epsilon": 0.0},
        )
    with pytest.raises(ValueError, match="boundary_radius_scale must be positive"):
        nex._add_external_reservoirs_to_network(
            without_precomputed_conductance(),
            **{**kwargs, "boundary_radius_scale": 0.0},
        )
    with pytest.raises(ValueError, match="precomputed throat.hydraulic_conductance"):
        nex._add_external_reservoirs_to_network(make_linear_chain_network(num_pores=2), **kwargs)

    missing_labels = without_precomputed_conductance()
    missing_labels.pore_labels.pop("outlet_xmax")
    with pytest.raises(KeyError, match="Missing pore boundary labels"):
        nex._add_external_reservoirs_to_network(missing_labels, **kwargs)

    no_boundary_pores = without_precomputed_conductance()
    no_boundary_pores.pore_labels["inlet_xmin"][:] = False
    no_boundary_pores.pore_labels["outlet_xmax"][:] = False
    assert nex._add_external_reservoirs_to_network(no_boundary_pores, **kwargs) is no_boundary_pores

    labeled = without_precomputed_conductance()
    labeled.pore_labels["interior_marker"] = np.array([False, True])
    augmented = nex._add_external_reservoirs_to_network(labeled, **kwargs)
    assert np.array_equal(
        augmented.pore_labels["interior_marker"],
        np.array([False, True, False, False]),
    )


def test_transport_geometry_rejects_missing_or_nonfinite_conduit_geometry() -> None:
    """Pyramids-and-cuboids size factors require complete finite conduit inputs."""

    with pytest.raises(KeyError, match="requires conduit lengths"):
        nex._assign_pyramids_and_cuboids_transport_geometry(
            make_linear_chain_network(num_pores=2),
            voxel_size=1.0,
        )

    bad = make_linear_chain_network(num_pores=2)
    bad.pore["radius_inscribed"] = np.array([1.0, 1.0])
    bad.throat["radius_inscribed"] = np.array([0.5])
    bad.throat["pore1_length"] = np.array([np.nan])
    bad.throat["core_length"] = np.array([1.0])
    bad.throat["pore2_length"] = np.array([1.0])
    with pytest.raises(ValueError, match="positive and finite"):
        nex._assign_pyramids_and_cuboids_transport_geometry(bad, voxel_size=1.0)

    with pytest.raises(ValueError, match="boundary_axis 'z' is not compatible"):
        extract_spanning_pore_network(
            np.ones((2, 2), dtype=int),
            voxel_size=1.0,
            backend="porespy",
            extraction_kwargs={"boundary_axis": "z"},
        )


def test_pyramids_and_cuboids_transport_geometry_adds_hydraulic_size_factors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The pyramids-and-cuboids option stores conduit size factors."""

    monkeypatch.setattr(nex, "_extract_network_dict", lambda *a, **kw: _make_minimal_network_dict())

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="prego",
        flow_axis="x",
        extraction_kwargs={
            "flow_boundary_mode": "external_reservoir",
            "transport_geometry": "pyramids_and_cuboids",
        },
        geometry_repairs=None,
    )

    sf = result.net_full.throat["hydraulic_size_factors"]
    assert sf.shape == (result.net_full.Nt, 3)
    assert np.all(np.isfinite(sf))
    assert np.all(sf > 0.0)
    assert result.net_full.extra["transport_geometry"]["mode"] == "pyramids_and_cuboids"
    assert (
        result.net_full.extra["transport_geometry"]["hydraulic_size_factors_location"]
        == "throat.hydraulic_size_factors"
    )

    path = tmp_path / "pyramids_and_cuboids_transport_geometry.h5"
    save_hdf5(result.net_full, path)
    loaded = load_hdf5(path)
    assert np.allclose(loaded.throat["hydraulic_size_factors"], sf)


def test_extract_spanning_pore_network_uses_voids_version_for_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native_maximal_ball backend should record the voids version, not porespy."""
    import porespy as ps

    from voids.version import __version__ as voids_version

    net = make_linear_chain_network(num_pores=2)

    monkeypatch.setattr(nex, "_extract_network_dict", lambda *a, **kw: _make_minimal_network_dict())
    monkeypatch.setattr(nex, "from_porespy", lambda *a, **kw: net)
    monkeypatch.setattr(
        nex,
        "spanning_subnetwork",
        lambda n, axis: (n, np.arange(n.Np, dtype=np.int64), np.ones(n.Nt, dtype=bool)),
    )

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="native_maximal_ball",
        flow_axis="x",
    )

    assert result.backend_version == voids_version
    assert result.provenance.source_version == voids_version
    porespy_ver = getattr(ps, "__version__", None)
    assert porespy_ver is not None, "porespy must expose __version__ for this test to be meaningful"
    assert result.backend_version != porespy_ver


def test_extract_spanning_pore_network_uses_voids_version_for_prego_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PREGO is implemented in voids, even though network geometry uses PoreSpy format."""

    from voids.version import __version__ as voids_version

    net = make_linear_chain_network(num_pores=2)

    monkeypatch.setattr(nex, "_extract_network_dict", lambda *a, **kw: _make_minimal_network_dict())
    monkeypatch.setattr(nex, "from_porespy", lambda *a, **kw: net)
    monkeypatch.setattr(
        nex,
        "spanning_subnetwork",
        lambda n, axis: (n, np.arange(n.Np, dtype=np.int64), np.ones(n.Nt, dtype=bool)),
    )
    monkeypatch.setattr(nex, "scale_porespy_geometry", lambda d, voxel_size: d)
    monkeypatch.setattr(nex, "ensure_cartesian_boundary_labels", lambda d, axes: d)

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="prego",
        flow_axis="x",
    )

    assert result.backend_version == voids_version
    assert result.provenance.source_version == voids_version


def test_extract_spanning_pore_network_uses_porespy_version_for_porespy_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PoreSpy-based backends should record porespy.__version__, not the voids version."""
    import porespy as ps

    from voids.version import __version__ as voids_version

    net = make_linear_chain_network(num_pores=2)

    monkeypatch.setattr(nex, "_extract_network_dict", lambda *a, **kw: _make_minimal_network_dict())
    monkeypatch.setattr(nex, "from_porespy", lambda *a, **kw: net)
    monkeypatch.setattr(
        nex,
        "spanning_subnetwork",
        lambda n, axis: (n, np.arange(n.Np, dtype=np.int64), np.ones(n.Nt, dtype=bool)),
    )
    monkeypatch.setattr(nex, "scale_porespy_geometry", lambda d, voxel_size: d)
    monkeypatch.setattr(nex, "ensure_cartesian_boundary_labels", lambda d, axes: d)

    result = extract_spanning_pore_network(
        np.ones((2, 2, 2), dtype=int),
        voxel_size=1.0,
        backend="porespy",
        flow_axis="x",
    )

    porespy_ver = getattr(ps, "__version__", None)
    assert porespy_ver is not None, "porespy must expose __version__ for this test to be meaningful"
    assert result.backend_version == porespy_ver
    assert result.provenance.source_version == porespy_ver
    assert porespy_ver != voids_version, (
        "porespy and voids versions must differ for this test to distinguish the two"
    )


def test_extract_spanning_pore_network_rejects_unsupported_backend() -> None:
    """Unsupported image-extraction backends should fail before backend work starts."""

    with pytest.raises(ValueError, match="Unsupported extraction backend"):
        extract_spanning_pore_network(
            np.ones((4, 5, 6), dtype=int),
            voxel_size=1.0,
            backend="pnextract_like",
        )


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
