from __future__ import annotations

import numpy as np
import pytest

from voids.core.sample import SampleGeometry
from voids.io.porespy import (
    _apply_imperial_export_geometry_repairs,
    _derive_missing_geometry,
    _ensure_inscribed_size_aliases,
    _imperial_export_random_shape_factors,
    _override_area_from_shape_factor_and_radius,
    ensure_cartesian_boundary_labels,
    from_porespy,
    scale_porespy_geometry,
)


def test_from_porespy_minimal() -> None:
    """Test minimal import from a PoreSpy-style mapping."""

    d = {
        "pore.coords": np.array([[0, 0], [1, 0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.volume": np.array([1.0, 1.0]),
        "throat.volume": np.array([0.1]),
        "throat.length": np.array([1.0]),
        "pore.left": np.array([True, False]),
        "pore.right": np.array([False, True]),
    }
    net = from_porespy(d, sample=SampleGeometry(bulk_volume=10.0))
    assert net.pore_coords.shape == (2, 3)
    assert "volume" in net.pore and "length" in net.throat
    assert net.pore_labels["left"].dtype == bool
    assert net.pore_labels["inlet_xmin"].sum() == 1
    assert net.pore_labels["outlet_xmax"].sum() == 1


def test_from_porespy_keymap_entry_with_null_canonical_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keys in _PORESPY_KEYMAP with canonical=None (and no reserved role) are skipped."""
    import voids.io.porespy as porespy_mod

    extended_keymap = {
        **porespy_mod._PORESPY_KEYMAP,
        "pore.deliberately_ignored": ("pore", None, None),
    }
    monkeypatch.setattr(porespy_mod, "_PORESPY_KEYMAP", extended_keymap)

    d = {
        "pore.coords": np.array([[0, 0], [1, 0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.deliberately_ignored": np.array([99.0, 100.0]),
    }
    net = from_porespy(d, sample=SampleGeometry(bulk_volume=10.0))
    assert "deliberately_ignored" not in net.pore
    assert "deliberately_ignored" not in net.pore_labels


def test_from_porespy_maps_openpnm_aliases_and_derives_fields() -> None:
    """Test alias normalization and derived-field construction during PoreSpy import."""

    d = {
        "pore.coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.volume": np.array([1.0, 1.0]),
        "throat.volume": np.array([0.1]),
        "throat.cross_sectional_area": np.array([4.0]),
        "throat.total_length": np.array([0.7]),
        "throat.conduit_lengths.pore1": np.array([0.1]),
        "throat.conduit_lengths.throat": np.array([0.5]),
        "throat.conduit_lengths.pore2": np.array([0.1]),
        "throat.perimeter": np.array([8.0]),
        "pore.radius_inscribed": np.array([1.0, 1.0]),
        "pore.left": np.array([True, False]),
        "pore.right": np.array([False, True]),
    }
    net = from_porespy(d, sample=SampleGeometry(bulk_volume=10.0))

    assert np.allclose(net.throat["area"], [4.0])
    assert np.allclose(net.throat["length"], [0.7])
    assert np.allclose(net.throat["core_length"], [0.5])
    assert np.allclose(net.throat["pore1_length"], [0.1])
    assert np.allclose(net.throat["pore2_length"], [0.1])
    assert np.allclose(net.throat["shape_factor"], [1 / 16])
    assert np.allclose(net.pore["diameter_inscribed"], [2.0, 2.0])
    assert np.allclose(net.pore["area"], [np.pi, np.pi])
    assert net.pore_labels["inlet_xmin"].sum() == 1
    assert net.pore_labels["outlet_xmax"].sum() == 1


def test_scale_porespy_geometry_and_infer_boundaries() -> None:
    """Test voxel-to-physical scaling and simple Cartesian boundary inference."""

    d = {
        "pore.coords": np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.region_volume": np.array([10.0, 12.0]),
        "throat.cross_sectional_area": np.array([3.0]),
        "throat.total_length": np.array([2.0]),
    }
    scaled = scale_porespy_geometry(d, voxel_size=2.0)
    labeled = ensure_cartesian_boundary_labels(scaled, axes=("x",))

    assert np.allclose(scaled["pore.coords"], [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert np.allclose(scaled["pore.volume"], [80.0, 96.0])
    assert np.allclose(scaled["throat.volume"], [48.0])
    assert labeled["pore.inlet_xmin"].tolist() == [True, False]
    assert labeled["pore.outlet_xmax"].tolist() == [False, True]
    assert labeled["pore.boundary"].tolist() == [True, True]


def test_from_porespy_imperial_export_repairs_shape_factors_and_recomputes_pore_values() -> None:
    """Imperial export mode should repair throat G and rebuild pore G from throat weights."""

    d = {
        "pore.coords": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
        "throat.conns": np.array([[0, 1], [1, 2]], dtype=int),
        "pore.radius_inscribed": np.array([1.0, 1.0, 1.0]),
        "throat.radius_inscribed": np.array([1.0, 1.0]),
        "throat.area": np.array([2.0, 100.0]),
        "throat.total_length": np.array([1.0, 1.0]),
    }

    net = from_porespy(
        d,
        sample=SampleGeometry(bulk_volume=10.0),
        geometry_repairs="imperial_export",
        repair_seed=0,
    )

    g_high, g_low = net.throat["shape_factor"]
    assert g_high == np.min([0.079, 0.125 / 2.0]) == 0.0625
    assert 0.01 <= g_low <= 0.0625

    expected_pore_mid = (g_high * 2.0 + g_low * 100.0 + 5.0e-38) / (102.0 + 1.0e-36)
    assert net.pore["shape_factor"][0] == pytest.approx(g_high)
    assert net.pore["shape_factor"][1] == pytest.approx(expected_pore_mid)
    assert net.pore["shape_factor"][2] == pytest.approx(g_low)

    assert np.allclose(net.throat["area"], 1.0 / (4.0 * net.throat["shape_factor"]))
    assert np.allclose(net.pore["area"], 1.0 / (4.0 * net.pore["shape_factor"]))
    assert net.extra["geometry_repairs"]["mode"] == "imperial_export"
    assert net.extra["geometry_repairs"]["throat_high_repairs"] == 1
    assert net.extra["geometry_repairs"]["throat_low_repairs"] == 1
    assert net.extra["geometry_repairs"]["pore_shape_factor_weighted"] is True


def test_from_porespy_accepts_legacy_geometry_repairs_alias_with_deprecation_warning() -> None:
    """Legacy repair-mode names should remain usable with a deprecation warning."""

    d = {
        "pore.coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.radius_inscribed": np.array([1.0, 1.0]),
        "throat.radius_inscribed": np.array([1.0]),
        "throat.area": np.array([2.0]),
        "throat.total_length": np.array([1.0]),
    }

    with pytest.warns(DeprecationWarning, match=r"geometry_repairs='pnextract'.*'imperial_export'"):
        net = from_porespy(
            d,
            sample=SampleGeometry(bulk_volume=10.0),
            geometry_repairs="pnextract",
            repair_seed=0,
        )

    assert net.extra["geometry_repairs"]["mode"] == "imperial_export"
    assert net.throat["shape_factor"][0] == pytest.approx(0.0625)


def test_scale_geometry_skips_nonnumeric_and_scales_perimeter() -> None:
    raw = {
        "pore.coords": np.array([[0.0, 1.0, 2.0]]),
        "throat.perimeter": np.array([2.0]),
        "meta": "keep-me",
    }
    out = scale_porespy_geometry(raw, voxel_size=3.0)
    assert np.allclose(out["pore.coords"], [[0.0, 3.0, 6.0]])
    assert np.allclose(out["throat.perimeter"], [6.0])
    assert out["meta"] == "keep-me"


def test_scale_geometry_requires_positive_voxel_size() -> None:
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        scale_porespy_geometry({}, voxel_size=0.0)


def test_ensure_cartesian_boundary_labels_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape \\(Np, 2\\) or \\(Np, 3\\)"):
        ensure_cartesian_boundary_labels({"pore.coords": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="nonnegative"):
        ensure_cartesian_boundary_labels(
            {"pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]])}, tol_fraction=-1.0
        )
    with pytest.raises(ValueError, match="drawn from"):
        ensure_cartesian_boundary_labels(
            {"pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]])}, axes=("u",)
        )
    with pytest.raises(ValueError, match="not available"):
        ensure_cartesian_boundary_labels(
            {"pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]])}, axes=("z",)
        )


def test_ensure_cartesian_boundary_labels_preserves_existing_labels() -> None:
    d = {
        "pore.coords": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "pore.inlet_xmin": np.array([False, True]),
    }
    out = ensure_cartesian_boundary_labels(d, axes=("x", "y"))
    assert out["pore.inlet_xmin"].tolist() == [False, True]
    assert out["pore.outlet_xmax"].tolist() == [False, True]
    assert out["pore.inlet_ymin"].tolist() == [True, False]
    assert out["pore.outlet_ymax"].tolist() == [False, True]
    assert out["pore.boundary"].tolist() == [True, True]


def test_derive_missing_geometry_radius_and_lengths() -> None:
    pore = {"radius_inscribed": np.array([2.0]), "perimeter": np.array([8.0])}
    throat = {
        "radius_inscribed": np.array([1.0]),
        "pore1_length": np.array([0.1]),
        "core_length": np.array([0.2]),
        "pore2_length": np.array([0.3]),
    }
    _derive_missing_geometry(pore, throat)
    assert np.allclose(pore["area"], [4.0 * np.pi])
    assert np.allclose(pore["diameter_inscribed"], [4.0])
    assert np.allclose(pore["shape_factor"], [4.0 * np.pi / 64.0])
    assert np.allclose(throat["area"], [np.pi])
    assert np.allclose(throat["diameter_inscribed"], [2.0])
    assert np.allclose(throat["length"], [0.6])


def test_ensure_inscribed_size_aliases_backfills_radius() -> None:
    d = {"diameter_inscribed": np.array([2.0, 4.0])}
    _ensure_inscribed_size_aliases(d)
    assert np.allclose(d["radius_inscribed"], [1.0, 2.0])


def test_imperial_random_shape_factors_handles_rejected_samples() -> None:
    class _FakeRng:
        def __init__(self) -> None:
            self.calls = 0

        def random(self, n: int) -> np.ndarray:
            self.calls += 1
            if self.calls <= 2:
                return np.full(n, 1.0)
            return np.full(n, 0.75)

    g = _imperial_export_random_shape_factors(3, _FakeRng())  # type: ignore[arg-type]
    assert g.shape == (3,)
    assert np.all(np.isfinite(g))
    assert np.all(g > 0.0)


def test_override_area_from_shape_factor_and_radius_false_when_missing() -> None:
    assert _override_area_from_shape_factor_and_radius({"shape_factor": np.array([0.1])}) is False


def test_apply_imperial_repairs_return_paths_without_weighting() -> None:
    summary = _apply_imperial_export_geometry_repairs(
        pore_data={},
        throat_data={},
        throat_conns=np.array([[0, 1]]),
        num_pores=2,
        random_seed=0,
    )
    assert summary["throat_shape_factor_source"] == "existing"
    assert summary["throat_high_repairs"] == 0
    assert summary["pore_shape_factor_weighted"] is False

    throat = {"shape_factor": np.array([0.2])}
    summary = _apply_imperial_export_geometry_repairs(
        pore_data={},
        throat_data=throat,
        throat_conns=np.array([[0, 1]]),
        num_pores=2,
        random_seed=0,
    )
    assert summary["throat_high_repairs"] == 1
    assert summary["pore_shape_factor_weighted"] is False
    assert np.allclose(throat["shape_factor"], [0.079])


def test_apply_imperial_repairs_sets_pore_shape_only_if_all_connected() -> None:
    throat_data = {"shape_factor": np.array([0.02]), "area": np.array([2.0])}
    pore_data: dict[str, np.ndarray] = {}
    summary = _apply_imperial_export_geometry_repairs(
        pore_data=pore_data,
        throat_data=throat_data,
        throat_conns=np.array([[0, 1]]),
        num_pores=3,
        random_seed=0,
    )
    assert "shape_factor" not in pore_data
    assert summary["pore_shape_factor_weighted"] is False


def test_apply_imperial_repairs_sets_pore_shape_if_all_connected() -> None:
    throat_data = {"shape_factor": np.array([0.02, 0.03]), "area": np.array([2.0, 2.0])}
    pore_data: dict[str, np.ndarray] = {}
    summary = _apply_imperial_export_geometry_repairs(
        pore_data=pore_data,
        throat_data=throat_data,
        throat_conns=np.array([[0, 1], [1, 2]]),
        num_pores=3,
        random_seed=0,
    )
    assert "shape_factor" in pore_data
    assert summary["pore_shape_factor_weighted"] is True


def test_apply_imperial_repairs_overwrites_existing_pore_shape_factor() -> None:
    throat_data = {"shape_factor": np.array([0.02, 0.03]), "area": np.array([2.0, 2.0])}
    pore_data = {"shape_factor": np.array([0.5, 0.5, 0.5])}
    summary = _apply_imperial_export_geometry_repairs(
        pore_data=pore_data,
        throat_data=throat_data,
        throat_conns=np.array([[0, 1], [1, 2]]),
        num_pores=3,
        random_seed=0,
    )
    assert summary["pore_shape_factor_weighted"] is True
    assert not np.allclose(pore_data["shape_factor"], [0.5, 0.5, 0.5])


def test_from_porespy_non_strict_missing_and_invalid_geometry_repairs() -> None:
    with pytest.raises(KeyError, match="Required keys"):
        from_porespy({}, strict=False)
    with pytest.raises(ValueError, match="geometry_repairs must be None or 'imperial_export'"):
        from_porespy(
            {"pore.coords": np.array([[0.0, 0.0]]), "throat.conns": np.array([[0, 0]])},
            geometry_repairs="invalid",
        )


def test_from_porespy_strict_missing_keys_raises() -> None:
    with pytest.raises(KeyError, match="must include 'throat.conns' and 'pore.coords'"):
        from_porespy({}, strict=True)


def test_from_porespy_maps_dotted_passthrough_and_extra_fields() -> None:
    d = {
        "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "throat.conns": np.array([[0, 1]]),
        "pore.custom.scalar": np.array([1.0, 2.0]),
        "throat.custom.scalar": np.array([3.0]),
        "pore.flag": np.array([True, False]),
        "throat.flag": np.array([True]),
        "unmapped": {"value": 1},
    }
    net = from_porespy(d)
    assert np.allclose(net.pore["custom_scalar"], [1.0, 2.0])
    assert np.allclose(net.throat["custom_scalar"], [3.0])
    assert net.pore_labels["flag"].tolist() == [True, False]
    assert net.throat_labels["flag"].tolist() == [True]
    assert net.extra["unmapped"] == {"value": 1}


@pytest.mark.parametrize("bad_g", [0.0, -0.1, float("nan"), float("inf")])
def test_override_area_raises_on_invalid_shape_factor(bad_g: float) -> None:
    data = {
        "shape_factor": np.array([0.1, bad_g]),
        "radius_inscribed": np.array([1.0, 1.0]),
    }
    with pytest.raises(ValueError, match="shape_factor must be positive and finite"):
        _override_area_from_shape_factor_and_radius(data)


@pytest.mark.parametrize("bad_area", [0.0, -1.0, float("nan"), float("inf")])
def test_imperial_repairs_raises_on_invalid_throat_area(bad_area: float) -> None:
    throat_data = {
        "radius_inscribed": np.array([1.0, 1.0]),
        "area": np.array([1.0, bad_area]),
    }
    with pytest.raises(ValueError, match="throat area values must be positive and finite"):
        _apply_imperial_export_geometry_repairs(
            pore_data={},
            throat_data=throat_data,
            throat_conns=np.array([[0, 1]]),
            num_pores=2,
            random_seed=0,
        )


def test_from_porespy_warns_and_stores_hydraulic_size_factors() -> None:
    d = {
        "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "throat.conns": np.array([[0, 1]]),
        "throat.hydraulic_size_factors": np.array([[1.0, 2.0, 3.0]]),
    }
    with pytest.warns(RuntimeWarning, match="Stored throat.hydraulic_size_factors"):
        net = from_porespy(d)
    assert "hydraulic_size_factors" not in net.throat
    assert "throat.hydraulic_size_factors" in net.extra
