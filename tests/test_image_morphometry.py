from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import voids.image.morphometry as morphometry
from voids.image.morphometry import (
    local_thickness_analysis,
    local_thickness_map,
    summarize_local_thickness_map,
)


def test_porespy_local_thickness_is_imported_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PoreSpy call path should be lazy and easy to isolate in tests."""

    fake_porespy = SimpleNamespace(
        filters=SimpleNamespace(local_thickness=lambda *a, **k: np.array([[2.0]]))
    )
    monkeypatch.setitem(sys.modules, "porespy", fake_porespy)

    radius = morphometry._porespy_local_thickness(np.array([[True]]), method="dt")

    assert np.allclose(radius, [[2.0]])


def test_local_thickness_map_converts_porespy_radius_to_physical_diameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PoreSpy local-thickness radii should become physical diameter values."""

    phase = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )
    radius = np.arange(8, dtype=float).reshape(2, 2, 2)
    calls: dict[str, object] = {}

    def fake_local_thickness(*args, **kwargs):
        calls.update(kwargs)
        return radius

    monkeypatch.setattr(morphometry, "_porespy_local_thickness", fake_local_thickness)

    distance_map = np.ones_like(radius)
    thickness = local_thickness_map(
        phase,
        voxel_size=2.5,
        method="imj",
        smooth=False,
        approx=True,
        sizes=None,
        distance_map=distance_map,
    )

    expected = np.where(phase.astype(bool), 2.0 * 2.5 * radius, 0.0)
    assert np.allclose(thickness, expected)
    assert calls["dt"] is distance_map
    assert calls["method"] == "imj"
    assert calls["smooth"] is False
    assert calls["approx"] is True
    assert calls["sizes"] is None


def test_local_thickness_analysis_summarizes_phase_voxels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined helper should return a map plus phase-only statistics."""

    phase = np.array([[True, True, False], [False, True, False]])
    radius = np.array([[1.0, 2.0, 9.0], [9.0, 3.0, 9.0]])
    monkeypatch.setattr(morphometry, "_porespy_local_thickness", lambda *a, **k: radius)

    result = local_thickness_analysis(
        phase,
        voxel_size=10.0,
        units="um",
        label="bone",
        method="dt",
        sizes=4,
    )

    values = np.array([20.0, 40.0, 60.0])
    assert np.allclose(result.thickness_map[phase], values)
    assert np.all(result.thickness_map[~phase] == 0.0)
    assert result.summary.label == "bone"
    assert result.summary.voxel_count == 3
    assert result.summary.mean == pytest.approx(float(np.mean(values)))
    assert result.summary.p50 == pytest.approx(float(np.median(values)))
    assert result.summary.max == pytest.approx(60.0)
    assert result.summary.units == "um"
    assert result.summary.method == "dt"


def test_summarize_local_thickness_map_handles_empty_phase() -> None:
    """Empty phases should yield NaN statistics rather than invoking PoreSpy."""

    phase = np.zeros((2, 2, 2), dtype=bool)
    thickness = local_thickness_map(phase, voxel_size=3.0)
    summary = summarize_local_thickness_map(
        thickness,
        phase,
        label="empty",
        units="um",
        method="dt",
        voxel_size=3.0,
    )

    assert np.all(thickness == 0.0)
    assert summary.voxel_count == 0
    assert np.isnan(summary.mean)
    assert np.isnan(summary.max)
    assert summary.as_dict()["label"] == "empty"


def test_local_thickness_rejects_nonbinary_and_anisotropic_inputs() -> None:
    """The API should fail loudly for ambiguous masks or anisotropic voxels."""

    with pytest.raises(ValueError, match="numeric 0/1"):
        local_thickness_map(np.array([[0, 2]], dtype=int))

    with pytest.raises(ValueError, match="numeric 0/1"):
        local_thickness_map(np.array([["bone", "marrow"]], dtype=object))

    with pytest.raises(ValueError, match="length 2"):
        local_thickness_map(np.ones((2, 2), dtype=bool), voxel_size=(1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="finite and positive"):
        local_thickness_map(np.ones((2, 2), dtype=bool), voxel_size=0.0)

    with pytest.raises(ValueError, match="isotropic"):
        local_thickness_map(np.ones((2, 2), dtype=bool), voxel_size=(1.0, 2.0))

    with pytest.raises(ValueError, match="same shape"):
        local_thickness_map(
            np.ones((2, 2), dtype=bool),
            distance_map=np.ones((2, 3), dtype=float),
        )

    with pytest.raises(ValueError, match="finite"):
        local_thickness_map(
            np.ones((2, 2), dtype=bool),
            distance_map=np.array([[1.0, np.nan], [1.0, 1.0]]),
        )

    with pytest.raises(ValueError, match="nonnegative"):
        local_thickness_map(
            np.ones((2, 2), dtype=bool),
            distance_map=np.array([[1.0, -1.0], [1.0, 1.0]]),
        )


def test_summarize_local_thickness_map_rejects_invalid_values() -> None:
    """Summary inputs should keep shape and value assumptions explicit."""

    phase = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="same shape"):
        summarize_local_thickness_map(np.ones((2, 3)), phase)

    with pytest.raises(ValueError, match="finite"):
        summarize_local_thickness_map(np.array([[1.0, np.inf], [1.0, 1.0]]), phase)

    with pytest.raises(ValueError, match="nonnegative"):
        summarize_local_thickness_map(np.array([[1.0, -1.0], [1.0, 1.0]]), phase)
