from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from voids.generators import MacroMicroPorousImage, generate_macro_micro_blobs_matrix
from voids.generators import generate_spanning_multiscale_blobs_matrix
from voids.generators import porous_image as pimg
from voids.image import has_spanning_cluster_2d


def test_generate_macro_micro_blobs_matrix_places_small_pores_in_matrix() -> None:
    """Micropores should be confined to the macro matrix phase."""

    case = generate_macro_micro_blobs_matrix(
        shape=(64, 64),
        macro_porosity=0.32,
        matrix_microporosity=0.15,
        macro_blobiness=1.2,
        micropore_blobiness=8.0,
        seed_start=101,
        max_tries=30,
    )

    assert isinstance(case, MacroMicroPorousImage)
    assert case.shape == (64, 64)
    assert case.void.dtype == bool
    assert np.array_equal(case.void, case.macro_void | case.micropore_void)
    assert not np.any(case.micropore_void & case.macro_void)
    assert abs(case.macro_porosity - 0.32) <= 2.0e-3
    assert abs(case.matrix_microporosity - 0.15) <= 1.0e-3
    expected_total = case.macro_porosity + (1.0 - case.macro_porosity) * case.matrix_microporosity
    assert case.porosity == pytest.approx(expected_total)
    assert case.metadata["source_kind"] == "macro_micro_porespy_blobs"
    assert case.metadata["macro_seed"] >= 101


def test_generate_macro_micro_blobs_matrix_supports_zero_microporosity() -> None:
    """A zero microporosity request should return the macro field unchanged."""

    case = generate_macro_micro_blobs_matrix(
        shape=(24, 24, 24),
        macro_porosity=0.25,
        matrix_microporosity=0.0,
        macro_blobiness=1.0,
        micropore_blobiness=6.0,
        seed_start=5,
        max_tries=1,
    )

    assert not np.any(case.micropore_void)
    assert np.array_equal(case.void, case.macro_void)
    assert np.isnan(
        MacroMicroPorousImage(
            void=np.ones((2, 2), dtype=bool),
            macro_void=np.ones((2, 2), dtype=bool),
            micropore_void=np.zeros((2, 2), dtype=bool),
        ).matrix_microporosity
    )


def test_generate_macro_micro_blobs_matrix_validation() -> None:
    """The two-porosity generator should fail loudly for invalid controls."""

    with pytest.raises(ValueError, match="macro_porosity"):
        generate_macro_micro_blobs_matrix(
            shape=(16, 16),
            macro_porosity=0.0,
            matrix_microporosity=0.1,
            macro_blobiness=1.0,
            micropore_blobiness=5.0,
            seed_start=0,
            max_tries=1,
        )
    with pytest.raises(ValueError, match="max_tries"):
        generate_macro_micro_blobs_matrix(
            shape=(16, 16),
            macro_porosity=0.2,
            matrix_microporosity=0.1,
            macro_blobiness=1.0,
            micropore_blobiness=5.0,
            seed_start=0,
            max_tries=0,
        )
    with pytest.raises(ValueError, match="matrix_microporosity"):
        generate_macro_micro_blobs_matrix(
            shape=(16, 16),
            macro_porosity=0.2,
            matrix_microporosity=1.1,
            macro_blobiness=1.0,
            micropore_blobiness=5.0,
            seed_start=0,
            max_tries=1,
        )
    with pytest.raises(ValueError, match="micropore_blobiness"):
        generate_macro_micro_blobs_matrix(
            shape=(16, 16),
            macro_porosity=0.2,
            matrix_microporosity=0.1,
            macro_blobiness=1.0,
            micropore_blobiness=(1.0, -1.0),
            seed_start=0,
            max_tries=1,
        )


def test_generate_macro_micro_blobs_matrix_can_fail_spanning_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spanning acceptance should fail cleanly when both scales are disconnected."""

    def fake_blobs(*, shape, porosity, blobiness, seed, periodic):
        del porosity, blobiness, seed, periodic
        field = np.ones(shape, dtype=float)
        field[:2, :] = 0.0
        return field

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(generators=SimpleNamespace(blobs=fake_blobs)),
    )

    with pytest.raises(RuntimeError, match="Could not generate accepted macro/micro"):
        generate_macro_micro_blobs_matrix(
            shape=(16, 16),
            macro_porosity=0.10,
            matrix_microporosity=0.10,
            macro_blobiness=1.0,
            micropore_blobiness=5.0,
            axis_index=0,
            seed_start=0,
            max_tries=1,
        )


def test_macro_micro_porosity_dataclass_rejects_inconsistent_masks() -> None:
    """The case wrapper keeps the macro/micro phase split explicit."""

    macro = np.array([[True, False]])
    micro = np.array([[True, False]])

    with pytest.raises(ValueError, match="confined"):
        MacroMicroPorousImage(void=macro | micro, macro_void=macro, micropore_void=micro)

    with pytest.raises(ValueError, match=r"macro_void \| micropore_void"):
        MacroMicroPorousImage(
            void=np.array([[False, False]]),
            macro_void=macro,
            micropore_void=np.array([[False, True]]),
        )

    with pytest.raises(ValueError, match="same shape"):
        MacroMicroPorousImage(
            void=np.zeros((2, 2), dtype=bool),
            macro_void=np.zeros((2, 3), dtype=bool),
            micropore_void=np.zeros((2, 2), dtype=bool),
        )


def test_macro_micro_private_quantile_edge_branches() -> None:
    """Private helper tests keep deterministic coverage on rare quantile ties."""

    case = MacroMicroPorousImage(
        void=np.array([[True, False], [False, True]]),
        macro_void=np.array([[True, False], [False, False]]),
        micropore_void=np.array([[False, False], [False, True]]),
    )
    assert case.ndim == 2
    assert case.total_microporosity == pytest.approx(0.25)

    with pytest.raises(ValueError, match="no matrix"):
        pimg._matrix_quantile_mask(
            np.ones((2, 2)),
            np.zeros((2, 2), dtype=bool),
            fraction=0.5,
        )

    selected = pimg._matrix_quantile_mask(
        np.array([[np.nan, 0.0], [1.0, 2.0]]),
        np.ones((2, 2), dtype=bool),
        fraction=0.5,
    )
    assert np.count_nonzero(selected) == 2


def test_generate_spanning_multiscale_blobs_matrix_controls_porosity_and_connectivity() -> None:
    """Multiscale blobs should honor target porosity and spanning acceptance."""

    im, seed = generate_spanning_multiscale_blobs_matrix(
        shape=(96, 256),
        porosity=0.40,
        blobiness_primary=(0.5, 1.2),
        blobiness_secondary=(2.5, 6.0),
        primary_weight=0.75,
        axis_index=0,
        seed_start=1234,
        max_tries=40,
    )

    assert seed >= 1234
    assert im.shape == (96, 256)
    assert im.dtype == bool
    assert abs(float(im.mean()) - 0.40) <= 5.0e-4
    assert has_spanning_cluster_2d(im, axis_index=0)


def test_generate_spanning_multiscale_blobs_matrix_validation() -> None:
    """Validation should reject invalid weights, porosities, and anisotropy lengths."""

    with pytest.raises(ValueError, match="porosity must be in"):
        generate_spanning_multiscale_blobs_matrix(
            shape=(32, 32),
            porosity=1.0,
            blobiness_primary=1.0,
            blobiness_secondary=2.0,
            primary_weight=0.75,
            axis_index=0,
            seed_start=0,
            max_tries=1,
        )

    with pytest.raises(ValueError, match="max_tries must be >= 1"):
        generate_spanning_multiscale_blobs_matrix(
            shape=(32, 32),
            porosity=0.4,
            blobiness_primary=1.0,
            blobiness_secondary=2.0,
            primary_weight=0.75,
            axis_index=0,
            seed_start=0,
            max_tries=0,
        )

    with pytest.raises(ValueError, match="primary_weight must be in"):
        generate_spanning_multiscale_blobs_matrix(
            shape=(32, 32),
            porosity=0.4,
            blobiness_primary=1.0,
            blobiness_secondary=2.0,
            primary_weight=1.5,
            axis_index=0,
            seed_start=0,
            max_tries=1,
        )

    with pytest.raises(ValueError, match="blobiness_primary must have length 2"):
        generate_spanning_multiscale_blobs_matrix(
            shape=(32, 32),
            porosity=0.4,
            blobiness_primary=(1.0, 2.0, 3.0),
            blobiness_secondary=2.0,
            primary_weight=0.75,
            axis_index=0,
            seed_start=0,
            max_tries=1,
        )


def test_coerce_blobiness_accepts_scalar_and_rejects_nonpositive_values() -> None:
    """Blobiness coercion should preserve scalar inputs and reject invalid values."""

    assert pimg._coerce_blobiness(1.5, ndim=2, name="blobiness") == pytest.approx(1.5)

    with pytest.raises(ValueError, match="blobiness must be positive"):
        pimg._coerce_blobiness(0.0, ndim=2, name="blobiness")

    with pytest.raises(ValueError, match="All entries in blobiness must be positive"):
        pimg._coerce_blobiness((1.0, -2.0), ndim=2, name="blobiness")


def test_generate_spanning_multiscale_blobs_matrix_can_fail_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multiscale generator should raise cleanly when the trial image cannot span."""

    def fake_blobs(*, shape, porosity, blobiness, seed, periodic):
        del porosity, blobiness, seed, periodic
        field = np.ones(shape, dtype=float)
        field[:3, :] = 0.0
        return field

    monkeypatch.setattr(
        pimg,
        "ps",
        SimpleNamespace(generators=SimpleNamespace(blobs=fake_blobs)),
    )

    with pytest.raises(RuntimeError, match="Could not generate spanning multiscale blobs matrix"):
        generate_spanning_multiscale_blobs_matrix(
            shape=(24, 24),
            porosity=0.10,
            blobiness_primary=(0.5, 0.5),
            blobiness_secondary=(2.0, 2.0),
            primary_weight=0.75,
            axis_index=0,
            seed_start=0,
            max_tries=1,
        )
