from __future__ import annotations

import pytest

from voids.generators import generate_spanning_multiscale_blobs_matrix
from voids.generators import porous_image as pimg
from voids.image import has_spanning_cluster_2d


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


def test_generate_spanning_multiscale_blobs_matrix_can_fail_cleanly() -> None:
    """A single low-porosity trial on a tiny 2D image may legitimately fail to span."""

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
