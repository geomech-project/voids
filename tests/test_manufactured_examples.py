from __future__ import annotations

from pathlib import Path

import numpy as np

from voids.examples.manufactured import make_manufactured_void_image


def test_make_manufactured_void_image_basic_properties() -> None:
    """Test the basic structure of the manufactured binary void image."""

    im = make_manufactured_void_image((32, 32, 32))
    assert im.shape == (32, 32, 32)
    assert im.dtype == bool
    assert im.any()
    assert (~im).any()


def test_repo_manufactured_image_exists_and_loads() -> None:
    """Test that the committed manufactured image dataset exists and loads cleanly."""

    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "examples" / "data" / "manufactured_void_image.npy"
    assert p.exists()
    im = np.load(p)
    assert im.ndim == 3
    assert im.dtype == bool
