"""Internal testing utilities for voids package."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seeds for the standard library and NumPy RNGs.

    Parameters
    ----------
    seed :
        Integer seed applied to :mod:`random` and :mod:`numpy.random`.

    Notes
    -----
    The helper is intentionally narrow and does not configure any external or
    accelerator-backed random-number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
