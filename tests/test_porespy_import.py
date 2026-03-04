from __future__ import annotations

import numpy as np

from voids.io.porespy import from_porespy
from voids.core.sample import SampleGeometry


def test_from_porespy_minimal():
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
