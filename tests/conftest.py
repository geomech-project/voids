from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voids.examples import make_linear_chain_network
from voids.core.network import Network
from voids.core.sample import SampleGeometry


@pytest.fixture()
def line_network() -> Network:
    """Return the canonical three-pore line network used across tests."""

    return make_linear_chain_network()


@pytest.fixture()
def branched_network() -> Network:
    """Return a small branched network with one disconnected pore."""

    pore_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [5, 5, 0]], dtype=float)
    throat_conns = np.array([[0, 1], [1, 2], [1, 3]], dtype=int)
    pore = {"volume": np.ones(5)}
    throat = {"volume": np.ones(3) * 0.2, "length": np.ones(3), "hydraulic_conductance": np.ones(3)}
    labels = {
        "inlet_xmin": np.array([True, False, False, False, False]),
        "outlet_xmax": np.array([False, False, True, False, False]),
        "boundary": np.array([True, False, True, False, False]),
    }
    sample = SampleGeometry(bulk_volume=20.0, lengths={"x": 2.0}, cross_sections={"x": 1.0})
    return Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample,
        pore=pore,
        throat=throat,
        pore_labels=labels,
    )
