from __future__ import annotations

import numpy as np
import pytest

from voids.core.network import Network
from voids.core.sample import SampleGeometry
from voids.core.validation import validate_network


def test_validate_network_ok(line_network):
    """Test that the canonical line network passes validation."""

    validate_network(line_network)


def test_validate_network_bad_self_loop():
    """Test rejection of a self-loop throat."""

    net = Network(
        throat_conns=np.array([[0, 0]]),
        pore_coords=np.array([[0, 0, 0.0]]),
        sample=SampleGeometry(bulk_volume=1.0),
    )
    with pytest.raises(ValueError, match="self-loop"):
        validate_network(net)


def test_validate_network_bad_label_shape(line_network):
    """Test rejection of incorrectly shaped pore-label arrays."""

    line_network.pore_labels["bad"] = np.array([True, False])
    with pytest.raises(ValueError, match="wrong shape"):
        validate_network(line_network)
