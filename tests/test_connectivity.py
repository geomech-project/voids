from __future__ import annotations

from voids.graph.connectivity import connected_components, spanning_component_mask
from voids.graph.metrics import connectivity_metrics


def test_connected_components(branched_network):
    """Test connected-component counting on a branched toy network."""

    n, labels = connected_components(branched_network)
    assert n == 2
    assert labels.shape == (branched_network.Np,)


def test_spanning_component_mask(branched_network):
    """Test spanning-component masking along the x axis."""

    mask = spanning_component_mask(branched_network, axis="x")
    assert mask.tolist() == [True, True, True, True, False]


def test_connectivity_metrics(branched_network):
    """Test graph connectivity summary statistics."""

    m = connectivity_metrics(branched_network)
    assert m.n_components == 2
    assert m.spans["x"] is True
    assert 0 in m.coordination_histogram
