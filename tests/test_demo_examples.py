from __future__ import annotations

import numpy as np

from voids.examples import make_linear_chain_network


def test_make_linear_chain_network_default_values() -> None:
    """Test default geometry, labels, and conductance values of the demo chain network."""

    net = make_linear_chain_network()

    assert net.Np == 3
    assert net.Nt == 2
    assert np.allclose(net.pore_coords[:, 0], [0.0, 0.5, 1.0])
    assert net.pore_labels["inlet_xmin"].tolist() == [True, False, False]
    assert net.pore_labels["outlet_xmax"].tolist() == [False, False, True]
    assert net.pore_labels["boundary"].tolist() == [True, False, True]
    assert np.allclose(net.throat["hydraulic_conductance"], [1.0, 1.0])


def test_make_linear_chain_network_axis_support() -> None:
    """Test coordinate placement and sample metadata for non-x demo chains."""

    net = make_linear_chain_network(num_pores=4, axis="z", length=3.0)

    assert np.allclose(net.pore_coords[:, 2], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(net.pore_coords[:, :2], 0.0)
    assert np.isclose(net.sample.length_for_axis("z"), 3.0)
