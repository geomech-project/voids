from __future__ import annotations

import numpy as np

from voids.physics.petrophysics import absolute_porosity, effective_porosity


def test_absolute_porosity(line_network):
    """Test absolute porosity on the canonical line network."""

    phi = absolute_porosity(line_network)
    # void volume = 3*1 + 2*0.5 = 4, bulk = 10
    assert np.isclose(phi, 0.4)


def test_effective_porosity_axis(branched_network):
    """Test axis-based effective porosity on a branched network."""

    # spanning pores: 0,1,2,3 (vol 4) + throats 3*0.2 = 0.6 ; bulk=20 => 0.23
    phi_eff = effective_porosity(branched_network, axis="x")
    assert np.isclose(phi_eff, 0.23)


def test_effective_porosity_boundary_mode(line_network):
    """Test boundary-connected effective porosity on the line network."""

    phi_eff = effective_porosity(line_network)
    assert np.isclose(phi_eff, 0.4)
