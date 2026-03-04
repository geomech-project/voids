from __future__ import annotations

import numpy as np

from voids.core.network import Network
from voids.geom.hydraulic import (
    available_conductance_models,
    generic_poiseuille_conductance,
    valvatne_blunt_baseline_conductance,
)


def test_available_models_contains_expected_names() -> None:
    """Test the public list of available conductance model names."""

    models = available_conductance_models()
    assert "generic_poiseuille" in models
    assert "valvatne_blunt_baseline" in models


def test_valvatne_baseline_uses_conduit_lengths_and_pore_geometry(line_network: Network) -> None:
    """Test conduit-based Valvatne-style conductance assembly on circular-like geometry."""

    net = line_network.copy()
    # Remove precomputed conductance to force geometric path.
    net.throat.pop("hydraulic_conductance", None)
    # Conduit lengths per throat (sum = total length = 1.0)
    net.throat["pore1_length"] = np.array([0.25, 0.25])
    net.throat["core_length"] = np.array([0.50, 0.50])
    net.throat["pore2_length"] = np.array([0.25, 0.25])
    # Circular-like geometry: choose shape factor of a circle and compatible areas.
    gref = 1.0 / (4.0 * np.pi)
    net.throat["shape_factor"] = np.array([gref, gref])
    net.pore["shape_factor"] = np.array([gref, gref, gref])
    # Pick areas so each segment gives unit conductance when mu=1.
    # g = 0.5 * G * A^2 / (mu * L) = 1  => A = sqrt(2L/G)
    net.throat["area"] = np.sqrt(2.0 * net.throat["core_length"] / gref)
    net.pore["area"] = np.sqrt(2.0 * 0.25 / gref) * np.ones(net.Np)

    gv = valvatne_blunt_baseline_conductance(net, viscosity=1.0)
    # Harmonic(1,1,1) = 1/3 for each throat.
    assert np.allclose(gv, [1 / 3, 1 / 3])


def test_valvatne_baseline_falls_back_to_generic_when_shape_missing(line_network: Network) -> None:
    """Test fallback to generic Poiseuille conductance when shape data are missing."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["diameter_inscribed"] = np.array([1.0, 1.0])
    gg = generic_poiseuille_conductance(net, viscosity=1.0)
    gv = valvatne_blunt_baseline_conductance(net, viscosity=1.0)
    assert np.allclose(gv, gg)
