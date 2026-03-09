from __future__ import annotations

import numpy as np

from voids.core.network import Network
from voids.geom.hydraulic import (
    DEFAULT_G_REF,
    TRIANGLE_MAX_G,
    _segment_conductance_valvatne_blunt,
    available_conductance_models,
    generic_poiseuille_conductance,
    valvatne_blunt_conductance,
    valvatne_blunt_baseline_conductance,
    valvatne_blunt_throat_conductance,
)


def test_available_models_contains_expected_names() -> None:
    """Test the public list of available conductance model names."""

    models = available_conductance_models()
    assert "generic_poiseuille" in models
    assert "valvatne_blunt_throat" in models
    assert "valvatne_blunt" in models
    assert "valvatne_blunt_baseline" in models


def test_valvatne_segment_coefficients_follow_reference_shape_classes() -> None:
    """Test triangle, square, and circle coefficients against the reference values."""

    area = np.ones(3)
    shape_factor = np.array([0.5 * TRIANGLE_MAX_G, 0.06, DEFAULT_G_REF])
    length = np.ones(3)

    g = _segment_conductance_valvatne_blunt(area, shape_factor, length, viscosity=1.0)

    expected = np.array(
        [
            (3.0 / 5.0) * shape_factor[0],
            0.5623 * shape_factor[1],
            0.5 * shape_factor[2],
        ]
    )
    assert np.allclose(g, expected)


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


def test_valvatne_uses_area_and_diameter_to_recover_missing_shape_factor(
    line_network: Network,
) -> None:
    """Test pore shape factor derivation from area and inscribed diameter."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["pore1_length"] = np.array([0.25, 0.25])
    net.throat["core_length"] = np.array([0.50, 0.50])
    net.throat["pore2_length"] = np.array([0.25, 0.25])
    net.throat["shape_factor"] = np.array([DEFAULT_G_REF, DEFAULT_G_REF])
    net.throat["area"] = np.sqrt(2.0 * net.throat["core_length"] / DEFAULT_G_REF)

    pore_diameter = np.ones(net.Np)
    pore_shape_factor = 0.04
    net.pore["diameter_inscribed"] = pore_diameter
    net.pore["area"] = (pore_diameter**2) / (16.0 * pore_shape_factor)
    net.pore.pop("shape_factor", None)
    net.pore.pop("perimeter", None)

    g = valvatne_blunt_conductance(net, viscosity=1.0)

    pore_segment = (3.0 / 5.0) * pore_shape_factor * net.pore["area"][0] ** 2 / 0.25
    throat_segment = np.ones(net.Nt)
    expected = 1.0 / (2.0 / pore_segment + 1.0 / throat_segment)
    assert np.allclose(g, np.full(net.Nt, expected))


def test_valvatne_throat_model_derives_area_from_shape_factor_and_diameter(
    line_network: Network,
) -> None:
    """Test throat-only shape-aware conductance when only inscribed size and G are known."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat["diameter_inscribed"] = np.ones(net.Nt)
    net.throat["shape_factor"] = np.full(net.Nt, 0.04)
    net.throat.pop("area", None)

    g = valvatne_blunt_throat_conductance(net, viscosity=1.0)

    area = 1.0 / (16.0 * 0.04)
    expected = (3.0 / 5.0) * 0.04 * area**2
    assert np.allclose(g, np.full(net.Nt, expected))


def test_valvatne_clips_nonphysical_shape_factor_to_circle_limit(line_network: Network) -> None:
    """Test clipping of shape factors above the circular upper bound."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat["diameter_inscribed"] = np.ones(net.Nt)
    net.throat["shape_factor"] = np.full(net.Nt, 1.0)
    net.throat.pop("area", None)

    g = valvatne_blunt_throat_conductance(net, viscosity=1.0)

    expected = np.pi / 128.0
    assert np.allclose(g, np.full(net.Nt, expected))


def test_valvatne_baseline_falls_back_to_generic_when_shape_missing(line_network: Network) -> None:
    """Test fallback to generic Poiseuille conductance when shape data are missing."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["diameter_inscribed"] = np.array([1.0, 1.0])
    gg = generic_poiseuille_conductance(net, viscosity=1.0)
    gv = valvatne_blunt_baseline_conductance(net, viscosity=1.0)
    assert np.allclose(gv, gg)
