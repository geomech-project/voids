from __future__ import annotations

import numpy as np
import pytest

from voids.core.network import Network
from voids.geom.hydraulic import (
    DEFAULT_G_REF,
    TRIANGLE_MAX_G,
    _conductance_coefficient_from_shape_factor,
    _require,
    _segment_conductance_from_agl,
    _segment_conductance_valvatne_blunt,
    _shape_factor_from_area_perimeter,
    _shape_factor_from_area_inscribed_radius,
    _sanitize_shape_factor,
    available_conductance_models,
    generic_poiseuille_conductance,
    throat_conductance,
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


def test_radius_based_geometry_paths_are_supported(line_network: Network) -> None:
    """Area and shape factor can be inferred from radius-inscribed data."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat.pop("area", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat["radius_inscribed"] = np.full(net.Nt, 0.5)
    net.throat["shape_factor"] = np.full(net.Nt, 0.04)

    g = valvatne_blunt_throat_conductance(net, viscosity=1.0)

    area = (0.5**2) / (4.0 * 0.04)
    expected = (3.0 / 5.0) * 0.04 * area**2
    assert np.allclose(g, np.full(net.Nt, expected))


def test_shape_factor_from_area_radius_formula() -> None:
    """The radius-based shape-factor helper follows the analytical relation."""

    area = np.array([2.0])
    radius = np.array([1.0])
    assert np.allclose(_shape_factor_from_area_inscribed_radius(area, radius), np.array([0.125]))


def test_shape_factor_from_area_perimeter_formula() -> None:
    """Perimeter-based shape-factor helper follows G=A/P^2."""

    area = np.array([2.0])
    perimeter = np.array([4.0])
    assert np.allclose(_shape_factor_from_area_perimeter(area, perimeter), np.array([0.125]))


def test_shape_factor_sanitization_rejects_negative_values() -> None:
    """Negative geometric shape factors are rejected."""

    with pytest.raises(ValueError, match="shape_factor contains negative values"):
        _sanitize_shape_factor(np.array([-1e-6]))


def test_require_raises_for_missing_fields(line_network: Network) -> None:
    """Required-field helper reports missing items."""

    with pytest.raises(KeyError, match="Missing required throat fields"):
        _require(line_network, "throat", ("missing",))


def test_get_entity_shape_factor_raises_when_only_area_exists(line_network: Network) -> None:
    """Throat-only model raises when no shape-factor surrogate is present."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat["area"] = np.ones(net.Nt)
    net.throat.pop("shape_factor", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat.pop("radius_inscribed", None)
    net.throat.pop("perimeter", None)

    with pytest.raises(KeyError, match="Need throat.shape_factor"):
        valvatne_blunt_throat_conductance(net, viscosity=1.0)


def test_get_entity_area_uses_radius_when_shape_missing(line_network: Network) -> None:
    """Radius-inscribed fallback computes area as pi*r^2."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat.pop("area", None)
    net.throat.pop("shape_factor", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat["radius_inscribed"] = np.full(net.Nt, 0.5)
    g = valvatne_blunt_throat_conductance(net, viscosity=1.0)
    area = np.pi * 0.5**2
    shape_factor = (0.5**2) / (4.0 * area)
    expected = 0.5 * shape_factor * area**2
    assert np.allclose(g, np.full(net.Nt, expected))


def test_get_entity_area_raises_when_no_geometric_surrogate(line_network: Network) -> None:
    """Area derivation raises when all geometric surrogates are removed."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat.pop("area", None)
    net.throat.pop("shape_factor", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat.pop("radius_inscribed", None)
    with pytest.raises(KeyError, match="Need throat.area or throat.diameter_inscribed"):
        valvatne_blunt_throat_conductance(net, viscosity=1.0)


def test_get_entity_shape_factor_from_perimeter(line_network: Network) -> None:
    """Perimeter is used as a shape-factor surrogate when explicit G is absent."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat["area"] = np.ones(net.Nt)
    net.throat["perimeter"] = np.full(net.Nt, 4.0)
    net.throat.pop("shape_factor", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat.pop("radius_inscribed", None)
    g = valvatne_blunt_throat_conductance(net, viscosity=1.0)
    expected = 0.5623 * (1.0 / 16.0)
    assert np.allclose(g, np.full(net.Nt, expected))


def test_segment_conductance_from_agl_clips_shape_and_handles_zero_length() -> None:
    """Low shape factors are clipped and zero-length segments map to +inf conductance."""

    g = _segment_conductance_from_agl(
        area=np.array([2.0, 2.0]),
        shape_factor=np.array([1e-30, 0.5]),
        length=np.array([0.0, 2.0]),
        viscosity=1.0,
    )
    assert np.isinf(g[0])
    assert g[1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("area", "shape_factor", "length", "viscosity", "message"),
    [
        (np.array([1.0]), np.array([0.04]), np.array([1.0]), 0.0, "viscosity must be positive"),
        (np.array([-1.0]), np.array([0.04]), np.array([1.0]), 1.0, "area contains negative values"),
        (
            np.array([1.0]),
            np.array([0.04]),
            np.array([-1.0]),
            1.0,
            "length contains negative values",
        ),
        (
            np.array([1.0]),
            np.array([-0.04]),
            np.array([1.0]),
            1.0,
            "shape_factor contains negative values",
        ),
    ],
)
def test_segment_conductance_from_agl_input_validation(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float,
    message: str,
) -> None:
    """Input checks for A-G-L segment conductance are enforced."""

    with pytest.raises(ValueError, match=message):
        _segment_conductance_from_agl(area, shape_factor, length, viscosity)


def test_conductance_coefficient_rejects_negative_shape_factor() -> None:
    """Coefficient classifier validates shape-factor sign."""

    with pytest.raises(ValueError, match="shape_factor contains negative values"):
        _conductance_coefficient_from_shape_factor(np.array([-0.1]))


@pytest.mark.parametrize(
    ("area", "shape_factor", "length", "viscosity", "message"),
    [
        (np.array([1.0]), np.array([0.04]), np.array([1.0]), 0.0, "viscosity must be positive"),
        (np.array([-1.0]), np.array([0.04]), np.array([1.0]), 1.0, "area contains negative values"),
        (
            np.array([1.0]),
            np.array([0.04]),
            np.array([-1.0]),
            1.0,
            "length contains negative values",
        ),
        (
            np.array([1.0]),
            np.array([-0.04]),
            np.array([1.0]),
            1.0,
            "shape_factor contains negative values",
        ),
    ],
)
def test_segment_conductance_valvatne_input_validation(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float,
    message: str,
) -> None:
    """Input validation errors are raised with explicit messages."""

    with pytest.raises(ValueError, match=message):
        _segment_conductance_valvatne_blunt(area, shape_factor, length, viscosity)


def test_valvatne_throat_model_validates_viscosity_and_uses_precomputed(
    line_network: Network,
) -> None:
    """Throat-only wrapper validates viscosity and returns precomputed values when present."""

    with pytest.raises(ValueError, match="viscosity must be positive"):
        valvatne_blunt_throat_conductance(line_network, viscosity=0.0)

    precomputed = valvatne_blunt_throat_conductance(line_network, viscosity=1.0)
    assert np.allclose(precomputed, line_network.throat["hydraulic_conductance"])


def test_valvatne_baseline_warns_and_falls_back_when_shape_geometry_missing(
    line_network: Network,
) -> None:
    """Baseline model warns and falls back to Poiseuille when shape surrogates are absent."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["length"] = np.ones(net.Nt)
    net.throat["area"] = np.full(net.Nt, np.pi)
    net.throat.pop("diameter_inscribed", None)
    net.throat.pop("radius_inscribed", None)
    net.throat.pop("shape_factor", None)
    net.throat.pop("perimeter", None)

    with pytest.warns(RuntimeWarning, match="Insufficient geometry for shape-factor model"):
        g = valvatne_blunt_conductance(net, viscosity=1.0)
    assert np.allclose(g, generic_poiseuille_conductance(net, viscosity=1.0))


def test_generic_poiseuille_validation_and_missing_geometry(line_network: Network) -> None:
    """Generic model validates viscosity, precomputed sign, and geometry availability."""

    with pytest.raises(ValueError, match="viscosity must be positive"):
        generic_poiseuille_conductance(line_network, viscosity=0.0)

    negative = line_network.copy()
    negative.throat["hydraulic_conductance"] = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="contains negative values"):
        generic_poiseuille_conductance(negative, viscosity=1.0)

    missing = line_network.copy()
    missing.throat.pop("hydraulic_conductance", None)
    missing.throat.pop("diameter_inscribed", None)
    missing.throat.pop("area", None)
    with pytest.raises(KeyError, match="Need throat.diameter_inscribed or throat.area"):
        generic_poiseuille_conductance(missing, viscosity=1.0)


def test_valvatne_baseline_validation_and_precomputed_path(line_network: Network) -> None:
    """Baseline wrapper validates viscosity and honors precomputed throat conductance."""

    with pytest.raises(ValueError, match="viscosity must be positive"):
        valvatne_blunt_conductance(line_network, viscosity=0.0)

    precomputed = valvatne_blunt_conductance(line_network, viscosity=1.0)
    assert np.allclose(precomputed, line_network.throat["hydraulic_conductance"])


def test_throat_conductance_default_model_dispatch(line_network: Network) -> None:
    """Default dispatcher path maps to generic Poiseuille model."""

    g = throat_conductance(line_network, viscosity=1.0)
    assert np.allclose(g, generic_poiseuille_conductance(line_network, viscosity=1.0))


def test_throat_conductance_dispatches_all_models_and_validates_name(
    line_network: Network,
) -> None:
    """Dispatcher routes each named model and rejects unknown names."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["diameter_inscribed"] = np.ones(net.Nt)
    net.throat["length"] = np.ones(net.Nt)

    assert np.allclose(
        throat_conductance(net, viscosity=1.0, model="generic_poiseuille"),
        generic_poiseuille_conductance(net, viscosity=1.0),
    )
    assert np.allclose(
        throat_conductance(net, viscosity=1.0, model="valvatne_blunt_throat"),
        valvatne_blunt_throat_conductance(net, viscosity=1.0),
    )
    assert np.allclose(
        throat_conductance(net, viscosity=1.0, model="valvatne_blunt"),
        valvatne_blunt_conductance(net, viscosity=1.0),
    )
    assert np.allclose(
        throat_conductance(net, viscosity=1.0, model="valvatne_blunt_baseline"),
        valvatne_blunt_baseline_conductance(net, viscosity=1.0),
    )
    with pytest.raises(ValueError, match="Unknown conductance model"):
        throat_conductance(net, viscosity=1.0, model="unknown")
