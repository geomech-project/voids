from __future__ import annotations

import numpy as np
import pytest

from voids.core.network import Network
from voids.geom.hydraulic import (
    DEFAULT_G_REF,
    TRIANGLE_MAX_G,
    _broadcast_finite,
    _broadcast_viscosity,
    _conductance_coefficient_from_shape_factor,
    _require,
    _resolve_pore_throat_viscosities,
    _segment_conductance_from_agl,
    _segment_conductance_valvatne_blunt,
    _shape_factor_from_area_perimeter,
    _shape_factor_from_area_inscribed_radius,
    _throat_only_shape_factor_conductance,
    _sanitize_shape_factor,
    available_conductance_models,
    generic_poiseuille_conductance,
    throat_conductance,
    throat_conductance_with_sensitivities,
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


def test_valvatne_baseline_accepts_distinct_pore_and_throat_viscosities(
    line_network: Network,
) -> None:
    """Conduit conductance can use different pore and throat viscosity fields."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["pore1_length"] = np.array([0.25, 0.25])
    net.throat["core_length"] = np.array([0.50, 0.50])
    net.throat["pore2_length"] = np.array([0.25, 0.25])
    gref = 1.0 / (4.0 * np.pi)
    net.throat["shape_factor"] = np.array([gref, gref])
    net.pore["shape_factor"] = np.array([gref, gref, gref])
    net.throat["area"] = np.sqrt(2.0 * net.throat["core_length"] / gref)
    net.pore["area"] = np.sqrt(2.0 * 0.25 / gref) * np.ones(net.Np)

    pore_viscosity = np.array([1.0, 2.0, 4.0])
    throat_viscosity = np.array([3.0, 5.0])
    g = valvatne_blunt_baseline_conductance(
        net,
        viscosity=None,
        pore_viscosity=pore_viscosity,
        throat_viscosity=throat_viscosity,
    )

    expected = np.array(
        [
            1.0 / (1.0 / 1.0 + 1.0 / (1.0 / 3.0) + 1.0 / (1.0 / 2.0)),
            1.0 / (1.0 / (1.0 / 2.0) + 1.0 / (1.0 / 5.0) + 1.0 / (1.0 / 4.0)),
        ]
    )
    assert np.allclose(g, expected)


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


def test_generic_poiseuille_sensitivity_matches_analytic_expression(line_network: Network) -> None:
    """Generic Poiseuille sensitivity follows the expected chain rule."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["diameter_inscribed"] = np.ones(net.Nt)
    net.throat["length"] = np.ones(net.Nt)

    mu_t = np.array([2.0, 4.0])
    dmu_t = np.array([6.0, 10.0])
    g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="generic_poiseuille",
        throat_viscosity=mu_t,
        throat_dviscosity_dpressure=dmu_t,
    )

    expected_g = np.pi / (128.0 * mu_t)
    expected_dg = -(expected_g / mu_t) * (0.5 * dmu_t)
    assert np.allclose(g, expected_g)
    assert np.allclose(dg_dpi, expected_dg)
    assert np.allclose(dg_dpj, expected_dg)


def test_conductance_sensitivities_accept_negative_viscosity_derivatives(
    line_network: Network,
) -> None:
    """Pressure derivatives of viscosity may be negative and should remain valid."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["diameter_inscribed"] = np.ones(net.Nt)
    net.throat["length"] = np.ones(net.Nt)

    g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="generic_poiseuille",
        throat_viscosity=np.array([2.0, 2.0]),
        throat_dviscosity_dpressure=np.array([-1.0, -2.0]),
    )

    assert np.isfinite(g).all()
    assert np.isfinite(dg_dpi).all()
    assert np.isfinite(dg_dpj).all()


@pytest.mark.parametrize(
    ("helper", "value", "message"),
    [
        (_broadcast_viscosity, np.array([1.0, np.nan]), "finite values"),
        (_broadcast_viscosity, np.array([1.0, 0.0]), "must be positive"),
        (_broadcast_finite, np.array([1.0, np.nan]), "finite values"),
    ],
)
def test_broadcast_helpers_validate_values(helper, value: np.ndarray, message: str) -> None:
    """Broadcast helpers reject nonphysical or nonfinite arrays."""

    kwargs = {"name": "trial"} if helper is _broadcast_finite else {}
    with pytest.raises(ValueError, match=message):
        helper(value, (2,), **kwargs)


@pytest.mark.parametrize("helper", [_broadcast_viscosity, _broadcast_finite])
def test_broadcast_helpers_validate_shapes(helper) -> None:
    """Broadcast helpers reject arrays that cannot match the requested shape."""

    kwargs = {"name": "trial"} if helper is _broadcast_finite else {}
    with pytest.raises(ValueError, match="broadcastable to shape"):
        helper(np.ones(3), (2,), **kwargs)


def test_resolve_pore_throat_viscosities_requires_at_least_one_source(
    line_network: Network,
) -> None:
    """Missing viscosity inputs are rejected separately for pores and throats."""

    with pytest.raises(ValueError, match="Need either viscosity or pore_viscosity"):
        _resolve_pore_throat_viscosities(
            line_network,
            viscosity=None,
            pore_viscosity=None,
            throat_viscosity=np.ones(line_network.Nt),
        )
    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        _resolve_pore_throat_viscosities(
            line_network,
            viscosity=None,
            pore_viscosity=np.ones(line_network.Np),
            throat_viscosity=None,
        )


def test_conductance_models_require_explicit_viscosity_input(line_network: Network) -> None:
    """Conductance models reject missing scalar and throat viscosity inputs."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["shape_factor"] = np.full(net.Nt, 1.0 / (4.0 * np.pi))
    net.throat["area"] = np.ones(net.Nt)

    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        generic_poiseuille_conductance(net, viscosity=None, throat_viscosity=None)
    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        _throat_only_shape_factor_conductance(net, viscosity=None, throat_viscosity=None)
    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        valvatne_blunt_throat_conductance(net, viscosity=None, throat_viscosity=None)


def test_sensitivity_models_require_explicit_viscosity_input(line_network: Network) -> None:
    """Sensitivity branches reject missing scalar and throat viscosity inputs."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["shape_factor"] = np.full(net.Nt, 1.0 / (4.0 * np.pi))
    net.throat["area"] = np.ones(net.Nt)

    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        throat_conductance_with_sensitivities(
            net,
            viscosity=None,
            model="generic_poiseuille",
            throat_viscosity=None,
        )
    with pytest.raises(ValueError, match="Need either viscosity or throat_viscosity"):
        throat_conductance_with_sensitivities(
            net,
            viscosity=None,
            model="valvatne_blunt_throat",
            throat_viscosity=None,
        )


def test_conductance_sensitivities_return_zero_for_precomputed_hydraulic_conductance(
    line_network: Network,
) -> None:
    """Precomputed throat conductance bypasses geometric sensitivities."""

    g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
        line_network,
        viscosity=1.0,
        model="valvatne_blunt",
    )
    assert np.allclose(g, line_network.throat["hydraulic_conductance"])
    assert np.allclose(dg_dpi, 0.0)
    assert np.allclose(dg_dpj, 0.0)


def test_generic_poiseuille_sensitivity_requires_geometry_when_no_precomputed_conductance(
    line_network: Network,
) -> None:
    """The sensitivity path raises when geometric inputs are absent."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat.pop("area", None)
    with pytest.raises(KeyError, match="Need throat.diameter_inscribed or throat.area"):
        throat_conductance_with_sensitivities(
            net,
            viscosity=None,
            model="generic_poiseuille",
            throat_viscosity=np.ones(net.Nt),
        )


def test_valvatne_blunt_throat_sensitivity_matches_analytic_expression(
    line_network: Network,
) -> None:
    """Throat-only Valvatne-Blunt sensitivity follows the expected local chain rule."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    gref = 1.0 / (4.0 * np.pi)
    net.throat["shape_factor"] = np.full(net.Nt, gref)
    net.throat["area"] = np.full(net.Nt, 2.0)
    net.throat["length"] = np.ones(net.Nt)

    throat_mu = np.array([2.0, 4.0])
    throat_dmu = np.array([6.0, 10.0])
    g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt_throat",
        throat_viscosity=throat_mu,
        throat_dviscosity_dpressure=throat_dmu,
    )

    expected_g = _segment_conductance_valvatne_blunt(
        net.throat["area"],
        net.throat["shape_factor"],
        net.throat["length"],
        throat_mu,
    )
    expected_dg = -(expected_g / throat_mu) * (0.5 * throat_dmu)
    assert np.allclose(g, expected_g)
    assert np.allclose(dg_dpi, expected_dg)
    assert np.allclose(dg_dpj, expected_dg)


def test_valvatne_conduit_sensitivity_matches_finite_difference(line_network: Network) -> None:
    """Conduit sensitivity agrees with a finite-difference perturbation."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat["pore1_length"] = np.array([0.25, 0.25])
    net.throat["core_length"] = np.array([0.50, 0.50])
    net.throat["pore2_length"] = np.array([0.25, 0.25])
    gref = 1.0 / (4.0 * np.pi)
    net.throat["shape_factor"] = np.array([gref, gref])
    net.pore["shape_factor"] = np.array([gref, gref, gref])
    net.throat["area"] = np.sqrt(2.0 * net.throat["core_length"] / gref)
    net.pore["area"] = np.sqrt(2.0 * 0.25 / gref) * np.ones(net.Np)

    pore_mu = np.array([2.0, 3.0, 5.0])
    throat_mu = np.array([7.0, 11.0])
    pore_dmu = np.array([13.0, 17.0, 19.0])
    throat_dmu = np.array([23.0, 29.0])
    g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt",
        pore_viscosity=pore_mu,
        throat_viscosity=throat_mu,
        pore_dviscosity_dpressure=pore_dmu,
        throat_dviscosity_dpressure=throat_dmu,
    )

    eps = 1.0e-7
    g_plus = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt",
        pore_viscosity=np.array([pore_mu[0] + pore_dmu[0] * eps, pore_mu[1], pore_mu[2]]),
        throat_viscosity=np.array([throat_mu[0] + 0.5 * throat_dmu[0] * eps, throat_mu[1]]),
    )[0]
    g_minus = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt",
        pore_viscosity=np.array([pore_mu[0] - pore_dmu[0] * eps, pore_mu[1], pore_mu[2]]),
        throat_viscosity=np.array([throat_mu[0] - 0.5 * throat_dmu[0] * eps, throat_mu[1]]),
    )[0]
    fd = (g_plus - g_minus) / (2.0 * eps)
    assert np.allclose(dg_dpi[0], fd[0], rtol=1.0e-6, atol=1.0e-9)
    assert np.isfinite(g).all()


def test_valvatne_sensitivity_falls_back_to_throat_only_when_conduit_lengths_are_missing(
    line_network: Network,
) -> None:
    """Missing conduit segmentation triggers the throat-only sensitivity path."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    gref = 1.0 / (4.0 * np.pi)
    net.throat["shape_factor"] = np.full(net.Nt, gref)
    net.throat["area"] = np.full(net.Nt, 2.0)
    net.throat["length"] = np.ones(net.Nt)
    net.pore["shape_factor"] = np.full(net.Np, gref)
    net.pore["area"] = np.full(net.Np, 2.0)
    for key in ("pore1_length", "core_length", "pore2_length"):
        net.throat.pop(key, None)

    g_direct, dg_direct_i, dg_direct_j = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt_throat",
        throat_viscosity=np.array([2.0, 4.0]),
        throat_dviscosity_dpressure=np.array([6.0, 10.0]),
    )
    g_fallback, dg_fallback_i, dg_fallback_j = throat_conductance_with_sensitivities(
        net,
        viscosity=None,
        model="valvatne_blunt",
        pore_viscosity=np.array([1.0, 1.0, 1.0]),
        throat_viscosity=np.array([2.0, 4.0]),
        pore_dviscosity_dpressure=np.zeros(net.Np),
        throat_dviscosity_dpressure=np.array([6.0, 10.0]),
    )

    assert np.allclose(g_fallback, g_direct)
    assert np.allclose(dg_fallback_i, dg_direct_i)
    assert np.allclose(dg_fallback_j, dg_direct_j)


def test_valvatne_sensitivity_warns_and_falls_back_to_generic_poiseuille(
    line_network: Network,
) -> None:
    """If shape data are unavailable, the sensitivity path warns and uses Poiseuille."""

    net = line_network.copy()
    net.throat.pop("hydraulic_conductance", None)
    net.throat.pop("shape_factor", None)
    net.throat.pop("perimeter", None)
    net.pore.pop("shape_factor", None)
    net.throat.pop("diameter_inscribed", None)
    net.throat["area"] = np.ones(net.Nt)

    with pytest.warns(RuntimeWarning, match="falling back to generic_poiseuille"):
        g, dg_dpi, dg_dpj = throat_conductance_with_sensitivities(
            net,
            viscosity=None,
            model="valvatne_blunt",
            pore_viscosity=np.ones(net.Np),
            throat_viscosity=np.ones(net.Nt),
            throat_dviscosity_dpressure=np.zeros(net.Nt),
        )

    expected = generic_poiseuille_conductance(net, viscosity=1.0)
    assert np.allclose(g, expected)
    assert np.allclose(dg_dpi, 0.0)
    assert np.allclose(dg_dpj, 0.0)
