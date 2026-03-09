from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voids.examples import make_linear_chain_network
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve
import voids.physics.thermo as thermo_module
from voids.physics.thermo import (
    CoolPropWaterViscosityBackend,
    PressureViscosityTable,
    TabulatedWaterViscosityModel,
    ThermoWaterViscosityBackend,
)


class LinearPressureViscosityBackend:
    """Analytic backend used to regression-test pressure-coupled viscosity."""

    name = "linear-test"

    def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
        del temperature
        return np.asarray(pressure, dtype=float)


class InvalidThermoViscosityModel:
    """Stub thermo object that returns a nonphysical viscosity value."""

    def __init__(self, value: float | None) -> None:
        self.value = value

    def TP_dependent_property(self, temperature: float, pressure: float) -> float | None:
        del temperature, pressure
        return self.value


@pytest.fixture
def linear_model() -> TabulatedWaterViscosityModel:
    """Compact analytic model used across variable-viscosity tests."""

    return TabulatedWaterViscosityModel(
        backend=LinearPressureViscosityBackend(),
        temperature=300.0,
        pressure_points=128,
        pressure_padding_fraction=0.0,
    )


@pytest.mark.parametrize(
    ("pressure", "viscosity", "message"),
    [
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), "one-dimensional"),
        (np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), "one-dimensional"),
        (np.array([1.0, 2.0]), np.array([1.0]), "same length"),
        (np.array([]), np.array([]), "at least one point"),
        (np.array([1.0, 1.0]), np.array([1.0, 2.0]), "strictly increasing"),
        (np.array([1.0, np.inf]), np.array([1.0, 2.0]), "finite values"),
        (np.array([1.0, 2.0]), np.array([1.0, 0.0]), "positive values"),
    ],
)
def test_pressure_viscosity_table_validates_inputs(
    pressure: np.ndarray, viscosity: np.ndarray, message: str
) -> None:
    """Pressure tables reject malformed grids and invalid state values."""

    with pytest.raises(ValueError, match=message):
        PressureViscosityTable(pressure=pressure, viscosity=viscosity)


def test_pressure_viscosity_table_single_point_returns_constant_value_and_zero_slope() -> None:
    """A one-point table behaves as a constant constitutive law."""

    table = PressureViscosityTable(pressure=np.array([2.0]), viscosity=np.array([5.0]))
    pressure = np.array([1.0, 2.0, 3.0])
    assert np.allclose(table.evaluate(pressure), np.array([5.0, 5.0, 5.0]))
    assert np.allclose(table.derivative(pressure), np.zeros_like(pressure))


def test_pressure_viscosity_table_clips_queries_to_bounds() -> None:
    """Pressure-table lookups are clipped at the tabulated bounds."""

    table = PressureViscosityTable(
        pressure=np.array([1.0, 2.0, 3.0]),
        viscosity=np.array([10.0, 20.0, 30.0]),
    )
    values = table.evaluate(np.array([0.5, 1.5, 4.0]))
    assert values[0] == pytest.approx(10.0)
    assert values[1] == pytest.approx(15.0)
    assert values[2] == pytest.approx(30.0)


def test_pressure_viscosity_table_returns_differentiable_slope() -> None:
    """The pressure table exposes a usable derivative for Newton linearization."""

    table = PressureViscosityTable(
        pressure=np.array([1.0, 2.0, 3.0]),
        viscosity=np.array([10.0, 20.0, 30.0]),
    )
    dmu = table.derivative(np.array([0.5, 1.5, 2.5, 4.0]))
    assert dmu[0] == pytest.approx(0.0)
    assert dmu[1] == pytest.approx(10.0)
    assert dmu[2] == pytest.approx(10.0)
    assert dmu[3] == pytest.approx(0.0)


def test_thermo_backend_changes_viscosity_with_pressure_for_liquid_water() -> None:
    """The thermo backend responds to pressure, even if the sign depends on the backend model."""

    backend = ThermoWaterViscosityBackend()
    mu = backend.evaluate(np.array([1.0e5, 5.0e6]), temperature=298.15)
    assert np.all(mu > 0.0)
    assert not np.isclose(mu[1], mu[0], rtol=1.0e-6, atol=0.0)


def test_tabulated_thermo_model_matches_direct_backend_queries() -> None:
    """The cached interpolator stays close to direct thermo evaluations."""

    backend = ThermoWaterViscosityBackend()
    model = TabulatedWaterViscosityModel(
        backend=backend,
        temperature=298.15,
        pressure_points=64,
        pressure_padding_fraction=0.0,
    )
    query_pressure = np.array([1.5e5, 3.0e5, 8.0e5])
    table_values = model.evaluate(query_pressure, pin=1.0e5, pout=1.0e6)
    direct_values = backend.evaluate(query_pressure, temperature=298.15)
    assert np.allclose(table_values, direct_values, rtol=2.0e-4, atol=0.0)


def test_thermo_backend_validates_temperature_and_backend_response() -> None:
    """The thermo backend rejects invalid temperatures and nonphysical responses."""

    backend = ThermoWaterViscosityBackend()
    with pytest.raises(ValueError, match="temperature must be a positive finite value"):
        backend.evaluate(np.array([1.0e5]), temperature=0.0)

    backend._viscosity_liquid = InvalidThermoViscosityModel(None)
    with pytest.raises(ValueError, match="could not evaluate a positive finite water viscosity"):
        backend.evaluate(np.array([1.0e5]), temperature=298.15)


def test_coolprop_backend_raises_if_dependency_is_missing(monkeypatch) -> None:
    """A clear import error is raised when CoolProp is unavailable."""

    original_import_module = thermo_module.importlib.import_module

    def _fake_import(name: str):
        if name == "CoolProp.CoolProp":
            raise ImportError("simulated missing CoolProp")
        return original_import_module(name)

    monkeypatch.setattr(thermo_module.importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="CoolProp"):
        CoolPropWaterViscosityBackend()


def test_coolprop_backend_uses_propssi_when_dependency_is_available(monkeypatch) -> None:
    """The CoolProp backend delegates viscosity evaluation to PropsSI."""

    package = types.ModuleType("CoolProp")
    package.__path__ = []
    submodule = types.ModuleType("CoolProp.CoolProp")

    def _props_si(
        output: str, key1: str, t_value: float, key2: str, p_value: float, fluid: str
    ) -> float:
        assert output == "VISCOSITY"
        assert key1 == "T"
        assert key2 == "P"
        assert fluid == "Water"
        return float(t_value) * 1.0e-6 + float(p_value) * 1.0e-12

    submodule.PropsSI = _props_si
    monkeypatch.setitem(sys.modules, "CoolProp", package)
    monkeypatch.setitem(sys.modules, "CoolProp.CoolProp", submodule)

    backend = CoolPropWaterViscosityBackend()
    values = backend.evaluate(np.array([1.0e5, 2.0e5]), temperature=300.0)
    assert np.allclose(values, [3.001e-4, 3.002e-4])


def test_coolprop_backend_validates_temperature_and_backend_response(monkeypatch) -> None:
    """The CoolProp backend rejects invalid temperatures and nonphysical outputs."""

    package = types.ModuleType("CoolProp")
    package.__path__ = []
    submodule = types.ModuleType("CoolProp.CoolProp")

    def _bad_props_si(
        output: str, key1: str, t_value: float, key2: str, p_value: float, fluid: str
    ) -> float:
        del output, key1, t_value, key2, p_value, fluid
        return 0.0

    submodule.PropsSI = _bad_props_si
    monkeypatch.setitem(sys.modules, "CoolProp", package)
    monkeypatch.setitem(sys.modules, "CoolProp.CoolProp", submodule)

    backend = CoolPropWaterViscosityBackend()
    with pytest.raises(ValueError, match="temperature must be a positive finite value"):
        backend.evaluate(np.array([1.0e5]), temperature=-1.0)
    with pytest.raises(ValueError, match="could not evaluate a positive finite water viscosity"):
        backend.evaluate(np.array([1.0e5]), temperature=300.0)


def test_singlephase_solver_converges_for_pressure_dependent_viscosity(
    linear_model: TabulatedWaterViscosityModel,
) -> None:
    """The Picard loop reproduces the analytic midpoint for mu(p)=p on a 3-pore chain."""

    net = make_linear_chain_network()
    net.throat.pop("hydraulic_conductance")
    net.throat["area"] = np.sqrt(8.0 * np.pi) * np.ones(net.Nt)

    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity_model=linear_model),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=2.0, pout=1.0),
        axis="x",
        options=SinglePhaseOptions(
            conductance_model="generic_poiseuille",
            nonlinear_max_iterations=50,
            nonlinear_pressure_tolerance=1.0e-12,
        ),
    )

    expected_midpoint = np.sqrt(2.0)
    assert result.pore_pressure[0] == pytest.approx(2.0)
    assert result.pore_pressure[2] == pytest.approx(1.0)
    assert result.pore_pressure[1] == pytest.approx(expected_midpoint, rel=1.0e-6)
    assert result.throat_viscosity is not None
    assert np.allclose(
        result.throat_viscosity[:2],
        np.array([(2.0 + expected_midpoint) / 2.0, (1.0 + expected_midpoint) / 2.0]),
        rtol=1.0e-6,
    )
    assert result.reference_viscosity == pytest.approx(1.5)
    assert int(result.solver_info["nonlinear_iterations"]) >= 1


def test_tabulated_model_exposes_value_and_derivative_consistently() -> None:
    """Tabulated viscosity model returns value and derivative from the same table."""

    model = TabulatedWaterViscosityModel(
        backend=LinearPressureViscosityBackend(),
        temperature=300.0,
        pressure_points=16,
        pressure_padding_fraction=0.0,
    )
    pressure = np.array([1.25, 1.75])
    mu, dmu = model.evaluate_with_derivative(pressure, pin=1.0, pout=2.0)
    assert np.allclose(mu, pressure)
    assert np.allclose(dmu, np.ones_like(pressure))


@pytest.mark.parametrize(
    ("temperature", "pressure_points", "pressure_padding_fraction", "message"),
    [
        (0.0, 16, 0.0, "temperature must be a positive finite value"),
        (300.0, 1, 0.0, "pressure_points must be at least 2"),
        (300.0, 16, -1.0, "pressure_padding_fraction must be nonnegative"),
    ],
)
def test_tabulated_model_validates_configuration(
    temperature: float, pressure_points: int, pressure_padding_fraction: float, message: str
) -> None:
    """Tabulated model guards reject invalid thermodynamic tabulation controls."""

    with pytest.raises(ValueError, match=message):
        TabulatedWaterViscosityModel(
            backend=LinearPressureViscosityBackend(),
            temperature=temperature,
            pressure_points=pressure_points,
            pressure_padding_fraction=pressure_padding_fraction,
        )


def test_tabulated_model_validates_backend_name() -> None:
    """Unknown thermodynamic backends are rejected explicitly."""

    with pytest.raises(ValueError, match="Unknown viscosity backend 'unknown'"):
        TabulatedWaterViscosityModel.from_backend("unknown", temperature=300.0)


def test_tabulated_model_from_backend_dispatches_known_backend_names(monkeypatch) -> None:
    """Backend-name dispatch builds thermo and coolprop models through the expected classes."""

    class DummyThermoBackend:
        name = "thermo"

        def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
            del temperature
            return np.ones_like(pressure, dtype=float)

    class DummyCoolPropBackend:
        name = "coolprop"

        def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
            del temperature
            return 2.0 * np.ones_like(pressure, dtype=float)

    monkeypatch.setattr(thermo_module, "ThermoWaterViscosityBackend", DummyThermoBackend)
    monkeypatch.setattr(thermo_module, "CoolPropWaterViscosityBackend", DummyCoolPropBackend)

    thermo_model = TabulatedWaterViscosityModel.from_backend("thermo", temperature=300.0)
    coolprop_model = TabulatedWaterViscosityModel.from_backend("coolprop", temperature=300.0)

    assert thermo_model.backend_name == "thermo"
    assert coolprop_model.backend_name == "coolprop"


def test_tabulated_model_derivative_uses_cached_table(
    linear_model: TabulatedWaterViscosityModel,
) -> None:
    """The direct derivative accessor delegates to the cached pressure table."""

    dmu = linear_model.derivative(np.array([1.25, 1.75]), pin=1.0, pout=2.0)
    assert np.allclose(dmu, np.ones(2))


def test_tabulated_model_requires_positive_absolute_boundary_pressures(
    linear_model: TabulatedWaterViscosityModel,
) -> None:
    """Pressure-dependent tabulation is defined only for positive absolute pressures."""

    with pytest.raises(ValueError, match="require absolute positive pressures"):
        linear_model.table_for_bounds(pin=0.0, pout=1.0)


def test_singlephase_solver_newton_converges_for_variable_viscosity(
    linear_model: TabulatedWaterViscosityModel,
) -> None:
    """Variable-viscosity Newton solves converge and report the Newton branch."""

    net = make_linear_chain_network()
    net.throat.pop("hydraulic_conductance")
    net.throat["area"] = np.sqrt(8.0 * np.pi) * np.ones(net.Nt)
    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity_model=linear_model),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=2.0, pout=1.0),
        axis="x",
        options=SinglePhaseOptions(
            conductance_model="generic_poiseuille",
            nonlinear_solver="newton",
            nonlinear_max_iterations=20,
            nonlinear_pressure_tolerance=1.0e-12,
        ),
    )

    expected_midpoint = np.sqrt(2.0)
    assert result.solver_info["nonlinear_solver"] == "newton"
    assert result.pore_pressure[1] == pytest.approx(expected_midpoint, rel=1.0e-8)
