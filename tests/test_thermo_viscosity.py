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


def test_singlephase_solver_converges_for_pressure_dependent_viscosity() -> None:
    """The Picard loop reproduces the analytic midpoint for mu(p)=p on a 3-pore chain."""

    net = make_linear_chain_network()
    net.throat.pop("hydraulic_conductance")
    net.throat["area"] = np.sqrt(8.0 * np.pi) * np.ones(net.Nt)

    model = TabulatedWaterViscosityModel(
        backend=LinearPressureViscosityBackend(),
        temperature=300.0,
        pressure_points=128,
        pressure_padding_fraction=0.0,
    )
    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity_model=model),
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
