from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Protocol, cast

import numpy as np
from scipy.interpolate import PchipInterpolator


class ViscosityBackend(Protocol):
    """Protocol implemented by pressure-temperature viscosity backends."""

    name: str

    def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
        """Return dynamic viscosity for the requested absolute pressures."""


class _ThermoViscosityModel(Protocol):
    """Minimal typed surface needed from ``thermo`` liquid-viscosity objects."""

    def TP_dependent_property(self, temperature: float, pressure: float) -> float | None:
        """Return viscosity at the requested state."""


class _ChemicalFactory(Protocol):
    """Typed constructor interface for ``thermo.Chemical``."""

    ViscosityLiquid: _ThermoViscosityModel


class _CoolPropPropsSI(Protocol):
    """Typed callable surface used from ``CoolProp.CoolProp.PropsSI``."""

    def __call__(
        self,
        output: str,
        key1: str,
        t_value: float,
        key2: str,
        p_value: float,
        fluid: str,
    ) -> float:
        """Return the requested CoolProp property."""


def _as_float_array(values: float | np.ndarray) -> np.ndarray:
    """Return the input as a NumPy float array."""

    return np.asarray(values, dtype=float)


def _require_positive_finite(values: np.ndarray, *, name: str) -> None:
    """Validate that an array contains only finite positive values."""

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must contain only positive values")


@dataclass(slots=True)
class PressureViscosityTable:
    """Tabulated and interpolated viscosity values on a 1D pressure grid.

    Notes
    -----
    Pressure is assumed to be absolute and expressed in Pa. Queries outside the
    tabulated interval are clipped to the interval bounds rather than
    extrapolated.
    """

    pressure: np.ndarray
    viscosity: np.ndarray
    _interpolator: PchipInterpolator | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.pressure = _as_float_array(self.pressure)
        self.viscosity = _as_float_array(self.viscosity)
        if self.pressure.ndim != 1 or self.viscosity.ndim != 1:
            raise ValueError("pressure and viscosity must be one-dimensional arrays")
        if self.pressure.size != self.viscosity.size:
            raise ValueError("pressure and viscosity must have the same length")
        if self.pressure.size == 0:
            raise ValueError("pressure table must contain at least one point")
        _require_positive_finite(self.pressure, name="pressure")
        _require_positive_finite(self.viscosity, name="viscosity")
        if self.pressure.size > 1 and np.any(np.diff(self.pressure) <= 0.0):
            raise ValueError("pressure grid must be strictly increasing")
        self._interpolator = (
            PchipInterpolator(self.pressure, self.viscosity, extrapolate=True)
            if self.pressure.size > 1
            else None
        )

    def evaluate(self, pressure: float | np.ndarray) -> np.ndarray:
        """Evaluate viscosity by clipped interpolation on the pressure grid."""

        p = _as_float_array(pressure)
        _require_positive_finite(p, name="pressure")
        p_clipped = np.clip(p, self.pressure[0], self.pressure[-1])
        if self._interpolator is None:
            return np.full_like(p_clipped, self.viscosity[0], dtype=float)
        return np.asarray(self._interpolator(p_clipped), dtype=float)

    def derivative(self, pressure: float | np.ndarray) -> np.ndarray:
        """Evaluate ``dmu/dp`` from the clipped pressure interpolant.

        Notes
        -----
        Because out-of-range queries are clipped to the tabulated interval, the
        effective derivative is zero outside the interval.
        """

        p = _as_float_array(pressure)
        _require_positive_finite(p, name="pressure")
        if self._interpolator is None:
            return np.zeros_like(p, dtype=float)
        out = np.zeros_like(p, dtype=float)
        inside = (p >= self.pressure[0]) & (p <= self.pressure[-1])
        if np.any(inside):
            out[inside] = np.asarray(self._interpolator.derivative()(p[inside]), dtype=float)
        return out

    def evaluate_with_derivative(
        self, pressure: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return viscosity and pressure derivative from the clipped interpolant."""

        return self.evaluate(pressure), self.derivative(pressure)

    __call__ = evaluate


@dataclass(slots=True)
class ThermoWaterViscosityBackend:
    """Water viscosity backend implemented with the ``thermo`` package."""

    chemical_name: str = "water"
    name: str = field(init=False, default="thermo")
    _viscosity_liquid: _ThermoViscosityModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from thermo import Chemical  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "The 'thermo' package is required for the thermo viscosity backend."
            ) from exc

        chemical = cast(_ChemicalFactory, Chemical(self.chemical_name))
        self._viscosity_liquid = chemical.ViscosityLiquid

    def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
        """Evaluate water viscosity at the requested absolute pressures."""

        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature must be a positive finite value in K")
        p = _as_float_array(pressure)
        _require_positive_finite(p, name="pressure")
        out = np.empty_like(p, dtype=float)
        for idx, value in np.ndenumerate(p):
            mu = self._viscosity_liquid.TP_dependent_property(float(temperature), float(value))
            if mu is None or not np.isfinite(mu) or mu <= 0.0:
                raise ValueError(
                    "The thermo backend could not evaluate a positive finite "
                    f"water viscosity at T={temperature} K and P={float(value)} Pa"
                )
            out[idx] = float(mu)
        return out


@dataclass(slots=True)
class CoolPropWaterViscosityBackend:
    """Water viscosity backend implemented with the ``CoolProp`` package."""

    fluid_name: str = "Water"
    name: str = field(init=False, default="coolprop")
    _props_si: _CoolPropPropsSI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            module = importlib.import_module("CoolProp.CoolProp")
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "The 'CoolProp' package is required for the coolprop viscosity backend."
            ) from exc
        self._props_si = cast(_CoolPropPropsSI, getattr(module, "PropsSI"))

    def evaluate(self, pressure: np.ndarray, *, temperature: float) -> np.ndarray:
        """Evaluate water viscosity at the requested absolute pressures."""

        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature must be a positive finite value in K")
        p = _as_float_array(pressure)
        _require_positive_finite(p, name="pressure")
        out = np.empty_like(p, dtype=float)
        for idx, value in np.ndenumerate(p):
            mu = self._props_si(
                "VISCOSITY",
                "T",
                float(temperature),
                "P",
                float(value),
                self.fluid_name,
            )
            if not np.isfinite(mu) or mu <= 0.0:
                raise ValueError(
                    "The CoolProp backend could not evaluate a positive finite "
                    f"water viscosity at T={temperature} K and P={float(value)} Pa"
                )
            out[idx] = float(mu)
        return out


def _expanded_pressure_bounds(
    pin: float,
    pout: float,
    *,
    padding_fraction: float,
) -> tuple[float, float]:
    """Return a padded pressure interval enclosing the Dirichlet bounds."""

    pmin = min(float(pin), float(pout))
    pmax = max(float(pin), float(pout))
    if pmin <= 0.0:
        raise ValueError(
            "Thermodynamic viscosity backends require absolute positive pressures in Pa. "
            "Use absolute boundary pressures when coupling viscosity to pressure."
        )
    span = pmax - pmin
    reference = max(abs(pmax), 1.0)
    pad = max(span * float(padding_fraction), reference * 1.0e-12)
    return pmin - pad, pmax + pad


@dataclass(slots=True)
class TabulatedWaterViscosityModel:
    """Pressure-dependent water viscosity with cached tabulation and interpolation.

    Parameters
    ----------
    backend :
        Backend used to generate the pressure-viscosity table.
    temperature :
        Absolute temperature in K.
    pressure_points :
        Number of tabulation points generated inside the Dirichlet pressure
        interval used by the flow solve.
    pressure_padding_fraction :
        Relative padding added around the boundary-pressure range before
        tabulation. This guards against tiny roundoff excursions near the
        interval ends while keeping the lookup interval tight.
    """

    backend: ViscosityBackend
    temperature: float
    pressure_points: int = 128
    pressure_padding_fraction: float = 0.02
    _cache: dict[tuple[float, float, int], PressureViscosityTable] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.temperature <= 0.0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be a positive finite value in K")
        if self.pressure_points < 2:
            raise ValueError("pressure_points must be at least 2")
        if self.pressure_padding_fraction < 0.0:
            raise ValueError("pressure_padding_fraction must be nonnegative")

    @classmethod
    def from_backend(
        cls,
        backend: str,
        *,
        temperature: float,
        pressure_points: int = 128,
        pressure_padding_fraction: float = 0.02,
    ) -> "TabulatedWaterViscosityModel":
        """Construct a tabulated water-viscosity model from a backend name."""

        key = backend.strip().lower()
        if key == "thermo":
            backend_obj: ViscosityBackend = ThermoWaterViscosityBackend()
        elif key == "coolprop":
            backend_obj = CoolPropWaterViscosityBackend()
        else:
            raise ValueError(
                f"Unknown viscosity backend '{backend}'. Expected 'thermo' or 'coolprop'."
            )
        return cls(
            backend=backend_obj,
            temperature=float(temperature),
            pressure_points=int(pressure_points),
            pressure_padding_fraction=float(pressure_padding_fraction),
        )

    @property
    def backend_name(self) -> str:
        """Return the backend identifier used by this model."""

        return self.backend.name

    def table_for_bounds(self, *, pin: float, pout: float) -> PressureViscosityTable:
        """Return a cached pressure table spanning the Dirichlet pressure range."""

        pmin, pmax = _expanded_pressure_bounds(
            pin,
            pout,
            padding_fraction=self.pressure_padding_fraction,
        )
        key = (float(pmin), float(pmax), int(self.pressure_points))
        if key not in self._cache:
            pressure = np.linspace(pmin, pmax, self.pressure_points, dtype=float)
            viscosity = self.backend.evaluate(pressure, temperature=float(self.temperature))
            self._cache[key] = PressureViscosityTable(pressure=pressure, viscosity=viscosity)
        return self._cache[key]

    def evaluate(self, pressure: float | np.ndarray, *, pin: float, pout: float) -> np.ndarray:
        """Evaluate viscosity through the cached pressure interpolator."""

        table = self.table_for_bounds(pin=pin, pout=pout)
        return table.evaluate(pressure)

    def derivative(self, pressure: float | np.ndarray, *, pin: float, pout: float) -> np.ndarray:
        """Evaluate ``dmu/dp`` through the cached pressure interpolator."""

        table = self.table_for_bounds(pin=pin, pout=pout)
        return table.derivative(pressure)

    def evaluate_with_derivative(
        self,
        pressure: float | np.ndarray,
        *,
        pin: float,
        pout: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return viscosity and ``dmu/dp`` through the cached interpolator."""

        table = self.table_for_bounds(pin=pin, pout=pout)
        return table.evaluate_with_derivative(pressure)

    def reference_viscosity(self, *, pin: float, pout: float) -> float:
        """Return the midpoint viscosity across the imposed pressure interval."""

        p_ref = 0.5 * (float(pin) + float(pout))
        return float(self.evaluate(np.asarray([p_ref], dtype=float), pin=pin, pout=pout)[0])


__all__ = [
    "CoolPropWaterViscosityBackend",
    "PressureViscosityTable",
    "TabulatedWaterViscosityModel",
    "ThermoWaterViscosityBackend",
    "ViscosityBackend",
]
