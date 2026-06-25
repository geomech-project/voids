from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voids.fem.singlephase import FEMMapProblem, _common
from voids.fem.singlephase import solve_brinkman_usfem
from voids.fem.singlephase.upscaling import _backend_from_name, _default_axes
from voids.image.porosity import PermeabilityMap, PorosityMap


def test_fem_backend_reports_clean_missing_dolfinx_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dolfinx" or name.startswith("dolfinx."):
            raise ImportError("simulated missing dolfinx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="full DOLFINx/PETSc Python stack"):
        _common._require_dolfinx()


def test_fem_backend_reports_native_windows_limitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dolfinx.fem.petsc":
            raise ImportError("simulated missing petsc4py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(ImportError) as exc_info:
        _common._require_dolfinx()

    message = str(exc_info.value)
    assert "full DOLFINx/PETSc Python stack" in message
    assert "Native Windows is currently not fully supported" in message
    assert "Use Linux, macOS, WSL2, or Docker" in message


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"viscosity": 0.0}, "viscosity must be positive and finite"),
        ({"porosity_floor": float("nan")}, "porosity_floor must be positive and finite"),
        ({"permeability_floor": -1.0}, "permeability_floor must be positive and finite"),
    ],
)
def test_fem_map_problem_rejects_nonphysical_coefficients(
    kwargs: dict[str, float],
    message: str,
) -> None:
    permeability = PermeabilityMap(np.ones((2, 2)), cell_size=1.0)

    with pytest.raises(ValueError, match=message):
        FEMMapProblem(permeability, **kwargs)


def test_fem_map_problem_rejects_bad_map_geometry() -> None:
    with pytest.raises(ValueError, match="permeability_map must be 2D or 3D"):
        FEMMapProblem(SimpleNamespace(ndim=1, shape=(2,), cell_size=(1.0,)))

    with pytest.raises(ValueError, match="same cell_size"):
        FEMMapProblem(
            PermeabilityMap(np.ones((2, 2)), cell_size=(1.0, 1.0)),
            PorosityMap(np.ones((2, 2)), cell_size=(1.0, 2.0)),
        )


def test_fem_axis_and_dispatch_validation_branches() -> None:
    assert _common._axis_index("y", 2) == 1
    assert _default_axes(2) == ("x", "y")
    assert _default_axes(3) == ("x", "y", "z")
    assert _backend_from_name("brinkman taylor hood").__name__ == "solve_brinkman_taylor_hood"
    assert _backend_from_name("darcy-darcy").__name__ == "solve_darcy_taylor_hood"

    with pytest.raises(ValueError, match="flow_axis must be one of"):
        _common._axis_index("z", 2)
    with pytest.raises(ValueError, match="permeability maps must be 2D or 3D"):
        _default_axes(1)
    with pytest.raises(ValueError, match="backend must be one of"):
        _backend_from_name("not a solver")


def test_fem_validate_pressure_drop() -> None:
    _common._validate_pressure_drop(1.0, 0.0)

    with pytest.raises(ValueError, match="pressure values must be finite"):
        _common._validate_pressure_drop(float("inf"), 0.0)
    with pytest.raises(ValueError, match="pressure_inlet must be greater"):
        _common._validate_pressure_drop(1.0, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tau_factor": 0.0}, "tau_factor must be positive"),
        ({"m_t": 0.0}, "m_t must be positive"),
        ({"alpha_edge": 0.0}, "alpha_edge must be positive"),
    ],
)
def test_usfem_rejects_nonpositive_stabilization_controls(
    kwargs: dict[str, float],
    message: str,
) -> None:
    problem = FEMMapProblem(PermeabilityMap(np.ones((2, 2)), cell_size=1.0))

    with pytest.raises(ValueError, match=message):
        solve_brinkman_usfem(problem, **kwargs)
