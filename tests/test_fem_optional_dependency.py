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

    with pytest.raises(ImportError, match="DOLFINx, Basix, UFL, and mpi4py"):
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
    assert "PETSc FEM linear backend requires" in message
    assert "linear_backend='auto' falls back to the SciPy direct backend" in message


def test_fem_auto_linear_backend_uses_scipy_when_windows_lacks_petsc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dolfinx.fem.petsc":
            raise ImportError("simulated missing petsc4py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "platform", "win32")

    api = _common._require_dolfinx_core()

    assert _common._resolve_linear_backend("auto", api) == "scipy"


def test_fem_linear_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="linear_backend must be one of"):
        _common._resolve_linear_backend("not-a-backend", SimpleNamespace())  # type: ignore[arg-type]


def test_scipy_fem_backend_rejects_distributed_mesh() -> None:
    context = SimpleNamespace(mesh=SimpleNamespace(comm=SimpleNamespace(size=2)))

    with pytest.raises(NotImplementedError, match="serial-only"):
        _common._solve_mixed_problem_scipy(
            context,
            mixed_space=None,
            form=None,
            rhs=None,
            bcs=[],
            linear_backend="scipy",
        )


def test_fem_dirichlet_bc_values_use_modern_bc_set_path() -> None:
    calls: list[float] = []
    array = np.array([1.0, 2.0])

    class FakeBC:
        def __init__(self, value: float) -> None:
            self.value = value

        def set(self, target: np.ndarray) -> None:
            calls.append(self.value)
            target[:] = self.value

    fem = SimpleNamespace(set_bc=lambda _array, _bcs: calls.append(-1.0))

    _common._set_dirichlet_bc_values(fem, array, [FakeBC(3.0), FakeBC(4.0)])

    assert calls == [3.0, 4.0]
    assert np.array_equal(array, np.array([4.0, 4.0]))


def test_fem_dirichlet_bc_values_fall_back_for_older_dolfinx() -> None:
    calls: list[int] = []
    array = np.array([1.0, 2.0])

    def fake_set_bc(target: np.ndarray, bcs: list[object]) -> None:
        calls.append(len(bcs))
        target[:] = 0.0

    fem = SimpleNamespace(set_bc=fake_set_bc)

    _common._set_dirichlet_bc_values(fem, array, [object()])

    assert calls == [1]
    assert np.array_equal(array, np.array([0.0, 0.0]))


def test_umfpack_fem_backend_dispatches_optional_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMatrix:
        def scatter_reverse(self) -> None:
            return None

        def to_scipy(self) -> Any:
            return SimpleNamespace(copy=lambda: "matrix")

    vector = SimpleNamespace(
        array=np.array([1.0]),
        scatter_reverse=lambda _mode: None,
    )
    solution = SimpleNamespace(
        x=SimpleNamespace(
            array=np.zeros(2),
            scatter_forward=lambda: None,
        )
    )
    fem = SimpleNamespace(
        form=lambda value: value,
        assemble_matrix=lambda _form, bcs: FakeMatrix(),
        assemble_vector=lambda _rhs: vector,
        apply_lifting=lambda _array, _forms, _bcs: None,
        set_bc=lambda _array, _bcs: None,
        Function=lambda _space: solution,
    )
    la = SimpleNamespace(InsertMode=SimpleNamespace(add="add"))
    context = SimpleNamespace(
        mesh=SimpleNamespace(comm=SimpleNamespace(size=1)),
        api=SimpleNamespace(fem=fem, la=la),
    )
    fake_umfpack = SimpleNamespace(spsolve=lambda _matrix, _rhs: np.array([2.0]))

    monkeypatch.setattr(_common, "import_module", lambda _name: fake_umfpack)

    with pytest.raises(RuntimeError, match="incompatible size"):
        _common._solve_mixed_problem_scipy(
            context,
            mixed_space=None,
            form=None,
            rhs=None,
            bcs=[],
            linear_backend="umfpack",
        )


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
