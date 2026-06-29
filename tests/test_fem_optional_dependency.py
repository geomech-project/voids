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
    assert "linear_backend='auto' falls back to the SciPy/SuperLU direct backend" in message


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


def test_fem_thread_environment_metadata_reports_solver_relevant_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

    metadata = _common._thread_environment_metadata()

    assert metadata["OMP_NUM_THREADS"] == "4"
    assert "MKL_NUM_THREADS" in metadata


def test_petsc_solver_diagnostics_extracts_available_ksp_metadata() -> None:
    solver = SimpleNamespace(
        getType=lambda: "fgmres",
        getConvergedReason=lambda: 2,
        getIterationNumber=lambda: 17,
        getResidualNorm=lambda: 1.0e-9,
    )
    problem = SimpleNamespace(solver=solver)

    assert _common._petsc_solver_diagnostics(problem) == {
        "petsc_ksp_type": "fgmres",
        "petsc_converged_reason": 2,
        "petsc_iteration_number": 17,
        "petsc_residual_norm": 1.0e-9,
    }


def test_petsc_solver_diagnostics_tolerates_missing_solver() -> None:
    assert _common._petsc_solver_diagnostics(SimpleNamespace()) == {}


def test_petsc_solver_diagnostics_skips_missing_ksp_methods() -> None:
    problem = SimpleNamespace(solver=SimpleNamespace(getType=lambda: "preonly"))

    assert _common._petsc_solver_diagnostics(problem) == {"petsc_ksp_type": "preonly"}


def test_petsc_solve_raises_when_returned_ksp_diverged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearProblem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.solver = SimpleNamespace(
                getType=lambda: "fgmres",
                getConvergedReason=lambda: -3,
                getIterationNumber=lambda: 300,
                getResidualNorm=lambda: 1.0e-5,
            )

        def solve(self) -> object:
            return object()

    fake_api = SimpleNamespace(petsc=SimpleNamespace(LinearProblem=FakeLinearProblem))
    monkeypatch.setattr(_common, "_require_dolfinx_petsc", lambda _api: fake_api)
    context = SimpleNamespace(api=SimpleNamespace())
    options = _common.FEniCSSolverOptions(
        linear_backend="petsc",
        petsc_options={
            "ksp_type": "fgmres",
            "ksp_error_if_not_converged": True,
        },
    )

    with pytest.raises(RuntimeError, match="PETSc linear solve did not converge"):
        _common._solve_mixed_problem(
            context,
            form=None,
            rhs=None,
            bcs=[],
            options=options,
            prefix_suffix="probe",
        )


def test_petsc_block_solve_normalizes_tuple_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearProblem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.solver = SimpleNamespace(
                getConvergedReason=lambda: 4,
                getIterationNumber=lambda: 1,
                getResidualNorm=lambda: 0.0,
            )

        def solve(self) -> tuple[str, str]:
            return ("u", "p")

    fake_api = SimpleNamespace(petsc=SimpleNamespace(LinearProblem=FakeLinearProblem))
    monkeypatch.setattr(_common, "_require_dolfinx_petsc", lambda _api: fake_api)
    context = SimpleNamespace(api=SimpleNamespace())

    solution, _, metadata = _common._solve_block_problem_petsc(
        context,
        forms=[[None, None], [None, None]],
        rhs=[None, None],
        bcs=[],
        solution_functions=[],
        options=_common.FEniCSSolverOptions(linear_backend="petsc"),
        prefix_suffix="probe",
        matrix_kind="mpi",
    )

    assert solution == ["u", "p"]
    assert metadata["petsc_matrix_kind"] == "mpi"


def test_petsc_block_solve_wraps_single_solution_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    class FakeLinearProblem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.solver = SimpleNamespace(getConvergedReason=lambda: 4)

        def solve(self) -> object:
            return sentinel

    fake_api = SimpleNamespace(petsc=SimpleNamespace(LinearProblem=FakeLinearProblem))
    monkeypatch.setattr(_common, "_require_dolfinx_petsc", lambda _api: fake_api)
    context = SimpleNamespace(api=SimpleNamespace())

    solution, _, _ = _common._solve_block_problem_petsc(
        context,
        forms=[[None]],
        rhs=[None],
        bcs=[],
        solution_functions=[],
        options=_common.FEniCSSolverOptions(linear_backend="petsc"),
        prefix_suffix="probe",
        matrix_kind="mpi",
    )

    assert solution == [sentinel]


def test_petsc_block_solve_raises_when_returned_ksp_diverged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOptions(dict):
        def prefixPush(self, _prefix: str) -> None:  # noqa: N802
            return None

        def prefixPop(self) -> None:  # noqa: N802
            return None

    class FakeLinearProblem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.solver = SimpleNamespace(
                getOptionsPrefix=lambda: "fake_",
                setFromOptions=lambda: None,
                getConvergedReason=lambda: -3,
                getIterationNumber=lambda: 10,
                getResidualNorm=lambda: 1.0e-5,
            )

        def solve(self) -> list[object]:
            return [object(), object()]

    fake_api = SimpleNamespace(
        petsc=SimpleNamespace(
            LinearProblem=FakeLinearProblem,
            PETSc=SimpleNamespace(Options=FakeOptions),
        )
    )
    monkeypatch.setattr(_common, "_require_dolfinx_petsc", lambda _api: fake_api)
    context = SimpleNamespace(api=SimpleNamespace())
    options = _common.FEniCSSolverOptions(
        linear_backend="petsc",
        petsc_options={
            "ksp_type": "fgmres",
            "ksp_error_if_not_converged": True,
        },
    )

    with pytest.raises(RuntimeError, match="PETSc linear solve did not converge"):
        _common._solve_block_problem_petsc(
            context,
            forms=[[None, None], [None, None]],
            rhs=[None, None],
            bcs=[],
            solution_functions=[],
            options=options,
            prefix_suffix="probe",
            matrix_kind="nest",
        )


def test_petsc_block_solve_defers_nested_options_until_after_fields_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    class FakeOptions(dict):
        def prefixPush(self, prefix: str) -> None:  # noqa: N802
            events.append(("prefixPush", prefix))

        def __setitem__(self, key: str, value: Any) -> None:
            events.append(("set", key, value))
            super().__setitem__(key, value)

        def __delitem__(self, key: str) -> None:
            events.append(("del", key))
            super().__delitem__(key)

        def prefixPop(self) -> None:  # noqa: N802
            events.append(("prefixPop", None))

    class FakeLinearProblem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append(("init_petsc_options", kwargs.get("petsc_options")))
            pc = SimpleNamespace(
                setFieldSplitIS=lambda *fields: events.append(("setFieldSplitIS", fields))
            )
            self.solver = SimpleNamespace(
                getOptionsPrefix=lambda: "voids_block_",
                setFromOptions=lambda: events.append(("setFromOptions", None)),
                getConvergedReason=lambda: 4,
                getPC=lambda: pc,
            )
            self.A = SimpleNamespace(getNestISs=lambda: (("u_is", "p_is"), None))
            self.u = [SimpleNamespace(name="u"), SimpleNamespace(name="p")]

        def solve(self) -> list[str]:
            return ["u", "p"]

    fake_api = SimpleNamespace(
        petsc=SimpleNamespace(
            LinearProblem=FakeLinearProblem,
            PETSc=SimpleNamespace(Options=FakeOptions),
        )
    )
    monkeypatch.setattr(_common, "_require_dolfinx_petsc", lambda _api: fake_api)
    context = SimpleNamespace(api=SimpleNamespace())

    _, _, metadata = _common._solve_block_problem_petsc(
        context,
        forms=[[None, None], [None, None]],
        rhs=[None, None],
        bcs=[],
        solution_functions=[],
        options=_common.FEniCSSolverOptions(
            linear_backend="petsc",
            petsc_options={"pc_type": "fieldsplit", "fieldsplit_velocity_pc_type": "lu"},
        ),
        prefix_suffix="probe",
        matrix_kind="nest",
    )

    assert ("init_petsc_options", None) in events
    assert events.index(("init_petsc_options", None)) < events.index(("setFromOptions", None))
    assert ("prefixPush", "voids_block_") in events
    assert ("set", "pc_type", "fieldsplit") in events
    assert ("set", "fieldsplit_velocity_pc_type", "lu") in events
    assert ("setFieldSplitIS", (("u_0", "u_is"), ("p_1", "p_is"))) in events
    assert events.count(("setFromOptions", None)) == 2
    assert ("del", "pc_type") in events
    assert events[-1] == ("prefixPop", None)
    assert metadata["petsc_options_applied_after_block_setup"] is True
    assert metadata["petsc_nest_fieldsplit_is_reapplied"] is True


def test_standalone_pressure_gauge_requires_petsc() -> None:
    context = SimpleNamespace(api=SimpleNamespace(petsc=None))

    with pytest.raises(ImportError, match="standalone pressure gauge"):
        _common._standalone_pressure_gauge_bc(context, pressure_space=None)


def test_standalone_pressure_gauge_falls_back_to_first_dof_when_origin_not_found() -> None:
    captured: dict[str, Any] = {}
    comm = SimpleNamespace(allreduce=lambda value, op=None: value)
    context = SimpleNamespace(
        mesh=SimpleNamespace(
            geometry=SimpleNamespace(x=np.array([[0.0, 0.0], [1.0, 1.0]]), dim=2),
            comm=comm,
        ),
        api=SimpleNamespace(
            MPI=SimpleNamespace(MIN="min", MAX="max"),
            fem=SimpleNamespace(
                locate_dofs_geometrical=lambda _space, _marker: np.array([], dtype=np.int32),
                dirichletbc=lambda value, dofs, space: (
                    captured.update(
                        value=value,
                        dofs=dofs,
                        space=space,
                    )
                    or "bc"
                ),
            ),
            petsc=SimpleNamespace(PETSc=SimpleNamespace(ScalarType=float)),
        ),
    )

    assert _common._standalone_pressure_gauge_bc(context, pressure_space="Q") == "bc"
    assert np.array_equal(captured["dofs"], np.array([0], dtype=np.int32))
    assert captured["space"] == "Q"


def test_serial_direct_fem_backend_rejects_distributed_mesh() -> None:
    context = SimpleNamespace(mesh=SimpleNamespace(comm=SimpleNamespace(size=2)))

    with pytest.raises(NotImplementedError, match="serial-only"):
        _common._solve_mixed_problem_serial_direct(
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
    class FakeSparseMatrix:
        nnz = 1

        def __init__(self) -> None:
            self.data = np.array([1.0])
            self.indices = np.array([0], dtype=np.int32)
            self.indptr = np.array([0, 1], dtype=np.int32)

        def tocsc(self) -> FakeSparseMatrix:
            return self

        def tocsr(self) -> FakeSparseMatrix:
            return self

        def copy(self) -> FakeSparseMatrix:
            copied = FakeSparseMatrix()
            copied.data = self.data.copy()
            copied.indices = self.indices.copy()
            copied.indptr = self.indptr.copy()
            return copied

    class FakeMatrix:
        def scatter_reverse(self) -> None:
            return None

        def to_scipy(self) -> Any:
            return FakeSparseMatrix()

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
    calls: dict[str, Any] = {}

    class FakeUmfpackContext:
        def __init__(self, family: str) -> None:
            calls["family"] = family
            self.control = np.zeros(20)

        def numeric(self, matrix: Any) -> None:
            calls["numeric_indices_dtype"] = str(matrix.indices.dtype)
            calls["numeric_indptr_dtype"] = str(matrix.indptr.dtype)
            calls["ordering_control"] = self.control[10]
            calls["pivot_tolerance_control"] = self.control[3]

        def solve(
            self,
            system: Any,
            matrix: Any,
            rhs: Any,
            *,
            autoTranspose: bool,
        ) -> np.ndarray:
            calls["system"] = system
            calls["solve_indices_dtype"] = str(matrix.indices.dtype)
            calls["solve_indptr_dtype"] = str(matrix.indptr.dtype)
            calls["autoTranspose"] = autoTranspose
            return np.array([2.0])

    fake_umfpack = SimpleNamespace(
        UMFPACK_A="A",
        UMFPACK_ORDERING=10,
        UMFPACK_ORDERING_METIS_GUARD=7,
        UMFPACK_PIVOT_TOLERANCE=3,
        UmfpackContext=FakeUmfpackContext,
    )

    monkeypatch.setattr(_common, "import_module", lambda _name: fake_umfpack)

    with pytest.raises(RuntimeError, match="incompatible size"):
        _common._solve_mixed_problem_serial_direct(
            context,
            mixed_space=None,
            form=None,
            rhs=None,
            bcs=[],
            linear_backend="umfpack",
            umfpack_controls={"ordering": "metis_guard", "pivot_tolerance": 1.0e-2},
        )
    assert calls == {
        "family": "dl",
        "numeric_indices_dtype": "int64",
        "numeric_indptr_dtype": "int64",
        "ordering_control": 7.0,
        "pivot_tolerance_control": 1.0e-2,
        "system": "A",
        "solve_indices_dtype": "int64",
        "solve_indptr_dtype": "int64",
        "autoTranspose": True,
    }


def test_pardiso_fem_backend_dispatches_optional_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSparseMatrix:
        nnz = 1

        def tocsc(self) -> FakeSparseMatrix:
            return self

        def tocsr(self) -> FakeSparseMatrix:
            return self

        def copy(self) -> str:
            return "matrix"

    class FakeMatrix:
        def scatter_reverse(self) -> None:
            return None

        def to_scipy(self) -> Any:
            return FakeSparseMatrix()

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
    fake_pypardiso = SimpleNamespace(spsolve=lambda _matrix, _rhs: np.array([2.0]))

    monkeypatch.setattr(_common, "import_module", lambda _name: fake_pypardiso)

    with pytest.raises(RuntimeError, match="incompatible size"):
        _common._solve_mixed_problem_serial_direct(
            context,
            mixed_space=None,
            form=None,
            rhs=None,
            bcs=[],
            linear_backend="pardiso",
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
