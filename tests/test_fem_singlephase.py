from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest

from voids.fem.singlephase import (  # noqa: E402
    FEMMapProblem,
    FEniCSSolverOptions,
    solve_brinkman_taylor_hood,
    solve_brinkman_usfem,
    solve_brinkman_usfem_block,
    solve_darcy_taylor_hood,
    upscale_permeability_fem,
    upscale_principal_permeabilities_fem,
)
from voids.fem.singlephase._common import (  # noqa: E402
    _FEM_THREAD_ENV_DEFAULTS,
    _apply_fem_thread_defaults,
    _require_dolfinx_core,
)
import voids.fem.singlephase.usfem as usfem_module  # noqa: E402
from voids.image.porosity import PermeabilityMap, PorosityMap  # noqa: E402

try:
    _require_dolfinx_core()
except ImportError as exc:
    requires_fem_stack = pytest.mark.skip(reason=str(exc))
else:
    requires_fem_stack = pytest.mark.skipif(False, reason="")


def _constant_problem(shape: tuple[int, ...], permeability: float = 2.0) -> FEMMapProblem:
    return FEMMapProblem(
        permeability_map=PermeabilityMap(np.full(shape, permeability), cell_size=1.0),
        porosity_map=PorosityMap(np.ones(shape), cell_size=1.0),
        viscosity=1.0,
    )


def test_fem_thread_defaults_preserve_existing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    _apply_fem_thread_defaults()

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == _FEM_THREAD_ENV_DEFAULTS["OPENBLAS_NUM_THREADS"]
    assert (
        os.environ["VECLIB_MAXIMUM_THREADS"] == _FEM_THREAD_ENV_DEFAULTS["VECLIB_MAXIMUM_THREADS"]
    )


def test_fenics_solver_options_direct_lu_builder() -> None:
    options = FEniCSSolverOptions.direct_lu("superlu_dist")

    assert options.linear_backend == "petsc"
    assert options.solver_preset == "direct_reference"
    assert options.petsc_options == {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
        "ksp_error_if_not_converged": True,
        "pc_factor_shift_type": "nonzero",
        "pc_factor_shift_amount": 1.0e-12,
    }

    mumps_options = FEniCSSolverOptions.direct_lu(
        "mumps",
        mumps_memory_relaxation_percent=500,
        mumps_workspace_mb=20000,
    )

    assert mumps_options.petsc_options["mat_mumps_icntl_14"] == 500
    assert mumps_options.petsc_options["mat_mumps_icntl_23"] == 20000
    assert FEniCSSolverOptions.scipy_direct().linear_backend == "scipy"
    assert FEniCSSolverOptions.scipy_direct().solver_preset == "direct_reference"
    assert FEniCSSolverOptions.superlu_direct().linear_backend == "superlu"
    assert FEniCSSolverOptions.superlu_direct().solver_preset == "direct_reference"
    assert FEniCSSolverOptions.umfpack_direct().linear_backend == "umfpack"
    assert FEniCSSolverOptions.umfpack_direct().solver_preset == "direct_reference"
    tuned_umfpack = FEniCSSolverOptions.umfpack_direct(
        ordering="metis_guard",
        strategy="unsymmetric",
        pivot_tolerance=1.0e-2,
    )
    assert tuned_umfpack.umfpack_controls == {
        "ordering": "metis_guard",
        "strategy": "unsymmetric",
        "pivot_tolerance": 1.0e-2,
    }
    assert FEniCSSolverOptions.pardiso_direct().linear_backend == "pardiso"
    assert FEniCSSolverOptions.pardiso_direct().solver_preset == "direct_reference"


def test_fenics_solver_options_parallel_and_iterative_presets() -> None:
    reference = FEniCSSolverOptions.direct_reference("mumps", mumps_memory_relaxation_percent=300)
    assert reference.solver_preset == "direct_reference"
    assert reference.petsc_options["mat_mumps_icntl_14"] == 300

    parallel = FEniCSSolverOptions.direct_parallel("mumps", mumps_workspace_mb=16000)

    assert parallel.linear_backend == "petsc"
    assert parallel.solver_preset == "direct_parallel"
    assert parallel.petsc_options["pc_type"] == "lu"
    assert parallel.petsc_options["pc_factor_mat_solver_type"] == "mumps"
    assert parallel.petsc_options["mat_mumps_icntl_14"] == 500
    assert parallel.petsc_options["mat_mumps_icntl_23"] == 16000

    superlu = FEniCSSolverOptions.direct_parallel("superlu_dist")
    assert superlu.petsc_options["pc_factor_mat_solver_type"] == "superlu_dist"
    assert "mat_mumps_icntl_14" not in superlu.petsc_options

    iterative = FEniCSSolverOptions.iterative_fieldsplit_experimental(rtol=1.0e-7, max_it=123)
    assert iterative.linear_backend == "petsc"
    assert iterative.solver_preset == "iterative_fieldsplit_experimental"
    assert iterative.petsc_options["ksp_type"] == "fgmres"
    assert iterative.petsc_options["ksp_rtol"] == 1.0e-7
    assert iterative.petsc_options["ksp_max_it"] == 123
    assert iterative.petsc_options["pc_type"] == "fieldsplit"

    block_iterative = FEniCSSolverOptions.iterative_block_lgmres_experimental(
        atol=1.0e-9,
        max_it=456,
        block_lu_backend="mumps",
    )
    assert block_iterative.linear_backend == "petsc"
    assert block_iterative.solver_preset == "iterative_block_lgmres_experimental"
    assert block_iterative.petsc_options["ksp_type"] == "lgmres"
    assert block_iterative.petsc_options["ksp_atol"] == 1.0e-9
    assert block_iterative.petsc_options["ksp_max_it"] == 456
    assert block_iterative.petsc_options["ksp_norm_type"] == "unpreconditioned"
    assert block_iterative.petsc_options["pc_fieldsplit_type"] == "multiplicative"
    assert block_iterative.petsc_options["fieldsplit_u_0_pc_type"] == "lu"
    assert block_iterative.petsc_options["fieldsplit_u_0_pc_factor_mat_solver_type"] == "mumps"
    assert block_iterative.petsc_options["fieldsplit_p_1_pc_type"] == "lu"
    assert block_iterative.petsc_options["fieldsplit_p_1_pc_factor_mat_solver_type"] == "mumps"


def test_usfem_block_rejects_unknown_matrix_kind_before_backend_import() -> None:
    with pytest.raises(ValueError, match="matrix_kind"):
        solve_brinkman_usfem_block(
            _constant_problem((2, 2), permeability=2.0),
            matrix_kind="bad",  # type: ignore[arg-type]
        )


def test_usfem_block_rejects_unknown_preconditioner_before_backend_import() -> None:
    with pytest.raises(ValueError, match="preconditioner"):
        solve_brinkman_usfem_block(
            _constant_problem((2, 2), permeability=2.0),
            preconditioner="bad",  # type: ignore[arg-type]
        )


@requires_fem_stack
def test_usfem_block_rejects_serial_linear_backend() -> None:
    with pytest.raises(NotImplementedError, match="supports only the PETSc backend"):
        solve_brinkman_usfem_block(
            _constant_problem((2, 2), permeability=2.0),
            options=FEniCSSolverOptions.scipy_direct(),
        )


@pytest.mark.parametrize(
    "solver",
    [
        solve_darcy_taylor_hood,
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_backends_recover_constant_2d_permeability(
    solver: Callable[..., Any],
) -> None:
    result = solver(_constant_problem((3, 3), permeability=2.0), flow_axis="x")

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.flow_rate > 0.0
    assert result.solve_seconds >= 0.0
    assert result.metadata["linear_backend"] in {"petsc", "scipy"}
    assert result.metadata["petsc_options"]["pc_factor_mat_solver_type"] == "mumps"
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@pytest.mark.parametrize(
    "solver",
    [
        solve_darcy_taylor_hood,
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_backends_recover_constant_2d_permeability_with_scipy_direct(
    solver: Callable[..., Any],
) -> None:
    result = solver(
        _constant_problem((3, 3), permeability=2.0),
        flow_axis="x",
        options=FEniCSSolverOptions.scipy_direct(),
    )

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.metadata["linear_backend"] == "scipy"
    assert result.metadata["serial_sparse_solver_backend"] == "scipy.sparse.linalg.splu"
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@requires_fem_stack
def test_fem_usfem_brinkman_recovers_constant_3d_permeability_with_superlu_direct() -> None:
    result = solve_brinkman_usfem(
        _constant_problem((2, 2, 2), permeability=1.5),
        flow_axis="z",
        options=FEniCSSolverOptions.superlu_direct(),
    )

    assert result.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert result.metadata["linear_backend"] == "superlu"
    assert result.metadata["serial_sparse_solver_backend"] == "scipy.sparse.linalg.splu"
    assert result.metadata["serial_sparse_matrix_format"] == "csc"
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@requires_fem_stack
def test_fem_usfem_brinkman_recovers_constant_3d_permeability_with_umfpack_direct() -> None:
    result = solve_brinkman_usfem(
        _constant_problem((2, 2, 2), permeability=1.5),
        flow_axis="x",
        options=FEniCSSolverOptions.umfpack_direct(),
    )

    assert result.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert result.metadata["linear_backend"] == "umfpack"
    assert result.metadata["serial_sparse_solver_backend"] == "scikits.umfpack.UmfpackContext(dl)"
    assert result.metadata["serial_sparse_matrix_format"] == "csc"
    assert result.metadata["serial_sparse_umfpack_family"] == "dl"
    assert result.metadata["serial_sparse_matrix_indices_dtype"] == "int64"
    assert result.metadata["serial_sparse_matrix_indptr_dtype"] == "int64"
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@requires_fem_stack
def test_fem_usfem_brinkman_recovers_constant_3d_permeability_with_pardiso_direct() -> None:
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        pytest.skip("pypardiso is not installed")

    result = solve_brinkman_usfem(
        _constant_problem((2, 2, 2), permeability=1.5),
        flow_axis="y",
        options=FEniCSSolverOptions.pardiso_direct(),
    )

    assert result.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert result.metadata["linear_backend"] == "pardiso"
    assert result.metadata["serial_sparse_solver_backend"] == "pypardiso.spsolve"
    assert result.metadata["serial_sparse_matrix_format"] == "csr"
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@requires_fem_stack
def test_fem_taylor_hood_brinkman_supports_3d_constant_map() -> None:
    result = solve_brinkman_taylor_hood(
        _constant_problem((2, 2, 2), permeability=1.5),
        flow_axis="z",
    )

    assert result.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert result.flow_axis == "z"


@requires_fem_stack
def test_fem_usfem_block_mpi_matches_monolithic_usfem_constant_3d() -> None:
    problem = _constant_problem((2, 2, 2), permeability=1.5)
    options = FEniCSSolverOptions.direct_parallel(
        "superlu_dist",
        petsc_options_prefix="test_usfem_block_superlu_",
    )

    monolithic = solve_brinkman_usfem(
        problem,
        flow_axis="x",
        options=FEniCSSolverOptions.direct_parallel(
            "superlu_dist",
            petsc_options_prefix="test_usfem_monolithic_superlu_",
        ),
    )
    block = solve_brinkman_usfem_block(
        problem,
        flow_axis="x",
        options=options,
        matrix_kind="mpi",
        preconditioner="none",
    )

    assert block.permeability == pytest.approx(monolithic.permeability, rel=1.0e-10)
    assert block.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert block.metadata["block_matrix_kind"] == "mpi"
    assert block.metadata["block_preconditioner"] == "none"
    assert block.metadata["petsc_matrix_kind"] == "mpi"


@requires_fem_stack
def test_fem_usfem_block_builds_diagonal_preconditioner_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_solve_block_problem_petsc(
        _context: Any,
        **kwargs: Any,
    ) -> tuple[list[Any], float, dict[str, Any]]:
        captured["preconditioner_forms"] = kwargs["preconditioner_forms"]
        return kwargs["solution_functions"], 0.0, {"petsc_matrix_kind": kwargs["matrix_kind"]}

    monkeypatch.setattr(
        usfem_module,
        "_solve_block_problem_petsc",
        fake_solve_block_problem_petsc,
    )

    result = solve_brinkman_usfem_block(
        _constant_problem((2, 2), permeability=2.0),
        options=FEniCSSolverOptions.direct_parallel("superlu_dist"),
        matrix_kind="nest",
        preconditioner="diagonal",
    )

    assert captured["preconditioner_forms"][0][0] is not None
    assert captured["preconditioner_forms"][0][1] is None
    assert captured["preconditioner_forms"][1][0] is None
    assert captured["preconditioner_forms"][1][1] is not None
    assert result.metadata["block_preconditioner"] == "diagonal"
    assert result.metadata["petsc_matrix_kind"] == "nest"


@requires_fem_stack
def test_fem_brinkman_uses_unit_porosity_when_porosity_map_is_absent() -> None:
    problem = FEMMapProblem(
        permeability_map=PermeabilityMap(np.full((3, 3), 2.0), cell_size=1.0),
        porosity_map=None,
        viscosity=1.0,
    )

    result = solve_brinkman_taylor_hood(problem, flow_axis="x")

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert np.allclose(result.metadata["porosity_floor"], problem.porosity_floor)


@requires_fem_stack
def test_fem_upscaling_dispatches_backends() -> None:
    problem = _constant_problem((3, 3), permeability=3.0)

    result = upscale_permeability_fem(
        problem,
        backend="taylor_hood_darcy",
        axes=("x", "y"),
    )

    assert result.backend == "taylor_hood_darcy"
    assert set(result.results) == {"x", "y"}
    assert result.permeability["x"] == pytest.approx(3.0, rel=5.0e-4)
    assert result.permeability["y"] == pytest.approx(3.0, rel=5.0e-4)
    assert set(result.solve_seconds) == {"x", "y"}
    assert upscale_principal_permeabilities_fem(
        problem,
        backend="usfem_brinkman",
        axes=("x",),
    ) == {"x": pytest.approx(3.0, rel=5.0e-4)}


def test_fem_problem_validates_map_compatibility() -> None:
    with pytest.raises(ValueError, match="same shape"):
        FEMMapProblem(
            PermeabilityMap(np.ones((2, 2)), cell_size=1.0),
            PorosityMap(np.ones((2, 3)), cell_size=1.0),
        )
