from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from voids.fem.singlephase import (  # noqa: E402
    BrinkmanNondimensionalization,
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
    _brinkman_nondimensional_scales,
    _constant_permeability_value,
    _require_dolfinx_core,
    _require_dolfinx_petsc,
    _resolve_brinkman_nondimensionalization,
)
import voids.fem.singlephase.usfem as usfem_module  # noqa: E402
from voids.image.porosity import PermeabilityMap, PorosityMap  # noqa: E402

try:
    _require_dolfinx_core()
except ImportError as exc:
    requires_fem_stack = pytest.mark.skip(reason=str(exc))
else:
    requires_fem_stack = pytest.mark.skipif(False, reason="")

try:
    _require_dolfinx_petsc()
except ImportError as exc:
    requires_petsc_fem_stack = pytest.mark.skip(reason=str(exc))
else:
    requires_petsc_fem_stack = pytest.mark.skipif(False, reason="")


def _constant_problem(shape: tuple[int, ...], permeability: float = 2.0) -> FEMMapProblem:
    return FEMMapProblem(
        permeability_map=PermeabilityMap(np.full(shape, permeability), cell_size=1.0),
        porosity_map=PorosityMap(np.ones(shape), cell_size=1.0),
        viscosity=1.0,
    )


def _heterogeneous_problem() -> FEMMapProblem:
    permeability = np.array(
        [
            [1.0, 2.0, 4.0],
            [3.0, 5.0, 7.0],
            [2.5, 4.5, 6.5],
        ],
        dtype=float,
    )
    porosity = np.array(
        [
            [0.45, 0.55, 0.70],
            [0.60, 0.80, 0.65],
            [0.50, 0.75, 0.85],
        ],
        dtype=float,
    )
    return FEMMapProblem(
        permeability_map=PermeabilityMap(permeability, cell_size=1.0),
        porosity_map=PorosityMap(porosity, cell_size=1.0),
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
    assert options.linear_system_dtype == "float64"
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
    assert (
        FEniCSSolverOptions.superlu_direct(linear_system_dtype="float32").linear_system_dtype
        == "float32"
    )
    tuned_superlu = FEniCSSolverOptions.superlu_direct(
        permc_spec="COLAMD",
        diag_pivot_thresh=0.0,
    )
    assert tuned_superlu.superlu_controls == {
        "permc_spec": "COLAMD",
        "diag_pivot_thresh": 0.0,
    }
    tuned_scipy_alias = FEniCSSolverOptions.scipy_direct(
        controls={"permc-spec": "MMD_ATA", "equil": False},
    )
    assert tuned_scipy_alias.superlu_controls == {
        "permc_spec": "MMD_ATA",
        "equil": False,
    }
    assert FEniCSSolverOptions.umfpack_direct().linear_backend == "umfpack"
    assert FEniCSSolverOptions.umfpack_direct().solver_preset == "direct_reference"
    assert (
        FEniCSSolverOptions.umfpack_direct(linear_system_dtype="float32").linear_system_dtype
        == "float32"
    )
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
    assert (
        FEniCSSolverOptions.pardiso_direct(linear_system_dtype="float32").linear_system_dtype
        == "float32"
    )
    assert FEniCSSolverOptions.nvmath_cudss_direct().linear_backend == "nvmath_cudss"
    assert FEniCSSolverOptions.nvmath_cudss_direct().solver_preset == "direct_reference"
    cudss_options = FEniCSSolverOptions.nvmath_cudss_direct(
        device_ids=0,
        dtype="float64",
        ir_steps=5,
    )
    assert cudss_options.nvmath_cudss_controls == {
        "dtype": "float64",
        "ir_steps": 5,
        "use_matching": True,
        "check_residual": True,
        "device_ids": (0,),
        "residual_rtol": 1.0e-8,
    }
    assert cudss_options.linear_system_dtype == "float64"
    assert FEniCSSolverOptions.nvmath_cudss_direct(dtype="float32").linear_system_dtype == "float32"
    robust_cudss_options = FEniCSSolverOptions.nvmath_cudss_direct(
        dtype="float32",
        reordering_alg="alg_1",
        matching_alg="alg_2",
        pivot_type="row",
        pivot_threshold=0.2,
        pivot_epsilon=1.0e-6,
        pivot_epsilon_alg="alg_3",
        nd_nlevels=2,
        host_nthreads=4,
        hybrid_mode=True,
        hybrid_device_memory_limit=20_000_000_000,
        hybrid_execute_mode=False,
        use_cuda_register_memory=True,
        use_superpanels=False,
        deterministic_mode=True,
        residual_rtol=1.0e-3,
    )
    assert robust_cudss_options.linear_system_dtype == "float32"
    assert robust_cudss_options.nvmath_cudss_controls["reordering_alg"] == 1
    assert robust_cudss_options.nvmath_cudss_controls["matching_alg"] == 2
    assert robust_cudss_options.nvmath_cudss_controls["pivot_type"] == "row"
    assert robust_cudss_options.nvmath_cudss_controls["pivot_threshold"] == 0.2
    assert robust_cudss_options.nvmath_cudss_controls["pivot_epsilon"] == 1.0e-6
    assert robust_cudss_options.nvmath_cudss_controls["pivot_epsilon_alg"] == 3
    assert robust_cudss_options.nvmath_cudss_controls["nd_nlevels"] == 2
    assert robust_cudss_options.nvmath_cudss_controls["host_nthreads"] == 4
    assert robust_cudss_options.nvmath_cudss_controls["hybrid_mode"] is True
    assert (
        robust_cudss_options.nvmath_cudss_controls["hybrid_device_memory_limit"] == 20_000_000_000
    )
    assert robust_cudss_options.nvmath_cudss_controls["hybrid_execute_mode"] is False
    assert robust_cudss_options.nvmath_cudss_controls["use_cuda_register_memory"] is True
    assert robust_cudss_options.nvmath_cudss_controls["use_superpanels"] is False
    assert robust_cudss_options.nvmath_cudss_controls["deterministic_mode"] is True


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

    schurdiag_cudss = FEniCSSolverOptions.usfem_schurdiag_cudss_experimental(
        device_ids=(0, 1),
        rtol=1.0e-7,
        max_it=789,
        restart=80,
        velocity_solver="exact",
        schur_drop_rel=1.0e-4,
    )
    assert schurdiag_cudss.linear_backend == "petsc"
    assert schurdiag_cudss.solver_preset == "iterative_schurdiag_cudss_experimental"
    assert schurdiag_cudss.petsc_options == {}
    assert schurdiag_cudss.linear_system_dtype == "float64"
    assert schurdiag_cudss.nvmath_cudss_controls["device_ids"] == (0, 1)
    assert schurdiag_cudss.nvmath_cudss_controls["dtype"] == "float64"
    assert schurdiag_cudss.nvmath_cudss_controls["check_residual"] is False
    assert schurdiag_cudss.iterative_solver_controls == {
        "gmres_rtol": 1.0e-7,
        "gmres_atol": 0.0,
        "gmres_maxiter": 789,
        "gmres_restart": 80,
        "velocity_solver": "exact",
        "schurdiag_drop_rel": 1.0e-4,
        "error_if_not_converged": True,
    }


def test_brinkman_nondimensional_constant_permeability_resolution() -> None:
    assert _constant_permeability_value(np.array([2.0, 2.0, 2.0])) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least one"):
        _constant_permeability_value(np.array([], dtype=float))
    with pytest.raises(ValueError, match="positive and finite"):
        _constant_permeability_value(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="globally constant floored permeability"):
        _constant_permeability_value(np.array([1.0, 2.0]))


def test_brinkman_nondimensional_option_resolution() -> None:
    assert _resolve_brinkman_nondimensionalization(False) is None
    assert isinstance(_resolve_brinkman_nondimensionalization(True), BrinkmanNondimensionalization)
    options = BrinkmanNondimensionalization(velocity_scale="unit_darcy")
    assert _resolve_brinkman_nondimensionalization(options) is options

    with pytest.raises(TypeError, match="nondimensional"):
        _resolve_brinkman_nondimensionalization("yes")  # type: ignore[arg-type]


def test_brinkman_nondimensional_scales_validate_pressure_and_velocity_scale() -> None:
    problem = _constant_problem((2, 2), permeability=2.0)
    context = SimpleNamespace(
        coefficients={"permeability_values": np.array([2.0])},
        domain_length=2.0,
    )

    with pytest.raises(ValueError, match="pressure scale"):
        _brinkman_nondimensional_scales(
            context,
            problem,
            pressure_inlet=0.0,
            pressure_outlet=1.0,
            velocity_scale="viscous",
        )

    context_with_bad_length = SimpleNamespace(
        coefficients={"permeability_values": np.array([2.0])},
        domain_length=-1.0,
    )
    with pytest.raises(ValueError, match="length scale"):
        _brinkman_nondimensional_scales(
            context_with_bad_length,
            problem,
            pressure_inlet=1.0,
            pressure_outlet=0.0,
            velocity_scale="viscous",
        )

    with pytest.raises(ValueError, match="velocity_scale"):
        _brinkman_nondimensional_scales(
            context,
            problem,
            pressure_inlet=1.0,
            pressure_outlet=0.0,
            velocity_scale="bad",  # type: ignore[arg-type]
        )


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


@pytest.mark.parametrize(
    "solver",
    [
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_brinkman_nondimensional_recovers_constant_permeability(
    solver: Callable[..., Any],
) -> None:
    result = solver(
        _constant_problem((3, 3), permeability=2.0),
        flow_axis="x",
        options=FEniCSSolverOptions.scipy_direct(),
        nondimensional=True,
    )

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.metadata["nondimensional"] is True
    assert result.metadata["nondimensional_velocity_scale_type"] == "viscous"
    assert result.metadata["nondimensional_pressure_scale"] == pytest.approx(1.0)
    assert result.metadata["nondimensional_velocity_scale"] == pytest.approx(3.0)
    assert result.metadata["nondimensional_constant_permeability"] is None
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@pytest.mark.parametrize(
    "solver",
    [
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_brinkman_unit_darcy_nondimensional_recovers_constant_permeability(
    solver: Callable[..., Any],
) -> None:
    result = solver(
        _constant_problem((3, 3), permeability=2.0),
        flow_axis="x",
        options=FEniCSSolverOptions.scipy_direct(),
        nondimensional=BrinkmanNondimensionalization(velocity_scale="unit_darcy"),
    )

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.metadata["nondimensional_velocity_scale_type"] == "unit_darcy"
    assert result.metadata["nondimensional_velocity_scale"] == pytest.approx(2.0 / 3.0)
    assert result.metadata["nondimensional_constant_permeability"] == pytest.approx(2.0)
    assert result.metadata["nondimensional_darcy_number"] == pytest.approx(2.0 / 9.0)


@requires_fem_stack
def test_fem_brinkman_nondimensional_recovers_nonunit_pressure_drop() -> None:
    result = solve_brinkman_taylor_hood(
        _constant_problem((3, 3), permeability=2.0),
        flow_axis="x",
        pressure_inlet=2.0,
        pressure_outlet=0.0,
        options=FEniCSSolverOptions.scipy_direct(),
        nondimensional=True,
    )

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.metadata["nondimensional_pressure_scale"] == pytest.approx(2.0)
    assert result.metadata["nondimensional_velocity_scale"] == pytest.approx(6.0)
    assert np.max(np.abs(result.pressure.x.array)) > 1.0


@pytest.mark.parametrize(
    "solver",
    [
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_brinkman_nondimensional_matches_dimensional_heterogeneous_map(
    solver: Callable[..., Any],
) -> None:
    problem = _heterogeneous_problem()
    options = FEniCSSolverOptions.scipy_direct()

    dimensional = solver(problem, flow_axis="x", options=options)
    nondimensional = solver(
        problem,
        flow_axis="x",
        options=options,
        nondimensional=True,
    )

    assert nondimensional.permeability == pytest.approx(dimensional.permeability, rel=1.0e-9)
    assert nondimensional.flow_rate == pytest.approx(dimensional.flow_rate, rel=1.0e-9)
    np.testing.assert_allclose(
        nondimensional.velocity.x.array,
        dimensional.velocity.x.array,
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        nondimensional.pressure.x.array,
        dimensional.pressure.x.array,
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    assert nondimensional.metadata["nondimensional_velocity_scale_type"] == "viscous"


@pytest.mark.parametrize(
    "solver",
    [
        solve_brinkman_taylor_hood,
        solve_brinkman_usfem,
    ],
)
@requires_fem_stack
def test_fem_brinkman_unit_darcy_nondimensional_rejects_heterogeneous_map(
    solver: Callable[..., Any],
) -> None:
    with pytest.raises(ValueError, match="globally constant floored permeability"):
        solver(
            _heterogeneous_problem(),
            flow_axis="x",
            options=FEniCSSolverOptions.scipy_direct(),
            nondimensional=BrinkmanNondimensionalization(velocity_scale="unit_darcy"),
        )


@requires_petsc_fem_stack
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


@requires_petsc_fem_stack
def test_fem_usfem_block_nondimensional_matches_monolithic_constant_3d() -> None:
    problem = _constant_problem((2, 2, 2), permeability=1.5)
    options = FEniCSSolverOptions.direct_parallel(
        "superlu_dist",
        petsc_options_prefix="test_usfem_block_nondim_superlu_",
    )

    monolithic = solve_brinkman_usfem(
        problem,
        flow_axis="x",
        options=FEniCSSolverOptions.direct_parallel(
            "superlu_dist",
            petsc_options_prefix="test_usfem_monolithic_nondim_superlu_",
        ),
        nondimensional=BrinkmanNondimensionalization(velocity_scale="unit_darcy"),
    )
    block = solve_brinkman_usfem_block(
        problem,
        flow_axis="x",
        options=options,
        matrix_kind="mpi",
        preconditioner="none",
        nondimensional=BrinkmanNondimensionalization(velocity_scale="unit_darcy"),
    )

    assert block.permeability == pytest.approx(monolithic.permeability, rel=1.0e-10)
    assert block.permeability == pytest.approx(1.5, rel=5.0e-4)
    assert block.metadata["nondimensional"] is True
    assert block.metadata["nondimensional_velocity_scale_type"] == "unit_darcy"
    assert block.metadata["block_matrix_kind"] == "mpi"


@requires_petsc_fem_stack
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


@requires_petsc_fem_stack
def test_fem_usfem_block_dispatches_schurdiag_cudss_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_solve_usfem_schurdiag_cudss(
        _context: Any,
        **kwargs: Any,
    ) -> tuple[list[Any], float, dict[str, Any]]:
        captured["forms"] = kwargs["forms"]
        captured["rhs"] = kwargs["rhs"]
        captured["bcs"] = kwargs["bcs"]
        captured["options"] = kwargs["options"]
        return kwargs["solution_functions"], 0.0, {"usfem_schurdiag_cudss_gmres_info": 0}

    monkeypatch.setattr(
        usfem_module,
        "_solve_usfem_schurdiag_cudss",
        fake_solve_usfem_schurdiag_cudss,
    )

    options = FEniCSSolverOptions.usfem_schurdiag_cudss_experimental(
        device_ids=(0, 1),
    )
    result = solve_brinkman_usfem_block(
        _constant_problem((2, 2), permeability=2.0),
        options=options,
    )

    assert captured["options"] is options
    assert len(captured["forms"]) == 2
    assert len(captured["rhs"]) == 2
    assert captured["bcs"]
    assert result.metadata["solver_preset"] == "iterative_schurdiag_cudss_experimental"
    assert result.metadata["nvmath_cudss_controls"]["device_ids"] == (0, 1)
    assert result.metadata["usfem_schurdiag_cudss_gmres_info"] == 0


@requires_fem_stack
def test_fem_usfem_block_rejects_external_preconditioner_for_schurdiag_cudss() -> None:
    with pytest.raises(ValueError, match="lower-Schur preconditioner"):
        solve_brinkman_usfem_block(
            _constant_problem((2, 2), permeability=2.0),
            options=FEniCSSolverOptions.usfem_schurdiag_cudss_experimental(),
            preconditioner="diagonal",
        )


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
