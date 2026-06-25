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
    solve_darcy_taylor_hood,
    upscale_permeability_fem,
    upscale_principal_permeabilities_fem,
)
from voids.fem.singlephase._common import (  # noqa: E402
    _FEM_THREAD_ENV_DEFAULTS,
    _apply_fem_thread_defaults,
    _require_dolfinx_core,
)
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
    assert FEniCSSolverOptions.umfpack_direct().linear_backend == "umfpack"


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
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))


@requires_fem_stack
def test_fem_darcy_backend_recovers_constant_2d_permeability_with_umfpack_direct() -> None:
    result = solve_darcy_taylor_hood(
        _constant_problem((3, 3), permeability=2.0),
        flow_axis="x",
        options=FEniCSSolverOptions.umfpack_direct(),
    )

    assert result.permeability == pytest.approx(2.0, rel=5.0e-4)
    assert result.metadata["linear_backend"] == "umfpack"
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
