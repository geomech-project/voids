from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import voids.linalg.solve as solve_mod
from voids.core import provenance as provenance_mod
from voids.core import sample as sample_mod
from voids._logging import logger
from voids._testing import set_seed
from voids.core.sample import SampleGeometry
from voids.graph.incidence import incidence_matrix
from voids.linalg.backends import SCIPY, SciPyBackend
from voids.linalg.solve import solve_linear_system, _import_pypardiso, _import_umfpack
from voids.paths import (
    DATA_PATH_ENV,
    EXAMPLES_PATH_ENV,
    PROJECT_ROOT_ENV,
    _repo_root_from_source_tree,
    data_path,
    examples_path,
    project_root,
)


def test_logger_uses_package_namespace() -> None:
    """Test that the package logger uses the expected namespace."""

    assert logger.name == "voids"
    assert isinstance(logger, logging.Logger)


def test_set_seed_resets_python_and_numpy_rngs() -> None:
    """Test deterministic reseeding of Python and NumPy random generators."""

    set_seed(123)
    first = (random.random(), float(np.random.random()))

    set_seed(123)
    second = (random.random(), float(np.random.random()))

    assert first == second


def test_metadata_conversion_helpers_cover_scalar_tuple_and_invalid_values() -> None:
    assert provenance_mod._voxel_size_original(None) is None
    assert provenance_mod._voxel_size_original(2) == 2.0
    assert provenance_mod._voxel_size_original([1, "2", 3.0]) == (1.0, 2.0, 3.0)
    assert provenance_mod._voxel_size_original("bad") is None
    with pytest.raises(TypeError, match="Expected numeric metadata"):
        provenance_mod._float_value({})

    assert sample_mod._scalar_or_tuple(None) is None
    assert sample_mod._scalar_or_tuple("2") == 2.0
    assert sample_mod._scalar_or_tuple([1, "2", 3.0]) == (1.0, 2.0, 3.0)
    assert sample_mod._scalar_or_tuple([1, 2]) is None
    with pytest.raises(TypeError, match="Expected numeric metadata"):
        sample_mod._float_value({})


def test_sparse_solver_control_helper_validation_and_forwarding() -> None:
    with pytest.raises(ValueError, match="dtype"):
        solve_mod._resolve_linear_system_dtype(1)
    with pytest.raises(ValueError, match="dtype"):
        solve_mod._resolve_linear_system_dtype("not-a-dtype")
    assert solve_mod._superlu_kwargs(
        {
            "permc_spec": "COLAMD",
            "diag_pivot_thresh": 0.5,
            "relax": 4,
            "panel_size": 8,
            "equil": False,
        }
    ) == {
        "permc_spec": "COLAMD",
        "diag_pivot_thresh": 0.5,
        "relax": 4,
        "panel_size": 8,
        "options": {"Equil": False},
    }


def test_sample_geometry_resolves_tuple_voxel_volume() -> None:
    """Test bulk-volume recovery from anisotropic voxel geometry."""

    sample = SampleGeometry(voxel_size=(1.5, 2.0, 3.0), bulk_shape_voxels=(2, 3, 4))

    assert sample.resolved_bulk_volume() == pytest.approx(216.0)


def test_sample_geometry_axis_lookups_raise_for_missing_entries() -> None:
    """Test axis lookup failures when sample metadata is incomplete."""

    sample = SampleGeometry(bulk_volume=1.0)

    with pytest.raises(KeyError, match="Missing sample length"):
        sample.length_for_axis("x")
    with pytest.raises(KeyError, match="Missing sample cross-section"):
        sample.area_for_axis("x")


def test_network_missing_field_helpers_raise(line_network) -> None:
    """Test pore/throat array access helpers and their error paths."""

    assert np.array_equal(line_network.get_pore_array("volume"), line_network.pore["volume"])
    assert np.array_equal(line_network.get_throat_array("length"), line_network.throat["length"])

    with pytest.raises(KeyError, match="Missing pore field"):
        line_network.get_pore_array("missing")
    with pytest.raises(KeyError, match="Missing throat field"):
        line_network.get_throat_array("missing")


def test_sample_geometry_resolves_scalar_voxel_volume() -> None:
    """Test bulk-volume recovery from isotropic voxel geometry."""

    sample = SampleGeometry(voxel_size=2.0, bulk_shape_voxels=(2, 3, 4))

    assert sample.resolved_bulk_volume() == pytest.approx(192.0)


def test_incidence_matrix_sign_convention(line_network) -> None:
    """Test the orientation sign convention of the incidence matrix."""

    incidence = incidence_matrix(line_network).toarray()

    assert incidence.tolist() == [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]


def test_scipy_backend_exports_expected_callables() -> None:
    """Test that the SciPy linear-algebra backend exposes the expected callables."""

    assert isinstance(SCIPY, SciPyBackend)
    assert SCIPY.coo_matrix is not None
    assert SCIPY.csr_matrix is not None
    assert SCIPY.spsolve is not None
    assert SCIPY.cg is not None
    assert SCIPY.gmres is not None


@pytest.mark.parametrize("method", ["direct", "superlu", "cg", "gmres"])
def test_solve_linear_system_supports_all_methods(method: str) -> None:
    """Test all supported linear-solver backends on an identity system."""

    A = sparse.csr_matrix(np.eye(2))
    b = np.array([1.0, -2.0])

    x, info = solve_linear_system(A, b, method=method)

    assert np.allclose(x, b)
    assert info["method"] == method
    assert info["info"] == 0
    assert info["linear_system_dtype"] == "float64"


@pytest.mark.parametrize("method", ["direct", "superlu", "cg", "gmres"])
def test_solve_linear_system_supports_cpu_single_precision(method: str) -> None:
    """Portable CPU sparse backends can factor or iterate on float32 values."""

    A = sparse.csr_matrix(np.array([[3.0, -1.0], [-1.0, 3.0]], dtype=np.float64))
    b = np.array([2.0, 4.0], dtype=np.float64)

    x, info = solve_linear_system(
        A,
        b,
        method=method,
        solver_parameters={"dtype": "float32"},
    )

    assert x.dtype == np.dtype("float32")
    assert np.allclose(A @ x, b, rtol=5.0e-6, atol=5.0e-7)
    assert info["method"] == method
    assert info["info"] == 0
    assert info["linear_system_dtype"] == "float32"


def test_solve_linear_system_rejects_unsupported_dtype() -> None:
    """Precision selection is intentionally limited to real single/double values."""

    with pytest.raises(ValueError, match="dtype must be 'float32' or 'float64'"):
        solve_linear_system(
            sparse.csr_matrix(np.eye(2)),
            np.array([1.0, 2.0]),
            solver_parameters={"dtype": "float16"},
        )


def test_solve_linear_system_umfpack_available_or_raises_import_error() -> None:
    """UMFPACK is exposed as an explicit reusable sparse direct backend."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    try:
        x, info = solve_linear_system(A, b, method="umfpack")
    except ImportError as exc:
        assert "umfpack" in str(exc).lower()
        assert "scikit-umfpack" in str(exc).lower()
        pytest.skip("UMFPACK solver not available in this environment")

    assert np.allclose(A @ x, b)
    assert info["method"] == "umfpack"
    assert info["backend"] == "scikits.umfpack.spsolve"
    assert info["info"] == 0


def test_solve_linear_system_rejects_unknown_method() -> None:
    """Test rejection of unsupported linear-solver names."""

    with pytest.raises(ValueError, match="Unknown solver method"):
        solve_linear_system(sparse.csr_matrix(np.eye(1)), np.array([1.0]), method="bicgstab")


def test_solve_linear_system_supports_pyamg_preconditioning() -> None:
    """PyAMG can be attached as a preconditioner to Krylov solves."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    x, info = solve_linear_system(
        A,
        b,
        method="cg",
        solver_parameters={"preconditioner": "pyamg"},
    )

    assert np.allclose(A @ x, b)
    assert info["method"] == "cg"
    assert info["preconditioner"] == "pyamg"
    assert info["pyamg_solver"] == "smoothed_aggregation"
    assert info["pyamg_levels"] >= 1


def test_solve_linear_system_rejects_unknown_preconditioner() -> None:
    """Unsupported preconditioner names are rejected explicitly."""

    with pytest.raises(ValueError, match="Unknown preconditioner"):
        solve_linear_system(
            sparse.csr_matrix(np.eye(2)),
            np.array([1.0, 2.0]),
            method="cg",
            solver_parameters={"preconditioner": "ilu"},
        )


def test_solve_linear_system_rejects_invalid_pyamg_kwargs() -> None:
    """PyAMG keyword arguments must be passed as a dictionary."""

    with pytest.raises(ValueError, match="pyamg_kwargs must be a dictionary"):
        solve_linear_system(
            sparse.csr_matrix(np.eye(2)),
            np.array([1.0, 2.0]),
            method="cg",
            solver_parameters={"preconditioner": "pyamg", "pyamg_kwargs": "invalid"},
        )


def test_solve_linear_system_rejects_unknown_pyamg_solver() -> None:
    """Unsupported PyAMG hierarchy builders are rejected explicitly."""

    with pytest.raises(ValueError, match="Unknown pyamg_solver"):
        solve_linear_system(
            sparse.csr_matrix(np.eye(2)),
            np.array([1.0, 2.0]),
            method="cg",
            solver_parameters={"preconditioner": "pyamg", "pyamg_solver": "unsupported"},
        )


def test_solve_linear_system_supports_gmres_with_pyamg_preconditioning() -> None:
    """GMRES also accepts the PyAMG preconditioner path."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    x, info = solve_linear_system(
        A,
        b,
        method="gmres",
        solver_parameters={"preconditioner": "pyamg"},
    )

    assert np.allclose(A @ x, b)
    assert info["method"] == "gmres"
    assert info["preconditioner"] == "pyamg"


@pytest.mark.parametrize("amg_solver", ["rootnode", "ruge_stuben"])
def test_solve_linear_system_supports_multiple_pyamg_hierarchies(amg_solver: str) -> None:
    """PyAMG preconditioning supports alternate hierarchy builders."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    x, info = solve_linear_system(
        A,
        b,
        method="cg",
        solver_parameters={"preconditioner": "pyamg", "pyamg_solver": amg_solver},
    )

    assert np.allclose(A @ x, b)
    assert info["pyamg_solver"] == amg_solver


def test_solve_linear_system_pardiso_available_or_raises_import_error() -> None:
    """Test that PARDISO solver either works or raises helpful ImportError."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    try:
        x, info = solve_linear_system(A, b, method="pardiso")
        # If pypardiso is available (Linux), verify it works correctly
        assert np.allclose(A @ x, b)
        assert info["method"] == "pardiso"
        assert info["info"] == 0
    except ImportError as exc:
        # Expected on non-Linux platforms or when pypardiso is not installed
        assert "pypardiso" in str(exc).lower()
        assert "linux" in str(exc).lower()


def test_import_pypardiso_raises_clear_error_when_unavailable() -> None:
    """Test that _import_pypardiso raises a clear ImportError when pypardiso is missing."""

    try:
        solver = _import_pypardiso()
        # If we get here, pypardiso is available (Linux)
        assert callable(solver)
    except ImportError as exc:
        # Expected on non-Linux platforms
        error_msg = str(exc).lower()
        assert "pypardiso" in error_msg
        assert "linux" in error_msg


def test_import_umfpack_raises_clear_error_when_unavailable() -> None:
    """Missing scikit-umfpack produces a backend-specific diagnostic."""

    try:
        solver = _import_umfpack()
    except ImportError as exc:
        error_msg = str(exc).lower()
        assert "umfpack" in error_msg
        assert "scikit-umfpack" in error_msg
    else:
        assert callable(solver)


def test_solve_linear_system_pardiso_produces_same_result_as_direct() -> None:
    """Test that PARDISO produces identical results to the default direct solver."""

    # Create a simple SPD system
    A = sparse.csr_matrix(np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]]))
    b = np.array([1.0, 2.0, 3.0])

    # Solve with direct
    x_direct, _ = solve_linear_system(A, b, method="direct")

    # Try PARDISO
    try:
        x_pardiso, info = solve_linear_system(A, b, method="pardiso")
        # Results should match to machine precision
        assert np.allclose(x_pardiso, x_direct, rtol=1e-12, atol=1e-14)
        assert info["method"] == "pardiso"
    except ImportError:
        # Expected on non-Linux platforms
        pytest.skip("PARDISO solver not available on this platform")


def test_solve_linear_system_pardiso_success_metadata_with_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PARDISO success metadata is stable when the optional backend is present."""

    def fake_pardiso_spsolve(A: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
        return np.asarray(sparse.linalg.spsolve(A, b), dtype=float)

    monkeypatch.setattr(solve_mod, "_import_pypardiso", lambda: fake_pardiso_spsolve)

    A = sparse.csr_matrix(np.array([[3.0, -1.0], [-1.0, 3.0]]))
    b = np.array([2.0, 4.0])

    x, info = solve_linear_system(A, b, method="pardiso")

    assert np.allclose(A @ x, b)
    assert info == {
        "method": "pardiso",
        "backend": "pypardiso",
        "info": 0,
        "linear_system_dtype": "float64",
    }


@pytest.mark.parametrize("method", ["umfpack", "pardiso"])
def test_solve_linear_system_rejects_float32_for_double_only_wrappers(method: str) -> None:
    """Some optional CPU wrappers expose only double precision at runtime."""

    with pytest.raises(ValueError, match="supports float64 only"):
        solve_linear_system(
            sparse.csr_matrix(np.eye(2)),
            np.array([1.0, 2.0]),
            method=method,
            solver_parameters={"dtype": "float32"},
        )


def test_solve_linear_system_nvmath_cudss_dispatches_shared_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA direct backend is exposed through the shared sparse solver API."""

    calls: dict[str, object] = {}

    def fake_solve_nvmath_cudss(
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        controls: object = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        calls["matrix_format"] = matrix.getformat()
        calls["rhs"] = rhs.copy()
        calls["controls"] = dict(controls or {})
        return np.asarray(sparse.linalg.spsolve(matrix, rhs), dtype=float), {
            "serial_sparse_nvmath_cudss_relative_residual": 0.0,
        }

    monkeypatch.setattr(solve_mod, "solve_nvmath_cudss", fake_solve_nvmath_cudss)

    A = sparse.csr_matrix(np.array([[3.0, -1.0], [-1.0, 3.0]]))
    b = np.array([2.0, 4.0])
    x, info = solve_linear_system(
        A,
        b,
        method="nvmath_cudss",
        solver_parameters={"device_ids": (0,), "dtype": "float64"},
    )

    assert np.allclose(A @ x, b)
    assert calls["matrix_format"] == "csr"
    assert np.array_equal(calls["rhs"], b)
    assert calls["controls"] == {"device_ids": (0,), "dtype": "float64"}
    assert info["method"] == "nvmath_cudss"
    assert info["backend"] == "nvmath.bindings.cudss"
    assert info["info"] == 0
    assert info["serial_sparse_nvmath_cudss_relative_residual"] == 0.0


def test_solve_linear_system_umfpack_produces_same_result_as_direct() -> None:
    """UMFPACK matches the default direct sparse solve on a small SPD system."""

    A = sparse.csr_matrix(np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]]))
    b = np.array([1.0, 2.0, 3.0])
    x_direct, _ = solve_linear_system(A, b, method="direct")

    try:
        x_umfpack, info = solve_linear_system(A, b, method="umfpack")
    except ImportError:
        pytest.skip("UMFPACK solver not available in this environment")

    assert np.allclose(x_umfpack, x_direct, rtol=1.0e-12, atol=1.0e-14)
    assert info["method"] == "umfpack"


def test_project_and_examples_paths_use_env_overrides(monkeypatch, tmp_path: Path) -> None:
    """Test environment-variable overrides for project and examples paths."""

    root = tmp_path / "root"
    examples = tmp_path / "examples"
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(root))
    monkeypatch.setenv(EXAMPLES_PATH_ENV, str(examples))

    assert project_root() == root.resolve()
    assert examples_path() == examples.resolve()


def test_repo_root_from_source_tree_raises_when_layout_is_not_repo(
    monkeypatch, tmp_path: Path
) -> None:
    """Test source-tree root detection failure outside the expected repo layout."""

    fake_module_path = tmp_path / "site-packages" / "voids" / "paths.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr("voids.paths.Path", lambda *_args, **_kwargs: fake_module_path)

    with pytest.raises(RuntimeError, match="Could not resolve the project paths"):
        _repo_root_from_source_tree()


def test_data_path_fallback_is_repo_relative_when_env_missing(monkeypatch) -> None:
    """Test default example-data path resolution without environment overrides."""

    monkeypatch.delenv(DATA_PATH_ENV, raising=False)
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)

    resolved = data_path()

    assert resolved.name == "data"
    assert resolved.is_dir()
