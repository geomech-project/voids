from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from voids._logging import logger
from voids._testing import set_seed
from voids.core.sample import SampleGeometry
from voids.graph.incidence import incidence_matrix
from voids.linalg.backends import SCIPY, SciPyBackend
from voids.linalg.solve import solve_linear_system
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


@pytest.mark.parametrize("method", ["direct", "cg", "gmres"])
def test_solve_linear_system_supports_all_methods(method: str) -> None:
    """Test all supported linear-solver backends on an identity system."""

    A = sparse.csr_matrix(np.eye(2))
    b = np.array([1.0, -2.0])

    x, info = solve_linear_system(A, b, method=method)

    assert np.allclose(x, b)
    assert info["method"] == method
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

    resolved = data_path()

    assert resolved.name == "data"
    assert resolved.is_dir()
