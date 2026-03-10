from __future__ import annotations

from typing import Protocol, TypeAlias, cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, gmres, spsolve


class _PyAMGHierarchy(Protocol):
    """Minimal typed surface used from a PyAMG multilevel hierarchy."""

    levels: list[object]

    def aspreconditioner(self) -> LinearOperator[np.float64]:
        """Return a SciPy-compatible preconditioner."""

    def operator_complexity(self) -> float:
        """Return the operator complexity."""


class _PyAMGModule(Protocol):
    """Minimal typed subset of the top-level ``pyamg`` module."""

    def smoothed_aggregation_solver(
        self, matrix: sparse.csr_matrix, **kwargs: object
    ) -> _PyAMGHierarchy:
        """Build a smoothed-aggregation hierarchy."""

    def rootnode_solver(self, matrix: sparse.csr_matrix, **kwargs: object) -> _PyAMGHierarchy:
        """Build a root-node hierarchy."""

    def ruge_stuben_solver(self, matrix: sparse.csr_matrix, **kwargs: object) -> _PyAMGHierarchy:
        """Build a classical AMG hierarchy."""


class _PyPardisoSolver(Protocol):
    """Minimal typed surface for pypardiso.spsolve function."""

    def __call__(
        self,
        A: sparse.csr_matrix | sparse.csc_matrix,
        b: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Solve sparse linear system using PARDISO."""


SolverParameterValue: TypeAlias = (
    str | float | int | bool | dict[str, object] | LinearOperator[np.float64]
)
SolverParameters: TypeAlias = dict[str, SolverParameterValue]


def _import_pyamg() -> _PyAMGModule:
    """Import PyAMG lazily so the dependency remains easy to diagnose."""

    try:
        import pyamg  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyAMG preconditioning requires the 'pyamg' package to be installed."
        ) from exc
    return cast(_PyAMGModule, pyamg)


def _import_pypardiso() -> _PyPardisoSolver:
    """Import pypardiso lazily so the dependency remains easy to diagnose."""

    try:
        import pypardiso  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PARDISO solver requires the 'pypardiso' package to be installed. "
            "This is currently only supported on Linux systems."
        ) from exc
    return cast(_PyPardisoSolver, pypardiso.spsolve)


def _build_preconditioner(
    A: sparse.csr_matrix,
    *,
    solver_parameters: SolverParameters | None,
) -> tuple[LinearOperator[np.float64] | None, dict[str, str | float | int]]:
    """Build an optional Krylov preconditioner from solver parameters."""

    parameters = dict(solver_parameters or {})
    name = parameters.get("preconditioner")
    if name is None:
        return None, {}
    if name != "pyamg":
        raise ValueError(f"Unknown preconditioner '{name}'")

    pyamg = _import_pyamg()
    amg_kind = str(parameters.get("pyamg_solver", "smoothed_aggregation"))
    amg_kwargs = parameters.get("pyamg_kwargs", {})
    if not isinstance(amg_kwargs, dict):
        raise ValueError("pyamg_kwargs must be a dictionary")

    matrix = sparse.csr_matrix(A, dtype=float)
    if amg_kind == "smoothed_aggregation":
        hierarchy = pyamg.smoothed_aggregation_solver(matrix, **amg_kwargs)
    elif amg_kind == "rootnode":
        hierarchy = pyamg.rootnode_solver(matrix, **amg_kwargs)
    elif amg_kind == "ruge_stuben":
        hierarchy = pyamg.ruge_stuben_solver(matrix, **amg_kwargs)
    else:
        raise ValueError(
            f"Unknown pyamg_solver '{amg_kind}'. Expected 'smoothed_aggregation', "
            "'rootnode', or 'ruge_stuben'."
        )
    return (
        hierarchy.aspreconditioner(),
        {
            "preconditioner": "pyamg",
            "pyamg_solver": amg_kind,
            "pyamg_levels": int(len(hierarchy.levels)),
            "pyamg_operator_complexity": float(hierarchy.operator_complexity()),
        },
    )


def solve_linear_system(
    A: sparse.csr_matrix,
    b: np.ndarray,
    *,
    method: str = "direct",
    solver_parameters: SolverParameters | None = None,
) -> tuple[np.ndarray, dict[str, str | float | int]]:
    """Solve a sparse linear system with one of the supported SciPy backends.

    Parameters
    ----------
    A :
        Sparse system matrix.
    b :
        Right-hand-side vector.
    method :
        Solver backend. Supported values are ``"direct"``, ``"pardiso"``,
        ``"cg"``, and ``"gmres"``.
    solver_parameters :
        Optional backend-specific solver options. For SciPy Krylov methods this
        maps directly to supported keyword arguments such as ``rtol``,
        ``atol``, ``restart``, and ``maxiter``. Setting
        ``{"preconditioner": "pyamg"}`` attaches a PyAMG preconditioner to
        ``cg`` or ``gmres``.

    Returns
    -------
    numpy.ndarray
        Solution vector.
    dict[str, Any]
        Solver metadata containing the method name and the iterative solver
        status code ``info``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.

    Notes
    -----
    The ``"pardiso"`` method uses the Intel MKL PARDISO solver via the
    ``pypardiso`` package. This is typically only available on Linux systems
    and may provide better performance for large systems compared to the
    default SciPy direct solver.
    """

    if method == "direct":
        x = spsolve(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": 0}
    if method == "pardiso":
        pardiso_spsolve = _import_pypardiso()
        x = pardiso_spsolve(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": 0}
    if method == "cg":
        parameters = dict(solver_parameters or {})
        preconditioner, preconditioner_info = _build_preconditioner(A, solver_parameters=parameters)
        cg_kwargs = {
            key: parameters[key] for key in ("rtol", "atol", "maxiter", "M") if key in parameters
        }
        if preconditioner is not None and "M" not in cg_kwargs:
            cg_kwargs["M"] = preconditioner
        x, info = cg(A, b, **cg_kwargs)
        return np.asarray(x, dtype=float), {
            "method": method,
            "info": int(info),
            **preconditioner_info,
        }
    if method == "gmres":
        parameters = dict(solver_parameters or {})
        preconditioner, preconditioner_info = _build_preconditioner(A, solver_parameters=parameters)
        gmres_kwargs = {
            key: parameters[key]
            for key in ("rtol", "atol", "restart", "maxiter", "M")
            if key in parameters
        }
        if preconditioner is not None and "M" not in gmres_kwargs:
            gmres_kwargs["M"] = preconditioner
        x, info = gmres(A, b, **gmres_kwargs)
        return np.asarray(x, dtype=float), {
            "method": method,
            "info": int(info),
            **preconditioner_info,
        }
    raise ValueError(f"Unknown solver method '{method}'")
