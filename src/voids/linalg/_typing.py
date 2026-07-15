from __future__ import annotations

from typing import Literal, Protocol, TypeAlias

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


class PyAMGHierarchy(Protocol):
    """Minimal typed surface used from a PyAMG multilevel hierarchy."""

    levels: list[object]

    def aspreconditioner(self) -> LinearOperator[np.float64]:
        """Return a SciPy-compatible preconditioner."""

    def operator_complexity(self) -> float:
        """Return the operator complexity."""


class PyAMGModule(Protocol):
    """Minimal typed subset of the top-level ``pyamg`` module."""

    def smoothed_aggregation_solver(
        self, matrix: sparse.csr_matrix, **kwargs: object
    ) -> PyAMGHierarchy:
        """Build a smoothed-aggregation hierarchy."""

    def rootnode_solver(self, matrix: sparse.csr_matrix, **kwargs: object) -> PyAMGHierarchy:
        """Build a root-node hierarchy."""

    def ruge_stuben_solver(self, matrix: sparse.csr_matrix, **kwargs: object) -> PyAMGHierarchy:
        """Build a classical AMG hierarchy."""


class PyPardisoSolver(Protocol):
    """Minimal typed surface for the ``pypardiso.spsolve`` function."""

    def __call__(
        self,
        A: sparse.csr_matrix | sparse.csc_matrix,
        b: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Solve a sparse linear system using PARDISO."""


class UmfpackSolver(Protocol):
    """Minimal typed surface for ``scikits.umfpack.spsolve``."""

    def __call__(
        self,
        A: sparse.csr_matrix | sparse.csc_matrix,
        b: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Solve a sparse linear system using UMFPACK."""


SolverParameterValue: TypeAlias = (
    str
    | float
    | int
    | bool
    | None
    | tuple[int, ...]
    | list[int]
    | dict[str, object]
    | LinearOperator[np.float64]
)
SolverParameters: TypeAlias = dict[str, SolverParameterValue]
LinearSystemDType: TypeAlias = Literal["float32", "float64"]
SolverInfo: TypeAlias = dict[str, str | float | int | bool]
