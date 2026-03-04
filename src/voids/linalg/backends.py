from __future__ import annotations

from dataclasses import dataclass

from scipy import sparse
from scipy.sparse.linalg import cg, gmres, spsolve


@dataclass(frozen=True, slots=True)
class SciPyBackend:
    """Namespace collecting SciPy sparse constructors and solvers.

    Attributes
    ----------
    coo_matrix, csr_matrix :
        Sparse matrix constructors.
    spsolve, cg, gmres :
        Direct and iterative sparse linear solvers used by the package.
    """

    coo_matrix = staticmethod(sparse.coo_matrix)
    csr_matrix = staticmethod(sparse.csr_matrix)
    spsolve = staticmethod(spsolve)
    cg = staticmethod(cg)
    gmres = staticmethod(gmres)


SCIPY = SciPyBackend()
