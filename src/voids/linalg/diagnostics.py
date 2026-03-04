from __future__ import annotations

import numpy as np
from scipy import sparse


def residual_norm(A: sparse.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean norm of the linear-system residual.

    Parameters
    ----------
    A :
        System matrix.
    x :
        Trial or converged solution vector.
    b :
        Right-hand-side vector.

    Returns
    -------
    float
        Value of ``||A x - b||_2``.
    """

    r = A @ x - b
    return float(np.linalg.norm(r))
