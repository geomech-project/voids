from __future__ import annotations

import numpy as np
from scipy import sparse


def apply_dirichlet_rowcol(
    A: sparse.csr_matrix, b: np.ndarray, values: np.ndarray, mask: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Apply Dirichlet conditions by row and column elimination.

    Parameters
    ----------
    A :
        System matrix with shape ``(N, N)``.
    b :
        Right-hand-side vector with shape ``(N,)``.
    values :
        Full-length vector of prescribed values. Only entries selected by
        ``mask`` are enforced.
    mask :
        Boolean array selecting the Dirichlet degrees of freedom.

    Returns
    -------
    scipy.sparse.csr_matrix
        Modified system matrix in CSR format.
    numpy.ndarray
        Modified right-hand-side vector.

    Raises
    ------
    ValueError
        If ``values``, ``mask``, and ``b`` do not have the same shape.

    Notes
    -----
    For each constrained degree of freedom ``k``, the routine enforces

    ``A[k, :] = 0``
    ``A[:, k] = 0``
    ``A[k, k] = 1``
    ``b[k] = values[k]``

    after first subtracting the eliminated column contribution from the
    unconstrained rows of ``b``.
    """

    A = A.tolil(copy=True)
    b2 = np.asarray(b, dtype=float).copy()
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != b2.shape or mask.shape != b2.shape:
        raise ValueError("values, mask and b must have the same shape")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return A.tocsr(), b2

    A_csr = A.tocsr()
    b2 = b2 - A_csr[:, idx] @ values[idx]

    for k in idx:
        A[:, k] = 0.0
        A[k, :] = 0.0
        A[k, k] = 1.0
        b2[k] = values[k]
    return A.tocsr(), b2
