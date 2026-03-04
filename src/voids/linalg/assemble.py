from __future__ import annotations

import numpy as np
from scipy import sparse

from voids.core.network import Network


def assemble_pressure_system(net: Network, throat_conductance: np.ndarray) -> sparse.csr_matrix:
    """Assemble the pore-pressure matrix for steady single-phase flow.

    Parameters
    ----------
    net :
        Network defining the pore-throat topology.
    throat_conductance :
        Conductance array with shape ``(Nt,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric matrix ``A`` with shape ``(Np, Np)``.

    Raises
    ------
    ValueError
        If the conductance array has the wrong shape or contains negative
        entries.

    Notes
    -----
    The assembled matrix is the conductance-weighted graph Laplacian. For a
    throat with conductance ``g_t`` connecting pores ``i`` and ``j``, the local
    contribution is

    ``A[i, i] += g_t``
    ``A[j, j] += g_t``
    ``A[i, j] -= g_t``
    ``A[j, i] -= g_t``
    """

    g = np.asarray(throat_conductance, dtype=float)
    if g.shape != (net.Nt,):
        raise ValueError("throat_conductance must have shape (Nt,)")
    if (g < 0).any():
        raise ValueError("throat_conductance must be nonnegative")
    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([-g, -g])
    diag = np.zeros(net.Np, dtype=float)
    np.add.at(diag, i, g)
    np.add.at(diag, j, g)
    rows = np.concatenate([rows, np.arange(net.Np)])
    cols = np.concatenate([cols, np.arange(net.Np)])
    data = np.concatenate([data, diag])
    return sparse.coo_matrix((data, (rows, cols)), shape=(net.Np, net.Np)).tocsr()
