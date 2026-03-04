from __future__ import annotations

import numpy as np
from scipy import sparse

from voids.core.network import Network


def incidence_matrix(net: Network) -> sparse.csr_matrix:
    """Build the throat-to-pore incidence matrix.

    Parameters
    ----------
    net :
        Network whose incidence structure is requested.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix ``B`` with shape ``(Nt, Np)``. For each throat ``t``
        connecting pores ``i`` and ``j``, row ``t`` stores ``+1`` at one
        endpoint and ``-1`` at the other.

    Notes
    -----
    The orientation is arbitrary but fixed by the ordering in
    ``net.throat_conns``. This is sufficient to define discrete pressure
    differences and fluxes consistently.
    """

    rows = np.repeat(np.arange(net.Nt), 2)
    cols = net.throat_conns.reshape(-1)
    data = np.tile(np.array([1.0, -1.0]), net.Nt)
    return sparse.coo_matrix((data, (rows, cols)), shape=(net.Nt, net.Np)).tocsr()
