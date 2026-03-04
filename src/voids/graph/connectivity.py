from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components as _cc

from voids.core.network import Network


def adjacency_matrix(net: Network) -> sparse.csr_matrix:
    """Build the undirected pore adjacency matrix.

    Parameters
    ----------
    net :
        Network whose pore connectivity is to be represented.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse symmetric matrix ``A`` with ``A[i, j] = 1`` when pores ``i`` and
        ``j`` are connected by at least one throat.
    """

    i = net.throat_conns[:, 0]
    j = net.throat_conns[:, 1]
    data = np.ones(net.Nt, dtype=float)
    A = sparse.coo_matrix((data, (i, j)), shape=(net.Np, net.Np))
    return (A + A.T).tocsr()


def connected_components(net: Network) -> tuple[int, np.ndarray]:
    """Compute connected components of the pore graph.

    Parameters
    ----------
    net :
        Network whose pore graph is analyzed.

    Returns
    -------
    int
        Number of connected components.
    numpy.ndarray
        Integer component labels with shape ``(Np,)``.
    """

    A = adjacency_matrix(net)
    n, labels = _cc(A, directed=False, return_labels=True)
    return int(n), labels.astype(np.int64)


def _axis_boundary_labels(axis: str) -> tuple[str, str]:
    """Return canonical inlet and outlet label names for one axis.

    Parameters
    ----------
    axis :
        Axis identifier.

    Returns
    -------
    tuple[str, str]
        Pair ``(inlet_label, outlet_label)``.

    Raises
    ------
    ValueError
        If the axis is not one of ``"x"``, ``"y"``, or ``"z"``.
    """

    amap = {
        "x": ("inlet_xmin", "outlet_xmax"),
        "y": ("inlet_ymin", "outlet_ymax"),
        "z": ("inlet_zmin", "outlet_zmax"),
    }
    if axis not in amap:
        raise ValueError(f"Unsupported axis '{axis}'")
    return amap[axis]


def spanning_component_ids(net: Network, axis: str, labels: np.ndarray | None = None) -> np.ndarray:
    """Return component identifiers that span a given sample axis.

    Parameters
    ----------
    net :
        Network to analyze.
    axis :
        Axis whose inlet and outlet boundaries define the spanning criterion.
    labels :
        Optional precomputed connected-component labels.

    Returns
    -------
    numpy.ndarray
        Sorted array of component identifiers touching both the inlet and outlet
        boundary sets for the requested axis.

    Raises
    ------
    KeyError
        If the required inlet or outlet labels are missing.
    """

    if labels is None:
        _, labels = connected_components(net)
    inlet_name, outlet_name = _axis_boundary_labels(axis)
    if inlet_name not in net.pore_labels or outlet_name not in net.pore_labels:
        raise KeyError(f"Missing pore labels '{inlet_name}'/'{outlet_name}'")
    inlet_mask = net.pore_labels[inlet_name]
    outlet_mask = net.pore_labels[outlet_name]
    inlet_ids = np.unique(labels[inlet_mask])
    outlet_ids = np.unique(labels[outlet_mask])
    return np.intersect1d(inlet_ids, outlet_ids)


def spanning_component_mask(
    net: Network, axis: str, labels: np.ndarray | None = None
) -> np.ndarray:
    """Return a pore mask selecting axis-spanning connected components.

    Parameters
    ----------
    net :
        Network to analyze.
    axis :
        Axis whose boundary labels define the spanning criterion.
    labels :
        Optional precomputed connected-component labels.

    Returns
    -------
    numpy.ndarray
        Boolean array with shape ``(Np,)`` selecting pores that belong to any
        spanning component.
    """

    if labels is None:
        _, labels = connected_components(net)
    comp_ids = spanning_component_ids(net, axis=axis, labels=labels)
    return np.isin(labels, comp_ids)
