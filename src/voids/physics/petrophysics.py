from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voids.core.network import Network
from voids.graph.connectivity import connected_components, spanning_component_mask
from voids.graph.metrics import ConnectivitySummary, connectivity_metrics as _connectivity_metrics


@dataclass(slots=True)
class PorosityBreakdown:
    """Minimal porosity bookkeeping container.

    Attributes
    ----------
    void_volume :
        Total connected or absolute void volume used in the calculation.
    bulk_volume :
        Reference bulk volume of the sample.
    porosity :
        Ratio ``void_volume / bulk_volume``.
    """

    void_volume: float
    bulk_volume: float
    porosity: float


def _void_volume(net: Network, pore_mask: np.ndarray | None = None) -> float:
    """Compute void volume from pore and throat volumes.

    Parameters
    ----------
    net :
        Network containing ``pore.volume`` and ``throat.volume``.
    pore_mask :
        Optional boolean mask defining the subset of pores to include.

    Returns
    -------
    float
        Total void volume.

    Raises
    ------
    KeyError
        If pore or throat volume data are missing.

    Notes
    -----
    When a pore mask is supplied, a throat contributes only if both of its end pores
    are inside the selected subset. This corresponds to summing the volume of the
    induced subnetwork.
    """

    if "volume" not in net.pore or "volume" not in net.throat:
        raise KeyError("Both pore.volume and throat.volume are required for porosity calculations")
    pv = np.asarray(net.pore["volume"], dtype=float)
    tv = np.asarray(net.throat["volume"], dtype=float)
    if pore_mask is None:
        throat_mask = np.ones(net.Nt, dtype=bool)
        pore_mask_local = np.ones(net.Np, dtype=bool)
    else:
        pore_mask_local = np.asarray(pore_mask, dtype=bool)
        c = net.throat_conns
        throat_mask = pore_mask_local[c[:, 0]] & pore_mask_local[c[:, 1]]
    return float(pv[pore_mask_local].sum() + tv[throat_mask].sum())


def absolute_porosity(net: Network) -> float:
    """Compute absolute porosity of the networked sample.

    Parameters
    ----------
    net :
        Network with pore and throat volumes plus sample bulk-volume metadata.

    Returns
    -------
    float
        Absolute porosity defined as
        ``phi_abs = V_void / V_bulk``.
    """

    return _void_volume(net) / net.sample.resolved_bulk_volume()


def effective_porosity(net: Network, axis: str | None = None, mode: str | None = None) -> float:
    """Compute an effective porosity based on connected void space.

    Parameters
    ----------
    net :
        Network with pore and throat volumes.
    axis :
        Cartesian axis used for spanning connectivity. When provided, only pores in
        components spanning the requested inlet/outlet labels contribute.
    mode :
        Effective-porosity mode used when ``axis`` is omitted. Currently only
        ``"boundary_connected"`` is supported.

    Returns
    -------
    float
        Effective porosity based on the selected connected subset.

    Raises
    ------
    ValueError
        If an unsupported mode is requested.

    Notes
    -----
    Two selection rules are supported:

    - ``axis is not None``: use components spanning inlet/outlet labels along that axis.
    - ``axis is None`` and ``mode == "boundary_connected"``: include any component
      touching a pore label whose name starts with ``inlet`` or ``outlet``, or the
      generic ``boundary`` label.
    """

    _, comp_labels = connected_components(net)
    if axis is not None:
        pore_mask = spanning_component_mask(net, axis=axis, labels=comp_labels)
    else:
        if mode is None:
            mode = "boundary_connected"
        if mode != "boundary_connected":
            raise ValueError(f"Unsupported effective porosity mode '{mode}'")
        boundary_ids: list[int] = []
        for name, mask in net.pore_labels.items():
            lname = name.lower()
            if lname.startswith("inlet") or lname.startswith("outlet") or lname == "boundary":
                boundary_ids.extend(np.unique(comp_labels[np.asarray(mask, dtype=bool)]).tolist())
        pore_mask = (
            np.isin(comp_labels, np.unique(boundary_ids))
            if boundary_ids
            else np.zeros(net.Np, dtype=bool)
        )
    return _void_volume(net, pore_mask=pore_mask) / net.sample.resolved_bulk_volume()


def connectivity_metrics(net: Network) -> ConnectivitySummary:
    """Return graph-level connectivity metrics for a network.

    Parameters
    ----------
    net :
        Network to analyze.

    Returns
    -------
    ConnectivitySummary
        Convenience wrapper around :func:`voids.graph.metrics.connectivity_metrics`.
    """

    return _connectivity_metrics(net)
