from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voids.core.network import Network
from voids.graph.connectivity import connected_components, spanning_component_ids


@dataclass(slots=True)
class ConnectivitySummary:
    """Summarize graph-level connectivity statistics for a pore network.

    Attributes
    ----------
    n_components :
        Number of connected components in the pore graph.
    giant_component_fraction :
        Fraction of pores belonging to the largest connected component.
    isolated_pore_fraction :
        Fraction of pores with coordination number zero.
    dead_end_fraction :
        Fraction of pores with coordination number one.
    mean_coordination :
        Mean pore coordination number.
    coordination_histogram :
        Histogram mapping coordination number to pore count.
    spans :
        Dictionary indicating whether any component spans each principal axis.
    """

    n_components: int
    giant_component_fraction: float
    isolated_pore_fraction: float
    dead_end_fraction: float
    mean_coordination: float
    coordination_histogram: dict[int, int]
    spans: dict[str, bool]


def coordination_numbers(net: Network) -> np.ndarray:
    """Compute pore coordination numbers.

    Parameters
    ----------
    net :
        Network whose pore degrees are requested.

    Returns
    -------
    numpy.ndarray
        Integer array with shape ``(Np,)`` where each entry equals the number of
        throats incident on the corresponding pore.
    """

    deg = np.zeros(net.Np, dtype=np.int64)
    np.add.at(deg, net.throat_conns[:, 0], 1)
    np.add.at(deg, net.throat_conns[:, 1], 1)
    return deg


def connectivity_metrics(net: Network) -> ConnectivitySummary:
    """Compute a compact set of connectivity diagnostics for a pore graph.

    Parameters
    ----------
    net :
        Network to analyze.

    Returns
    -------
    ConnectivitySummary
        Aggregate summary of component counts, degree statistics, and axis
        spanning information.
    """

    n_comp, labels = connected_components(net)
    counts = np.bincount(labels, minlength=n_comp)
    deg = coordination_numbers(net)
    hist_keys, hist_counts = np.unique(deg, return_counts=True)
    spans: dict[str, bool] = {}
    for ax in ("x", "y", "z"):
        try:
            spans[ax] = spanning_component_ids(net, ax, labels=labels).size > 0
        except KeyError:
            continue
    return ConnectivitySummary(
        n_components=n_comp,
        giant_component_fraction=float(counts.max() / net.Np if net.Np else 0.0),
        isolated_pore_fraction=float(np.mean(deg == 0)),
        dead_end_fraction=float(np.mean(deg == 1)),
        mean_coordination=float(np.mean(deg) if deg.size else 0.0),
        coordination_histogram={int(k): int(v) for k, v in zip(hist_keys, hist_counts)},
        spans=spans,
    )
