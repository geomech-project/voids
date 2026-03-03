from __future__ import annotations

from typing import Any

import numpy as np

from ..core.network import Network
from .porespy import from_porespy


def to_openpnm_dict(net: Network, *, include_extra: bool = False) -> dict[str, Any]:
    """Export a :class:`voids.core.network.Network` to an OpenPNM/PoreSpy-style dict.

    This is intended for lightweight interoperability and cross-check workflows.
    It preserves the canonical keys used by :func:`voids.io.porespy.from_porespy`.
    """
    out: dict[str, Any] = {
        "pore.coords": np.asarray(net.pore_coords, dtype=float).copy(),
        "throat.conns": np.asarray(net.throat_conns, dtype=int).copy(),
    }
    for k, v in net.pore.items():
        out[f"pore.{k}"] = np.asarray(v).copy()
    for k, v in net.throat.items():
        if k in {"pore1_length", "core_length", "pore2_length"}:
            # emit OpenPNM conduit aliases in addition to canonical keys for convenience
            alias_map = {
                "pore1_length": "throat.conduit_lengths.pore1",
                "core_length": "throat.conduit_lengths.throat",
                "pore2_length": "throat.conduit_lengths.pore2",
            }
            out[alias_map[k]] = np.asarray(v).copy()
        out[f"throat.{k}"] = np.asarray(v).copy()
    for k, v in net.pore_labels.items():
        out[f"pore.{k}"] = np.asarray(v, dtype=bool).copy()
    for k, v in net.throat_labels.items():
        out[f"throat.{k}"] = np.asarray(v, dtype=bool).copy()
    if include_extra:
        out.update(net.extra)
    return out


__all__ = ["to_openpnm_dict", "from_porespy"]
