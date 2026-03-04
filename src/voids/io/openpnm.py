from __future__ import annotations

from typing import Any

import numpy as np

from voids.core.network import Network
from voids.io.porespy import from_porespy


def to_openpnm_dict(net: Network, *, include_extra: bool = False) -> dict[str, Any]:
    """Export a network to an OpenPNM/PoreSpy-style flat dictionary.

    Parameters
    ----------
    net :
        Network to export.
    include_extra :
        If ``True``, merge ``net.extra`` into the output dictionary.

    Returns
    -------
    dict of str to Any
        Flat mapping using keys such as ``"pore.coords"`` and ``"throat.conns"``.

    Notes
    -----
    The mapping preserves aliases expected by :func:`voids.io.porespy.from_porespy`.
    Conduit-length fields are emitted both under their internal names
    (for round-tripping within ``voids``) and under OpenPNM-style names
    such as ``throat.conduit_lengths.pore1``.
    """

    out: dict[str, Any] = {
        "pore.coords": np.asarray(net.pore_coords, dtype=float).copy(),
        "throat.conns": np.asarray(net.throat_conns, dtype=int).copy(),
    }
    for k, v in net.pore.items():
        out[f"pore.{k}"] = np.asarray(v).copy()
    for k, v in net.throat.items():
        if k in {"pore1_length", "core_length", "pore2_length"}:
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


def to_openpnm_network(
    net: Network,
    *,
    copy_properties: bool = True,
    copy_labels: bool = True,
    include_extra: bool = False,
):
    """Convert a :class:`Network` into an OpenPNM network object.

    Parameters
    ----------
    net :
        Network to convert.
    copy_properties :
        If ``True``, copy numeric pore and throat properties into the OpenPNM object.
    copy_labels :
        If ``True``, copy pore and throat boolean labels.
    include_extra :
        If ``True``, attempt to copy entries from ``net.extra`` whose keys already
        follow the ``pore.*`` or ``throat.*`` naming convention.

    Returns
    -------
    Any
        OpenPNM network object. The precise class depends on the installed OpenPNM version.

    Raises
    ------
    ImportError
        If OpenPNM is not installed.
    RuntimeError
        If a compatible OpenPNM network object cannot be instantiated.

    Notes
    -----
    OpenPNM's constructor signatures vary across versions. This helper tries a small
    set of known call patterns and always assigns ``pore.coords`` and
    ``throat.conns`` explicitly afterward so that topology transfer is version-robust.
    """

    try:
        import openpnm as op
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("OpenPNM is not installed") from exc

    coords = np.asarray(net.pore_coords, dtype=float)
    conns = np.asarray(net.throat_conns, dtype=int)

    pn = None
    errs: list[Exception] = []
    for ctor in (
        lambda: op.network.Network(coords=coords, conns=conns),
        lambda: op.network.Network(conns=conns, coords=coords),
        lambda: op.network.Network(),
    ):
        try:
            pn = ctor()
            break
        except Exception as e:  # pragma: no cover - depends on OpenPNM version
            errs.append(e)
    if pn is None:  # pragma: no cover
        raise RuntimeError(f"Unable to instantiate OpenPNM Network: {errs!r}")

    pn["pore.coords"] = coords
    pn["throat.conns"] = conns

    if copy_properties:
        for k, v in net.pore.items():
            pn[f"pore.{k}"] = np.asarray(v)
        for k, v in net.throat.items():
            pn[f"throat.{k}"] = np.asarray(v)
    if copy_labels:
        for k, v in net.pore_labels.items():
            pn[f"pore.{k}"] = np.asarray(v, dtype=bool)
        for k, v in net.throat_labels.items():
            pn[f"throat.{k}"] = np.asarray(v, dtype=bool)
    if include_extra:
        for k, v in net.extra.items():
            if isinstance(k, str) and (k.startswith("pore.") or k.startswith("throat.")):
                try:
                    pn[k] = np.asarray(v)
                except Exception:
                    pass
    return pn


__all__ = ["to_openpnm_dict", "to_openpnm_network", "from_porespy"]
