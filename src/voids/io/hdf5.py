from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


def _write_json_attr(obj: h5py.Group, name: str, value: Any) -> None:
    """Write a JSON-serializable value into an HDF5 attribute.

    Parameters
    ----------
    obj :
        HDF5 group or dataset receiving the attribute.
    name :
        Attribute name.
    value :
        JSON-serializable payload.
    """

    obj.attrs[name] = json.dumps(value)


def _read_json_attr(obj: h5py.Group, name: str, default: Any = None) -> Any:
    """Read a JSON-serialized HDF5 attribute.

    Parameters
    ----------
    obj :
        HDF5 group or dataset containing the attribute.
    name :
        Attribute name.
    default :
        Value returned when the attribute is absent.

    Returns
    -------
    Any
        Decoded JSON payload or ``default`` when the attribute does not exist.
    """

    if name not in obj.attrs:
        return default
    raw = obj.attrs[name]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def save_hdf5(net: Network, path: str | Path) -> None:
    """Serialize a network to the project HDF5 interchange format.

    Parameters
    ----------
    net :
        Network to store.
    path :
        Destination file path. Parent directories must already exist.

    Notes
    -----
    The file layout is intentionally explicit:

    - ``/meta`` stores schema and provenance metadata.
    - ``/sample`` stores the sample geometry payload.
    - ``/network/pore`` and ``/network/throat`` store arrays.
    - ``/labels`` stores boolean pore and throat labels as ``uint8`` datasets.
    - ``/`` attribute ``extra`` stores JSON-compatible auxiliary metadata.
    """

    path = Path(path)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("schema_version", data=np.bytes_(net.schema_version))
        _write_json_attr(meta, "provenance", net.provenance.to_metadata())

        sample = f.create_group("sample")
        _write_json_attr(sample, "payload", net.sample.to_metadata())

        ng = f.create_group("network")
        pg = ng.create_group("pore")
        tg = ng.create_group("throat")
        pg.create_dataset("coords", data=net.pore_coords)
        tg.create_dataset("conns", data=net.throat_conns)
        for k, v in net.pore.items():
            pg.create_dataset(k, data=v)
        for k, v in net.throat.items():
            tg.create_dataset(k, data=v)

        labels = f.create_group("labels")
        lpg = labels.create_group("pore")
        ltg = labels.create_group("throat")
        for k, v in net.pore_labels.items():
            lpg.create_dataset(k, data=v.astype(np.uint8))
        for k, v in net.throat_labels.items():
            ltg.create_dataset(k, data=v.astype(np.uint8))

        _write_json_attr(f, "extra", net.extra)


def load_hdf5(path: str | Path) -> Network:
    """Load a network from the project HDF5 interchange format.

    Parameters
    ----------
    path :
        Path to an HDF5 file produced by :func:`save_hdf5`.

    Returns
    -------
    Network
        Reconstructed network object.

    Notes
    -----
    Boolean labels are stored on disk as ``uint8`` arrays for portability and are
    converted back to ``bool`` arrays during import.
    """

    path = Path(path)
    with h5py.File(path, "r") as f:
        schema_version = f["meta"]["schema_version"][()].decode("utf-8")
        prov = Provenance.from_metadata(_read_json_attr(f["meta"], "provenance", {}))
        sample = SampleGeometry.from_metadata(_read_json_attr(f["sample"], "payload", {}))

        pore_coords = f["network"]["pore"]["coords"][()]
        throat_conns = f["network"]["throat"]["conns"][()]
        pore = {k: ds[()] for k, ds in f["network"]["pore"].items() if k != "coords"}
        throat = {k: ds[()] for k, ds in f["network"]["throat"].items() if k != "conns"}
        pore_labels = (
            {k: ds[()].astype(bool) for k, ds in f["labels"]["pore"].items()}
            if "labels" in f
            else {}
        )
        throat_labels = (
            {k: ds[()].astype(bool) for k, ds in f["labels"]["throat"].items()}
            if "labels" in f
            else {}
        )
        extra = _read_json_attr(f, "extra", {})

    return Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample,
        provenance=prov,
        schema_version=schema_version,
        pore=pore,
        throat=throat,
        pore_labels=pore_labels,
        throat_labels=throat_labels,
        extra=extra,
    )
