from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np

from ..core.network import Network
from ..core.provenance import Provenance
from ..core.sample import SampleGeometry
from ..core.validation import validate_network

_PORESPY_KEYMAP = {
    "throat.conns": ("throat", None, "conns"),
    "pore.coords": ("pore", None, "coords"),
    # Core geometric quantities
    "pore.volume": ("pore", "volume", None),
    "throat.volume": ("throat", "volume", None),
    "throat.length": ("throat", "length", None),
    "throat.total_length": ("throat", "length", None),
    "throat.direct_length": ("throat", "direct_length", None),
    "throat.area": ("throat", "area", None),
    "throat.cross_sectional_area": ("throat", "area", None),
    "pore.area": ("pore", "area", None),
    "pore.cross_sectional_area": ("pore", "area", None),
    "throat.perimeter": ("throat", "perimeter", None),
    "pore.perimeter": ("pore", "perimeter", None),
    "throat.shape_factor": ("throat", "shape_factor", None),
    "pore.shape_factor": ("pore", "shape_factor", None),
    # Diameter/radius aliases frequently seen in OpenPNM/PoreSpy pipelines
    "throat.diameter": ("throat", "diameter_inscribed", None),
    "pore.diameter": ("pore", "diameter_inscribed", None),
    "throat.inscribed_diameter": ("throat", "diameter_inscribed", None),
    "pore.inscribed_diameter": ("pore", "diameter_inscribed", None),
    "throat.diameter_inscribed": ("throat", "diameter_inscribed", None),
    "pore.diameter_inscribed": ("pore", "diameter_inscribed", None),
    "throat.equivalent_diameter": ("throat", "diameter_equivalent", None),
    "pore.equivalent_diameter": ("pore", "diameter_equivalent", None),
    "throat.radius": ("throat", "radius_inscribed", None),
    "pore.radius": ("pore", "radius_inscribed", None),
    "throat.radius_inscribed": ("throat", "radius_inscribed", None),
    "pore.radius_inscribed": ("pore", "radius_inscribed", None),
    # Conduit lengths (OpenPNM style)
    "throat.conduit_lengths.pore1": ("throat", "pore1_length", None),
    "throat.conduit_lengths.throat": ("throat", "core_length", None),
    "throat.conduit_lengths.pore2": ("throat", "pore2_length", None),
    # Precomputed conductance
    "throat.hydraulic_conductance": ("throat", "hydraulic_conductance", None),
}


_AXIS_LABEL_ALIASES = [
    ("left", "inlet_xmin"),
    ("right", "outlet_xmax"),
    ("front", "inlet_ymin"),
    ("back", "outlet_ymax"),
    ("bottom", "inlet_zmin"),
    ("top", "outlet_zmax"),
]


def _normalize_value(value: object) -> np.ndarray:
    return np.asarray(value)


def _derive_missing_geometry(
    pore_data: dict[str, np.ndarray], throat_data: dict[str, np.ndarray]
) -> None:
    for data in (pore_data, throat_data):
        if "area" not in data:
            if "diameter_inscribed" in data:
                d = np.asarray(data["diameter_inscribed"], dtype=float)
                data["area"] = np.pi * (0.5 * d) ** 2
            elif "radius_inscribed" in data:
                r = np.asarray(data["radius_inscribed"], dtype=float)
                data["area"] = np.pi * r**2
        if "diameter_inscribed" not in data and "radius_inscribed" in data:
            r = np.asarray(data["radius_inscribed"], dtype=float)
            data["diameter_inscribed"] = 2.0 * r
        if "shape_factor" not in data and "area" in data and "perimeter" in data:
            A = np.asarray(data["area"], dtype=float)
            P = np.asarray(data["perimeter"], dtype=float)
            data["shape_factor"] = A / np.maximum(P, 1e-30) ** 2

    if "length" not in throat_data and all(
        k in throat_data for k in ("pore1_length", "core_length", "pore2_length")
    ):
        throat_data["length"] = (
            np.asarray(throat_data["pore1_length"], dtype=float)
            + np.asarray(throat_data["core_length"], dtype=float)
            + np.asarray(throat_data["pore2_length"], dtype=float)
        )


def from_porespy(
    network_dict: Mapping[str, object],
    *,
    sample: SampleGeometry | None = None,
    provenance: Provenance | None = None,
    strict: bool = True,
) -> Network:
    if "throat.conns" not in network_dict or "pore.coords" not in network_dict:
        if strict:
            raise KeyError(
                "PoreSpy/OpenPNM-style dict must include 'throat.conns' and 'pore.coords'"
            )

    pore_data: dict[str, np.ndarray] = {}
    throat_data: dict[str, np.ndarray] = {}
    pore_labels: dict[str, np.ndarray] = {}
    throat_labels: dict[str, np.ndarray] = {}
    extra: dict[str, object] = {}

    throat_conns = None
    pore_coords = None

    for key, value in network_dict.items():
        arr = _normalize_value(value)
        if key in _PORESPY_KEYMAP:
            family, canonical, reserved = _PORESPY_KEYMAP[key]
            if reserved == "conns":
                throat_conns = np.asarray(arr, dtype=int)
                continue
            if reserved == "coords":
                pore_coords = np.asarray(arr, dtype=float)
                continue
            if family == "pore":
                pore_data[canonical] = np.asarray(arr)
            else:
                throat_data[canonical] = np.asarray(arr)
            continue

        if key.startswith("pore."):
            sub = key[5:]
            if np.asarray(arr).dtype == bool:
                pore_labels[sub] = np.asarray(arr, dtype=bool)
            else:
                pore_data[sub.replace(".", "_")] = np.asarray(arr)
        elif key.startswith("throat."):
            sub = key[7:]
            if np.asarray(arr).dtype == bool:
                throat_labels[sub] = np.asarray(arr, dtype=bool)
            else:
                throat_data[sub.replace(".", "_")] = np.asarray(arr)
        else:
            extra[key] = value

    if throat_conns is None or pore_coords is None:
        if strict:
            raise KeyError("Required keys 'throat.conns' and/or 'pore.coords' missing")
        throat_conns = np.asarray(network_dict.get("throat.conns"))
        pore_coords = np.asarray(network_dict.get("pore.coords"), dtype=float)

    if pore_coords.ndim == 2 and pore_coords.shape[1] == 2:
        pore_coords = np.column_stack([pore_coords, np.zeros(pore_coords.shape[0])])

    for alias, canonical in _AXIS_LABEL_ALIASES:
        if alias in pore_labels and canonical not in pore_labels:
            pore_labels[canonical] = pore_labels[alias]

    _derive_missing_geometry(pore_data, throat_data)

    # Gentle warning for unsupported nested conduit arrays often present in OpenPNM
    if (
        "throat.hydraulic_size_factors" in network_dict
        and "hydraulic_size_factors" not in throat_data
    ):
        extra["throat.hydraulic_size_factors"] = network_dict["throat.hydraulic_size_factors"]
        warnings.warn(
            "Stored throat.hydraulic_size_factors in net.extra (no v0.1 solver integration yet)",
            RuntimeWarning,
            stacklevel=2,
        )

    net = Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample or SampleGeometry(),
        provenance=provenance or Provenance(source_kind="porespy"),
        pore=pore_data,
        throat=throat_data,
        pore_labels=pore_labels,
        throat_labels=throat_labels,
        extra=extra,
    )
    validate_network(net)
    return net
