from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.core.validation import validate_network

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
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AREA_KEYS = frozenset(
    {
        "pore.area",
        "pore.cross_sectional_area",
        "pore.surface_area",
        "throat.area",
        "throat.cross_sectional_area",
    }
)
_LENGTH_KEYS = frozenset(
    {
        "pore.coords",
        "pore.geometric_centroid",
        "pore.global_peak",
        "pore.local_peak",
        "pore.equivalent_diameter",
        "pore.extended_diameter",
        "pore.inscribed_diameter",
        "throat.direct_length",
        "throat.total_length",
        "throat.equivalent_diameter",
        "throat.inscribed_diameter",
    }
)
_VOLUME_KEYS = frozenset({"pore.volume", "pore.region_volume", "throat.volume"})
_PERIMETER_KEYS = frozenset({"pore.perimeter", "throat.perimeter"})


def _normalize_value(value: object) -> np.ndarray:
    """Convert an importer payload to a NumPy array.

    Parameters
    ----------
    value :
        Arbitrary mapping value extracted from a PoreSpy/OpenPNM-style dictionary.

    Returns
    -------
    numpy.ndarray
        Array view or copy produced by :func:`numpy.asarray`.
    """

    return np.asarray(value)


def scale_porespy_geometry(
    network_dict: Mapping[str, object], *, voxel_size: float
) -> dict[str, object]:
    """Scale common PoreSpy geometry fields from voxel units to physical units.

    Parameters
    ----------
    network_dict :
        PoreSpy/OpenPNM-style mapping containing keys such as ``pore.coords``,
        ``throat.cross_sectional_area`` and ``pore.volume``.
    voxel_size :
        Edge length of one voxel in physical units.

    Returns
    -------
    dict of str to object
        New mapping with common geometric fields rescaled.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.

    Notes
    -----
    This helper assumes isotropic voxels. The conversion factors are:

    - lengths: ``L_phys = L_vox * voxel_size``
    - areas: ``A_phys = A_vox * voxel_size**2``
    - volumes: ``V_phys = V_vox * voxel_size**3``
    - perimeters: ``P_phys = P_vox * voxel_size``

    When ``throat.volume`` is absent but ``throat.cross_sectional_area`` and
    ``throat.total_length`` are available, a simple conduit approximation is used:

    ``throat.volume = throat.cross_sectional_area * throat.total_length``

    This is convenient for manufactured examples and notebook workflows, but it is
    still a geometric approximation rather than an exact segmented volume.
    """

    L = float(voxel_size)
    if L <= 0:
        raise ValueError("voxel_size must be positive")

    scaled = dict(network_dict)
    for key, value in list(scaled.items()):
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if key in _AREA_KEYS:
            scaled[key] = arr.astype(float) * L**2
        elif key in _LENGTH_KEYS:
            scaled[key] = arr.astype(float) * L
        elif key in _VOLUME_KEYS:
            scaled[key] = arr.astype(float) * L**3
        elif key in _PERIMETER_KEYS:
            scaled[key] = arr.astype(float) * L

    if "throat.volume" not in scaled and all(
        key in scaled for key in ("throat.cross_sectional_area", "throat.total_length")
    ):
        scaled["throat.volume"] = np.asarray(
            scaled["throat.cross_sectional_area"], dtype=float
        ) * np.asarray(scaled["throat.total_length"], dtype=float)
    if "pore.volume" not in scaled and "pore.region_volume" in scaled:
        scaled["pore.volume"] = np.asarray(scaled["pore.region_volume"], dtype=float)
    return scaled


def ensure_cartesian_boundary_labels(
    network_dict: Mapping[str, object],
    *,
    axes: tuple[str, ...] | None = None,
    tol_fraction: float = 0.05,
) -> dict[str, object]:
    """Infer Cartesian inlet and outlet pore labels from coordinates.

    Parameters
    ----------
    network_dict :
        Mapping containing at least ``pore.coords``.
    axes :
        Axes to label. If omitted, all axes present in the coordinate array are used.
    tol_fraction :
        Fraction of the domain span used as a geometric tolerance near each boundary.

    Returns
    -------
    dict of str to object
        Updated mapping with labels such as ``pore.inlet_xmin`` and ``pore.outlet_xmax``.

    Raises
    ------
    ValueError
        If the coordinate array has invalid shape, if ``tol_fraction`` is negative,
        or if an invalid axis name is requested.

    Notes
    -----
    For each active axis, the helper marks pores satisfying

    - ``x <= x_min + tol`` as inlet pores
    - ``x >= x_max - tol`` as outlet pores

    where ``tol = tol_fraction * max(x_max - x_min, 1e-12)``.
    """

    coords = np.asarray(network_dict["pore.coords"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in {2, 3}:
        raise ValueError("pore.coords must have shape (Np, 2) or (Np, 3)")
    if tol_fraction < 0:
        raise ValueError("tol_fraction must be nonnegative")

    ndim = coords.shape[1]
    active_axes = axes if axes is not None else tuple(("x", "y", "z")[:ndim])
    updated = dict(network_dict)
    boundary = np.asarray(
        updated.get("pore.boundary", np.zeros(coords.shape[0], dtype=bool)), dtype=bool
    ).copy()

    for axis in active_axes:
        if axis not in _AXIS_INDEX:
            raise ValueError("axes entries must be drawn from {'x', 'y', 'z'}")
        axis_index = _AXIS_INDEX[axis]
        if axis_index >= ndim:
            raise ValueError(
                f"axis '{axis}' is not available in pore.coords with shape {coords.shape}"
            )
        values = coords[:, axis_index]
        lo = float(values.min())
        hi = float(values.max())
        tol = tol_fraction * max(hi - lo, 1e-12)
        inlet_key = f"pore.inlet_{axis}min"
        outlet_key = f"pore.outlet_{axis}max"
        updated.setdefault(inlet_key, values <= lo + tol)
        updated.setdefault(outlet_key, values >= hi - tol)
        boundary |= np.asarray(updated[inlet_key], dtype=bool) | np.asarray(
            updated[outlet_key], dtype=bool
        )

    updated["pore.boundary"] = boundary
    return updated


def _derive_missing_geometry(
    pore_data: dict[str, np.ndarray], throat_data: dict[str, np.ndarray]
) -> None:
    """Derive secondary geometric fields from more primitive ones.

    Parameters
    ----------
    pore_data, throat_data :
        Mutable pore and throat property dictionaries. They are updated in place.

    Notes
    -----
    The helper infers a small set of frequently needed fields:

    - ``area`` from ``diameter_inscribed`` or ``radius_inscribed``
    - ``diameter_inscribed`` from ``radius_inscribed``
    - ``shape_factor`` from ``area / perimeter**2``
    - ``length`` from ``pore1_length + core_length + pore2_length``
    """

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
    """Build a :class:`Network` from a PoreSpy/OpenPNM-style mapping.

    Parameters
    ----------
    network_dict :
        Mapping containing PoreSpy/OpenPNM keys such as ``pore.coords`` and
        ``throat.conns``.
    sample :
        Sample geometry metadata attached to the resulting network. If omitted,
        a default empty :class:`SampleGeometry` is used.
    provenance :
        Provenance metadata. If omitted, a default record with
        ``source_kind="porespy"`` is created.
    strict :
        If ``True``, missing topology keys immediately raise an error.

    Returns
    -------
    Network
        Imported network in the canonical ``voids`` representation.

    Raises
    ------
    KeyError
        If the required topology keys are missing.

    Notes
    -----
    The importer performs several normalizations:

    - PoreSpy/OpenPNM aliases are mapped to canonical ``voids`` names.
    - Two-dimensional coordinates are embedded into 3D as ``(x, y, 0)``.
    - Basic missing geometry is derived when possible.
    - Common boundary aliases such as ``left`` and ``right`` are mirrored to
      ``inlet_xmin`` and ``outlet_xmax``.

    Unsupported nested arrays such as ``throat.hydraulic_size_factors`` are
    preserved in ``net.extra`` so that information is not silently lost.
    """

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
        throat_conns = np.asarray(network_dict.get("throat.conns"))
        pore_coords = np.asarray(network_dict.get("pore.coords"), dtype=float)
        if throat_conns.ndim == 0 or pore_coords.ndim == 0:
            raise KeyError("Required keys 'throat.conns' and/or 'pore.coords' missing")

    if pore_coords.ndim == 2 and pore_coords.shape[1] == 2:
        pore_coords = np.column_stack([pore_coords, np.zeros(pore_coords.shape[0])])

    for alias, canonical in _AXIS_LABEL_ALIASES:
        if alias in pore_labels and canonical not in pore_labels:
            pore_labels[canonical] = pore_labels[alias]

    _derive_missing_geometry(pore_data, throat_data)

    if "hydraulic_size_factors" in throat_data:
        extra["throat.hydraulic_size_factors"] = throat_data.pop("hydraulic_size_factors")
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
