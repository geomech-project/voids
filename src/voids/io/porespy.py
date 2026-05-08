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
    "throat.shape_factor_radius": ("throat", "shape_factor_radius", None),
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
        "throat.conduit_lengths",
        "throat.conduit_lengths.pore1",
        "throat.conduit_lengths.throat",
        "throat.conduit_lengths.pore2",
        "throat.equivalent_diameter",
        "throat.inscribed_diameter",
        "throat.shape_factor_radius",
    }
)
_VOLUME_KEYS = frozenset({"pore.volume", "pore.region_volume", "throat.volume"})
_PERIMETER_KEYS = frozenset({"pore.perimeter", "throat.perimeter"})
_IMPERIAL_EXPORT_THROAT_G_HIGH_TRIGGER = 0.09
_IMPERIAL_EXPORT_THROAT_G_HIGH_CAP = 0.079
_IMPERIAL_EXPORT_THROAT_G_LOW_TRIGGER = 0.01
_IMPERIAL_EXPORT_RANDOM_G_SCALE = 0.00625
_IMPERIAL_EXPORT_RANDOM_G_SHIFT = 5.0
_IMPERIAL_EXPORT_RANDOM_G_CAP_TRIGGER = 0.049
_IMPERIAL_EXPORT_RANDOM_G_CAP_VALUE = 0.0625
_LEGACY_GEOMETRY_REPAIR_ALIASES = {"pnextract": "imperial_export"}


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
        # Use argmin/argmax instead of ndarray.min/max for compatibility with
        # environments where numpy reduction defaults are monkeypatched.
        lo = float(values[np.argmin(values)])
        hi = float(values[np.argmax(values)])
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


def _positive_epsilon_like(length: np.ndarray) -> np.ndarray:
    """Return a tiny positive scale-aware epsilon for conduit sub-lengths."""

    return np.maximum(1.0e-15 * np.asarray(length, dtype=float), np.finfo(float).tiny)


def _enforce_positive_conduit_partition(
    pore1_length: np.ndarray,
    core_length: np.ndarray,
    pore2_length: np.ndarray,
    total_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Clip a conduit partition to positive sub-lengths while preserving totals."""

    p1 = np.asarray(pore1_length, dtype=float).copy()
    core = np.asarray(core_length, dtype=float).copy()
    p2 = np.asarray(pore2_length, dtype=float).copy()
    total = np.asarray(total_length, dtype=float)
    eps = _positive_epsilon_like(total)
    adjusted = (p1 <= eps) | (core <= eps) | (p2 <= eps) | ((p1 + core + p2) > total + eps)
    if not np.any(adjusted):
        return p1, core, p2, 0

    available = np.maximum(total[adjusted] - eps[adjusted], 2.0 * eps[adjusted])
    side_sum = p1[adjusted] + p2[adjusted]
    side_sum = np.where(side_sum > 0.0, side_sum, 1.0)
    p1_fraction = np.clip(p1[adjusted] / side_sum, 0.0, 1.0)
    p2_fraction = 1.0 - p1_fraction
    p1[adjusted] = np.maximum(p1_fraction * available, eps[adjusted])
    p2[adjusted] = np.maximum(p2_fraction * available, eps[adjusted])

    side_total = p1[adjusted] + p2[adjusted]
    too_long = side_total >= total[adjusted] - eps[adjusted]
    if np.any(too_long):
        scale = (total[adjusted][too_long] - eps[adjusted][too_long]) / side_total[too_long]
        p1_adjusted = p1[adjusted]
        p2_adjusted = p2[adjusted]
        p1_adjusted[too_long] = np.maximum(p1_adjusted[too_long] * scale, eps[adjusted][too_long])
        p2_adjusted[too_long] = np.maximum(p2_adjusted[too_long] * scale, eps[adjusted][too_long])
        p1[adjusted] = p1_adjusted
        p2[adjusted] = p2_adjusted

    core[adjusted] = np.maximum(total[adjusted] - p1[adjusted] - p2[adjusted], eps[adjusted])
    return p1, core, p2, int(np.count_nonzero(adjusted))


def _derive_missing_conduit_lengths(
    pore_data: dict[str, np.ndarray],
    throat_data: dict[str, np.ndarray],
    throat_conns: np.ndarray,
    pore_coords: np.ndarray,
) -> dict[str, object] | None:
    """Derive pore-throat-pore conduit lengths when extraction did not provide them."""

    if "conduit_lengths" in throat_data:
        conduit = np.asarray(throat_data.pop("conduit_lengths"), dtype=float)
        if conduit.ndim != 2 or conduit.shape[1] != 3:
            raise ValueError("throat.conduit_lengths must have shape (Nt, 3)")
        throat_data.setdefault("pore1_length", conduit[:, 0])
        throat_data.setdefault("core_length", conduit[:, 1])
        throat_data.setdefault("pore2_length", conduit[:, 2])
        return {"mode": "provided_array", "source": "throat.conduit_lengths"}

    conduit_keys = ("pore1_length", "core_length", "pore2_length")
    if all(key in throat_data for key in conduit_keys):
        return None
    if any(key in throat_data for key in conduit_keys):
        return None
    if "diameter_inscribed" not in pore_data or "diameter_inscribed" not in throat_data:
        return None

    length_present = "length" in throat_data
    if "direct_length" in throat_data:
        total_length = np.asarray(throat_data["direct_length"], dtype=float)
        source_length = "direct_length"
    elif "length" in throat_data:
        total_length = np.asarray(throat_data["length"], dtype=float)
        source_length = "length"
    else:
        coords = np.asarray(pore_coords, dtype=float)
        conns = np.asarray(throat_conns, dtype=int)
        total_length = np.linalg.norm(coords[conns[:, 1]] - coords[conns[:, 0]], axis=1)
        source_length = "pore_coords"

    conns = np.asarray(throat_conns, dtype=int)
    total = np.asarray(total_length, dtype=float)
    pore_diameter = np.asarray(pore_data["diameter_inscribed"], dtype=float)
    throat_diameter = np.asarray(throat_data["diameter_inscribed"], dtype=float)
    if not (
        np.all(np.isfinite(total))
        and np.all(np.isfinite(pore_diameter))
        and np.all(np.isfinite(throat_diameter))
        and np.all(total > 0.0)
        and np.all(pore_diameter > 0.0)
        and np.all(throat_diameter > 0.0)
    ):
        return None

    d1 = pore_diameter[conns[:, 0]]
    d2 = pore_diameter[conns[:, 1]]
    dt_max = np.minimum(d1, d2) * (1.0 - 1.0e-12)
    clipped = throat_diameter > dt_max
    dt = np.minimum(throat_diameter, dt_max)

    p1 = 0.5 * np.sqrt(np.maximum(d1 * d1 - dt * dt, 0.0))
    p2 = 0.5 * np.sqrt(np.maximum(d2 * d2 - dt * dt, 0.0))
    core = total - p1 - p2

    overlap = core <= 0.0
    if np.any(overlap):
        d = total[overlap]
        r1 = 0.5 * d1[overlap]
        r2 = 0.5 * d2[overlap]
        l1_int = (d * d - r2 * r2 + r1 * r1) / (2.0 * d)
        valid_intersection = (l1_int > 0.0) & (l1_int < d)
        if np.any(valid_intersection):
            idx = np.flatnonzero(overlap)[valid_intersection]
            p1[idx] = l1_int[valid_intersection]
            p2[idx] = d[valid_intersection] - l1_int[valid_intersection]
            core[idx] = _positive_epsilon_like(total[idx])

    p1, core, p2, positive_repairs = _enforce_positive_conduit_partition(p1, core, p2, total)
    throat_data["pore1_length"] = p1
    throat_data["core_length"] = core
    throat_data["pore2_length"] = p2
    if not length_present:
        throat_data["length"] = p1 + core + p2
    return {
        "mode": "spheres_and_cylinders",
        "source_length": source_length,
        "throat_diameter_clipped": int(np.count_nonzero(clipped)),
        "overlap_repairs": int(np.count_nonzero(overlap)),
        "positive_partition_repairs": positive_repairs,
    }


def _ensure_inscribed_size_aliases(data: dict[str, np.ndarray]) -> None:
    """Backfill diameter/radius aliases without deriving areas."""

    if "diameter_inscribed" not in data and "radius_inscribed" in data:
        r = np.asarray(data["radius_inscribed"], dtype=float)
        data["diameter_inscribed"] = 2.0 * r
    if "radius_inscribed" not in data and "diameter_inscribed" in data:
        d = np.asarray(data["diameter_inscribed"], dtype=float)
        data["radius_inscribed"] = 0.5 * d


def _imperial_export_random_shape_factors(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the fallback distribution used by the Imperial export heuristics.

    Notes
    -----
    This follows the `randomG()` helper in the reference `pnextract`
    ``blockNet_write_cnm.cpp`` export path.
    """

    out = np.empty(int(size), dtype=float)
    filled = 0
    while filled < out.size:
        needed = out.size - filled
        x1 = 2.0 * rng.random(needed) - 1.0
        x2 = 2.0 * rng.random(needed) - 1.0
        w = x1 * x1 + x2 * x2
        keep = (w > 0.0) & (w < 1.0)
        if not np.any(keep):
            continue
        y = _IMPERIAL_EXPORT_RANDOM_G_SCALE * (
            x1[keep] * np.sqrt((-2.0 * np.log(w[keep])) / w[keep]) + _IMPERIAL_EXPORT_RANDOM_G_SHIFT
        )
        y = np.where(
            y > _IMPERIAL_EXPORT_RANDOM_G_CAP_TRIGGER,
            _IMPERIAL_EXPORT_RANDOM_G_CAP_VALUE,
            y,
        )
        n_take = min(int(y.size), needed)
        out[filled : filled + n_take] = y[:n_take]
        filled += n_take
    return out


def _override_area_from_shape_factor_and_radius(data: dict[str, np.ndarray]) -> bool:
    """Set `area = r^2 / (4G)` when both inscribed radius and shape factor exist."""

    if "shape_factor" not in data or "radius_inscribed" not in data:
        return False
    g = np.asarray(data["shape_factor"], dtype=float)
    if not np.all(np.isfinite(g)) or not np.all(g > 0.0):
        raise ValueError(
            "shape_factor must be positive and finite for all elements before area can be"
            " overridden; found non-positive or non-finite values in shape_factor"
        )
    r = np.asarray(data["radius_inscribed"], dtype=float)
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            area = r * r / (4.0 * g)
        except FloatingPointError as exc:
            raise ValueError(
                "Failed to compute area from radius_inscribed and shape_factor:"
                " non-finite result due to numerical overflow or invalid operation"
            ) from exc
    if not np.all(np.isfinite(area)) or not np.all(area > 0.0):
        raise ValueError(
            "Computed area must be positive and finite for all elements; found non-positive"
            " or non-finite values in area derived from radius_inscribed and shape_factor"
        )
    data["area"] = area
    return True


def _normalize_geometry_repairs_mode(geometry_repairs: str | None) -> str | None:
    """Normalize geometry-repair mode names, accepting legacy aliases."""

    if geometry_repairs in _LEGACY_GEOMETRY_REPAIR_ALIASES:
        normalized = _LEGACY_GEOMETRY_REPAIR_ALIASES[geometry_repairs]
        warnings.warn(
            f"geometry_repairs={geometry_repairs!r} is deprecated; use {normalized!r} instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return normalized
    return geometry_repairs


def _apply_imperial_export_geometry_repairs(
    pore_data: dict[str, np.ndarray],
    throat_data: dict[str, np.ndarray],
    throat_conns: np.ndarray,
    *,
    num_pores: int,
    random_seed: int | None,
) -> dict[str, object]:
    """Apply Imperial export-style shape-factor preprocessing and repair heuristics.

    Notes
    -----
    This follows the reference Imperial College `pnextract`
    `blockNet_write_cnm.cpp` logic closely:

    - throats prefer ``G = r^2 / (4A)`` when a shape-factor radius and
      ``area`` are available; when no separate shape-factor radius is provided,
      ``radius_inscribed`` is used
    - very large throat shape factors are repaired with ``min(0.079, G/2)``
    - very small throat shape factors are replaced by the reference-style
      randomized admissible distribution, floored at ``0.01``
    - pore shape factor is reassigned as a throat-area-weighted average of the
      repaired neighboring throat shape factors
    - when inscribed radii are available, pore and throat areas are overwritten
      by ``r^2 / (4G)`` to match the geometry used by `pnflow`
    """

    _ensure_inscribed_size_aliases(pore_data)
    _ensure_inscribed_size_aliases(throat_data)

    summary: dict[str, object] = {
        "mode": "imperial_export",
        "random_seed": random_seed,
        "throat_shape_factor_source": "existing",
        "throat_high_repairs": 0,
        "throat_low_repairs": 0,
        "pore_shape_factor_weighted": False,
        "throat_area_overridden": False,
        "pore_area_overridden": False,
    }

    throat_area_weight = throat_data.get("area")
    throat_radius = throat_data.get("radius_inscribed")
    throat_shape_factor_radius = throat_data.get("shape_factor_radius", throat_radius)

    if throat_area_weight is not None:
        throat_area_weight = np.asarray(throat_area_weight, dtype=float).copy()

    if throat_shape_factor_radius is not None and throat_area_weight is not None:
        if not np.all(np.isfinite(throat_area_weight)) or not np.all(throat_area_weight > 0.0):
            raise ValueError(
                "throat area values must be positive and finite for all elements when"
                " deriving shape factors via r^2/(4A); found non-positive or"
                " non-finite values in throat area"
            )
        throat_shape_factor_radius = np.asarray(throat_shape_factor_radius, dtype=float)
        if not np.all(np.isfinite(throat_shape_factor_radius)) or not np.all(
            throat_shape_factor_radius > 0.0
        ):
            raise ValueError(
                "throat shape-factor radius values must be positive and finite for all"
                " elements when deriving shape factors via r^2/(4A)"
            )
        throat_shape = throat_shape_factor_radius**2 / (4.0 * throat_area_weight)
        summary["throat_shape_factor_source"] = (
            "shape_factor_radius_area" if "shape_factor_radius" in throat_data else "radius_area"
        )
    elif "shape_factor" in throat_data:
        throat_shape = np.asarray(throat_data["shape_factor"], dtype=float).copy()
    else:
        return summary

    high = throat_shape >= _IMPERIAL_EXPORT_THROAT_G_HIGH_TRIGGER
    if np.any(high):
        throat_shape[high] = np.minimum(
            _IMPERIAL_EXPORT_THROAT_G_HIGH_CAP, throat_shape[high] / 2.0
        )
    summary["throat_high_repairs"] = int(np.count_nonzero(high))

    low = throat_shape < _IMPERIAL_EXPORT_THROAT_G_LOW_TRIGGER
    if np.any(low):
        rng = np.random.default_rng(random_seed)
        repl = np.maximum(
            _imperial_export_random_shape_factors(int(np.count_nonzero(low)), rng),
            _IMPERIAL_EXPORT_THROAT_G_LOW_TRIGGER,
        )
        throat_shape[low] = repl
    summary["throat_low_repairs"] = int(np.count_nonzero(low))
    throat_data["shape_factor"] = throat_shape

    if _override_area_from_shape_factor_and_radius(throat_data):
        summary["throat_area_overridden"] = True

    if throat_area_weight is None:
        return summary

    accum = np.full(int(num_pores), 5.0e-38, dtype=float)
    weights = np.full(int(num_pores), 1.0e-36, dtype=float)
    counts = np.zeros(int(num_pores), dtype=int)

    p1 = np.asarray(throat_conns[:, 0], dtype=int)
    p2 = np.asarray(throat_conns[:, 1], dtype=int)
    contrib = throat_shape * throat_area_weight
    np.add.at(accum, p1, contrib)
    np.add.at(accum, p2, contrib)
    np.add.at(weights, p1, throat_area_weight)
    np.add.at(weights, p2, throat_area_weight)
    np.add.at(counts, p1, 1)
    np.add.at(counts, p2, 1)

    if np.any(counts > 0):
        connected = counts > 0
        if "shape_factor" in pore_data:
            pore_shape = np.asarray(pore_data["shape_factor"], dtype=float).copy()
            pore_shape[connected] = accum[connected] / weights[connected]
            pore_data["shape_factor"] = pore_shape
            summary["pore_shape_factor_weighted"] = True
        elif np.all(connected):
            pore_data["shape_factor"] = accum / weights
            summary["pore_shape_factor_weighted"] = True

        if summary["pore_shape_factor_weighted"] and _override_area_from_shape_factor_and_radius(
            pore_data
        ):
            summary["pore_area_overridden"] = True

    return summary


def from_porespy(
    network_dict: Mapping[str, object],
    *,
    sample: SampleGeometry | None = None,
    provenance: Provenance | None = None,
    strict: bool = True,
    geometry_repairs: str | None = None,
    repair_seed: int | None = 0,
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
    geometry_repairs :
        Optional extraction-style preprocessing mode. Set to
        ``"imperial_export"`` to apply the Imperial College export heuristics
        for throat shape-factor repair and pore shape-factor reconstruction.
        The default ``None`` preserves the imported geometry as-is apart from
        basic alias normalization. The legacy name ``"pnextract"`` is still
        accepted as a deprecated alias for backward compatibility.
    repair_seed :
        Seed for any stochastic repair branch. Only used when
        ``geometry_repairs="imperial_export"``.

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

    OpenPNM-style arrays that are not part of the formal schema, such as
    ``throat.hydraulic_size_factors``, are preserved in ``net.extra`` so that
    information is not silently lost.
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
            if canonical is None:
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

    _ensure_inscribed_size_aliases(pore_data)
    _ensure_inscribed_size_aliases(throat_data)

    geometry_repairs = _normalize_geometry_repairs_mode(geometry_repairs)

    if geometry_repairs not in {None, "imperial_export"}:
        raise ValueError("geometry_repairs must be None or 'imperial_export'")
    if geometry_repairs == "imperial_export":
        extra["geometry_repairs"] = _apply_imperial_export_geometry_repairs(
            pore_data,
            throat_data,
            np.asarray(throat_conns, dtype=int),
            num_pores=int(pore_coords.shape[0]),
            random_seed=repair_seed,
        )

    conduit_summary = _derive_missing_conduit_lengths(
        pore_data,
        throat_data,
        np.asarray(throat_conns, dtype=int),
        np.asarray(pore_coords, dtype=float),
    )
    if conduit_summary is not None:
        extra["conduit_lengths"] = conduit_summary

    _derive_missing_geometry(pore_data, throat_data)

    if "hydraulic_size_factors" in throat_data:
        extra["throat.hydraulic_size_factors"] = throat_data.pop("hydraulic_size_factors")
        warnings.warn(
            "Stored throat.hydraulic_size_factors in net.extra; the auto conductance model can use it",
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
