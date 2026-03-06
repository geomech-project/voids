from __future__ import annotations

import numpy as np

from voids.core.network import Network
from voids.geom.hydraulic import DEFAULT_G_REF
from voids.graph import induced_subnetwork


def _equivalent_radius_2d(radii_xy: tuple[float, float]) -> float:
    """Return area-equivalent circular radius for ellipse radii.

    Parameters
    ----------
    radii_xy :
        Ellipse semi-axes ``(rx, ry)``.

    Returns
    -------
    float
        Equivalent radius ``r_eq`` satisfying ``pi * r_eq^2 = pi * rx * ry``.
    """

    rx, ry = (float(radii_xy[0]), float(radii_xy[1]))
    if min(rx, ry) <= 0:
        raise ValueError("Ellipse radii must be positive")
    return float((rx * ry) ** 0.5)


def _equivalent_radius_3d(radii_xyz: tuple[float, float, float]) -> float:
    """Return volume-equivalent spherical radius for ellipsoid radii.

    Parameters
    ----------
    radii_xyz :
        Ellipsoid semi-axes ``(rx, ry, rz)``.

    Returns
    -------
    float
        Equivalent radius ``r_eq`` satisfying
        ``(4/3) * pi * r_eq^3 = (4/3) * pi * rx * ry * rz``.
    """

    rx, ry, rz = (float(radii_xyz[0]), float(radii_xyz[1]), float(radii_xyz[2]))
    if min(rx, ry, rz) <= 0:
        raise ValueError("Ellipsoid radii must be positive")
    return float((rx * ry * rz) ** (1.0 / 3.0))


def _as_float_vector(values: np.ndarray, *, expected_size: int, name: str) -> np.ndarray:
    """Validate that one entity-wise field is a 1D float vector.

    Parameters
    ----------
    values :
        Input array-like values.
    expected_size :
        Expected first-dimension length.
    name :
        Field name used in error messages.

    Returns
    -------
    numpy.ndarray
        Float array with shape ``(expected_size,)``.
    """

    arr = np.asarray(values, dtype=float)
    if arr.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    return arr


def _validate_geometry_update_controls(
    *,
    shape_factor: float,
    pore_length_fraction: float,
    min_core_fraction: float,
) -> None:
    """Validate shared geometry-update control parameters."""

    if shape_factor <= 0:
        raise ValueError("shape_factor must be positive")
    if pore_length_fraction < 0:
        raise ValueError("pore_length_fraction must be non-negative")
    if min_core_fraction < 0:
        raise ValueError("min_core_fraction must be non-negative")


def _extend_entity_fields(
    store: dict[str, np.ndarray],
    *,
    n_before: int,
    n_append: int,
    append_fields: dict[str, np.ndarray],
) -> None:
    """Extend all entity fields whose leading dimension matches source size.

    Parameters
    ----------
    store :
        Pore or throat dictionary to be extended in place.
    n_before :
        Number of entities currently represented in the arrays.
    n_append :
        Number of entities to append.
    append_fields :
        Mapping with explicit appended arrays for selected keys. Missing keys
        are padded with zeros of matching dtype/shape.
    """

    for key in list(store.keys()):
        arr = np.asarray(store[key])
        if arr.ndim >= 1 and arr.shape[0] == n_before:
            ext = append_fields.get(key)
            if ext is None:
                ext = np.zeros((n_append,) + arr.shape[1:], dtype=arr.dtype)
            else:
                ext = np.asarray(ext, dtype=arr.dtype)
                expected = (n_append,) + arr.shape[1:]
                if ext.shape != expected:
                    raise ValueError(
                        f"append field '{key}' has shape {ext.shape}, expected {expected}"
                    )
            store[key] = np.concatenate([arr, ext], axis=0)


def _extend_labels(labels: dict[str, np.ndarray], *, n_before: int, n_append: int) -> None:
    """Extend boolean label arrays in place with ``False`` appended entries."""

    for key, mask in list(labels.items()):
        arr = np.asarray(mask, dtype=bool)
        if arr.shape == (n_before,):
            labels[key] = np.concatenate([arr, np.zeros(n_append, dtype=bool)])


def _ellipsoid_mask(
    coords: np.ndarray,
    *,
    center: np.ndarray,
    radii_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Select pore coordinates inside an axis-aligned ellipsoid.

    Parameters
    ----------
    coords :
        Pore-coordinate array with shape ``(Np, 3)``.
    center :
        Ellipsoid center coordinates.
    radii_xyz :
        Ellipsoid semi-axes.

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``(Np,)``.
    """

    rx, ry, rz = (float(radii_xyz[0]), float(radii_xyz[1]), float(radii_xyz[2]))
    if min(rx, ry, rz) <= 0:
        raise ValueError("Ellipsoid radii must be strictly positive")
    dx = (coords[:, 0] - center[0]) / rx
    dy = (coords[:, 1] - center[1]) / ry
    dz = (coords[:, 2] - center[2]) / rz
    mask = (dx * dx + dy * dy + dz * dz) <= 1.0
    return np.asarray(mask, dtype=bool)


def _ellipse_mask_2d(
    coords: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radii_xy: tuple[float, float],
) -> np.ndarray:
    """Select pore coordinates inside an axis-aligned ellipse on the XY plane."""

    cx, cy = center_xy
    rx, ry = (float(radii_xy[0]), float(radii_xy[1]))
    if min(rx, ry) <= 0:
        raise ValueError("Ellipse radii must be strictly positive")
    dx = (coords[:, 0] - cx) / rx
    dy = (coords[:, 1] - cy) / ry
    return (dx * dx + dy * dy) <= 1.0


def _median_throat_radius(net: Network, *, fallback: float) -> float:
    """Estimate representative throat radius from available geometry fields.

    Notes
    -----
    Priority is:
    1. ``throat['radius_inscribed']``
    2. ``0.5 * throat['diameter_inscribed']``
    3. explicit ``fallback`` value.
    """

    if "radius_inscribed" in net.throat:
        values = np.asarray(net.throat["radius_inscribed"], dtype=float)
        if values.size > 0:
            return float(np.median(values))
    if "diameter_inscribed" in net.throat:
        values = np.asarray(net.throat["diameter_inscribed"], dtype=float)
        if values.size > 0:
            return float(0.5 * np.median(values))
    return float(fallback)


def sample_depth(net: Network) -> float:
    """Infer slab thickness for quasi-2D network models.

    Parameters
    ----------
    net :
        Network whose sample geometry is inspected.

    Returns
    -------
    float
        Effective depth (thickness) used to convert 2D areas to volumes.

    Notes
    -----
    If ``sample.lengths['z']`` exists, it is used directly. Otherwise depth is
    inferred from

    ``depth = bulk_volume / (Lx * Ly)``.

    This inference assumes orthogonal sample axes and consistent sample
    metadata.
    """

    if "z" in net.sample.lengths:
        depth = float(net.sample.lengths["z"])
    else:
        lx = float(net.sample.length_for_axis("x"))
        ly = float(net.sample.length_for_axis("y"))
        bulk = float(net.sample.resolved_bulk_volume())
        depth = float(bulk / max(lx * ly, 1.0e-30))
    if depth <= 0:
        raise ValueError("sample depth must be positive")
    return depth


def update_network_geometry_from_radii(
    net: Network,
    *,
    pore_radius: np.ndarray,
    throat_radius: np.ndarray,
    shape_factor: float = DEFAULT_G_REF,
    pore_length_fraction: float = 0.45,
    min_core_fraction: float = 0.05,
) -> None:
    """Recompute 3D pore/throat geometric fields from prescribed radii.

    Parameters
    ----------
    net :
        Target network to update in place.
    pore_radius :
        Pore inscribed radii, shape ``(Np,)``.
    throat_radius :
        Throat inscribed radii, shape ``(Nt,)``.
    shape_factor :
        Shape factor assigned uniformly to pores and throats. For circular
        conduits, the common value is ``1 / (4 * pi)``.
    pore_length_fraction :
        Fraction of center-to-center throat length used as an upper bound for
        each pore-body contribution.
    min_core_fraction :
        Lower bound for throat core length as fraction of direct length.

    Returns
    -------
    None
        The network is modified in place.

    Notes
    -----
    Geometry model:
    - ``A = pi r^2``
    - ``P = 2 pi r``
    - pore volume ``V_p = (4/3) pi r^3``
    - throat core volume ``V_t = A_t * L_core``

    Here ``L_core`` is computed from center distance minus pore-body lengths and
    clipped by ``min_core_fraction``.

    Scientific caveat
    -----------------
    This is a synthetic geometric closure used for controlled studies; it is not
    a direct pore-space reconstruction from imaging.
    """

    _validate_geometry_update_controls(
        shape_factor=shape_factor,
        pore_length_fraction=pore_length_fraction,
        min_core_fraction=min_core_fraction,
    )
    pore_r = _as_float_vector(pore_radius, expected_size=net.Np, name="pore_radius")
    throat_r = _as_float_vector(throat_radius, expected_size=net.Nt, name="throat_radius")
    if np.any(pore_r <= 0) or np.any(throat_r <= 0):
        raise ValueError("pore_radius and throat_radius must be strictly positive")

    pore_area = np.pi * pore_r**2
    pore_perimeter = 2.0 * np.pi * pore_r
    pore_volume = (4.0 / 3.0) * np.pi * pore_r**3

    net.pore["radius_inscribed"] = pore_r
    net.pore["diameter_inscribed"] = 2.0 * pore_r
    net.pore["area"] = pore_area
    net.pore["perimeter"] = pore_perimeter
    net.pore["volume"] = pore_volume
    net.pore["shape_factor"] = np.full(net.Np, float(shape_factor), dtype=float)

    conns = np.asarray(net.throat_conns, dtype=int)
    c0 = net.pore_coords[conns[:, 0]]
    c1 = net.pore_coords[conns[:, 1]]
    direct_length = np.linalg.norm(c1 - c0, axis=1)

    p1_length = np.minimum(pore_r[conns[:, 0]], float(pore_length_fraction) * direct_length)
    p2_length = np.minimum(pore_r[conns[:, 1]], float(pore_length_fraction) * direct_length)
    core_length = np.maximum(
        direct_length - p1_length - p2_length,
        float(min_core_fraction) * direct_length,
    )

    throat_area = np.pi * throat_r**2
    throat_perimeter = 2.0 * np.pi * throat_r
    throat_volume = throat_area * core_length

    net.throat["radius_inscribed"] = throat_r
    net.throat["diameter_inscribed"] = 2.0 * throat_r
    net.throat["area"] = throat_area
    net.throat["perimeter"] = throat_perimeter
    net.throat["shape_factor"] = np.full(net.Nt, float(shape_factor), dtype=float)
    net.throat["length"] = direct_length
    net.throat["direct_length"] = direct_length
    net.throat["pore1_length"] = p1_length
    net.throat["core_length"] = core_length
    net.throat["pore2_length"] = p2_length
    net.throat["volume"] = throat_volume


def update_network_geometry_2d(
    net: Network,
    *,
    pore_radius: np.ndarray,
    throat_radius: np.ndarray,
    depth: float | None = None,
    shape_factor: float = DEFAULT_G_REF,
    pore_length_fraction: float = 0.45,
    min_core_fraction: float = 0.05,
) -> None:
    """Recompute geometry fields for 2D slab-like network models.

    Parameters
    ----------
    net :
        Target network modified in place.
    pore_radius :
        Pore radii, shape ``(Np,)``.
    throat_radius :
        Throat radii, shape ``(Nt,)``.
    depth :
        Slab thickness. If omitted, inferred via :func:`sample_depth`.
    shape_factor, pore_length_fraction, min_core_fraction :
        Same geometric controls as :func:`update_network_geometry_from_radii`.

    Returns
    -------
    None
        The network is updated in place.

    Notes
    -----
    In quasi-2D mode, pore volume is approximated as ``V_p = A_p * depth`` with
    ``A_p = pi r^2``. Throat volume uses the same ``A_t * L_core`` model used in
    the 3D helper.
    """

    _validate_geometry_update_controls(
        shape_factor=shape_factor,
        pore_length_fraction=pore_length_fraction,
        min_core_fraction=min_core_fraction,
    )
    pore_r = _as_float_vector(pore_radius, expected_size=net.Np, name="pore_radius")
    throat_r = _as_float_vector(throat_radius, expected_size=net.Nt, name="throat_radius")
    if np.any(pore_r <= 0) or np.any(throat_r <= 0):
        raise ValueError("pore_radius and throat_radius must be strictly positive")

    slab_depth = float(sample_depth(net) if depth is None else depth)
    if slab_depth <= 0:
        raise ValueError("depth must be positive")

    pore_area = np.pi * pore_r**2
    pore_perimeter = 2.0 * np.pi * pore_r
    pore_volume = pore_area * slab_depth

    net.pore["radius_inscribed"] = pore_r
    net.pore["diameter_inscribed"] = 2.0 * pore_r
    net.pore["area"] = pore_area
    net.pore["perimeter"] = pore_perimeter
    net.pore["volume"] = pore_volume
    net.pore["shape_factor"] = np.full(net.Np, float(shape_factor), dtype=float)

    conns = np.asarray(net.throat_conns, dtype=int)
    c0 = net.pore_coords[conns[:, 0]]
    c1 = net.pore_coords[conns[:, 1]]
    direct_length = np.linalg.norm(c1 - c0, axis=1)

    p1_length = np.minimum(pore_r[conns[:, 0]], float(pore_length_fraction) * direct_length)
    p2_length = np.minimum(pore_r[conns[:, 1]], float(pore_length_fraction) * direct_length)
    core_length = np.maximum(
        direct_length - p1_length - p2_length,
        float(min_core_fraction) * direct_length,
    )

    throat_area = np.pi * throat_r**2
    throat_perimeter = 2.0 * np.pi * throat_r
    throat_volume = throat_area * core_length

    net.throat["radius_inscribed"] = throat_r
    net.throat["diameter_inscribed"] = 2.0 * throat_r
    net.throat["area"] = throat_area
    net.throat["perimeter"] = throat_perimeter
    net.throat["shape_factor"] = np.full(net.Nt, float(shape_factor), dtype=float)
    net.throat["length"] = direct_length
    net.throat["direct_length"] = direct_length
    net.throat["pore1_length"] = p1_length
    net.throat["core_length"] = core_length
    net.throat["pore2_length"] = p2_length
    net.throat["volume"] = throat_volume


def insert_vug_superpore(
    net: Network,
    *,
    radii_xyz: tuple[float, float, float],
    center: np.ndarray | tuple[float, float, float] | None = None,
    shape_factor: float = DEFAULT_G_REF,
    connector_neighbor_weight: float = 0.35,
    connector_vug_weight: float = 0.10,
    connector_min_scale: float = 1.10,
    connector_max_scale: float = 2.80,
    pore_length_fraction: float = 0.40,
    pore_length_radius_scale: float = 0.80,
    min_core_fraction: float = 0.02,
) -> tuple[Network, dict[str, object]]:
    """Insert one 3D vug as a super-pore replacing an ellipsoidal region.

    Parameters
    ----------
    net :
        Source network.
    radii_xyz :
        Semi-axes of the ellipsoidal replacement region.
    center :
        Optional vug center coordinate. Defaults to bounding-box center.
    shape_factor :
        Shape factor assigned to inserted pore/throats.
    connector_neighbor_weight, connector_vug_weight :
        Linear weights used to estimate connector throat radii from neighboring
        pore radii and vug inscribed radius.
    connector_min_scale, connector_max_scale :
        Lower/upper clipping multipliers against median existing throat radius.
    pore_length_fraction, pore_length_radius_scale, min_core_fraction :
        Controls for partitioning connector throat length into pore-side and core
        contributions.

    Returns
    -------
    tuple[Network, dict[str, object]]
        ``(net_vug, metadata)`` where ``net_vug`` is the transformed network and
        ``metadata`` summarizes removed pores, boundary neighbors, and masks.

    Raises
    ------
    ValueError
        If radii or controls are invalid.
    RuntimeError
        If the selected region yields no usable interface connections.
    KeyError
        If required pore radius fields are absent.

    Algorithm summary
    -----------------
    1. Identify pores inside an ellipsoid.
    2. Build induced subnetwork excluding those pores.
    3. Add one new vug pore at ``center``.
    4. Connect vug pore to boundary neighbors of removed region.
    5. Rebuild/extend geometry fields and labels.

    Scientific caveat
    -----------------
    The connector geometry is a heuristic closure model. It preserves
    topological continuity and plausible scale relations, but is not a unique
    physical derivation from first principles or direct imaging.
    """

    rx, ry, rz = (float(radii_xyz[0]), float(radii_xyz[1]), float(radii_xyz[2]))
    if min(rx, ry, rz) <= 0:
        raise ValueError("All radii_xyz values must be positive")
    if shape_factor <= 0:
        raise ValueError("shape_factor must be positive")

    base = net.copy()
    coords = np.asarray(base.pore_coords, dtype=float)
    if center is None:
        center_arr = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    else:
        center_arr = np.asarray(center, dtype=float)
    if center_arr.shape != (3,):
        raise ValueError("center must have shape (3,)")

    inside = _ellipsoid_mask(coords, center=center_arr, radii_xyz=(rx, ry, rz))
    if not np.any(inside):
        nearest = int(np.argmin(np.linalg.norm(coords - center_arr[None, :], axis=1)))
        inside[nearest] = True

    conns = np.asarray(base.throat_conns, dtype=int)
    ci = inside[conns[:, 0]]
    cj = inside[conns[:, 1]]
    outside_neighbors = np.unique(np.concatenate([conns[ci & (~cj), 1], conns[(~ci) & cj, 0]]))
    if outside_neighbors.size == 0:
        raise RuntimeError("Vug insertion produced zero interface neighbors")

    subnet, kept_old_idx, _ = induced_subnetwork(base, ~inside)
    old_to_new = -np.ones(base.Np, dtype=int)
    old_to_new[kept_old_idx] = np.arange(kept_old_idx.size, dtype=int)
    boundary_new = old_to_new[outside_neighbors]
    boundary_new = np.unique(boundary_new[boundary_new >= 0])
    if boundary_new.size == 0:
        raise RuntimeError("No boundary pores survived after induced-subnetwork reduction")
    if "radius_inscribed" not in subnet.pore:
        raise KeyError("net.pore['radius_inscribed'] is required for super-pore insertion")

    r_eq = _equivalent_radius_3d((rx, ry, rz))
    r_ins = float(min(rx, ry, rz))

    vug_idx = subnet.Np
    new_coords = np.vstack([subnet.pore_coords, center_arr[None, :]])
    new_conns = np.column_stack([np.full(boundary_new.size, vug_idx, dtype=int), boundary_new])

    net_vug = subnet.copy()
    net_vug.pore_coords = new_coords
    net_vug.throat_conns = np.vstack([subnet.throat_conns, new_conns])

    pore_append = {
        "radius_inscribed": np.array([r_ins], dtype=float),
        "diameter_inscribed": np.array([2.0 * r_ins], dtype=float),
        "area": np.array([np.pi * r_eq**2], dtype=float),
        "perimeter": np.array([2.0 * np.pi * r_eq], dtype=float),
        "shape_factor": np.array([float(shape_factor)], dtype=float),
        "volume": np.array([(4.0 / 3.0) * np.pi * rx * ry * rz], dtype=float),
    }
    _extend_entity_fields(net_vug.pore, n_before=subnet.Np, n_append=1, append_fields=pore_append)

    boundary_coords = net_vug.pore_coords[boundary_new]
    direct_length = np.linalg.norm(boundary_coords - center_arr[None, :], axis=1)
    direct_length = np.maximum(direct_length, 1.0e-12)

    throat_r_median = _median_throat_radius(subnet, fallback=max(0.1 * r_ins, 1.0e-12))
    neigh_r = np.asarray(net_vug.pore["radius_inscribed"], dtype=float)[boundary_new]
    conn_radius = np.clip(
        connector_neighbor_weight * neigh_r + connector_vug_weight * r_ins,
        connector_min_scale * throat_r_median,
        connector_max_scale * throat_r_median,
    )

    p1_length = np.minimum(pore_length_fraction * direct_length, pore_length_radius_scale * r_ins)
    p2_length = np.minimum(pore_length_fraction * direct_length, pore_length_radius_scale * neigh_r)
    core_length = np.maximum(
        direct_length - p1_length - p2_length,
        min_core_fraction * direct_length,
    )

    conn_area = np.pi * conn_radius**2
    conn_perimeter = 2.0 * np.pi * conn_radius
    conn_volume = conn_area * core_length

    throat_append = {
        "radius_inscribed": conn_radius,
        "diameter_inscribed": 2.0 * conn_radius,
        "area": conn_area,
        "perimeter": conn_perimeter,
        "shape_factor": np.full(boundary_new.size, float(shape_factor), dtype=float),
        "length": direct_length,
        "direct_length": direct_length,
        "pore1_length": p1_length,
        "core_length": core_length,
        "pore2_length": p2_length,
        "volume": conn_volume,
    }
    _extend_entity_fields(
        net_vug.throat,
        n_before=subnet.Nt,
        n_append=boundary_new.size,
        append_fields=throat_append,
    )

    _extend_labels(net_vug.pore_labels, n_before=subnet.Np, n_append=1)
    vug_label = np.zeros(net_vug.Np, dtype=bool)
    vug_label[vug_idx] = True
    net_vug.pore_labels["vug"] = vug_label

    _extend_labels(net_vug.throat_labels, n_before=subnet.Nt, n_append=boundary_new.size)
    vug_conn = np.zeros(net_vug.Nt, dtype=bool)
    vug_conn[subnet.Nt :] = True
    net_vug.throat_labels["vug_connection"] = vug_conn

    net_vug.extra = {
        **net_vug.extra,
        "vug_radii_xyz_m": (rx, ry, rz),
        "vug_equivalent_radius_m": float(r_eq),
        "vug_removed_pores": int(inside.sum()),
        "vug_boundary_neighbors": int(boundary_new.size),
    }

    metadata: dict[str, object] = {
        "inside_mask_original": inside,
        "outside_neighbors_original": outside_neighbors,
        "removed_pores": int(inside.sum()),
        "boundary_neighbors": int(boundary_new.size),
        "equivalent_radius_m": float(r_eq),
    }
    return net_vug, metadata


def insert_vug_superpore_2d(
    net: Network,
    *,
    radii_xy: tuple[float, float],
    center_xy: tuple[float, float] | None = None,
    depth: float | None = None,
    shape_factor: float = DEFAULT_G_REF,
    connector_neighbor_weight: float = 0.40,
    connector_vug_weight: float = 0.12,
    connector_min_scale: float = 1.05,
    connector_max_scale: float = 3.00,
    pore_length_fraction: float = 0.42,
    pore_length_radius_scale: float = 0.85,
    min_core_fraction: float = 0.02,
) -> tuple[Network, dict[str, object]]:
    """Insert one 2D vug as a super-pore replacing an elliptical region.

    Parameters
    ----------
    net :
        Source network (typically quasi-2D mesh network).
    radii_xy :
        Semi-axes of the elliptical replacement region in the XY plane.
    center_xy :
        Optional XY center. Defaults to XY bounding-box center.
    depth :
        Optional slab thickness; inferred from sample metadata when omitted.
    shape_factor :
        Shape factor assigned to inserted pore/throats.
    connector_neighbor_weight, connector_vug_weight :
        Weights used to estimate connector throat radii.
    connector_min_scale, connector_max_scale :
        Connector radius clipping multipliers against median throat radius.
    pore_length_fraction, pore_length_radius_scale, min_core_fraction :
        Controls for connector throat length partitioning.

    Returns
    -------
    tuple[Network, dict[str, object]]
        Updated network and diagnostic metadata.

    Notes
    -----
    The inserted 2D vug pore uses:
    - equivalent radius from area matching,
    - pore volume ``pi * rx * ry * depth``,
    - connector geometry from the same heuristic strategy used in 3D.
    """

    rx, ry = (float(radii_xy[0]), float(radii_xy[1]))
    if min(rx, ry) <= 0:
        raise ValueError("All radii_xy values must be positive")
    if shape_factor <= 0:
        raise ValueError("shape_factor must be positive")

    base = net.copy()
    coords = np.asarray(base.pore_coords, dtype=float)
    if center_xy is None:
        center_xy = (
            float(0.5 * (coords[:, 0].min() + coords[:, 0].max())),
            float(0.5 * (coords[:, 1].min() + coords[:, 1].max())),
        )

    inside = _ellipse_mask_2d(coords, center_xy=center_xy, radii_xy=(rx, ry))
    if not np.any(inside):
        nearest = int(
            np.argmin((coords[:, 0] - center_xy[0]) ** 2 + (coords[:, 1] - center_xy[1]) ** 2)
        )
        inside[nearest] = True

    conns = np.asarray(base.throat_conns, dtype=int)
    ci = inside[conns[:, 0]]
    cj = inside[conns[:, 1]]
    outside_neighbors = np.unique(np.concatenate([conns[ci & (~cj), 1], conns[(~ci) & cj, 0]]))
    if outside_neighbors.size == 0:
        raise RuntimeError("Vug insertion produced zero interface neighbors")

    subnet, kept_old_idx, _ = induced_subnetwork(base, ~inside)
    old_to_new = -np.ones(base.Np, dtype=int)
    old_to_new[kept_old_idx] = np.arange(kept_old_idx.size, dtype=int)
    boundary_new = old_to_new[outside_neighbors]
    boundary_new = np.unique(boundary_new[boundary_new >= 0])
    if boundary_new.size == 0:
        raise RuntimeError("No interface pores remained after induced-subnetwork reduction")
    if "radius_inscribed" not in subnet.pore:
        raise KeyError("net.pore['radius_inscribed'] is required for super-pore insertion")

    r_eq = _equivalent_radius_2d((rx, ry))
    r_ins = float(min(rx, ry))
    slab_depth = float(sample_depth(subnet) if depth is None else depth)
    if slab_depth <= 0:
        raise ValueError("depth must be positive")

    vug_idx = subnet.Np
    zref = float(np.mean(subnet.pore_coords[:, 2]))
    center3 = np.array([center_xy[0], center_xy[1], zref], dtype=float)
    new_coords = np.vstack([subnet.pore_coords, center3[None, :]])
    new_conns = np.column_stack([np.full(boundary_new.size, vug_idx, dtype=int), boundary_new])

    net_vug = subnet.copy()
    net_vug.pore_coords = new_coords
    net_vug.throat_conns = np.vstack([subnet.throat_conns, new_conns])

    pore_append = {
        "radius_inscribed": np.array([r_ins], dtype=float),
        "diameter_inscribed": np.array([2.0 * r_ins], dtype=float),
        "area": np.array([np.pi * r_eq**2], dtype=float),
        "perimeter": np.array([2.0 * np.pi * r_eq], dtype=float),
        "shape_factor": np.array([float(shape_factor)], dtype=float),
        "volume": np.array([np.pi * rx * ry * slab_depth], dtype=float),
    }
    _extend_entity_fields(net_vug.pore, n_before=subnet.Np, n_append=1, append_fields=pore_append)

    boundary_coords = net_vug.pore_coords[boundary_new]
    direct_length = np.linalg.norm(boundary_coords - center3[None, :], axis=1)
    direct_length = np.maximum(direct_length, 1.0e-12)

    throat_r_median = _median_throat_radius(subnet, fallback=max(0.1 * r_ins, 1.0e-12))
    neigh_r = np.asarray(net_vug.pore["radius_inscribed"], dtype=float)[boundary_new]
    conn_radius = np.clip(
        connector_neighbor_weight * neigh_r + connector_vug_weight * r_ins,
        connector_min_scale * throat_r_median,
        connector_max_scale * throat_r_median,
    )

    p1_length = np.minimum(pore_length_fraction * direct_length, pore_length_radius_scale * r_ins)
    p2_length = np.minimum(pore_length_fraction * direct_length, pore_length_radius_scale * neigh_r)
    core_length = np.maximum(
        direct_length - p1_length - p2_length,
        min_core_fraction * direct_length,
    )

    conn_area = np.pi * conn_radius**2
    conn_perimeter = 2.0 * np.pi * conn_radius
    conn_volume = conn_area * core_length

    throat_append = {
        "radius_inscribed": conn_radius,
        "diameter_inscribed": 2.0 * conn_radius,
        "area": conn_area,
        "perimeter": conn_perimeter,
        "shape_factor": np.full(boundary_new.size, float(shape_factor), dtype=float),
        "length": direct_length,
        "direct_length": direct_length,
        "pore1_length": p1_length,
        "core_length": core_length,
        "pore2_length": p2_length,
        "volume": conn_volume,
    }
    _extend_entity_fields(
        net_vug.throat,
        n_before=subnet.Nt,
        n_append=boundary_new.size,
        append_fields=throat_append,
    )

    _extend_labels(net_vug.pore_labels, n_before=subnet.Np, n_append=1)
    vug_label = np.zeros(net_vug.Np, dtype=bool)
    vug_label[vug_idx] = True
    net_vug.pore_labels["vug"] = vug_label

    _extend_labels(net_vug.throat_labels, n_before=subnet.Nt, n_append=boundary_new.size)
    vug_conn = np.zeros(net_vug.Nt, dtype=bool)
    vug_conn[subnet.Nt :] = True
    net_vug.throat_labels["vug_connection"] = vug_conn

    net_vug.extra = {
        **net_vug.extra,
        "vug_radii_xy_m": (rx, ry),
        "vug_equivalent_radius_m": float(r_eq),
        "vug_removed_pores": int(inside.sum()),
        "vug_boundary_neighbors": int(boundary_new.size),
    }

    metadata: dict[str, object] = {
        "inside_mask_original": inside,
        "removed_pores": int(inside.sum()),
        "boundary_neighbors": int(boundary_new.size),
        "equivalent_radius_m": float(r_eq),
        "center_xy": center_xy,
    }
    return net_vug, metadata


insert_vug_superpore_3d = insert_vug_superpore


__all__ = [
    "sample_depth",
    "update_network_geometry_from_radii",
    "update_network_geometry_2d",
    "insert_vug_superpore",
    "insert_vug_superpore_2d",
    "insert_vug_superpore_3d",
]
