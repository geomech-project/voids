from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


_AXES = ("x", "y", "z")
_CIRCULAR_SHAPE_FACTOR = 1.0 / (4.0 * np.pi)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Normalize and validate the requested mesh shape.

    Parameters
    ----------
    shape :
        Sequence of grid sizes in each active dimension.

    Returns
    -------
    tuple[int, ...]
        Normalized shape tuple with length 2 or 3.

    Raises
    ------
    ValueError
        If the dimensionality is unsupported or any axis has fewer than two
        nodes.
    """

    dims = tuple(int(n) for n in shape)
    if len(dims) not in {2, 3}:
        raise ValueError("shape must have length 2 or 3, e.g. (20, 20) or (20, 20, 20)")
    if any(n < 2 for n in dims):
        raise ValueError(
            "each entry in shape must be >= 2 to define distinct inlet/outlet boundaries"
        )
    return dims


def _build_cartesian_connectivity(shape3: tuple[int, int, int], ndim: int) -> np.ndarray:
    """Build nearest-neighbor connectivity for a Cartesian node lattice.

    Parameters
    ----------
    shape3 :
        Three-dimensional shape, with inactive dimensions set to one.
    ndim :
        Number of active dimensions.

    Returns
    -------
    numpy.ndarray
        Integer throat connectivity array with shape ``(Nt, 2)``.
    """

    node_ids = np.arange(np.prod(shape3), dtype=np.int64).reshape(shape3)
    conns: list[np.ndarray] = []
    for axis in range(ndim):
        start = [slice(None)] * 3
        stop = [slice(None)] * 3
        start[axis] = slice(0, -1)
        stop[axis] = slice(1, None)
        conns.append(
            np.column_stack(
                [
                    node_ids[tuple(start)].ravel(),
                    node_ids[tuple(stop)].ravel(),
                ]
            )
        )
    return np.vstack(conns)


def _build_boundary_labels(shape3: tuple[int, int, int], ndim: int) -> dict[str, np.ndarray]:
    """Construct pore-label masks for Cartesian boundary planes.

    Parameters
    ----------
    shape3 :
        Three-dimensional shape, with inactive dimensions set to one.
    ndim :
        Number of active dimensions.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary of boolean pore masks including inlet, outlet, and boundary
        labels.
    """

    node_ids = np.arange(np.prod(shape3), dtype=np.int64).reshape(shape3)
    labels: dict[str, np.ndarray] = {"all": np.ones(node_ids.size, dtype=bool)}
    boundary = np.zeros(node_ids.size, dtype=bool)
    for axis_index, axis_name in enumerate(_AXES[:ndim]):
        inlet = np.zeros(node_ids.size, dtype=bool)
        outlet = np.zeros(node_ids.size, dtype=bool)
        inlet[node_ids.take(indices=0, axis=axis_index).ravel()] = True
        outlet[node_ids.take(indices=shape3[axis_index] - 1, axis=axis_index).ravel()] = True
        labels[f"{axis_name}min"] = inlet.copy()
        labels[f"{axis_name}max"] = outlet.copy()
        labels[f"inlet_{axis_name}min"] = inlet
        labels[f"outlet_{axis_name}max"] = outlet
        boundary |= inlet | outlet
    labels["boundary"] = boundary
    return labels


def make_cartesian_mesh_network(
    shape: Sequence[int],
    *,
    spacing: float = 1.0,
    pore_radius: float | None = None,
    throat_radius: float | None = None,
    thickness: float | None = None,
    units: dict[str, str] | None = None,
) -> Network:
    """Build a regular mesh-like pore network with one pore per mesh node.

    Parameters
    ----------
    shape :
        Number of pores along each active axis. Typical examples are ``(20, 20)``
        and ``(20, 20, 20)``.
    spacing :
        Center-to-center pore spacing.
    pore_radius, throat_radius :
        Synthetic geometric radii used to construct pore and throat attributes.
    thickness :
        Extrusion thickness for 2-D meshes. Ignored for 3-D meshes.
    units :
        Optional unit metadata stored in :class:`SampleGeometry`.

    Returns
    -------
    Network
        Synthetic Cartesian lattice network with geometry, labels, and sample
        metadata.

    Raises
    ------
    ValueError
        If the shape, spacing, or geometric radii are invalid.

    Notes
    -----
    Each mesh node becomes one pore, and each nearest-neighbor pair becomes one
    throat. The resulting graph is a regular square or cubic lattice. For the
    current synthetic geometry model, the throat core length is

    ``L_core = spacing - 2 * pore_radius``

    and the throat volume is approximated as

    ``V_throat = A_throat * L_core``.

    This makes the example useful for solver verification and scaling studies,
    while remaining intentionally simpler than an image-derived pore network.
    """

    dims = _normalize_shape(shape)
    ndim = len(dims)
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    pore_radius = 0.2 * spacing if pore_radius is None else float(pore_radius)
    throat_radius = 0.1 * spacing if throat_radius is None else float(throat_radius)
    if pore_radius <= 0 or throat_radius <= 0:
        raise ValueError("pore_radius and throat_radius must be positive")
    if pore_radius >= 0.5 * spacing:
        raise ValueError("pore_radius must be smaller than half the pore spacing")
    if throat_radius >= 0.5 * spacing:
        raise ValueError("throat_radius must be smaller than half the pore spacing")

    if ndim == 2:
        nz = 1
        depth = float(spacing if thickness is None else thickness)
        if depth <= 0:
            raise ValueError("thickness must be positive for 2D meshes")
        shape3 = (dims[0], dims[1], nz)
        x = (np.arange(dims[0], dtype=float) + 0.5) * spacing
        y = (np.arange(dims[1], dtype=float) + 0.5) * spacing
        z = np.array([0.5 * depth], dtype=float)
        pore_volume_scalar = np.pi * pore_radius**2 * depth
        cross_sections = {
            "x": dims[1] * spacing * depth,
            "y": dims[0] * spacing * depth,
        }
        lengths = {
            "x": dims[0] * spacing,
            "y": dims[1] * spacing,
        }
        bulk_volume = dims[0] * dims[1] * spacing**2 * depth
    else:
        shape3 = (dims[0], dims[1], dims[2])
        x = (np.arange(dims[0], dtype=float) + 0.5) * spacing
        y = (np.arange(dims[1], dtype=float) + 0.5) * spacing
        z = (np.arange(dims[2], dtype=float) + 0.5) * spacing
        pore_volume_scalar = (4.0 / 3.0) * np.pi * pore_radius**3
        cross_sections = {
            "x": dims[1] * dims[2] * spacing**2,
            "y": dims[0] * dims[2] * spacing**2,
            "z": dims[0] * dims[1] * spacing**2,
        }
        lengths = {
            "x": dims[0] * spacing,
            "y": dims[1] * spacing,
            "z": dims[2] * spacing,
        }
        bulk_volume = dims[0] * dims[1] * dims[2] * spacing**3

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pore_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    throat_conns = _build_cartesian_connectivity(shape3, ndim=ndim)
    pore_labels = _build_boundary_labels(shape3, ndim=ndim)

    throat_area_scalar = np.pi * throat_radius**2
    throat_perimeter_scalar = 2.0 * np.pi * throat_radius
    throat_core_length_scalar = spacing - 2.0 * pore_radius
    if throat_core_length_scalar <= 0:  # pragma: no cover - guarded by pore_radius < spacing / 2
        raise ValueError(
            "pore_radius is too large relative to spacing; throat core length must stay positive"
        )

    pore_area_scalar = np.pi * pore_radius**2
    pore_perimeter_scalar = 2.0 * np.pi * pore_radius
    n_pores = pore_coords.shape[0]
    n_throats = throat_conns.shape[0]

    pore = {
        "volume": np.full(n_pores, pore_volume_scalar, dtype=float),
        "area": np.full(n_pores, pore_area_scalar, dtype=float),
        "perimeter": np.full(n_pores, pore_perimeter_scalar, dtype=float),
        "shape_factor": np.full(n_pores, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "radius_inscribed": np.full(n_pores, pore_radius, dtype=float),
        "diameter_inscribed": np.full(n_pores, 2.0 * pore_radius, dtype=float),
    }
    throat = {
        "volume": np.full(n_throats, throat_area_scalar * throat_core_length_scalar, dtype=float),
        "area": np.full(n_throats, throat_area_scalar, dtype=float),
        "perimeter": np.full(n_throats, throat_perimeter_scalar, dtype=float),
        "shape_factor": np.full(n_throats, _CIRCULAR_SHAPE_FACTOR, dtype=float),
        "radius_inscribed": np.full(n_throats, throat_radius, dtype=float),
        "diameter_inscribed": np.full(n_throats, 2.0 * throat_radius, dtype=float),
        "length": np.full(n_throats, spacing, dtype=float),
        "direct_length": np.full(n_throats, spacing, dtype=float),
        "pore1_length": np.full(n_throats, pore_radius, dtype=float),
        "core_length": np.full(n_throats, throat_core_length_scalar, dtype=float),
        "pore2_length": np.full(n_throats, pore_radius, dtype=float),
    }

    sample = SampleGeometry(
        bulk_volume=float(bulk_volume),
        lengths={k: float(v) for k, v in lengths.items()},
        cross_sections={k: float(v) for k, v in cross_sections.items()},
        units=units or {"length": "m", "pressure": "Pa"},
    )
    provenance = Provenance(
        source_kind="synthetic_mesh",
        extraction_method="cartesian_lattice",
        voxel_size_original=float(spacing),
        user_notes={"shape": list(dims)},
    )

    return Network(
        throat_conns=throat_conns,
        pore_coords=pore_coords,
        sample=sample,
        provenance=provenance,
        pore=pore,
        throat=throat,
        pore_labels=pore_labels,
        extra={
            "mesh_shape": tuple(dims),
            "mesh_spacing": float(spacing),
            "mesh_ndim": ndim,
        },
    )
