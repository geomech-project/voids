from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import porespy as ps

from voids.core.network import Network
from voids.version import __version__ as _voids_version
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.core.validation import validate_network
from voids.geom.hydraulic import DEFAULT_G_REF
from voids.graph import spanning_subnetwork
from voids.image.maximal_ball import (
    MaximalBallSettings,
    extract_maximal_ball_network_dict,
)
from voids.image.prego import PregoSettings, extract_prego_network_dict
from voids.io.pnflow_cnm import load_pnflow_cnm
from voids.io.porespy import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry

_IMPERIAL_SNOW2_DEFAULTS: dict[str, object] = {
    "sigma": 1.0,
    "r_max": 4,
    "boundary_width": 1,
}
_PORESPY_STYLE_IMAGE_BACKENDS = frozenset({"porespy_snow2", "porespy_snow2_imperial", "prego"})
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(slots=True)
class NetworkExtractionResult:
    """Store outputs of an image-to-network extraction workflow.

    Attributes
    ----------
    image :
        Input phase image used for extraction.
    voxel_size :
        Physical voxel edge length.
    axis_lengths :
        Sample lengths by axis.
    axis_areas :
        Cross-sectional areas normal to each axis.
    flow_axis :
        Axis used for spanning subnetwork pruning.
    network_dict :
        Intermediate extracted network mapping before conversion to
        :class:`voids.core.network.Network`.
    sample :
        Sample geometry attached to the network.
    provenance :
        Extraction provenance metadata.
    net_full :
        Full imported network before spanning pruning.
    net :
        Axis-spanning subnetwork.
    pore_indices :
        Indices of retained pores in ``net_full``.
    throat_mask :
        Mask of retained throats in ``net_full``.
    backend :
        Extraction backend identifier (currently ``"porespy"``).
    backend_version :
        Backend version string when available.
    """

    image: np.ndarray
    voxel_size: float
    axis_lengths: dict[str, float]
    axis_areas: dict[str, float]
    flow_axis: str
    network_dict: dict[str, object]
    sample: SampleGeometry
    provenance: Provenance
    net_full: Network
    net: Network
    pore_indices: np.ndarray
    throat_mask: np.ndarray
    backend: str
    backend_version: str | None


@dataclass(slots=True)
class NetworkConstructionResult:
    """Store outputs of a general network-construction workflow.

    This result type covers both image-based extraction backends and imported
    external-network backends such as Imperial CNM text files.
    """

    backend: str
    flow_axis: str
    sample: SampleGeometry
    provenance: Provenance
    net_full: Network
    net: Network
    image: np.ndarray | None = None
    voxel_size: float | None = None
    axis_lengths: dict[str, float] | None = None
    axis_areas: dict[str, float] | None = None
    network_dict: dict[str, object] | None = None
    pore_indices: np.ndarray | None = None
    throat_mask: np.ndarray | None = None
    backend_version: str | None = None
    backend_details: dict[str, object] = field(default_factory=dict)


def infer_sample_axes(
    shape: tuple[int, ...],
    *,
    voxel_size: float,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> tuple[dict[str, int], dict[str, float], dict[str, float], str]:
    """Infer per-axis counts, lengths, areas, and the longest flow axis.

    Parameters
    ----------
    shape :
        Image shape in voxel counts.
    voxel_size :
        Edge length of one voxel in the target length unit.
    axis_names :
        Axis labels mapped onto the image shape.

    Returns
    -------
    tuple
        ``(axis_counts, axis_lengths, axis_areas, flow_axis)``.
    """

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(shape) not in {2, 3}:
        raise ValueError("shape must have length 2 or 3")
    if len(axis_names) < len(shape):
        raise ValueError("axis_names must cover every image dimension")

    active_axes = axis_names[: len(shape)]
    axis_counts = {ax: int(n) for ax, n in zip(active_axes, shape)}
    axis_lengths = {ax: count * float(voxel_size) for ax, count in axis_counts.items()}
    axis_areas: dict[str, float] = {}
    for ax in active_axes:
        others = [other for other in active_axes if other != ax]
        area = float(voxel_size) ** max(len(others), 1)
        for other in others:
            area *= axis_counts[other]
        axis_areas[ax] = area
    flow_axis = max(axis_lengths, key=lambda ax: axis_lengths[ax])
    return axis_counts, axis_lengths, axis_areas, flow_axis


def _snow2_network_dict(
    phases: np.ndarray,
    *,
    snow2_kwargs: dict[str, object] | None,
    porespy_module: Any = ps,
) -> dict[str, object]:
    """Run ``porespy.networks.snow2`` and normalize its network mapping output.

    Parameters
    ----------
    phases :
        Integer phase image passed to the extraction backend.
    snow2_kwargs :
        Keyword arguments forwarded to ``networks.snow2``.
    porespy_module :
        PoreSpy-like module object exposing ``networks.snow2`` and
        ``networks.regions_to_network``. Defaults to the installed ``porespy``
        module and is injectable for deterministic tests.
    """

    kwargs = dict(snow2_kwargs or {})
    snow = porespy_module.networks.snow2(phases=phases, **kwargs)
    if hasattr(snow, "network"):
        return dict(snow.network)
    if isinstance(snow, dict) and "network" in snow:
        return dict(snow["network"])
    if isinstance(snow, dict) and "throat.conns" in snow and "pore.coords" in snow:
        return dict(snow)
    regions = getattr(snow, "regions", None)
    if regions is None and isinstance(snow, dict):
        regions = snow.get("regions", None)
    if regions is None:
        raise RuntimeError("Could not find a network dict or regions in snow2 result")
    return dict(porespy_module.networks.regions_to_network(regions))


def _normalize_extraction_backend(backend: str) -> str:
    """Normalize public extraction-backend aliases."""

    normalized = str(backend).strip().lower()
    aliases = {
        "porespy": "porespy_snow2",
        "porespy_snow2": "porespy_snow2",
        "snow2": "porespy_snow2",
        "porespy_snow2_imperial": "porespy_snow2_imperial",
        "porespy_imperial": "porespy_snow2_imperial",
        "imperial_snow2": "porespy_snow2_imperial",
        "snow2_imperial": "porespy_snow2_imperial",
        "prego": "prego",
        "maximal_ball": "native_maximal_ball",
        "native_maximal_ball": "native_maximal_ball",
        "maxball": "native_maximal_ball",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported extraction backend {backend!r}; expected one of {sorted(aliases)}"
        )
    return aliases[normalized]


def _normalize_construction_backend(backend: str) -> str:
    """Normalize public network-construction backend aliases."""

    normalized = str(backend).strip().lower()
    aliases = {
        "porespy": "porespy_snow2",
        "porespy_snow2": "porespy_snow2",
        "snow2": "porespy_snow2",
        "porespy_snow2_imperial": "porespy_snow2_imperial",
        "porespy_imperial": "porespy_snow2_imperial",
        "imperial_snow2": "porespy_snow2_imperial",
        "snow2_imperial": "porespy_snow2_imperial",
        "prego": "prego",
        "maximal_ball": "native_maximal_ball",
        "native_maximal_ball": "native_maximal_ball",
        "maxball": "native_maximal_ball",
        "pnflow_cnm": "pnflow_cnm",
        "imperial_cnm": "pnflow_cnm",
        "pnextract_cnm": "pnflow_cnm",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported construction backend {backend!r}; expected one of {sorted(aliases)}"
        )
    return aliases[normalized]


def _merge_provenance_notes(provenance: Provenance, notes: dict[str, object] | None) -> Provenance:
    """Return provenance with extra user notes merged in."""

    if not notes:
        return provenance
    merged = Provenance.from_metadata(provenance.to_metadata())
    merged.user_notes = {**merged.user_notes, **dict(notes)}
    return merged


def _resolve_flow_boundary_mode(flow_boundary_mode: str) -> str:
    """Normalize the network boundary treatment used for flow solves."""

    normalized = str(flow_boundary_mode).strip().lower()
    if normalized not in {"direct", "external_reservoir"}:
        raise ValueError("flow_boundary_mode must be one of {'direct', 'external_reservoir'}")
    return normalized


def _resolve_transport_geometry(transport_geometry: object) -> str | None:
    """Normalize optional post-extraction hydraulic geometry enrichment."""

    if transport_geometry is None:
        return None
    normalized = str(transport_geometry).strip().lower()
    aliases = {
        "none": None,
        "direct": None,
        "pyramids_and_cuboids": "pyramids_and_cuboids",
        "openpnm_pyramids_and_cuboids": "pyramids_and_cuboids",
        "truncated_pyramids_and_cuboids": "pyramids_and_cuboids",
    }
    if normalized not in aliases:
        raise ValueError("transport_geometry must be None or one of {'pyramids_and_cuboids'}")
    return aliases[normalized]


def _entity_radius_from_fields(net: Network, kind: str, *, fallback_radius: float) -> np.ndarray:
    """Return positive pore or throat radii from canonical fields when possible."""

    store = net.pore if kind == "pore" else net.throat
    count = net.Np if kind == "pore" else net.Nt
    if "radius_inscribed" in store:
        radius = np.asarray(store["radius_inscribed"], dtype=float)
    elif "diameter_inscribed" in store:
        radius = 0.5 * np.asarray(store["diameter_inscribed"], dtype=float)
    elif "area" in store:
        radius = np.sqrt(np.asarray(store["area"], dtype=float) / np.pi)
    else:
        radius = np.full(count, float(fallback_radius), dtype=float)
    return np.maximum(radius, float(fallback_radius))


def _entity_diameter_for_pyramids_and_cuboids(
    net: Network, kind: str, *, fallback_diameter: float
) -> np.ndarray:
    """Return positive square/pyramid side-length surrogates for size factors."""

    store = net.pore if kind == "pore" else net.throat
    count = net.Np if kind == "pore" else net.Nt
    minimum = float(np.finfo(float).tiny)
    if "diameter_inscribed" in store:
        diameter = np.asarray(store["diameter_inscribed"], dtype=float)
    elif "radius_inscribed" in store:
        diameter = 2.0 * np.asarray(store["radius_inscribed"], dtype=float)
    elif "area" in store:
        diameter = np.sqrt(np.asarray(store["area"], dtype=float))
    else:
        diameter = np.full(count, float(fallback_diameter), dtype=float)
        minimum = max(float(fallback_diameter), minimum)
    return np.maximum(diameter, minimum)


def _extend_pore_field(
    name: str,
    values: np.ndarray,
    *,
    helper_coords: np.ndarray,
    helper_radii: np.ndarray,
    helper_area: np.ndarray,
    helper_source_indices: np.ndarray,
) -> np.ndarray:
    """Append helper-pore values to an existing pore field."""

    arr = np.asarray(values)
    helper_count = int(helper_radii.size)
    fill_shape = (helper_count, *arr.shape[1:])
    if name in {"radius_inscribed"}:
        fill = helper_radii
    elif name in {"diameter_inscribed", "equivalent_diameter", "extended_diameter"}:
        fill = 2.0 * helper_radii
    elif name == "area":
        fill = helper_area
    elif name in {"volume", "region_volume", "surface_area"}:
        fill = np.zeros(fill_shape, dtype=arr.dtype)
    elif name == "shape_factor":
        fill = np.full(fill_shape, DEFAULT_G_REF, dtype=float)
    elif name in {"local_peak", "global_peak", "geometric_centroid"} and arr.ndim == 2:
        fill = np.zeros(fill_shape, dtype=arr.dtype)
        cols = min(arr.shape[1], helper_coords.shape[1])
        fill[:, :cols] = helper_coords[:, :cols]
    elif name == "phase" and arr.ndim == 1 and helper_source_indices.size:
        fill = arr[helper_source_indices]
    elif np.issubdtype(arr.dtype, np.bool_):
        fill = np.zeros(fill_shape, dtype=bool)
    else:
        fill = np.zeros(fill_shape, dtype=arr.dtype)
    return np.concatenate([arr, np.asarray(fill, dtype=arr.dtype)], axis=0)


def _extend_throat_field(
    name: str,
    values: np.ndarray,
    *,
    boundary_length: np.ndarray,
    boundary_area: np.ndarray,
    boundary_radius: np.ndarray,
    boundary_pore1_length: np.ndarray,
    boundary_core_length: np.ndarray,
    boundary_pore2_length: np.ndarray,
    boundary_centroid: np.ndarray,
) -> np.ndarray:
    """Append helper-throat values to an existing throat field."""

    arr = np.asarray(values)
    helper_count = int(boundary_length.size)
    fill_shape = (helper_count, *arr.shape[1:])
    if name in {"length", "total_length", "direct_length"}:
        fill = boundary_length
    elif name in {"pore1_length"}:
        fill = boundary_pore1_length
    elif name in {"core_length"}:
        fill = boundary_core_length
    elif name in {"pore2_length"}:
        fill = boundary_pore2_length
    elif name in {"area", "cross_sectional_area"}:
        fill = boundary_area
    elif name in {"radius_inscribed", "shape_factor_radius"}:
        fill = boundary_radius
    elif name in {"diameter_inscribed", "equivalent_diameter"}:
        fill = 2.0 * boundary_radius
    elif name == "shape_factor":
        fill = np.full(fill_shape, DEFAULT_G_REF, dtype=float)
    elif name == "volume":
        fill = boundary_area * boundary_core_length
    elif name in {"centroid", "global_peak"} and arr.ndim == 2:
        fill = np.zeros(fill_shape, dtype=arr.dtype)
        cols = min(arr.shape[1], boundary_centroid.shape[1])
        fill[:, :cols] = boundary_centroid[:, :cols]
    elif name == "face_count":
        fill = np.ones(fill_shape, dtype=arr.dtype)
    elif name == "hydraulic_conductance":
        raise ValueError(
            "external_reservoir boundary mode cannot extend precomputed "
            "throat.hydraulic_conductance"
        )
    elif np.issubdtype(arr.dtype, np.bool_):
        fill = np.zeros(fill_shape, dtype=bool)
    else:
        fill = np.zeros(fill_shape, dtype=arr.dtype)
    return np.concatenate([arr, np.asarray(fill, dtype=arr.dtype)], axis=0)


def _add_external_reservoirs_to_network(
    net: Network,
    *,
    axis: str,
    axis_length: float,
    voxel_size: float,
    boundary_length_epsilon: float,
    boundary_radius_scale: float,
) -> Network:
    """Attach zero-volume helper pores to PoreSpy-style boundary pores."""

    if axis not in _AXIS_INDEX:
        raise ValueError("boundary_axis must be one of {'x', 'y', 'z'}")
    if boundary_length_epsilon <= 0.0:
        raise ValueError("boundary_length_epsilon must be positive")
    if boundary_radius_scale <= 0.0:
        raise ValueError("boundary_radius_scale must be positive")
    if "hydraulic_conductance" in net.throat:
        raise ValueError(
            "external_reservoir boundary mode cannot extend networks with "
            "precomputed throat.hydraulic_conductance"
        )

    inlet_label = f"inlet_{axis}min"
    outlet_label = f"outlet_{axis}max"
    if inlet_label not in net.pore_labels or outlet_label not in net.pore_labels:
        raise KeyError(f"Missing pore boundary labels {inlet_label!r} and/or {outlet_label!r}")

    axis_index = _AXIS_INDEX[axis]
    inlet_mask = np.asarray(net.pore_labels[inlet_label], dtype=bool)
    outlet_mask = np.asarray(net.pore_labels[outlet_label], dtype=bool)
    boundary_specs: list[tuple[int, str]] = [
        *[(int(i), "lower") for i in np.flatnonzero(inlet_mask)],
        *[(int(i), "upper") for i in np.flatnonzero(outlet_mask)],
    ]
    if not boundary_specs:
        return net

    fallback_radius = max(0.5 * float(voxel_size), boundary_length_epsilon)
    pore_radii = _entity_radius_from_fields(net, "pore", fallback_radius=fallback_radius)
    helper_coords: list[np.ndarray] = []
    helper_radii: list[float] = []
    helper_source_indices: list[int] = []
    boundary_conns: list[tuple[int, int]] = []
    boundary_lengths: list[float] = []
    boundary_areas: list[float] = []
    boundary_radii: list[float] = []
    boundary_pore1_lengths: list[float] = []
    boundary_core_lengths: list[float] = []
    boundary_pore2_lengths: list[float] = []
    boundary_centroids: list[np.ndarray] = []
    helper_inlet = np.zeros(len(boundary_specs), dtype=bool)
    helper_outlet = np.zeros(len(boundary_specs), dtype=bool)

    for helper_offset, (pore_index, side) in enumerate(boundary_specs):
        helper_index = net.Np + helper_offset
        boundary_coordinate = 0.0 if side == "lower" else float(axis_length)
        source_coord = net.pore_coords[pore_index]
        helper_coord = source_coord.copy()
        helper_coord[axis_index] = boundary_coordinate
        contact_radius = max(float(pore_radii[pore_index]), fallback_radius)
        helper_radius = boundary_radius_scale * contact_radius
        center_to_boundary = abs(float(source_coord[axis_index] - boundary_coordinate))
        total_length = max(
            center_to_boundary, 3.01 * float(voxel_size), 3.0 * boundary_length_epsilon
        )
        internal_pore_length = max(0.67 * center_to_boundary, boundary_length_epsilon)
        core_length = max(
            total_length - boundary_length_epsilon - internal_pore_length,
            boundary_length_epsilon,
        )
        length = boundary_length_epsilon + core_length + internal_pore_length

        helper_coords.append(helper_coord)
        helper_radii.append(helper_radius)
        helper_source_indices.append(pore_index)
        boundary_lengths.append(length)
        boundary_areas.append(np.pi * contact_radius**2)
        boundary_radii.append(contact_radius)
        boundary_core_lengths.append(core_length)
        boundary_centroids.append(helper_coord.copy())
        if side == "lower":
            boundary_conns.append((helper_index, pore_index))
            boundary_pore1_lengths.append(boundary_length_epsilon)
            boundary_pore2_lengths.append(internal_pore_length)
            helper_inlet[helper_offset] = True
        else:
            boundary_conns.append((pore_index, helper_index))
            boundary_pore1_lengths.append(internal_pore_length)
            boundary_pore2_lengths.append(boundary_length_epsilon)
            helper_outlet[helper_offset] = True

    helper_coords_arr = np.asarray(helper_coords, dtype=float)
    helper_radii_arr = np.asarray(helper_radii, dtype=float)
    helper_source_arr = np.asarray(helper_source_indices, dtype=np.int64)
    boundary_conns_arr = np.asarray(boundary_conns, dtype=np.int64)
    boundary_lengths_arr = np.asarray(boundary_lengths, dtype=float)
    boundary_area_arr = np.asarray(boundary_areas, dtype=float)
    boundary_radius_arr = np.asarray(boundary_radii, dtype=float)
    boundary_pore1_arr = np.asarray(boundary_pore1_lengths, dtype=float)
    boundary_core_arr = np.asarray(boundary_core_lengths, dtype=float)
    boundary_pore2_arr = np.asarray(boundary_pore2_lengths, dtype=float)
    boundary_centroid_arr = np.asarray(boundary_centroids, dtype=float)
    helper_area_arr = np.pi * helper_radii_arr**2

    pore_labels: dict[str, np.ndarray] = {}
    for name, mask in net.pore_labels.items():
        old = np.asarray(mask, dtype=bool)
        if name == inlet_label:
            new = np.zeros(net.Np + helper_radii_arr.size, dtype=bool)
            new[net.Np :] = helper_inlet
        elif name == outlet_label:
            new = np.zeros(net.Np + helper_radii_arr.size, dtype=bool)
            new[net.Np :] = helper_outlet
        elif name == "boundary":
            new = np.concatenate([old, np.ones(helper_radii_arr.size, dtype=bool)])
        else:
            new = np.concatenate([old, np.zeros(helper_radii_arr.size, dtype=bool)])
        pore_labels[name] = new
    connected_inlet = np.concatenate([inlet_mask, np.zeros(helper_radii_arr.size, dtype=bool)])
    connected_outlet = np.concatenate([outlet_mask, np.zeros(helper_radii_arr.size, dtype=bool)])
    pore_labels[f"boundary_connected_{inlet_label}"] = connected_inlet
    pore_labels[f"boundary_connected_{outlet_label}"] = connected_outlet
    pore_labels.setdefault(
        "boundary",
        np.concatenate([np.zeros(net.Np, dtype=bool), np.ones(helper_radii_arr.size, dtype=bool)]),
    )

    throat_labels = {
        name: np.concatenate(
            [np.asarray(mask, dtype=bool), np.zeros(boundary_conns_arr.shape[0], dtype=bool)]
        )
        for name, mask in net.throat_labels.items()
    }
    throat_labels["boundary_reservoir"] = np.concatenate(
        [np.zeros(net.Nt, dtype=bool), np.ones(boundary_conns_arr.shape[0], dtype=bool)]
    )

    out = Network(
        throat_conns=np.vstack([net.throat_conns, boundary_conns_arr]),
        pore_coords=np.vstack([net.pore_coords, helper_coords_arr]),
        sample=net.sample,
        provenance=net.provenance,
        schema_version=net.schema_version,
        pore={
            name: _extend_pore_field(
                name,
                values,
                helper_coords=helper_coords_arr,
                helper_radii=helper_radii_arr,
                helper_area=helper_area_arr,
                helper_source_indices=helper_source_arr,
            )
            for name, values in net.pore.items()
        },
        throat={
            name: _extend_throat_field(
                name,
                values,
                boundary_length=boundary_lengths_arr,
                boundary_area=boundary_area_arr,
                boundary_radius=boundary_radius_arr,
                boundary_pore1_length=boundary_pore1_arr,
                boundary_core_length=boundary_core_arr,
                boundary_pore2_length=boundary_pore2_arr,
                boundary_centroid=boundary_centroid_arr,
            )
            for name, values in net.throat.items()
        },
        pore_labels=pore_labels,
        throat_labels=throat_labels,
        extra={
            **net.extra,
            "external_reservoirs": {
                "axis": axis,
                "helper_pore_count": int(helper_radii_arr.size),
                "boundary_length_epsilon": float(boundary_length_epsilon),
                "boundary_radius_scale": float(boundary_radius_scale),
            },
        },
    )
    validate_network(out)
    return out


def _assign_pyramids_and_cuboids_transport_geometry(net: Network, *, voxel_size: float) -> Network:
    """Attach OpenPNM-style pyramids-and-cuboids hydraulic size factors."""

    required_lengths = ("pore1_length", "core_length", "pore2_length")
    if not all(name in net.throat for name in required_lengths):
        raise KeyError(
            "pyramids-and-cuboids transport geometry requires conduit lengths "
            "(pore1_length, core_length, pore2_length)"
        )
    fallback_diameter = max(float(voxel_size), float(np.finfo(float).tiny))
    pore_diameter = _entity_diameter_for_pyramids_and_cuboids(
        net,
        "pore",
        fallback_diameter=fallback_diameter,
    )
    throat_diameter = _entity_diameter_for_pyramids_and_cuboids(
        net,
        "throat",
        fallback_diameter=fallback_diameter,
    )
    conns = net.throat_conns
    d1 = pore_diameter[conns[:, 0]]
    d2 = pore_diameter[conns[:, 1]]
    dt_limit = np.minimum(d1, d2) * (1.0 - 1.0e-12)
    dt = np.minimum(throat_diameter, dt_limit)
    dt = np.maximum(dt, np.finfo(float).tiny)
    l1 = np.asarray(net.throat["pore1_length"], dtype=float)
    lt = np.asarray(net.throat["core_length"], dtype=float)
    l2 = np.asarray(net.throat["pore2_length"], dtype=float)

    f1 = (l1 * (d1 * d1 + d1 * dt + dt * dt)) / (3.0 * d1**3 * dt**3)
    ft = lt / dt**4
    f2 = (l2 * (d2 * d2 + d2 * dt + dt * dt)) / (3.0 * d2**3 * dt**3)
    moment_factor = 1.0 / 6.0
    prefactor = 1.0 / (16.0 * np.pi**2 * moment_factor)
    sf = np.column_stack([prefactor / f1, prefactor / ft, prefactor / f2])
    if not np.all(np.isfinite(sf)) or np.any(sf <= 0.0):
        raise ValueError(
            "Computed pyramids-and-cuboids hydraulic size factors must be positive and finite"
        )

    out = net.copy()
    out.throat["hydraulic_size_factors"] = sf
    out.extra["transport_geometry"] = {
        "mode": "pyramids_and_cuboids",
        "size_factor_model": "pyramids_and_cuboids",
        "hydraulic_size_factors_location": "throat.hydraulic_size_factors",
        "throat_diameter_clipped": int(np.count_nonzero(throat_diameter > dt_limit)),
    }
    validate_network(out)
    return out


def _construction_result_from_extraction(
    result: NetworkExtractionResult,
) -> NetworkConstructionResult:
    """Lift an extraction result into the broader construction schema."""

    return NetworkConstructionResult(
        backend=result.backend,
        flow_axis=result.flow_axis,
        sample=result.sample,
        provenance=result.provenance,
        net_full=result.net_full,
        net=result.net,
        image=result.image,
        voxel_size=result.voxel_size,
        axis_lengths=result.axis_lengths,
        axis_areas=result.axis_areas,
        network_dict=result.network_dict,
        pore_indices=result.pore_indices,
        throat_mask=result.throat_mask,
        backend_version=result.backend_version,
    )


def _extract_network_dict(
    phases: np.ndarray,
    *,
    backend: str,
    voxel_size: float,
    extraction_kwargs: dict[str, object] | None,
    flow_axis: str | None,
) -> dict[str, object]:
    """Dispatch image extraction to the requested backend."""

    backend_normalized = _normalize_extraction_backend(backend)
    if backend_normalized == "porespy_snow2":
        return _snow2_network_dict(phases, snow2_kwargs=dict(extraction_kwargs or {}))
    if backend_normalized == "porespy_snow2_imperial":
        kwargs = {
            **_IMPERIAL_SNOW2_DEFAULTS,
            **dict(extraction_kwargs or {}),
        }
        return _snow2_network_dict(phases, snow2_kwargs=kwargs)
    if backend_normalized == "prego":
        kwargs = dict(extraction_kwargs or {})
        settings_value = kwargs.pop("settings", kwargs.pop("prego_settings", None))
        if isinstance(settings_value, dict):
            settings_value = PregoSettings(**settings_value)
        if settings_value is not None and not isinstance(settings_value, PregoSettings):
            raise TypeError(
                "PREGO extraction settings must be a PregoSettings instance, a mapping, or None"
            )
        distance_map = kwargs.pop("distance_map", None)
        peaks = kwargs.pop("peaks", None)
        regions_to_network_kwargs = kwargs.pop("regions_to_network_kwargs", None)
        if regions_to_network_kwargs is not None and not isinstance(
            regions_to_network_kwargs, dict
        ):
            raise TypeError("regions_to_network_kwargs must be a mapping or None")
        if kwargs:
            unexpected_keys = ", ".join(sorted(kwargs))
            raise ValueError(f"Unexpected extraction_kwargs for backend='prego': {unexpected_keys}")
        return extract_prego_network_dict(
            np.asarray(phases, dtype=bool),
            settings=settings_value,
            distance_map=None if distance_map is None else np.asarray(distance_map, dtype=float),
            peaks=None if peaks is None else np.asarray(peaks),
            regions_to_network_kwargs=regions_to_network_kwargs,
        ).network_dict
    if backend_normalized == "native_maximal_ball":
        kwargs = dict(extraction_kwargs or {})
        settings_value = kwargs.pop("settings", kwargs.pop("maximal_ball_settings", None))
        if isinstance(settings_value, dict):
            settings_value = MaximalBallSettings(**settings_value)
        if settings_value is not None and not isinstance(settings_value, MaximalBallSettings):
            raise TypeError(
                "maximal-ball extraction settings must be a MaximalBallSettings instance,"
                " a mapping, or None"
            )
        distance_map_backend = str(kwargs.pop("distance_map_backend", "auto"))
        edt_parallel_threads_value = kwargs.pop("edt_parallel_threads", None)
        edt_parallel_threads = (
            None
            if edt_parallel_threads_value is None
            else int(cast(int | str, edt_parallel_threads_value))
        )
        apply_boundary_clipping = bool(kwargs.pop("apply_boundary_clipping", True))
        flow_boundary_mode = str(kwargs.pop("flow_boundary_mode", "direct"))
        boundary_axis = kwargs.pop("boundary_axis", flow_axis)
        if boundary_axis is not None:
            boundary_axis = str(boundary_axis)
        boundary_length_epsilon = float(
            cast(float | int | str, kwargs.pop("boundary_length_epsilon", 1.0e-300))
        )
        boundary_radius_scale = float(
            cast(float | int | str, kwargs.pop("boundary_radius_scale", 1.1))
        )
        throat_area_mode = str(kwargs.pop("throat_area_mode", "face_count"))
        throat_shape_factor_radius_mode = str(
            kwargs.pop("throat_shape_factor_radius_mode", "inscribed")
        )
        throat_anchor_mode = str(kwargs.pop("throat_anchor_mode", "second_side"))
        if kwargs:
            unexpected_keys = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unexpected extraction_kwargs for backend='maximal_ball': {unexpected_keys}"
            )
        return cast(
            dict[str, object],
            extract_maximal_ball_network_dict(
                np.asarray(phases, dtype=bool),
                voxel_size=float(voxel_size),
                distance_map_backend=distance_map_backend,
                edt_parallel_threads=edt_parallel_threads,
                settings=settings_value,
                apply_boundary_clipping=apply_boundary_clipping,
                flow_boundary_mode=flow_boundary_mode,
                boundary_axis=boundary_axis,
                boundary_length_epsilon=boundary_length_epsilon,
                boundary_radius_scale=boundary_radius_scale,
                throat_area_mode=throat_area_mode,
                throat_shape_factor_radius_mode=throat_shape_factor_radius_mode,
                throat_anchor_mode=throat_anchor_mode,
            ).network_dict,
        )
    raise AssertionError(f"Unhandled normalized backend {backend_normalized!r}")


def extract_spanning_pore_network(
    phases: np.ndarray,
    *,
    voxel_size: float,
    backend: str = "porespy",
    flow_axis: str | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
    geometry_repairs: str | None = "imperial_export",
    repair_seed: int | None = 0,
) -> NetworkExtractionResult:
    """Extract, import, and prune an axis-spanning pore network from an image.

    Parameters
    ----------
    phases :
        Binary or integer-labeled phase image where nonzero values are active
        phases passed to the extraction backend.
    voxel_size :
        Edge length of one voxel in the declared ``length_unit``.
    backend :
        Image-to-network extraction backend. Currently supported values are
        ``"porespy"``, ``"snow2"``, ``"porespy_snow2"``, the calibrated
        approximation aliases ``"porespy_imperial"``, ``"imperial_snow2"``,
        and ``"snow2_imperial"``, the native PREGO backend ``"prego"``,
        plus the native maximal-ball aliases
        ``"maximal_ball"``, ``"native_maximal_ball"``, and ``"maxball"``.
    flow_axis :
        Requested spanning axis. When omitted, the longest image axis is used.
    length_unit, pressure_unit :
        Units stored in resulting :class:`SampleGeometry`.
    extraction_kwargs :
        Keyword arguments forwarded to the extraction backend call. For the
        Imperial-calibrated `snow2` aliases, user-supplied values override the
        built-in defaults ``sigma=1.0``, ``r_max=4``, and ``boundary_width=1``.
        For the PREGO backend, supported keys are ``settings`` or
        ``prego_settings``, ``distance_map``, ``peaks``, and
        ``regions_to_network_kwargs``. PoreSpy-style backends
        (``"porespy"``, the `snow2` aliases, and ``"prego"``) also accept
        post-import transport keys ``flow_boundary_mode``, ``boundary_axis``,
        ``boundary_length_epsilon``, ``boundary_radius_scale``, and
        ``transport_geometry``. For the native maximal-ball backend, supported
        keys are
        ``distance_map_backend``, ``edt_parallel_threads``,
        ``apply_boundary_clipping``,
        ``flow_boundary_mode``, ``boundary_axis``,
        ``boundary_length_epsilon``, ``boundary_radius_scale``,
        ``throat_area_mode``, ``throat_shape_factor_radius_mode``,
        ``throat_anchor_mode``, and either ``settings`` or
        ``maximal_ball_settings``.
    provenance_notes :
        Optional extra provenance metadata attached to the resulting network.
    strict :
        Forwarded to :func:`voids.io.porespy.from_porespy`.
    geometry_repairs :
        Optional importer preprocessing mode. The default
        ``"imperial_export"`` applies the Imperial College export-style
        shape-factor repair heuristics during the PoreSpy-to-``voids``
        conversion.
    repair_seed :
        Seed for any stochastic repair branch when ``geometry_repairs`` is not
        ``None``.

    Returns
    -------
    NetworkExtractionResult
        Full and pruned networks together with intermediate metadata.

    Notes
    -----
    Current implementation uses PoreSpy's ``snow2`` backend and normalizes
    accepted return styles into a standard network mapping before import. The
    calibrated Imperial-style aliases still use `snow2`, but start from a
    benchmark-tuned parameter profile that is closer to the committed
    `pnextract` reference cases than the plain default.
    """

    arr = np.asarray(phases, dtype=int)
    if arr.ndim not in {2, 3}:
        raise ValueError("phases must be a 2D or 3D integer image")

    _, axis_lengths, axis_areas, inferred_axis = infer_sample_axes(arr.shape, voxel_size=voxel_size)
    selected_axis = inferred_axis if flow_axis is None else flow_axis
    if selected_axis not in axis_lengths:
        raise ValueError(f"flow_axis '{selected_axis}' is not compatible with shape {arr.shape}")

    backend_normalized = _normalize_extraction_backend(backend)
    extraction_kwargs_for_backend = dict(extraction_kwargs or {})
    flow_boundary_mode = "direct"
    boundary_axis = selected_axis
    boundary_length_epsilon = 1.0e-300
    boundary_radius_scale = 1.1
    transport_geometry: str | None = None
    if backend_normalized in _PORESPY_STYLE_IMAGE_BACKENDS:
        flow_boundary_mode = _resolve_flow_boundary_mode(
            str(extraction_kwargs_for_backend.pop("flow_boundary_mode", "direct"))
        )
        boundary_axis = str(extraction_kwargs_for_backend.pop("boundary_axis", selected_axis))
        if boundary_axis not in axis_lengths:
            raise ValueError(
                f"boundary_axis '{boundary_axis}' is not compatible with shape {arr.shape}"
            )
        boundary_length_epsilon = float(
            cast(
                float | int | str,
                extraction_kwargs_for_backend.pop("boundary_length_epsilon", 1.0e-300),
            )
        )
        boundary_radius_scale = float(
            cast(float | int | str, extraction_kwargs_for_backend.pop("boundary_radius_scale", 1.1))
        )
        transport_geometry = _resolve_transport_geometry(
            extraction_kwargs_for_backend.pop("transport_geometry", None)
        )
    network_dict = _extract_network_dict(
        arr,
        backend=backend_normalized,
        voxel_size=float(voxel_size),
        extraction_kwargs=extraction_kwargs_for_backend or None,
        flow_axis=selected_axis,
    )
    importer_geometry_repairs = geometry_repairs
    if backend_normalized != "native_maximal_ball":
        network_dict = scale_porespy_geometry(network_dict, voxel_size=voxel_size)
        network_dict = ensure_cartesian_boundary_labels(network_dict, axes=(selected_axis,))
    else:
        importer_geometry_repairs = None

    shape_2d_or_3d = tuple(int(n) for n in arr.shape)
    bulk_shape: tuple[int, int, int] = (
        shape_2d_or_3d[0],
        shape_2d_or_3d[1],
        shape_2d_or_3d[2] if arr.ndim == 3 else 1,
    )
    sample = SampleGeometry(
        voxel_size=float(voxel_size),
        bulk_shape_voxels=bulk_shape,
        lengths=axis_lengths,
        cross_sections=axis_areas,
        units={"length": length_unit, "pressure": pressure_unit},
    )
    if backend_normalized in {"native_maximal_ball", "prego"}:
        source_version: str | None = _voids_version
    else:
        source_version = getattr(ps, "__version__", None)
    provenance = Provenance(
        source_kind="image_extraction",
        source_version=source_version,
        extraction_method=backend_normalized,
        random_seed=repair_seed if geometry_repairs is not None else None,
        user_notes=dict(provenance_notes or {}),
    )
    net_full = from_porespy(
        network_dict,
        sample=sample,
        provenance=provenance,
        strict=strict,
        geometry_repairs=importer_geometry_repairs,
        repair_seed=repair_seed,
    )
    if (
        backend_normalized in _PORESPY_STYLE_IMAGE_BACKENDS
        and flow_boundary_mode == "external_reservoir"
    ):
        net_full = _add_external_reservoirs_to_network(
            net_full,
            axis=boundary_axis,
            axis_length=axis_lengths[boundary_axis],
            voxel_size=float(voxel_size),
            boundary_length_epsilon=boundary_length_epsilon,
            boundary_radius_scale=boundary_radius_scale,
        )
    if transport_geometry == "pyramids_and_cuboids":
        net_full = _assign_pyramids_and_cuboids_transport_geometry(
            net_full,
            voxel_size=float(voxel_size),
        )
    net, pore_indices, throat_mask = spanning_subnetwork(net_full, axis=selected_axis)
    return NetworkExtractionResult(
        image=arr,
        voxel_size=float(voxel_size),
        axis_lengths=axis_lengths,
        axis_areas=axis_areas,
        flow_axis=selected_axis,
        network_dict=network_dict,
        sample=sample,
        provenance=provenance,
        net_full=net_full,
        net=net,
        pore_indices=pore_indices,
        throat_mask=throat_mask,
        backend=backend_normalized,
        backend_version=source_version,
    )


def construct_spanning_network(
    *,
    backend: str,
    phases: np.ndarray | None = None,
    voxel_size: float | None = None,
    pnflow_cnm_prefix: str | Path | None = None,
    pnflow_solver_box_compat: bool = False,
    flow_axis: str | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
    geometry_repairs: str | None = "imperial_export",
    repair_seed: int | None = 0,
) -> NetworkConstructionResult:
    """Construct a pore network from an image backend or imported CNM files.

    Parameters
    ----------
    backend :
        Construction backend identifier. Supported values include the existing
        image-extraction aliases ``"porespy"``, ``"snow2"``,
        ``"porespy_snow2"``, ``"prego"``, the native maximal-ball aliases
        ``"maximal_ball"``, ``"native_maximal_ball"``, ``"maxball"``, and the imported-network aliases
        ``"pnflow_cnm"``, ``"imperial_cnm"``, and ``"pnextract_cnm"``.
    phases, voxel_size :
        Required for image-based backends and forwarded to
        :func:`extract_spanning_pore_network`.
    pnflow_cnm_prefix :
        Required for the Imperial CNM backend. This is the shared path prefix
        before the ``*_node*.dat`` and ``*_link*.dat`` suffixes.
    pnflow_solver_box_compat :
        If ``True`` and ``backend`` selects the Imperial CNM path, reproduce
        the checked-in `pnflow` solver-box preprocessing quirk so the imported
        network matches Imperial single-phase benchmark behavior. Leave this
        ``False`` for a generic CNM import.
    flow_axis, length_unit, pressure_unit, extraction_kwargs, provenance_notes,
    strict, geometry_repairs, repair_seed :
        Forwarded to the selected backend where applicable.

    Returns
    -------
    NetworkConstructionResult
        Unified network-construction result.
    """

    backend_normalized = _normalize_construction_backend(backend)
    if backend_normalized in {
        "porespy_snow2",
        "porespy_snow2_imperial",
        "prego",
        "native_maximal_ball",
    }:
        if phases is None:
            raise ValueError("phases is required for the image-extraction backends")
        if voxel_size is None:
            raise ValueError("voxel_size is required for the image-extraction backends")
        extracted = extract_spanning_pore_network(
            phases,
            voxel_size=float(voxel_size),
            backend=backend_normalized,
            flow_axis=flow_axis,
            length_unit=length_unit,
            pressure_unit=pressure_unit,
            extraction_kwargs=extraction_kwargs,
            provenance_notes=provenance_notes,
            strict=strict,
            geometry_repairs=geometry_repairs,
            repair_seed=repair_seed,
        )
        return _construction_result_from_extraction(extracted)

    if pnflow_cnm_prefix is None:
        raise ValueError("pnflow_cnm_prefix is required for backend='pnflow_cnm'")
    selected_axis = "x" if flow_axis is None else str(flow_axis)
    if selected_axis != "x":
        raise ValueError("Imperial CNM construction currently supports only flow_axis='x'")

    imported = load_pnflow_cnm(
        pnflow_cnm_prefix,
        boundary_axis=selected_axis,
        length_unit=length_unit,
        pressure_unit=pressure_unit,
        pnflow_solver_box_compat=pnflow_solver_box_compat,
    )
    net = imported.net.copy()
    net.provenance = _merge_provenance_notes(net.provenance, provenance_notes)
    axis_lengths = dict(imported.box_lengths)
    axis_areas = {
        "x": imported.box_lengths["y"] * imported.box_lengths["z"],
        "y": imported.box_lengths["x"] * imported.box_lengths["z"],
        "z": imported.box_lengths["x"] * imported.box_lengths["y"],
    }
    return NetworkConstructionResult(
        backend=backend_normalized,
        flow_axis=selected_axis,
        sample=net.sample,
        provenance=net.provenance,
        net_full=net,
        net=net,
        image=None,
        voxel_size=None,
        axis_lengths=axis_lengths,
        axis_areas=axis_areas,
        network_dict=None,
        pore_indices=np.arange(net.Np, dtype=np.int64),
        throat_mask=np.ones(net.Nt, dtype=bool),
        backend_version=None,
        backend_details={
            "pnflow_cnm_prefix": str(Path(pnflow_cnm_prefix)),
            "n_physical_pores": int(imported.n_physical_pores),
            "n_boundary_mirror_pores": int(imported.n_boundary_mirror_pores),
        },
    )


__all__ = [
    "NetworkConstructionResult",
    "NetworkExtractionResult",
    "construct_spanning_network",
    "infer_sample_axes",
    "extract_spanning_pore_network",
]
