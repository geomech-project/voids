from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import porespy as ps

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.graph import spanning_subnetwork
from voids.io import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry


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


def extract_spanning_pore_network(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
) -> NetworkExtractionResult:
    """Extract, import, and prune an axis-spanning pore network from an image.

    Parameters
    ----------
    phases :
        Binary or integer-labeled phase image where nonzero values are active
        phases passed to the extraction backend.
    voxel_size :
        Edge length of one voxel in the declared ``length_unit``.
    flow_axis :
        Requested spanning axis. When omitted, the longest image axis is used.
    length_unit, pressure_unit :
        Units stored in resulting :class:`SampleGeometry`.
    extraction_kwargs :
        Keyword arguments forwarded to the extraction backend call.
    provenance_notes :
        Optional extra provenance metadata attached to the resulting network.
    strict :
        Forwarded to :func:`voids.io.from_porespy`.

    Returns
    -------
    NetworkExtractionResult
        Full and pruned networks together with intermediate metadata.

    Notes
    -----
    Current implementation uses PoreSpy's ``snow2`` backend and normalizes
    accepted return styles into a standard network mapping before import.
    """

    arr = np.asarray(phases, dtype=int)
    if arr.ndim not in {2, 3}:
        raise ValueError("phases must be a 2D or 3D integer image")

    _, axis_lengths, axis_areas, inferred_axis = infer_sample_axes(arr.shape, voxel_size=voxel_size)
    selected_axis = inferred_axis if flow_axis is None else flow_axis
    if selected_axis not in axis_lengths:
        raise ValueError(f"flow_axis '{selected_axis}' is not compatible with shape {arr.shape}")

    network_dict = _snow2_network_dict(arr, snow2_kwargs=extraction_kwargs)
    network_dict = scale_porespy_geometry(network_dict, voxel_size=voxel_size)
    network_dict = ensure_cartesian_boundary_labels(network_dict, axes=(selected_axis,))

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
    provenance = Provenance(
        source_kind="image_extraction",
        source_version=getattr(ps, "__version__", None),
        extraction_method="snow2",
        user_notes=dict(provenance_notes or {}),
    )
    net_full = from_porespy(network_dict, sample=sample, provenance=provenance, strict=strict)
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
        backend="porespy",
        backend_version=getattr(ps, "__version__", None),
    )


__all__ = [
    "NetworkExtractionResult",
    "infer_sample_axes",
    "extract_spanning_pore_network",
]
