from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import porespy as ps

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.graph import spanning_subnetwork
from voids.image.maximal_ball import (
    MaximalBallSettings,
    extract_maximal_ball_network_dict,
)
from voids.io.pnflow_cnm import load_pnflow_cnm
from voids.io.porespy import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry

_IMPERIAL_SNOW2_DEFAULTS: dict[str, object] = {
    "sigma": 1.0,
    "r_max": 4,
    "boundary_width": 1,
}


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
        and ``"snow2_imperial"``, plus the native maximal-ball aliases
        ``"maximal_ball"``, ``"native_maximal_ball"``, and ``"maxball"``.
    flow_axis :
        Requested spanning axis. When omitted, the longest image axis is used.
    length_unit, pressure_unit :
        Units stored in resulting :class:`SampleGeometry`.
    extraction_kwargs :
        Keyword arguments forwarded to the extraction backend call. For the
        Imperial-calibrated `snow2` aliases, user-supplied values override the
        built-in defaults ``sigma=1.0``, ``r_max=4``, and ``boundary_width=1``.
        For the native maximal-ball backend, supported keys are
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
    network_dict = _extract_network_dict(
        arr,
        backend=backend_normalized,
        voxel_size=float(voxel_size),
        extraction_kwargs=extraction_kwargs,
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
    provenance = Provenance(
        source_kind="image_extraction",
        source_version=getattr(ps, "__version__", None),
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
        backend_version=getattr(ps, "__version__", None),
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
        ``"porespy_snow2"``, the native maximal-ball aliases
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
