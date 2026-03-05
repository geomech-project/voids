from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numpy as np
from scipy import ndimage as ndi

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.graph import spanning_subnetwork
from voids.io import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry


_T = TypeVar("_T")


def _progress_iter(
    iterable: Iterable[_T],
    *,
    show_progress: bool,
    desc: str | None = None,
    total: int | None = None,
) -> Iterable[_T]:
    """Wrap an iterable with ``tqdm`` when available and requested."""

    if not show_progress:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        return cast(
            Iterable[_T],
            tqdm(
                iterable,
                desc=desc,
                total=total,
                dynamic_ncols=True,
                leave=False,
            ),
        )
    except Exception:  # pragma: no cover - optional dependency
        return iterable


def _require_skimage():
    """Import scikit-image lazily for optional grayscale preprocessing."""

    try:
        from skimage.filters import (
            threshold_isodata,
            threshold_li,
            threshold_otsu,
            threshold_triangle,
            threshold_yen,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-image is required for grayscale volume preprocessing. Use the 'test' pixi environment or install scikit-image."
        ) from exc
    return {
        "otsu": threshold_otsu,
        "li": threshold_li,
        "yen": threshold_yen,
        "isodata": threshold_isodata,
        "triangle": threshold_triangle,
    }


def _require_porespy():
    """Import PoreSpy lazily for image-to-network extraction."""

    try:
        import porespy as ps
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PoreSpy is required for volume extraction. Use the 'test' pixi environment or install porespy."
        ) from exc
    return ps


@dataclass(slots=True)
class VolumeCropResult:
    """Describe cylindrical-sample cropping from a grayscale volume."""

    raw: np.ndarray
    specimen_mask: np.ndarray
    common_mask: np.ndarray
    crop_bounds_yx: tuple[int, int, int, int]
    cropped: np.ndarray


@dataclass(slots=True)
class GrayscaleSegmentationResult:
    """Store grayscale preprocessing and binary segmentation outputs."""

    crop: VolumeCropResult
    threshold: float
    binary: np.ndarray
    void_phase: str
    threshold_method: str


@dataclass(slots=True)
class PoreSpyExtractionResult:
    """Store a PoreSpy-to-voids extraction workflow result."""

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
    porespy_version: str | None


def infer_sample_axes(
    shape: tuple[int, ...],
    *,
    voxel_size: float,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> tuple[dict[str, int], dict[str, float], dict[str, float], str]:
    """Infer per-axis counts, lengths, cross-sections, and the longest axis.

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


def largest_true_rectangle(mask2d: np.ndarray) -> tuple[int, int, int, int]:
    """Return the maximal-area axis-aligned rectangle fully contained in a mask.

    Parameters
    ----------
    mask2d :
        Two-dimensional boolean mask.

    Returns
    -------
    tuple
        Rectangle bounds ``(y0, y1, x0, x1)``.
    """

    mask = np.asarray(mask2d, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask2d must be a 2D boolean array")

    heights = [0] * mask.shape[1]
    best_area = 0
    best_bounds: tuple[int, int, int, int] | None = None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            heights[x] = heights[x] + 1 if mask[y, x] else 0
        stack: list[int] = []
        x = 0
        while x <= mask.shape[1]:
            cur = heights[x] if x < mask.shape[1] else 0
            if not stack or cur >= heights[stack[-1]]:
                stack.append(x)
                x += 1
            else:
                top = stack.pop()
                height = heights[top]
                left = stack[-1] + 1 if stack else 0
                width = x - left
                area = height * width
                if area > best_area:
                    best_area = area
                    best_bounds = (y + 1 - height, y + 1, left, x)
    if best_bounds is None:
        raise ValueError("mask2d does not contain any True pixels")
    return best_bounds


def crop_nonzero_cylindrical_volume(
    raw: np.ndarray,
    *,
    background_value: float = 0.0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> VolumeCropResult:
    """Crop a grayscale cylindrical sample to a common inscribed rectangle.

    Parameters
    ----------
    raw :
        Three-dimensional grayscale volume.
    background_value :
        Voxels strictly larger than this value are treated as specimen support
        before hole filling.
    show_progress :
        If ``True``, show a ``tqdm`` progress bar while filling holes slice-wise.
        When ``tqdm`` is unavailable, iteration proceeds silently.
    progress_desc :
        Optional progress-bar description string.

    Returns
    -------
    VolumeCropResult
        Raw volume, specimen masks, crop bounds, and cropped subvolume.

    Notes
    -----
    The helper assumes the sample is cylindrical in the last two dimensions and
    embedded in a constant-valued background. It fills holes on each slice to
    recover the specimen envelope, then finds the largest rectangle common to
    all slices.
    """

    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 3:
        raise ValueError("raw must be a 3D grayscale volume")

    specimen_mask = np.zeros_like(arr, dtype=bool)
    iterator = _progress_iter(
        range(arr.shape[0]),
        show_progress=show_progress,
        desc=progress_desc or "Filling support mask slices",
        total=int(arr.shape[0]),
    )
    for i in iterator:
        specimen_mask[i] = ndi.binary_fill_holes(arr[i] > background_value)

    common_mask = specimen_mask.all(axis=0)
    crop_bounds_yx = largest_true_rectangle(common_mask)
    y0, y1, x0, x1 = crop_bounds_yx
    cropped = arr[:, y0:y1, x0:x1]
    return VolumeCropResult(
        raw=arr,
        specimen_mask=specimen_mask,
        common_mask=common_mask,
        crop_bounds_yx=crop_bounds_yx,
        cropped=cropped,
    )


def binarize_grayscale_volume(
    cropped: np.ndarray,
    *,
    threshold: float | None = None,
    method: str = "otsu",
    void_phase: str = "dark",
) -> tuple[np.ndarray, float]:
    """Binarize a cropped grayscale volume into void and solid phases.

    Parameters
    ----------
    cropped :
        Cropped grayscale volume.
    threshold :
        Explicit threshold. When omitted, ``method`` is used.
    method :
        Automatic threshold method. Supported values are ``otsu``, ``li``,
        ``yen``, ``isodata``, and ``triangle``.
    void_phase :
        Either ``\"dark\"`` or ``\"bright\"`` to indicate which grayscale side is
        treated as void space.

    Returns
    -------
    tuple
        ``(binary, threshold)`` with ``binary`` encoded as ``void = 1``,
        ``solid = 0``.
    """

    arr = np.asarray(cropped, dtype=float)
    if arr.ndim != 3:
        raise ValueError("cropped must be a 3D grayscale volume")
    if void_phase not in {"dark", "bright"}:
        raise ValueError("void_phase must be either 'dark' or 'bright'")

    if threshold is None:
        methods = _require_skimage()
        if method not in methods:
            raise ValueError(f"Unsupported threshold method '{method}'")
        threshold = float(methods[method](arr))
    else:
        threshold = float(threshold)

    if void_phase == "dark":
        binary = (arr < threshold).astype(int)
    else:
        binary = (arr > threshold).astype(int)
    return binary, threshold


def preprocess_grayscale_cylindrical_volume(
    raw: np.ndarray,
    *,
    background_value: float = 0.0,
    threshold: float | None = None,
    threshold_method: str = "otsu",
    void_phase: str = "dark",
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> GrayscaleSegmentationResult:
    """Crop and binarize a grayscale cylindrical sample volume."""

    crop = crop_nonzero_cylindrical_volume(
        raw,
        background_value=background_value,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    binary, used_threshold = binarize_grayscale_volume(
        crop.cropped,
        threshold=threshold,
        method=threshold_method,
        void_phase=void_phase,
    )
    return GrayscaleSegmentationResult(
        crop=crop,
        threshold=used_threshold,
        binary=binary,
        void_phase=void_phase,
        threshold_method=threshold_method,
    )


def _snow2_network_dict(
    phases: np.ndarray, *, porespy_module: Any, snow2_kwargs: dict[str, object] | None
) -> dict[str, object]:
    """Run ``porespy.networks.snow2`` and normalize its returned network mapping."""

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


def extract_spanning_porespy_network(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    snow2_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
) -> PoreSpyExtractionResult:
    """Extract, import, and prune a PoreSpy network from a binary phase volume.

    Parameters
    ----------
    phases :
        Binary or integer-labeled phase image where nonzero values are active
        phases passed to ``snow2``.
    voxel_size :
        Edge length of one voxel in the declared ``length_unit``.
    flow_axis :
        Requested spanning axis. When omitted, the longest image axis is used.
    length_unit, pressure_unit :
        Units stored in the resulting :class:`SampleGeometry`.
    snow2_kwargs :
        Optional keyword arguments forwarded to ``porespy.networks.snow2``.
    provenance_notes :
        Optional extra provenance metadata attached to the resulting network.
    strict :
        Forwarded to :func:`voids.io.from_porespy`.

    Returns
    -------
    PoreSpyExtractionResult
        Full and pruned networks together with intermediate metadata.
    """

    arr = np.asarray(phases, dtype=int)
    if arr.ndim not in {2, 3}:
        raise ValueError("phases must be a 2D or 3D integer image")

    ps = _require_porespy()
    _, axis_lengths, axis_areas, inferred_axis = infer_sample_axes(arr.shape, voxel_size=voxel_size)
    selected_axis = inferred_axis if flow_axis is None else flow_axis
    if selected_axis not in axis_lengths:
        raise ValueError(f"flow_axis '{selected_axis}' is not compatible with shape {arr.shape}")

    network_dict = _snow2_network_dict(arr, porespy_module=ps, snow2_kwargs=snow2_kwargs)
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
        source_kind="porespy",
        source_version=getattr(ps, "__version__", None),
        extraction_method="snow2",
        user_notes=dict(provenance_notes or {}),
    )
    net_full = from_porespy(network_dict, sample=sample, provenance=provenance, strict=strict)
    net, pore_indices, throat_mask = spanning_subnetwork(net_full, axis=selected_axis)
    return PoreSpyExtractionResult(
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
        porespy_version=getattr(ps, "__version__", None),
    )
