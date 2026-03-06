from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

_T = TypeVar("_T")

_THRESHOLD_METHODS: dict[str, Callable[..., float]] = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "yen": threshold_yen,
    "isodata": threshold_isodata,
    "triangle": threshold_triangle,
}


@dataclass(slots=True)
class VolumeCropResult:
    """Store cylindrical-support cropping outputs from a grayscale volume.

    Attributes
    ----------
    raw :
        Original grayscale volume as float array.
    specimen_mask :
        Slice-wise support mask after hole filling.
    common_mask :
        Per-pixel intersection of support masks over all slices.
    crop_bounds_yx :
        Maximal common rectangle bounds ``(y0, y1, x0, x1)``.
    cropped :
        Cropped grayscale volume containing the common inscribed rectangle.
    """

    raw: np.ndarray
    specimen_mask: np.ndarray
    common_mask: np.ndarray
    crop_bounds_yx: tuple[int, int, int, int]
    cropped: np.ndarray


@dataclass(slots=True)
class GrayscaleSegmentationResult:
    """Store grayscale preprocessing and binary segmentation outputs.

    Attributes
    ----------
    crop :
        Cylindrical-support crop outputs.
    threshold :
        Threshold used for binarization.
    binary :
        Segmented binary volume encoded as ``void=1``, ``solid=0``.
    void_phase :
        Phase polarity used for thresholding (``"dark"`` or ``"bright"``).
    threshold_method :
        Automatic method used when threshold was not explicitly supplied.
    """

    crop: VolumeCropResult
    threshold: float
    binary: np.ndarray
    void_phase: str
    threshold_method: str


def _progress_iter(
    iterable: Iterable[_T],
    *,
    show_progress: bool,
    desc: str | None = None,
    total: int | None = None,
) -> Iterable[_T]:
    """Wrap an iterable with ``tqdm`` when progress reporting is requested."""

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
    except Exception:  # pragma: no cover - optional runtime fallback
        return iterable


def largest_true_rectangle(mask2d: np.ndarray) -> tuple[int, int, int, int]:
    """Return maximal-area axis-aligned rectangle fully contained in a mask.

    Parameters
    ----------
    mask2d :
        Two-dimensional boolean support mask.

    Returns
    -------
    tuple[int, int, int, int]
        Rectangle bounds ``(y0, y1, x0, x1)`` in NumPy slicing convention.

    Raises
    ------
    ValueError
        If ``mask2d`` is not 2D or contains no ``True`` pixels.
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
    """Crop cylindrical specimen support to a common rectangular field of view.

    Parameters
    ----------
    raw :
        Raw 3D grayscale volume.
    background_value :
        Voxels strictly above this value are interpreted as specimen support
        before hole filling.
    show_progress :
        Whether to show progress bars for slice-wise operations.
    progress_desc :
        Optional progress description string.

    Returns
    -------
    VolumeCropResult
        Structured crop result with masks, bounds, and cropped volume.
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
    """Segment grayscale volume into binary void/solid phases.

    Parameters
    ----------
    cropped :
        Cropped 3D grayscale volume.
    threshold :
        Explicit threshold; when omitted, an automatic threshold is computed.
    method :
        Automatic threshold method name. Supported values are ``"otsu"``,
        ``"li"``, ``"yen"``, ``"isodata"``, and ``"triangle"``.
    void_phase :
        Which side of threshold corresponds to void: ``"dark"`` or
        ``"bright"``.

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(binary, threshold_used)`` where ``binary`` is integer encoded as
        ``void=1`` and ``solid=0``.
    """

    arr = np.asarray(cropped, dtype=float)
    if arr.ndim != 3:
        raise ValueError("cropped must be a 3D grayscale volume")
    if void_phase not in {"dark", "bright"}:
        raise ValueError("void_phase must be either 'dark' or 'bright'")

    if threshold is None:
        if method not in _THRESHOLD_METHODS:
            raise ValueError(f"Unsupported threshold method '{method}'")
        threshold = float(_THRESHOLD_METHODS[method](arr))
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
    """Run cylindrical crop and grayscale segmentation in one workflow call.

    Parameters
    ----------
    raw :
        Raw 3D grayscale specimen volume.
    background_value :
        Background/support discriminator for cropping.
    threshold :
        Explicit segmentation threshold.
    threshold_method :
        Method used when ``threshold`` is omitted.
    void_phase :
        Phase polarity selector for thresholding.
    show_progress :
        Whether to request progress reporting.
    progress_desc :
        Optional progress message.

    Returns
    -------
    GrayscaleSegmentationResult
        Crop metadata plus segmented binary volume.
    """

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


def binarize_2d_with_voids(
    gray2d: np.ndarray,
    *,
    threshold: float | None = None,
    method: str = "otsu",
    void_phase: str = "dark",
) -> tuple[np.ndarray, float]:
    """Segment a 2D grayscale image using the same thresholding policy as 3D.

    Parameters
    ----------
    gray2d :
        Two-dimensional grayscale image.
    threshold :
        Explicit threshold value.
    method :
        Automatic threshold method when ``threshold`` is omitted.
    void_phase :
        Which side of threshold corresponds to void.

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(binary2d, threshold_used)`` with binary image encoded as integers
        in ``{0, 1}``.
    """

    gray = np.asarray(gray2d, dtype=float)
    if gray.ndim != 2:
        raise ValueError("gray2d must be a 2D array")
    seg3d, threshold_used = binarize_grayscale_volume(
        gray[None, :, :],
        threshold=threshold,
        method=method,
        void_phase=void_phase,
    )
    return np.asarray(seg3d[0], dtype=int), float(threshold_used)


__all__ = [
    "VolumeCropResult",
    "GrayscaleSegmentationResult",
    "largest_true_rectangle",
    "crop_nonzero_cylindrical_volume",
    "binarize_grayscale_volume",
    "preprocess_grayscale_cylindrical_volume",
    "binarize_2d_with_voids",
]
