from __future__ import annotations

import numpy as np

from voids.geom.characteristic import characteristic_size, normalize_characteristic_size


def resolve_size_values(
    values: str | np.ndarray | bool | None,
    *,
    store: dict[str, np.ndarray],
    expected_shape: tuple[int, ...],
    prefix: str,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve characteristic-size input from an explicit field, array, or auto mode.

    Parameters
    ----------
    values :
        One of:

        - ``None``: auto-detect a characteristic size field
        - ``False``: disable size-driven rendering
        - field name string
        - explicit array already interpreted as a characteristic size
    store :
        Property dictionary used for field-based resolution.
    expected_shape :
        Expected array shape.
    prefix :
        Prefix used in error messages and output labels.

    Returns
    -------
    tuple
        Pair ``(array, label)``. Both are ``None`` when no size data is available.
    """

    if values is False:
        return None, None

    if values is None:
        try:
            arr, field_name = characteristic_size(store, expected_shape=expected_shape)
        except KeyError:
            return None, None
        return arr, f"{prefix}.{field_name}"

    if isinstance(values, str):
        if values not in store:
            raise KeyError(f"Missing {prefix} field '{values}'")
        arr = np.asarray(store[values], dtype=float)
        if arr.shape != expected_shape:
            raise ValueError(f"{prefix} size field '{values}' must have shape {expected_shape}")
        return normalize_characteristic_size(arr, field_name=values), f"{prefix}.{values}"

    arr = np.asarray(values, dtype=float)
    if arr.shape != expected_shape:
        raise ValueError(f"{prefix} size array must have shape {expected_shape}")
    return arr, f"{prefix}.size"


def scale_sizes_to_pixels(
    values: np.ndarray,
    *,
    reference: float,
    scale: float = 1.0,
    min_size: float | None = None,
    max_size: float | None = None,
) -> np.ndarray:
    """Map physical characteristic sizes to screen-space sizes proportionally."""

    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, float(reference), dtype=float)
    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return out

    baseline = float(np.median(arr[valid]))
    if baseline <= 0.0 or not np.isfinite(baseline):
        return out

    out[valid] = float(scale) * float(reference) * arr[valid] / baseline
    if min_size is not None:
        out = np.maximum(out, float(min_size))
    if max_size is not None:
        out = np.minimum(out, float(max_size))
    return out
