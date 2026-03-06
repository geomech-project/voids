from __future__ import annotations

from collections.abc import Sequence


def normalize_shape(shape: Sequence[int], *, allowed_ndim: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and normalize image shape metadata.

    Parameters
    ----------
    shape :
        Iterable with voxel counts along each axis. Values are coerced to
        integers.
    allowed_ndim :
        Set of accepted dimensionalities, for example ``(2,)`` for strictly
        2D images or ``(2, 3)`` for functions that support both 2D and 3D.

    Returns
    -------
    tuple[int, ...]
        Canonical integer shape tuple.

    Raises
    ------
    ValueError
        If dimensionality is unsupported or if any axis length is non-positive.

    Notes
    -----
    This helper enforces a physically meaningful image domain where each axis
    has at least one voxel. It does not encode voxel spacing; only grid counts.
    """

    dims = tuple(int(v) for v in shape)
    if len(dims) not in allowed_ndim:
        allowed = ", ".join(str(v) for v in allowed_ndim)
        raise ValueError(f"shape must have dimensionality in {{{allowed}}}")
    if any(n <= 0 for n in dims):
        raise ValueError("all shape entries must be positive")
    return dims


def validate_axis_index(*, axis_index: int, ndim: int) -> int:
    """Validate and normalize an axis index for array-based workflows.

    Parameters
    ----------
    axis_index :
        Candidate axis index.
    ndim :
        Number of active dimensions in the target image/array.

    Returns
    -------
    int
        Normalized integer axis index.

    Raises
    ------
    ValueError
        If the axis index lies outside ``[0, ndim - 1]``.

    Notes
    -----
    Negative Python-style indices are intentionally rejected to keep axis
    semantics explicit in scientific workflows and avoid accidental misuse when
    porting formulas across 2D/3D datasets.
    """

    axis = int(axis_index)
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis_index={axis} is out of bounds for ndim={ndim}")
    return axis


__all__ = ["normalize_shape", "validate_axis_index"]
