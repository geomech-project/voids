from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from voids.image._utils import validate_axis_index


def has_spanning_cluster(void_mask: np.ndarray, axis_index: int) -> bool:
    """Test whether void space percolates from one boundary to the opposite.

    Parameters
    ----------
    void_mask :
        Binary image where ``True`` denotes void phase and ``False`` denotes
        solid phase. Supported dimensionalities are 2D and 3D.
    axis_index :
        Flow/percolation axis index. For a 3D array ``(nx, ny, nz)``, values
        are typically ``0`` (x), ``1`` (y), or ``2`` (z).

    Returns
    -------
    bool
        ``True`` if at least one connected void component intersects both
        opposite boundary planes along ``axis_index``.

    Raises
    ------
    ValueError
        If ``void_mask`` is not 2D/3D or if ``axis_index`` is invalid.

    Notes
    -----
    Connectivity is computed via :func:`scipy.ndimage.label` with its default
    structuring element (face-connected in 3D, edge-connected in 2D). The
    criterion is geometric percolation only; it does not assess hydraulic
    conductance magnitude.

    Assumptions and limitations
    ---------------------------
    - Boundaries are interpreted as the first and last index along the target
      axis.
    - Periodic boundaries are not considered.
    - Very thin connections may percolate topologically even if they are not
      representative of realistic transport in a given experiment.
    """

    mask = np.asarray(void_mask, dtype=bool)
    if mask.ndim not in {2, 3}:
        raise ValueError("void_mask must be 2D or 3D")
    axis = validate_axis_index(axis_index=axis_index, ndim=mask.ndim)

    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return False

    inlet_labels = np.unique(np.take(labels, indices=0, axis=axis))
    outlet_labels = np.unique(np.take(labels, indices=-1, axis=axis))
    inlet_labels = inlet_labels[inlet_labels > 0]
    outlet_labels = outlet_labels[outlet_labels > 0]
    return bool(np.intersect1d(inlet_labels, outlet_labels).size > 0)


def has_spanning_cluster_2d(void_mask: np.ndarray, axis_index: int) -> bool:
    """2D-specialized wrapper for axis-spanning connectivity checks.

    Parameters
    ----------
    void_mask :
        Two-dimensional binary void mask.
    axis_index :
        Axis along which percolation is tested (``0`` or ``1``).

    Returns
    -------
    bool
        ``True`` when at least one 2D connected void component spans the
        selected axis.

    Raises
    ------
    ValueError
        If ``void_mask`` is not 2D.

    Notes
    -----
    This function exists for notebook/API compatibility and delegates to
    :func:`has_spanning_cluster` after enforcing 2D inputs.
    """

    mask = np.asarray(void_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("void_mask must be 2D for has_spanning_cluster_2d")
    return has_spanning_cluster(mask, axis_index=axis_index)


__all__ = ["has_spanning_cluster", "has_spanning_cluster_2d"]
