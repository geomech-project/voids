from __future__ import annotations

from pathlib import Path

import numpy as np


def make_manufactured_void_image(shape: tuple[int, int, int] = (48, 48, 48)) -> np.ndarray:
    """Create a deterministic synthetic 3-D void-space image.

    Parameters
    ----------
    shape :
        Output image shape in voxels.

    Returns
    -------
    numpy.ndarray
        Boolean array with shape ``shape`` where ``True`` denotes void space.

    Notes
    -----
    The construction is intentionally simple: a chain of overlapping spheres
    spans the x-direction, while a few side branches create off-axis
    connectivity. The result is not intended as a geological model. It is a
    manufactured test image for extraction workflows such as ``porespy.snow2``.
    """

    nx, ny, nz = shape
    X, Y, Z = np.indices(shape)
    im = np.zeros(shape, dtype=bool)

    chain = [
        (6, ny // 2, nz // 2, 7),
        (14, ny // 2 + 1, nz // 2, 7),
        (22, ny // 2 - 1, nz // 2 + 1, 7),
        (30, ny // 2, nz // 2 - 1, 7),
        (38, ny // 2 + 1, nz // 2, 7),
    ]
    branches = [
        (20, ny // 2 + 10, nz // 2, 5),
        (28, ny // 2 - 10, nz // 2 + 2, 5),
        (34, ny // 2 + 6, nz // 2 + 8, 4),
    ]
    for cx, cy, cz, r in chain + branches:
        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r**2
        im |= mask

    y0 = ny // 2
    z0 = nz // 2
    im[12:17, y0 - 1 : y0 + 2, z0 - 1 : z0 + 2] = True

    return im


def save_default_manufactured_void_image(path: str | Path) -> Path:
    """Write the manufactured void image to a NumPy ``.npy`` file.

    Parameters
    ----------
    path :
        Destination file path.

    Returns
    -------
    pathlib.Path
        Resolved path that was written.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, make_manufactured_void_image())
    return path
