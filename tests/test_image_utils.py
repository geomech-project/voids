from __future__ import annotations

import pytest

from voids.image import _utils as iutils


def test_image_utils_shape_and_axis_validation() -> None:
    assert iutils.normalize_shape((4, 5), allowed_ndim=(2,)) == (4, 5)
    with pytest.raises(ValueError, match="dimensionality"):
        iutils.normalize_shape((4,), allowed_ndim=(2, 3))
    with pytest.raises(ValueError, match="all shape entries must be positive"):
        iutils.normalize_shape((4, 0), allowed_ndim=(2,))

    assert iutils.validate_axis_index(axis_index=1, ndim=3) == 1
    with pytest.raises(ValueError, match="out of bounds"):
        iutils.validate_axis_index(axis_index=-1, ndim=3)
