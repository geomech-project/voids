from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_CHARACTERISTIC_SIZE_FIELDS: tuple[str, ...] = (
    "diameter_equivalent",
    "diameter_inscribed",
    "radius_inscribed",
    "area",
)


def area_equivalent_diameter(area: np.ndarray) -> np.ndarray:
    """Return the circular-equivalent diameter associated with an area array.

    Parameters
    ----------
    area :
        Cross-sectional area array.

    Returns
    -------
    numpy.ndarray
        Diameter defined by ``d = 2 * sqrt(area / pi)``.
    """

    area = np.asarray(area, dtype=float)
    return 2.0 * np.sqrt(area / np.pi)


def normalize_characteristic_size(
    values: np.ndarray,
    *,
    field_name: str | None,
) -> np.ndarray:
    """Normalize a size-like field to a characteristic diameter surrogate.

    Parameters
    ----------
    values :
        Raw size-like field values.
    field_name :
        Source field name, used to convert radii and areas to diameters.

    Returns
    -------
    numpy.ndarray
        Characteristic-size array interpreted as a diameter-like quantity.
    """

    arr = np.asarray(values, dtype=float)
    if field_name == "radius_inscribed":
        return 2.0 * arr
    if field_name == "area":
        return area_equivalent_diameter(arr)
    return arr


def characteristic_size(
    store: Mapping[str, object],
    *,
    expected_shape: tuple[int, ...] | None = None,
    fields: tuple[str, ...] = _CHARACTERISTIC_SIZE_FIELDS,
) -> tuple[np.ndarray, str]:
    """Return a preferred characteristic size array from a pore/throat store.

    Parameters
    ----------
    store :
        Mapping such as ``net.pore`` or ``net.throat``.
    expected_shape :
        Optional expected array shape.
    fields :
        Ordered field priority. The default is
        ``diameter_equivalent -> diameter_inscribed -> radius_inscribed -> area``.

    Returns
    -------
    tuple
        Pair ``(values, label)`` where ``values`` is a characteristic-size array
        and ``label`` is the originating field name.

    Raises
    ------
    KeyError
        If none of the requested fields exists.
    ValueError
        If a selected field does not match ``expected_shape``.
    """

    for field_name in fields:
        if field_name not in store:
            continue
        arr = normalize_characteristic_size(
            np.asarray(store[field_name], dtype=float), field_name=field_name
        )
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(f"field '{field_name}' must have shape {expected_shape}")
        return arr, field_name
    raise KeyError(f"Need one of the characteristic size fields: {fields}")
