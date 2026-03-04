from __future__ import annotations

import warnings
from typing import Final

import numpy as np

from voids.core.network import Network


DEFAULT_G_REF: Final[float] = 1.0 / (4.0 * np.pi)  # circular duct shape factor A/P^2
_CIRCLE_COEFF_AG2: Final[float] = 0.5  # gives Hagen-Poiseuille when G=1/(4π)


def _require(net: Network, kind: str, names: tuple[str, ...]) -> None:
    """Require a set of pore or throat fields.

    Parameters
    ----------
    net :
        Network containing the requested arrays.
    kind :
        Either ``"pore"`` or ``"throat"``.
    names :
        Field names that must exist.

    Raises
    ------
    KeyError
        If at least one requested field is absent.
    """

    store = net.throat if kind == "throat" else net.pore
    missing = [n for n in names if n not in store]
    if missing:
        raise KeyError(f"Missing required {kind} fields: {missing}")


def _diameter_from_area(area: np.ndarray) -> np.ndarray:
    """Compute circular-equivalent diameter from area.

    Parameters
    ----------
    area :
        Cross-sectional areas.

    Returns
    -------
    numpy.ndarray
        Diameter defined by ``d = 2 * sqrt(area / pi)``.
    """

    return 2.0 * np.sqrt(area / np.pi)


def _area_from_diameter(d: np.ndarray) -> np.ndarray:
    """Compute circular area from diameter.

    Parameters
    ----------
    d :
        Diameters.

    Returns
    -------
    numpy.ndarray
        Areas defined by ``A = pi * (d / 2)**2``.
    """

    r = 0.5 * d
    return np.pi * r**2


def _shape_factor_from_area_perimeter(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    """Compute duct shape factor from area and perimeter.

    Parameters
    ----------
    area :
        Cross-sectional areas.
    perimeter :
        Wetted perimeters.

    Returns
    -------
    numpy.ndarray
        Shape factor defined by ``G = A / P**2``.
    """

    return area / np.maximum(perimeter, 1e-30) ** 2


def _get_entity_area(net: Network, kind: str) -> np.ndarray:
    """Return area data for pores or throats, deriving it when possible.

    Parameters
    ----------
    net :
        Network containing the geometric data.
    kind :
        Either ``"pore"`` or ``"throat"``.

    Returns
    -------
    numpy.ndarray
        Area array.

    Raises
    ------
    KeyError
        If no area or radius/diameter surrogate is available.
    """

    store = net.throat if kind == "throat" else net.pore
    if "area" in store:
        return np.asarray(store["area"], dtype=float)
    if "diameter_inscribed" in store:
        d = np.asarray(store["diameter_inscribed"], dtype=float)
        return _area_from_diameter(d)
    if "radius_inscribed" in store:
        r = np.asarray(store["radius_inscribed"], dtype=float)
        return np.pi * r**2
    raise KeyError(f"Need {kind}.area or {kind}.diameter_inscribed (or radius_inscribed)")


def _get_entity_shape_factor(net: Network, kind: str, area: np.ndarray | None = None) -> np.ndarray:
    """Return shape-factor data for pores or throats.

    Parameters
    ----------
    net :
        Network containing the geometric data.
    kind :
        Either ``"pore"`` or ``"throat"``.
    area :
        Optional precomputed area array to avoid recomputation.

    Returns
    -------
    numpy.ndarray
        Shape-factor array.

    Raises
    ------
    KeyError
        If neither ``shape_factor`` nor the required surrogate fields are available.
    """

    store = net.throat if kind == "throat" else net.pore
    if "shape_factor" in store:
        return np.asarray(store["shape_factor"], dtype=float)
    if "perimeter" in store:
        a = _get_entity_area(net, kind) if area is None else np.asarray(area, dtype=float)
        p = np.asarray(store["perimeter"], dtype=float)
        return _shape_factor_from_area_perimeter(a, p)
    raise KeyError(f"Need {kind}.shape_factor or {kind}.perimeter (with area/diameter)")


def _segment_conductance_from_agl(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float,
    *,
    clip_shape_factor: bool = True,
) -> np.ndarray:
    """Compute segment conductance from area, shape factor, and length.

    Parameters
    ----------
    area :
        Segment cross-sectional area.
    shape_factor :
        Segment shape factor ``G = A / P**2``.
    length :
        Segment length.
    viscosity :
        Dynamic viscosity.
    clip_shape_factor :
        If ``True``, clip shape factors to ``[1e-12, 1]`` to avoid extreme values
        caused by noisy geometry extraction.

    Returns
    -------
    numpy.ndarray
        Hydraulic conductance array.

    Raises
    ------
    ValueError
        If viscosity is non-positive or if the inputs contain negative values.

    Notes
    -----
    The scaling used is

    ``g = C * G * A**2 / (mu * L)``

    with ``C = 0.5``. For a circular duct, ``G = 1 / (4 * pi)``, so the expression
    recovers Hagen-Poiseuille conductance.
    """

    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    a = np.asarray(area, dtype=float)
    gsf = np.asarray(shape_factor, dtype=float)
    L = np.asarray(length, dtype=float)
    if np.any(a < 0):
        raise ValueError("area contains negative values")
    if np.any(L < 0):
        raise ValueError("length contains negative values")
    if np.any(gsf < 0):
        raise ValueError("shape_factor contains negative values")
    if clip_shape_factor:
        gsf = np.clip(gsf, 1e-12, 1.0)
    out = np.full_like(a, np.inf, dtype=float)
    nz = L > 0
    out[nz] = (_CIRCLE_COEFF_AG2 * gsf[nz] * a[nz] ** 2) / (viscosity * L[nz])
    return out


def generic_poiseuille_conductance(net: Network, viscosity: float) -> np.ndarray:
    """Compute throat conductance using a circular Poiseuille approximation.

    Parameters
    ----------
    net :
        Network containing throat geometry or precomputed hydraulic conductance.
    viscosity :
        Dynamic viscosity.

    Returns
    -------
    numpy.ndarray
        Conductance array for all throats.

    Raises
    ------
    ValueError
        If viscosity is non-positive or if precomputed conductance is negative.
    KeyError
        If required geometry is missing.

    Notes
    -----
    When no precomputed conductance is supplied, the model uses

    ``g = pi * r**4 / (8 * mu * L)``

    with radius inferred from either ``throat.diameter_inscribed`` or
    ``throat.area``.
    """

    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    if "hydraulic_conductance" in net.throat:
        g = np.asarray(net.throat["hydraulic_conductance"], dtype=float)
        if (g < 0).any():
            raise ValueError("throat.hydraulic_conductance contains negative values")
        return g.copy()

    _require(net, "throat", ("length",))
    L = np.asarray(net.throat["length"], dtype=float)
    if "diameter_inscribed" in net.throat:
        d = np.asarray(net.throat["diameter_inscribed"], dtype=float)
    elif "area" in net.throat:
        d = _diameter_from_area(np.asarray(net.throat["area"], dtype=float))
    else:
        raise KeyError(
            "Need throat.diameter_inscribed or throat.area (or precomputed hydraulic_conductance)"
        )
    r = 0.5 * d
    return (np.pi * r**4) / (8.0 * viscosity * L)


def _conduit_lengths_available(net: Network) -> bool:
    """Return whether conduit subsegment lengths are available.

    Parameters
    ----------
    net :
        Network to inspect.

    Returns
    -------
    bool
        ``True`` when ``pore1_length``, ``core_length``, and ``pore2_length`` exist.
    """

    keys = ("pore1_length", "core_length", "pore2_length")
    return all(k in net.throat for k in keys)


def _harmonic_combine_segments(*segments: np.ndarray) -> np.ndarray:
    """Combine segment conductances in series.

    Parameters
    ----------
    *segments :
        Segment conductance arrays defined on the same throats.

    Returns
    -------
    numpy.ndarray
        Equivalent series conductance satisfying
        ``1 / g_eq = sum_k 1 / g_k`` over positive conductances.
    """

    recip = np.zeros_like(np.asarray(segments[0], dtype=float))
    for s in segments:
        arr = np.asarray(s, dtype=float)
        positive = arr > 0
        recip[positive] += 1.0 / arr[positive]
    out = np.zeros_like(recip)
    positive = recip > 0
    out[positive] = 1.0 / recip[positive]
    return out


def _throat_only_shape_factor_conductance(net: Network, viscosity: float) -> np.ndarray:
    """Compute conductance using only throat geometry.

    Parameters
    ----------
    net :
        Network containing throat length and cross-sectional geometry.
    viscosity :
        Dynamic viscosity.

    Returns
    -------
    numpy.ndarray
        Conductance based on throat ``area``, ``shape_factor`` and ``length`` only.
    """

    _require(net, "throat", ("length",))
    L = np.asarray(net.throat["length"], dtype=float)
    A = _get_entity_area(net, "throat")
    G = _get_entity_shape_factor(net, "throat", area=A)
    return _segment_conductance_from_agl(A, G, L, viscosity)


def _valvatne_conduit_baseline(net: Network, viscosity: float) -> np.ndarray:
    """Compute a conduit-based conductance baseline using pore and throat segments.

    Parameters
    ----------
    net :
        Network containing conduit-length decomposition and pore/throat geometry.
    viscosity :
        Dynamic viscosity.

    Returns
    -------
    numpy.ndarray
        Equivalent throat conductance after harmonic combination of pore1, throat,
        and pore2 segments.

    Raises
    ------
    KeyError
        If conduit lengths are unavailable.
    """

    if not _conduit_lengths_available(net):
        raise KeyError("Missing conduit lengths (pore1_length, core_length, pore2_length)")

    At = _get_entity_area(net, "throat")
    Gt = _get_entity_shape_factor(net, "throat", area=At)
    Lt = np.asarray(net.throat["core_length"], dtype=float)
    gt = _segment_conductance_from_agl(At, Gt, Lt, viscosity)

    conns = net.throat_conns
    p1_idx = conns[:, 0]
    p2_idx = conns[:, 1]
    Ap = _get_entity_area(net, "pore")
    Gp = _get_entity_shape_factor(net, "pore", area=Ap)
    g1 = _segment_conductance_from_agl(
        Ap[p1_idx], Gp[p1_idx], np.asarray(net.throat["pore1_length"], dtype=float), viscosity
    )
    g2 = _segment_conductance_from_agl(
        Ap[p2_idx], Gp[p2_idx], np.asarray(net.throat["pore2_length"], dtype=float), viscosity
    )
    return _harmonic_combine_segments(g1, gt, g2)


def valvatne_blunt_baseline_conductance(net: Network, viscosity: float) -> np.ndarray:
    """Compute a shape-factor-informed single-phase conductance baseline.

    Parameters
    ----------
    net :
        Network containing throat and, ideally, pore geometry.
    viscosity :
        Dynamic viscosity.

    Returns
    -------
    numpy.ndarray
        Throat conductance array.

    Raises
    ------
    ValueError
        If viscosity is not positive.

    Notes
    -----
    This is a pragmatic single-phase baseline inspired by conduit decomposition
    commonly used in pore-network models. The selection logic is:

    1. If ``throat.hydraulic_conductance`` is explicitly present, return it.
    2. Else, if conduit lengths and pore/throat shape data are available, compute a
       harmonic pore1-core-pore2 conductance.
    3. Else, if throat-only shape data are available, use a throat-only model.
    4. Else, warn and fall back to circular Poiseuille conductance.

    It is not a full reproduction of Valvatne-Blunt multiphase physics.
    """

    if viscosity <= 0:
        raise ValueError("viscosity must be positive")

    if "hydraulic_conductance" in net.throat:
        return generic_poiseuille_conductance(net, viscosity)

    try:
        return _valvatne_conduit_baseline(net, viscosity)
    except KeyError:
        pass

    try:
        return _throat_only_shape_factor_conductance(net, viscosity)
    except KeyError:
        warnings.warn(
            "Insufficient geometry for shape-factor model; falling back to generic_poiseuille",
            RuntimeWarning,
            stacklevel=2,
        )
        return generic_poiseuille_conductance(net, viscosity)


def available_conductance_models() -> tuple[str, ...]:
    """Return the names of built-in hydraulic conductance models.

    Returns
    -------
    tuple of str
        Available model names.
    """

    return ("generic_poiseuille", "valvatne_blunt_baseline")


def throat_conductance(
    net: Network, viscosity: float, model: str = "generic_poiseuille"
) -> np.ndarray:
    """Dispatch to a throat hydraulic conductance model.

    Parameters
    ----------
    net :
        Network containing the required geometry.
    viscosity :
        Dynamic viscosity.
    model :
        Conductance model name.

    Returns
    -------
    numpy.ndarray
        Throat conductance array.

    Raises
    ------
    ValueError
        If ``model`` is unknown.
    """

    if model == "generic_poiseuille":
        return generic_poiseuille_conductance(net, viscosity)
    if model == "valvatne_blunt_baseline":
        return valvatne_blunt_baseline_conductance(net, viscosity)
    raise ValueError(f"Unknown conductance model '{model}'")
