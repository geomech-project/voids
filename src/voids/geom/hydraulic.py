from __future__ import annotations

import warnings
from typing import Final

import numpy as np

from voids.core.network import Network


DEFAULT_G_REF: Final[float] = 1.0 / (4.0 * np.pi)  # circular duct shape factor A/P^2
_CIRCLE_COEFF_AG2: Final[float] = 0.5  # gives Hagen-Poiseuille when G=1/(4π)
TRIANGLE_MAX_G: Final[float] = np.sqrt(3.0) / 36.0
SQUARE_G_REF: Final[float] = 1.0 / 16.0
_SQUARE_CIRCLE_TRANSITION_G: Final[float] = 0.07
_MAX_PHYSICAL_G: Final[float] = DEFAULT_G_REF
_TRIANGLE_COEFF_AG2: Final[float] = 3.0 / 5.0
_SQUARE_COEFF_AG2: Final[float] = 0.5623


def _broadcast_viscosity(viscosity: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a viscosity scalar or array to the requested shape."""

    mu = np.asarray(viscosity, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("viscosity must contain only finite values")
    if np.any(mu <= 0.0):
        raise ValueError("viscosity must be positive")
    try:
        return np.broadcast_to(mu, shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(f"viscosity is not broadcastable to shape {shape}") from exc


def _broadcast_finite(
    values: float | np.ndarray, shape: tuple[int, ...], *, name: str
) -> np.ndarray:
    """Broadcast a finite scalar or array to the requested shape."""

    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    try:
        return np.broadcast_to(arr, shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(f"{name} is not broadcastable to shape {shape}") from exc


def _resolve_pore_throat_viscosities(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve pore-wise and throat-wise viscosity arrays."""

    if pore_viscosity is None:
        if viscosity is None:
            raise ValueError("Need either viscosity or pore_viscosity")
        pore_viscosity = viscosity
    if throat_viscosity is None:
        if viscosity is None:
            raise ValueError("Need either viscosity or throat_viscosity")
        throat_viscosity = viscosity
    return (
        _broadcast_viscosity(pore_viscosity, (net.Np,)),
        _broadcast_viscosity(throat_viscosity, (net.Nt,)),
    )


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

    return np.asarray(2.0 * np.sqrt(area / np.pi))


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

    return np.asarray(area / np.maximum(perimeter, 1e-30) ** 2)


def _shape_factor_from_area_inscribed_radius(area: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Compute shape factor from area and inscribed radius.

    Parameters
    ----------
    area :
        Cross-sectional areas.
    radius :
        Inscribed radii.

    Returns
    -------
    numpy.ndarray
        Shape factor defined by ``G = r**2 / (4 * A)``.

    Notes
    -----
    This relation is exact for the equivalent circle/square/triangle ducts used by
    Valvatne-Blunt style network models and by the Imperial College extraction code.
    """

    return np.asarray(radius**2 / np.maximum(4.0 * area, 1e-30))


def _shape_factor_from_area_inscribed_diameter(
    area: np.ndarray, diameter: np.ndarray
) -> np.ndarray:
    """Compute shape factor from area and inscribed diameter."""

    return np.asarray(diameter**2 / np.maximum(16.0 * area, 1e-30))


def _area_from_shape_factor_radius(shape_factor: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Compute area from shape factor and inscribed radius."""

    return np.asarray(radius**2 / np.maximum(4.0 * shape_factor, 1e-30))


def _area_from_shape_factor_diameter(shape_factor: np.ndarray, diameter: np.ndarray) -> np.ndarray:
    """Compute area from shape factor and inscribed diameter."""

    return np.asarray(diameter**2 / np.maximum(16.0 * shape_factor, 1e-30))


def _sanitize_shape_factor(
    shape_factor: np.ndarray,
    *,
    clip_min: float = 1e-12,
    clip_max: float = _MAX_PHYSICAL_G,
) -> np.ndarray:
    """Clip shape factors to a physically admissible range.

    Parameters
    ----------
    shape_factor :
        Input shape-factor array.
    clip_min :
        Lower clip bound.
    clip_max :
        Upper clip bound. The default is the circular-duct maximum ``1 / (4 * pi)``.

    Returns
    -------
    numpy.ndarray
        Clipped shape factors.
    """

    gsf = np.asarray(shape_factor, dtype=float)
    if np.any(gsf < 0):
        raise ValueError("shape_factor contains negative values")
    return np.clip(gsf, clip_min, clip_max)


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
    if "shape_factor" in store:
        gsf = _sanitize_shape_factor(np.asarray(store["shape_factor"], dtype=float))
        if "diameter_inscribed" in store:
            d = np.asarray(store["diameter_inscribed"], dtype=float)
            return _area_from_shape_factor_diameter(gsf, d)
        if "radius_inscribed" in store:
            r = np.asarray(store["radius_inscribed"], dtype=float)
            return _area_from_shape_factor_radius(gsf, r)
    if "diameter_inscribed" in store:
        d = np.asarray(store["diameter_inscribed"], dtype=float)
        return _area_from_diameter(d)
    if "radius_inscribed" in store:
        r = np.asarray(store["radius_inscribed"], dtype=float)
        return np.pi * r**2
    raise KeyError(
        f"Need {kind}.area or {kind}.diameter_inscribed (or radius_inscribed); "
        f"shape_factor + inscribed size is also accepted"
    )


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
    a = _get_entity_area(net, kind) if area is None else np.asarray(area, dtype=float)
    if "diameter_inscribed" in store:
        d = np.asarray(store["diameter_inscribed"], dtype=float)
        return _shape_factor_from_area_inscribed_diameter(a, d)
    if "radius_inscribed" in store:
        r = np.asarray(store["radius_inscribed"], dtype=float)
        return _shape_factor_from_area_inscribed_radius(a, r)
    raise KeyError(
        f"Need {kind}.shape_factor, {kind}.perimeter (with area/diameter), "
        f"or {kind}.area + inscribed size"
    )


def _segment_conductance_from_agl(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float | np.ndarray,
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

    a = np.asarray(area, dtype=float)
    gsf = np.asarray(shape_factor, dtype=float)
    L = np.asarray(length, dtype=float)
    mu = _broadcast_viscosity(viscosity, a.shape)
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
    out[nz] = (_CIRCLE_COEFF_AG2 * gsf[nz] * a[nz] ** 2) / (mu[nz] * L[nz])
    return out


def _conductance_coefficient_from_shape_factor(shape_factor: np.ndarray) -> np.ndarray:
    """Return the Valvatne-Blunt single-phase coefficient for each shape factor.

    Parameters
    ----------
    shape_factor :
        Shape-factor array.

    Returns
    -------
    numpy.ndarray
        Coefficient array ``k`` in ``g = k * G * A**2 / (mu * L)``.

    Notes
    -----
    The triangular coefficient ``3/5`` and square coefficient ``0.5623`` follow
    Valvatne and Blunt (2004) and Patzek and Silin (2001). The square/circle
    transition at ``G = 0.07`` follows the Imperial College `pnflow` reference code.
    """

    gsf = np.asarray(shape_factor, dtype=float)
    if np.any(gsf < 0):
        raise ValueError("shape_factor contains negative values")
    coeff = np.full_like(gsf, _CIRCLE_COEFF_AG2, dtype=float)
    coeff[gsf <= TRIANGLE_MAX_G + 1e-12] = _TRIANGLE_COEFF_AG2
    square = (gsf > TRIANGLE_MAX_G + 1e-12) & (gsf < _SQUARE_CIRCLE_TRANSITION_G)
    coeff[square] = _SQUARE_COEFF_AG2
    return coeff


def _segment_conductance_valvatne_blunt(
    area: np.ndarray,
    shape_factor: np.ndarray,
    length: np.ndarray,
    viscosity: float | np.ndarray,
    *,
    clip_shape_factor: bool = True,
) -> np.ndarray:
    """Compute segment conductance using the Valvatne-Blunt single-phase closure.

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
        If ``True``, clip shape factors to the physically admissible interval
        ``[1e-12, 1 / (4 * pi)]`` before classification and evaluation.

    Returns
    -------
    numpy.ndarray
        Hydraulic conductance array.
    """

    a = np.asarray(area, dtype=float)
    gsf = np.asarray(shape_factor, dtype=float)
    L = np.asarray(length, dtype=float)
    mu = _broadcast_viscosity(viscosity, a.shape)
    if np.any(a < 0):
        raise ValueError("area contains negative values")
    if np.any(L < 0):
        raise ValueError("length contains negative values")
    if np.any(gsf < 0):
        raise ValueError("shape_factor contains negative values")
    if clip_shape_factor:
        gsf = _sanitize_shape_factor(gsf)
    coeff = _conductance_coefficient_from_shape_factor(gsf)
    out = np.full_like(a, np.inf, dtype=float)
    nz = L > 0
    out[nz] = (coeff[nz] * gsf[nz] * a[nz] ** 2) / (mu[nz] * L[nz])
    return out


def generic_poiseuille_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
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

    selected_viscosity = throat_viscosity if throat_viscosity is not None else viscosity
    if selected_viscosity is None:
        raise ValueError("Need either viscosity or throat_viscosity")
    mu_t = _broadcast_viscosity(selected_viscosity, (net.Nt,))
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
    return np.asarray((np.pi * r**4) / (8.0 * mu_t * L), dtype=float)


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


def _harmonic_sensitivity(
    equivalent: np.ndarray,
    segments: tuple[np.ndarray, ...],
    segment_derivatives: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Return the derivative of a harmonic segment combination."""

    g_eq = np.asarray(equivalent, dtype=float)
    term = np.zeros_like(g_eq, dtype=float)
    for g_seg, dg_seg in zip(segments, segment_derivatives, strict=True):
        g_arr = np.asarray(g_seg, dtype=float)
        dg_arr = np.asarray(dg_seg, dtype=float)
        positive = np.isfinite(g_arr) & (g_arr > 0.0)
        term[positive] += dg_arr[positive] / (g_arr[positive] ** 2)
    out = np.zeros_like(g_eq, dtype=float)
    positive_eq = np.isfinite(g_eq) & (g_eq > 0.0)
    out[positive_eq] = (g_eq[positive_eq] ** 2) * term[positive_eq]
    return out


def _throat_only_shape_factor_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
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
    selected_viscosity = throat_viscosity if throat_viscosity is not None else viscosity
    if selected_viscosity is None:
        raise ValueError("Need either viscosity or throat_viscosity")
    return _segment_conductance_valvatne_blunt(A, G, L, selected_viscosity)


def _valvatne_conduit_baseline(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
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
    mu_p, mu_t = _resolve_pore_throat_viscosities(
        net,
        viscosity,
        pore_viscosity=pore_viscosity,
        throat_viscosity=throat_viscosity,
    )

    At = _get_entity_area(net, "throat")
    Gt = _get_entity_shape_factor(net, "throat", area=At)
    Lt = np.asarray(net.throat["core_length"], dtype=float)
    gt = _segment_conductance_valvatne_blunt(At, Gt, Lt, mu_t)

    conns = net.throat_conns
    p1_idx = conns[:, 0]
    p2_idx = conns[:, 1]
    Ap = _get_entity_area(net, "pore")
    Gp = _get_entity_shape_factor(net, "pore", area=Ap)
    g1 = _segment_conductance_valvatne_blunt(
        Ap[p1_idx],
        Gp[p1_idx],
        np.asarray(net.throat["pore1_length"], dtype=float),
        mu_p[p1_idx],
    )
    g2 = _segment_conductance_valvatne_blunt(
        Ap[p2_idx],
        Gp[p2_idx],
        np.asarray(net.throat["pore2_length"], dtype=float),
        mu_p[p2_idx],
    )
    return _harmonic_combine_segments(g1, gt, g2)


def valvatne_blunt_throat_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
    """Compute shape-aware throat conductance using throat geometry only.

    Parameters
    ----------
    net :
        Network containing throat length and cross-sectional geometry.
    viscosity :
        Dynamic viscosity.

    Returns
    -------
    numpy.ndarray
        Shape-aware throat conductance array.

    Raises
    ------
    ValueError
        If viscosity is not positive.
    KeyError
        If required throat geometry is unavailable.
    """

    selected_viscosity = throat_viscosity if throat_viscosity is not None else viscosity
    if selected_viscosity is None:
        raise ValueError("Need either viscosity or throat_viscosity")
    _broadcast_viscosity(selected_viscosity, (net.Nt,))
    if "hydraulic_conductance" in net.throat:
        return generic_poiseuille_conductance(net, viscosity, throat_viscosity=throat_viscosity)
    return _throat_only_shape_factor_conductance(
        net,
        viscosity,
        throat_viscosity=throat_viscosity,
    )


def valvatne_blunt_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
    """Compute a shape-factor-aware single-phase conductance following Valvatne-Blunt.

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
    This implements the single-phase geometric closure used in the Imperial
    College Valvatne-Blunt style network model:

    - segment conductance is evaluated as ``g = k * G * A**2 / (mu * L)``
    - ``k = 3/5`` for triangular ducts
    - ``k = 0.5623`` for square ducts
    - ``k = 1/2`` for circular ducts

    The selection logic is:

    1. If ``throat.hydraulic_conductance`` is explicitly present, return it.
    2. Else, if conduit lengths and pore/throat shape data are available, compute a
       harmonic pore1-core-pore2 conductance.
    3. Else, if throat-only shape data are available, use a throat-only model.
    4. Else, warn and fall back to circular Poiseuille conductance.

    This is still a single-phase closure; corner films and multiphase occupancy are
    intentionally out of scope here.
    """

    _resolve_pore_throat_viscosities(
        net,
        viscosity,
        pore_viscosity=pore_viscosity,
        throat_viscosity=throat_viscosity,
    )

    if "hydraulic_conductance" in net.throat:
        return generic_poiseuille_conductance(net, viscosity, throat_viscosity=throat_viscosity)

    try:
        return _valvatne_conduit_baseline(
            net,
            viscosity,
            pore_viscosity=pore_viscosity,
            throat_viscosity=throat_viscosity,
        )
    except KeyError:
        pass

    try:
        return _throat_only_shape_factor_conductance(
            net,
            viscosity,
            throat_viscosity=throat_viscosity,
        )
    except KeyError:
        warnings.warn(
            "Insufficient geometry for shape-factor model; falling back to generic_poiseuille",
            RuntimeWarning,
            stacklevel=2,
        )
        return generic_poiseuille_conductance(net, viscosity, throat_viscosity=throat_viscosity)


def valvatne_blunt_baseline_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible alias for :func:`valvatne_blunt_conductance`."""

    return valvatne_blunt_conductance(
        net,
        viscosity,
        pore_viscosity=pore_viscosity,
        throat_viscosity=throat_viscosity,
    )


def available_conductance_models() -> tuple[str, ...]:
    """Return the names of built-in hydraulic conductance models.

    Returns
    -------
    tuple of str
        Available model names.
    """

    return (
        "generic_poiseuille",
        "valvatne_blunt_throat",
        "valvatne_blunt",
        "valvatne_blunt_baseline",
    )


def throat_conductance(
    net: Network,
    viscosity: float | np.ndarray | None,
    model: str = "generic_poiseuille",
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
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
        return generic_poiseuille_conductance(net, viscosity, throat_viscosity=throat_viscosity)
    if model == "valvatne_blunt_throat":
        return valvatne_blunt_throat_conductance(
            net,
            viscosity,
            throat_viscosity=throat_viscosity,
        )
    if model == "valvatne_blunt":
        return valvatne_blunt_conductance(
            net,
            viscosity,
            pore_viscosity=pore_viscosity,
            throat_viscosity=throat_viscosity,
        )
    if model == "valvatne_blunt_baseline":
        return valvatne_blunt_conductance(
            net,
            viscosity,
            pore_viscosity=pore_viscosity,
            throat_viscosity=throat_viscosity,
        )
    raise ValueError(f"Unknown conductance model '{model}'")


def throat_conductance_with_sensitivities(
    net: Network,
    viscosity: float | np.ndarray | None,
    model: str = "generic_poiseuille",
    *,
    pore_viscosity: float | np.ndarray | None = None,
    throat_viscosity: float | np.ndarray | None = None,
    pore_dviscosity_dpressure: float | np.ndarray | None = None,
    throat_dviscosity_dpressure: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return throat conductance and endpoint pressure sensitivities.

    Notes
    -----
    The returned arrays are ``(g, dg_dpi, dg_dpj)`` where ``i`` and ``j`` are
    the pore indices in ``net.throat_conns[:, 0]`` and ``net.throat_conns[:, 1]``.
    """

    conns = net.throat_conns
    i_idx = conns[:, 0]
    j_idx = conns[:, 1]

    if "hydraulic_conductance" in net.throat:
        g = generic_poiseuille_conductance(net, viscosity, throat_viscosity=throat_viscosity)
        zeros = np.zeros_like(g, dtype=float)
        return g, zeros, zeros

    if model == "generic_poiseuille":
        selected_viscosity = throat_viscosity if throat_viscosity is not None else viscosity
        if selected_viscosity is None:
            raise ValueError("Need either viscosity or throat_viscosity")
        mu_t = _broadcast_viscosity(selected_viscosity, (net.Nt,))
        dmu_t = (
            _broadcast_finite(
                throat_dviscosity_dpressure,
                (net.Nt,),
                name="throat_dviscosity_dpressure",
            )
            if throat_dviscosity_dpressure is not None
            else np.zeros(net.Nt, dtype=float)
        )
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
        g = (np.pi * r**4) / (8.0 * mu_t * L)
        dg_dmu = -g / mu_t
        factor = 0.5 * dmu_t
        return g, dg_dmu * factor, dg_dmu * factor

    if model == "valvatne_blunt_throat":
        selected_viscosity = throat_viscosity if throat_viscosity is not None else viscosity
        if selected_viscosity is None:
            raise ValueError("Need either viscosity or throat_viscosity")
        mu_t = _broadcast_viscosity(selected_viscosity, (net.Nt,))
        dmu_t = (
            _broadcast_finite(
                throat_dviscosity_dpressure,
                (net.Nt,),
                name="throat_dviscosity_dpressure",
            )
            if throat_dviscosity_dpressure is not None
            else np.zeros(net.Nt, dtype=float)
        )
        _require(net, "throat", ("length",))
        A = _get_entity_area(net, "throat")
        G = _get_entity_shape_factor(net, "throat", area=A)
        L = np.asarray(net.throat["length"], dtype=float)
        g = _segment_conductance_valvatne_blunt(A, G, L, mu_t)
        dg_dmu = np.zeros_like(g, dtype=float)
        positive = np.isfinite(g) & (g > 0.0)
        dg_dmu[positive] = -g[positive] / mu_t[positive]
        factor = 0.5 * dmu_t
        return g, dg_dmu * factor, dg_dmu * factor

    if model in {"valvatne_blunt", "valvatne_blunt_baseline"}:
        mu_p, mu_t = _resolve_pore_throat_viscosities(
            net,
            viscosity,
            pore_viscosity=pore_viscosity,
            throat_viscosity=throat_viscosity,
        )
        dmu_p = (
            _broadcast_finite(
                pore_dviscosity_dpressure,
                (net.Np,),
                name="pore_dviscosity_dpressure",
            )
            if pore_dviscosity_dpressure is not None
            else np.zeros(net.Np, dtype=float)
        )
        dmu_t = (
            _broadcast_finite(
                throat_dviscosity_dpressure,
                (net.Nt,),
                name="throat_dviscosity_dpressure",
            )
            if throat_dviscosity_dpressure is not None
            else np.zeros(net.Nt, dtype=float)
        )
        try:
            if not _conduit_lengths_available(net):
                raise KeyError("Missing conduit lengths (pore1_length, core_length, pore2_length)")
            At = _get_entity_area(net, "throat")
            Gt = _get_entity_shape_factor(net, "throat", area=At)
            Lt = np.asarray(net.throat["core_length"], dtype=float)
            gt = _segment_conductance_valvatne_blunt(At, Gt, Lt, mu_t)
            dgt_dmu = np.zeros_like(gt, dtype=float)
            positive_t = np.isfinite(gt) & (gt > 0.0)
            dgt_dmu[positive_t] = -gt[positive_t] / mu_t[positive_t]
            dgt_dpi = dgt_dmu * (0.5 * dmu_t)
            dgt_dpj = dgt_dmu * (0.5 * dmu_t)

            Ap = _get_entity_area(net, "pore")
            Gp = _get_entity_shape_factor(net, "pore", area=Ap)
            L1 = np.asarray(net.throat["pore1_length"], dtype=float)
            L2 = np.asarray(net.throat["pore2_length"], dtype=float)
            g1 = _segment_conductance_valvatne_blunt(Ap[i_idx], Gp[i_idx], L1, mu_p[i_idx])
            g2 = _segment_conductance_valvatne_blunt(Ap[j_idx], Gp[j_idx], L2, mu_p[j_idx])
            dg1_dmu = np.zeros_like(g1, dtype=float)
            dg2_dmu = np.zeros_like(g2, dtype=float)
            positive_1 = np.isfinite(g1) & (g1 > 0.0)
            positive_2 = np.isfinite(g2) & (g2 > 0.0)
            dg1_dmu[positive_1] = -g1[positive_1] / mu_p[i_idx][positive_1]
            dg2_dmu[positive_2] = -g2[positive_2] / mu_p[j_idx][positive_2]
            dg1_dpi = dg1_dmu * dmu_p[i_idx]
            dg2_dpj = dg2_dmu * dmu_p[j_idx]

            g = _harmonic_combine_segments(g1, gt, g2)
            dg_dpi = _harmonic_sensitivity(g, (g1, gt, g2), (dg1_dpi, dgt_dpi, np.zeros_like(g2)))
            dg_dpj = _harmonic_sensitivity(g, (g1, gt, g2), (np.zeros_like(g1), dgt_dpj, dg2_dpj))
            return g, dg_dpi, dg_dpj
        except KeyError:
            try:
                A = _get_entity_area(net, "throat")
                G = _get_entity_shape_factor(net, "throat", area=A)
                L = np.asarray(net.throat["length"], dtype=float)
                g = _segment_conductance_valvatne_blunt(A, G, L, mu_t)
                dg_dmu = np.zeros_like(g, dtype=float)
                positive = np.isfinite(g) & (g > 0.0)
                dg_dmu[positive] = -g[positive] / mu_t[positive]
                factor = 0.5 * dmu_t
                return g, dg_dmu * factor, dg_dmu * factor
            except KeyError:
                warnings.warn(
                    "Insufficient geometry for shape-factor model; falling back to generic_poiseuille",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return throat_conductance_with_sensitivities(
                    net,
                    viscosity,
                    model="generic_poiseuille",
                    pore_viscosity=pore_viscosity,
                    throat_viscosity=throat_viscosity,
                    pore_dviscosity_dpressure=pore_dviscosity_dpressure,
                    throat_dviscosity_dpressure=throat_dviscosity_dpressure,
                )

    raise ValueError(f"Unknown conductance model '{model}'")
