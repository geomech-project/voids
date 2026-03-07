from __future__ import annotations

from collections.abc import Sequence


def format_radius_token(value: float) -> str:
    """Return a stable filename-safe token for radius values."""

    return f"{value:.2f}".replace(".", "p")


def equivalent_radius_2d(radii_xy: tuple[float, float]) -> float:
    """Return the area-equivalent circular radius for an ellipse."""

    rx, ry = radii_xy
    if min(rx, ry) <= 0:
        raise ValueError("Ellipse radii must be positive")
    return float((rx * ry) ** 0.5)


def equivalent_radius_3d(radii_xyz: tuple[float, float, float]) -> float:
    """Return the volume-equivalent spherical radius for an ellipsoid."""

    rx, ry, rz = radii_xyz
    if min(rx, ry, rz) <= 0:
        raise ValueError("Ellipsoid radii must be positive")
    return float((rx * ry * rz) ** (1.0 / 3.0))


def _validate_aspect_and_window(*, aspect: float, search_window: int) -> None:
    if aspect <= 1.0:
        raise ValueError("aspect must be > 1.0")
    if search_window < 1:
        raise ValueError("search_window must be >= 1")


def match_ellipse_to_circle(
    radius_vox: int,
    *,
    aspect: float,
    search_window: int,
) -> tuple[int, int]:
    """
    Match integer ellipse radii `(a, b)` to circular area `r^2`.

    The optimization prioritizes area error and then aspect-ratio error.
    """

    _validate_aspect_and_window(aspect=aspect, search_window=search_window)
    r = int(radius_vox)
    if r <= 0:
        raise ValueError("radius_vox must be positive")

    target = float(r**2)
    b_real = r / (aspect**0.5)
    a_real = aspect * b_real
    a0 = int(round(a_real))
    b0 = int(round(b_real))

    best_key: tuple[float, float, float, float] | None = None
    best_tuple: tuple[int, int] | None = None

    b_min = max(1, b0 - search_window)
    b_max = max(b_min, b0 + search_window)
    for b in range(b_min, b_max + 1):
        a_target = target / float(b)
        a_center = int(round(a_target))
        a_min = max(1, min(a0, a_center) - search_window)
        a_max = max(a_min, max(a0, a_center) + search_window)
        for a in range(a_min, a_max + 1):
            area = float(a * b)
            area_err = abs(area - target) / target
            asp = float(a / b)
            asp_err = abs(asp - aspect) / aspect
            center_err = abs(a - a_real) + abs(b - b_real)
            key = (area_err + 0.10 * asp_err, area_err, asp_err, center_err)
            if best_key is None or key < best_key:
                best_key = key
                best_tuple = (int(a), int(b))

    if best_tuple is None:  # pragma: no cover - finite search always yields candidates
        raise RuntimeError("Could not find matched ellipse radii")
    return best_tuple


def match_ellipsoid_to_sphere(
    radius_vox: int,
    *,
    aspect: float,
    search_window: int,
) -> tuple[int, int, int]:
    """
    Match integer ellipsoid radii `(a, b, b)` to spherical volume `r^3`.

    The optimization prioritizes volume error and then aspect-ratio error.
    """

    _validate_aspect_and_window(aspect=aspect, search_window=search_window)
    r = int(radius_vox)
    if r <= 0:
        raise ValueError("radius_vox must be positive")

    target = float(r**3)
    b_real = r / (aspect ** (1.0 / 3.0))
    a_real = aspect * b_real
    a0 = int(round(a_real))
    b0 = int(round(b_real))

    best_key: tuple[float, float, float, float] | None = None
    best_tuple: tuple[int, int, int] | None = None

    b_min = max(1, b0 - search_window)
    b_max = max(b_min, b0 + search_window)
    for b in range(b_min, b_max + 1):
        a_target = target / float(b * b)
        a_center = int(round(a_target))
        a_min = max(1, min(a0, a_center) - search_window)
        a_max = max(a_min, max(a0, a_center) + search_window)
        for a in range(a_min, a_max + 1):
            vol = float(a * b * b)
            vol_err = abs(vol - target) / target
            asp = float(a / b)
            asp_err = abs(asp - aspect) / aspect
            center_err = abs(a - a_real) + abs(b - b_real)
            key = (vol_err + 0.12 * asp_err, vol_err, asp_err, center_err)
            if best_key is None or key < best_key:
                best_key = key
                best_tuple = (int(a), int(b), int(b))

    if best_tuple is None:  # pragma: no cover - finite search always yields candidates
        raise RuntimeError("Could not find matched ellipsoid radii")
    return best_tuple


def build_image_vug_radii_2d(
    circle_radii_vox: Sequence[int],
    *,
    aspect: float,
    search_window: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, float, float]]]:
    """
    Build area-matched 2D anisotropic radii from circular base radii.

    Returns
    -------
    tuple
        ``(flow_radii, orth_radii, match_report)`` where ``match_report`` contains
        ``(config_index, flow_rel_error, orth_rel_error)``.
    """

    flow_radii: list[tuple[int, int]] = []
    orth_radii: list[tuple[int, int]] = []
    report: list[tuple[int, float, float]] = []

    for idx, radius in enumerate(circle_radii_vox, start=1):
        r = int(radius)
        if r <= 0:
            raise ValueError("All circle_radii_vox values must be positive")
        flow = match_ellipse_to_circle(r, aspect=aspect, search_window=search_window)
        orth = (flow[1], flow[0])
        target = float(r**2)
        flow_err = abs(float(flow[0] * flow[1]) - target) / target
        orth_err = abs(float(orth[0] * orth[1]) - target) / target
        flow_radii.append(flow)
        orth_radii.append(orth)
        report.append((idx, flow_err, orth_err))

    return flow_radii, orth_radii, report


def build_image_vug_radii_3d(
    sphere_radii_vox: Sequence[int],
    *,
    aspect: float,
    search_window: int,
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, float, float]],
]:
    """
    Build volume-matched 3D anisotropic radii from spherical base radii.

    Returns
    -------
    tuple
        ``(flow_radii, orth_radii, match_report)`` where ``match_report`` contains
        ``(config_index, flow_rel_error, orth_rel_error)``.
    """

    flow_radii: list[tuple[int, int, int]] = []
    orth_radii: list[tuple[int, int, int]] = []
    report: list[tuple[int, float, float]] = []

    for idx, radius in enumerate(sphere_radii_vox, start=1):
        r = int(radius)
        if r <= 0:
            raise ValueError("All sphere_radii_vox values must be positive")
        flow = match_ellipsoid_to_sphere(r, aspect=aspect, search_window=search_window)
        orth = (flow[1], flow[2], flow[0])
        target = float(r**3)
        flow_err = abs(float(flow[0] * flow[1] * flow[2]) - target) / target
        orth_err = abs(float(orth[0] * orth[1] * orth[2]) - target) / target
        flow_radii.append(flow)
        orth_radii.append(orth)
        report.append((idx, flow_err, orth_err))

    return flow_radii, orth_radii, report


def build_lattice_vug_templates_2d(
    *,
    equiv_radii_spacing: Sequence[float],
    spacing_m: float,
    aspect: float,
) -> tuple[list[dict[str, object]], list[tuple[int, float, float, float]]]:
    """
    Build 2D lattice vug templates with matched equivalent radius per config.

    Returns
    -------
    tuple
        ``(templates, match_report)``, where ``match_report`` contains
        ``(config_index, circle_err, flow_err, orth_err)`` relative to target area.
    """

    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if aspect <= 1.0:
        raise ValueError("aspect must be > 1.0")

    templates: list[dict[str, object]] = []
    report: list[tuple[int, float, float, float]] = []
    aspect_root = aspect**0.5

    for idx, req_over_spacing in enumerate(equiv_radii_spacing, start=1):
        req = float(req_over_spacing)
        if req <= 0:
            raise ValueError("All equiv_radii_spacing values must be positive")
        req_m = req * spacing_m
        target = float(req_m**2)

        circle = (req_m, req_m)
        b = req_m / aspect_root
        a = aspect * b
        flow = (a, b)
        orth = (b, a)

        circle_err = abs(float(circle[0] * circle[1]) - target) / target
        flow_err = abs(float(flow[0] * flow[1]) - target) / target
        orth_err = abs(float(orth[0] * orth[1]) - target) / target

        token = format_radius_token(req)
        templates.extend(
            [
                {
                    "case": f"circle_cfg{idx}_req{token}",
                    "family": "circular",
                    "orientation": "isotropic",
                    "config_index": idx,
                    "radii_xy_m": circle,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_area_rel_error": circle_err,
                },
                {
                    "case": f"ellipse_flow_cfg{idx}_req{token}",
                    "family": "elliptical",
                    "orientation": "flow_stretched",
                    "config_index": idx,
                    "radii_xy_m": flow,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_area_rel_error": flow_err,
                },
                {
                    "case": f"ellipse_orth_cfg{idx}_req{token}",
                    "family": "elliptical",
                    "orientation": "orthogonal_stretched",
                    "config_index": idx,
                    "radii_xy_m": orth,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_area_rel_error": orth_err,
                },
            ]
        )
        report.append((idx, circle_err, flow_err, orth_err))

    return templates, report


def build_lattice_vug_templates_3d(
    *,
    equiv_radii_spacing: Sequence[float],
    spacing_m: float,
    aspect: float,
) -> tuple[list[dict[str, object]], list[tuple[int, float, float, float]]]:
    """
    Build 3D lattice vug templates with matched equivalent radius per config.

    Returns
    -------
    tuple
        ``(templates, match_report)``, where ``match_report`` contains
        ``(config_index, sphere_err, flow_err, orth_err)`` relative to target
        volume.
    """

    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if aspect <= 1.0:
        raise ValueError("aspect must be > 1.0")

    templates: list[dict[str, object]] = []
    report: list[tuple[int, float, float, float]] = []
    aspect_root = aspect ** (1.0 / 3.0)

    for idx, req_over_spacing in enumerate(equiv_radii_spacing, start=1):
        req = float(req_over_spacing)
        if req <= 0:
            raise ValueError("All equiv_radii_spacing values must be positive")
        req_m = req * spacing_m
        target = float(req_m**3)

        sphere = (req_m, req_m, req_m)
        b = req_m / aspect_root
        a = aspect * b
        flow = (a, b, b)
        orth = (b, b, a)

        sphere_err = abs(float(sphere[0] * sphere[1] * sphere[2]) - target) / target
        flow_err = abs(float(flow[0] * flow[1] * flow[2]) - target) / target
        orth_err = abs(float(orth[0] * orth[1] * orth[2]) - target) / target

        token = format_radius_token(req)
        templates.extend(
            [
                {
                    "case": f"sphere_cfg{idx}_req{token}",
                    "family": "spherical",
                    "orientation": "isotropic",
                    "config_index": idx,
                    "radii_xyz_m": sphere,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_volume_rel_error": sphere_err,
                },
                {
                    "case": f"ellipsoid_flow_cfg{idx}_req{token}",
                    "family": "ellipsoidal",
                    "orientation": "flow_stretched",
                    "config_index": idx,
                    "radii_xyz_m": flow,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_volume_rel_error": flow_err,
                },
                {
                    "case": f"ellipsoid_orth_cfg{idx}_req{token}",
                    "family": "ellipsoidal",
                    "orientation": "orthogonal_stretched",
                    "config_index": idx,
                    "radii_xyz_m": orth,
                    "r_eq_spacing": req,
                    "target_equivalent_radius_m": req_m,
                    "template_volume_rel_error": orth_err,
                },
            ]
        )
        report.append((idx, sphere_err, flow_err, orth_err))

    return templates, report
