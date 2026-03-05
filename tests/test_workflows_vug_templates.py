from __future__ import annotations

import pytest

from voids.workflows.vug_templates import (
    build_image_vug_radii_2d,
    build_image_vug_radii_3d,
    build_lattice_vug_templates_2d,
    build_lattice_vug_templates_3d,
    equivalent_radius_2d,
    equivalent_radius_3d,
    format_radius_token,
    match_ellipse_to_circle,
    match_ellipsoid_to_sphere,
)


def test_format_radius_token() -> None:
    assert format_radius_token(1.4) == "1p40"
    assert format_radius_token(2.0) == "2p00"


def test_equivalent_radius_functions() -> None:
    assert equivalent_radius_2d((9.0, 16.0)) == pytest.approx(12.0)
    assert equivalent_radius_3d((8.0, 18.0, 8.0)) == pytest.approx(
        (8.0 * 18.0 * 8.0) ** (1.0 / 3.0)
    )


@pytest.mark.parametrize(
    "radii",
    [(0.0, 2.0), (-1.0, 2.0)],
)
def test_equivalent_radius_2d_rejects_nonpositive(radii: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        equivalent_radius_2d(radii)


@pytest.mark.parametrize(
    "radii",
    [(0.0, 2.0, 3.0), (-1.0, 2.0, 3.0)],
)
def test_equivalent_radius_3d_rejects_nonpositive(
    radii: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        equivalent_radius_3d(radii)


@pytest.mark.parametrize("aspect", [0.8, 1.0])
def test_match_ellipse_to_circle_rejects_invalid_aspect(aspect: float) -> None:
    with pytest.raises(ValueError, match="aspect"):
        match_ellipse_to_circle(10, aspect=aspect, search_window=5)


def test_match_ellipse_to_circle_rejects_invalid_radius_and_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        match_ellipse_to_circle(0, aspect=1.8, search_window=5)
    with pytest.raises(ValueError, match="search_window"):
        match_ellipse_to_circle(10, aspect=1.8, search_window=0)


def test_match_ellipse_to_circle_basic_properties() -> None:
    a, b = match_ellipse_to_circle(20, aspect=1.8, search_window=10)
    assert a >= b >= 1
    rel_area_err = abs(a * b - 20**2) / (20**2)
    assert rel_area_err < 0.03


@pytest.mark.parametrize("aspect", [0.8, 1.0])
def test_match_ellipsoid_to_sphere_rejects_invalid_aspect(aspect: float) -> None:
    with pytest.raises(ValueError, match="aspect"):
        match_ellipsoid_to_sphere(10, aspect=aspect, search_window=5)


def test_match_ellipsoid_to_sphere_rejects_invalid_radius_and_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        match_ellipsoid_to_sphere(0, aspect=1.8, search_window=5)
    with pytest.raises(ValueError, match="search_window"):
        match_ellipsoid_to_sphere(10, aspect=1.8, search_window=0)


def test_match_ellipsoid_to_sphere_basic_properties() -> None:
    a, b, c = match_ellipsoid_to_sphere(10, aspect=1.8, search_window=8)
    assert b == c
    assert a >= b >= 1
    rel_vol_err = abs(a * b * c - 10**3) / (10**3)
    assert rel_vol_err < 0.05


def test_build_image_vug_radii_2d_outputs() -> None:
    flow, orth, report = build_image_vug_radii_2d([12, 16], aspect=1.8, search_window=10)
    assert len(flow) == len(orth) == len(report) == 2
    for idx, (f, o, (cfg_idx, f_err, o_err)) in enumerate(zip(flow, orth, report), start=1):
        assert cfg_idx == idx
        assert o == (f[1], f[0])
        assert f_err >= 0.0
        assert o_err >= 0.0


def test_build_image_vug_radii_2d_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_image_vug_radii_2d([10, 0], aspect=1.8, search_window=4)


def test_build_image_vug_radii_3d_outputs() -> None:
    flow, orth, report = build_image_vug_radii_3d([4, 6], aspect=1.8, search_window=8)
    assert len(flow) == len(orth) == len(report) == 2
    for idx, (f, o, (cfg_idx, f_err, o_err)) in enumerate(zip(flow, orth, report), start=1):
        assert cfg_idx == idx
        assert o == (f[1], f[2], f[0])
        assert f_err >= 0.0
        assert o_err >= 0.0


def test_build_image_vug_radii_3d_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_image_vug_radii_3d([6, -1], aspect=1.8, search_window=4)


def test_build_lattice_vug_templates_2d_outputs() -> None:
    templates, report = build_lattice_vug_templates_2d(
        equiv_radii_spacing=[1.6, 2.0],
        spacing_m=4.0e-5,
        aspect=1.8,
    )
    assert len(templates) == 6
    assert len(report) == 2
    assert templates[0]["case"] == "circle_cfg1_req1p60"
    assert templates[1]["case"] == "ellipse_flow_cfg1_req1p60"
    assert templates[2]["case"] == "ellipse_orth_cfg1_req1p60"
    assert templates[0]["r_eq_spacing"] == pytest.approx(1.6)
    for _, c_err, f_err, o_err in report:
        assert c_err >= 0.0
        assert f_err >= 0.0
        assert o_err >= 0.0


def test_build_lattice_vug_templates_2d_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="spacing_m"):
        build_lattice_vug_templates_2d(
            equiv_radii_spacing=[1.0],
            spacing_m=0.0,
            aspect=1.8,
        )
    with pytest.raises(ValueError, match="aspect"):
        build_lattice_vug_templates_2d(
            equiv_radii_spacing=[1.0],
            spacing_m=1.0,
            aspect=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        build_lattice_vug_templates_2d(
            equiv_radii_spacing=[0.0, 1.0],
            spacing_m=1.0,
            aspect=1.8,
        )


def test_build_lattice_vug_templates_3d_outputs() -> None:
    templates, report = build_lattice_vug_templates_3d(
        equiv_radii_spacing=[1.4, 1.7],
        spacing_m=4.0e-5,
        aspect=1.8,
    )
    assert len(templates) == 6
    assert len(report) == 2
    assert templates[0]["case"] == "sphere_cfg1_req1p40"
    assert templates[1]["case"] == "ellipsoid_flow_cfg1_req1p40"
    assert templates[2]["case"] == "ellipsoid_orth_cfg1_req1p40"
    assert templates[0]["r_eq_spacing"] == pytest.approx(1.4)
    for _, s_err, f_err, o_err in report:
        assert s_err >= 0.0
        assert f_err >= 0.0
        assert o_err >= 0.0


def test_build_lattice_vug_templates_3d_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="spacing_m"):
        build_lattice_vug_templates_3d(
            equiv_radii_spacing=[1.0],
            spacing_m=0.0,
            aspect=1.8,
        )
    with pytest.raises(ValueError, match="aspect"):
        build_lattice_vug_templates_3d(
            equiv_radii_spacing=[1.0],
            spacing_m=1.0,
            aspect=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        build_lattice_vug_templates_3d(
            equiv_radii_spacing=[-1.0, 1.0],
            spacing_m=1.0,
            aspect=1.8,
        )
