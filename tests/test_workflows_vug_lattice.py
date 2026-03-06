from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from voids.examples import make_cartesian_mesh_network
from voids.generators import network as gnet


def test_sample_depth_3d_and_2d_inference() -> None:
    net3d = make_cartesian_mesh_network((4, 3, 2), spacing=2.0)
    assert gnet.sample_depth(net3d) == pytest.approx(4.0)

    net2d = make_cartesian_mesh_network((5, 4), spacing=1.2, thickness=0.35)
    assert gnet.sample_depth(net2d) == pytest.approx(0.35)


def test_update_network_geometry_from_radii_sets_expected_fields() -> None:
    net = make_cartesian_mesh_network((4, 4, 3), spacing=1.0)
    pore_radius = np.full(net.Np, 0.22, dtype=float)
    throat_radius = np.full(net.Nt, 0.10, dtype=float)
    gnet.update_network_geometry_from_radii(
        net,
        pore_radius=pore_radius,
        throat_radius=throat_radius,
    )

    assert np.allclose(net.pore["radius_inscribed"], pore_radius)
    assert np.allclose(net.throat["radius_inscribed"], throat_radius)
    assert np.all(net.throat["core_length"] > 0.0)
    assert np.all(net.pore["volume"] > 0.0)
    assert np.all(net.throat["volume"] > 0.0)


def test_update_network_geometry_2d_uses_sample_depth_by_default() -> None:
    net = make_cartesian_mesh_network((6, 5), spacing=1.0, thickness=0.4)
    pore_radius = np.full(net.Np, 0.18, dtype=float)
    throat_radius = np.full(net.Nt, 0.09, dtype=float)
    gnet.update_network_geometry_2d(
        net,
        pore_radius=pore_radius,
        throat_radius=throat_radius,
    )

    expected_pore_volume = np.pi * pore_radius**2 * 0.4
    assert np.allclose(net.pore["volume"], expected_pore_volume)
    assert np.all(net.throat["core_length"] > 0.0)


def test_update_network_geometry_rejects_invalid_shapes() -> None:
    net = make_cartesian_mesh_network((4, 4), spacing=1.0, thickness=1.0)
    with pytest.raises(ValueError, match="shape"):
        gnet.update_network_geometry_2d(
            net,
            pore_radius=np.ones(net.Np + 1, dtype=float),
            throat_radius=np.ones(net.Nt, dtype=float),
        )


def test_insert_vug_superpore_3d_adds_vug_and_connections() -> None:
    net = make_cartesian_mesh_network((9, 9, 9), spacing=1.0, pore_radius=0.2, throat_radius=0.1)
    net_vug, meta = gnet.insert_vug_superpore(net, radii_xyz=(2.2, 1.6, 1.2))

    assert net_vug.Np < net.Np
    assert net_vug.Nt > 0
    assert net_vug.pore_labels["vug"].sum() == 1
    assert net_vug.throat_labels["vug_connection"].sum() == meta["boundary_neighbors"]
    assert meta["removed_pores"] >= 1
    assert meta["boundary_neighbors"] >= 1
    assert "vug_equivalent_radius_m" in net_vug.extra


def test_insert_vug_superpore_2d_adds_vug_and_connections() -> None:
    net = make_cartesian_mesh_network((20, 20), spacing=1.0, thickness=0.5)
    net_vug, meta = gnet.insert_vug_superpore_2d(net, radii_xy=(4.0, 2.5))

    assert net_vug.Np < net.Np
    assert net_vug.Nt > 0
    assert net_vug.pore_labels["vug"].sum() == 1
    assert net_vug.throat_labels["vug_connection"].sum() == meta["boundary_neighbors"]
    assert meta["removed_pores"] >= 1
    assert meta["boundary_neighbors"] >= 1
    assert "vug_equivalent_radius_m" in net_vug.extra


def test_insert_vug_superpore_requires_pore_radius_field() -> None:
    net = make_cartesian_mesh_network((7, 7, 7), spacing=1.0, pore_radius=0.2, throat_radius=0.1)
    net.pore.pop("radius_inscribed")
    with pytest.raises(KeyError, match="radius_inscribed"):
        gnet.insert_vug_superpore(net, radii_xyz=(2.0, 1.5, 1.0))


def test_insert_vug_superpore_3d_alias_points_to_main_function() -> None:
    assert gnet.insert_vug_superpore_3d is gnet.insert_vug_superpore


def test_network_internal_helpers_validation_and_fallback_branches() -> None:
    with pytest.raises(ValueError, match="Ellipse radii must be positive"):
        gnet._equivalent_radius_2d((1.0, 0.0))
    with pytest.raises(ValueError, match="Ellipsoid radii must be positive"):
        gnet._equivalent_radius_3d((1.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="shape_factor must be positive"):
        gnet._validate_geometry_update_controls(
            shape_factor=0.0,
            pore_length_fraction=0.1,
            min_core_fraction=0.1,
        )
    with pytest.raises(ValueError, match="pore_length_fraction must be non-negative"):
        gnet._validate_geometry_update_controls(
            shape_factor=0.1,
            pore_length_fraction=-0.1,
            min_core_fraction=0.1,
        )
    with pytest.raises(ValueError, match="min_core_fraction must be non-negative"):
        gnet._validate_geometry_update_controls(
            shape_factor=0.1,
            pore_length_fraction=0.1,
            min_core_fraction=-0.1,
        )

    store = {
        "radius_inscribed": np.array([1.0, 2.0], dtype=float),
        "custom2d": np.ones((2, 2), dtype=float),
    }
    gnet._extend_entity_fields(store, n_before=2, n_append=1, append_fields={})
    assert store["radius_inscribed"].shape == (3,)
    assert store["custom2d"].shape == (3, 2)
    assert store["radius_inscribed"][-1] == pytest.approx(0.0)
    assert np.allclose(store["custom2d"][-1], 0.0)

    with pytest.raises(ValueError, match="append field 'radius_inscribed' has shape"):
        gnet._extend_entity_fields(
            {"radius_inscribed": np.array([1.0, 2.0], dtype=float)},
            n_before=2,
            n_append=1,
            append_fields={"radius_inscribed": np.ones((2,), dtype=float)},
        )

    with pytest.raises(ValueError, match="Ellipsoid radii must be strictly positive"):
        gnet._ellipsoid_mask(
            np.zeros((1, 3), dtype=float),
            center=np.zeros(3, dtype=float),
            radii_xyz=(1.0, 1.0, 0.0),
        )
    with pytest.raises(ValueError, match="Ellipse radii must be strictly positive"):
        gnet._ellipse_mask_2d(
            np.zeros((1, 3), dtype=float),
            center_xy=(0.0, 0.0),
            radii_xy=(1.0, 0.0),
        )

    net_like = SimpleNamespace(throat={"diameter_inscribed": np.array([2.0, 4.0], dtype=float)})
    assert gnet._median_throat_radius(net_like, fallback=0.9) == pytest.approx(1.5)
    assert gnet._median_throat_radius(SimpleNamespace(throat={}), fallback=0.9) == pytest.approx(
        0.9
    )


def test_network_sample_and_geometry_update_error_branches() -> None:
    net3d = make_cartesian_mesh_network((3, 3, 3), spacing=1.0)
    net3d.sample.lengths["z"] = 0.0
    with pytest.raises(ValueError, match="sample depth must be positive"):
        gnet.sample_depth(net3d)

    net_geo3d = make_cartesian_mesh_network((4, 4, 3), spacing=1.0)
    with pytest.raises(ValueError, match="must be strictly positive"):
        gnet.update_network_geometry_from_radii(
            net_geo3d,
            pore_radius=np.full(net_geo3d.Np, 0.2, dtype=float),
            throat_radius=np.zeros(net_geo3d.Nt, dtype=float),
        )

    net_geo2d = make_cartesian_mesh_network((4, 4), spacing=1.0, thickness=0.4)
    with pytest.raises(ValueError, match="must be strictly positive"):
        gnet.update_network_geometry_2d(
            net_geo2d,
            pore_radius=np.full(net_geo2d.Np, 0.2, dtype=float),
            throat_radius=np.zeros(net_geo2d.Nt, dtype=float),
        )
    with pytest.raises(ValueError, match="depth must be positive"):
        gnet.update_network_geometry_2d(
            net_geo2d,
            pore_radius=np.full(net_geo2d.Np, 0.2, dtype=float),
            throat_radius=np.full(net_geo2d.Nt, 0.1, dtype=float),
            depth=0.0,
        )


def test_insert_vug_superpore_3d_error_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = make_cartesian_mesh_network((5, 5, 5), spacing=1.0, pore_radius=0.2, throat_radius=0.1)

    with pytest.raises(ValueError, match="All radii_xyz values must be positive"):
        gnet.insert_vug_superpore(net, radii_xyz=(0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="shape_factor must be positive"):
        gnet.insert_vug_superpore(net, radii_xyz=(1.0, 1.0, 1.0), shape_factor=0.0)
    with pytest.raises(ValueError, match="center must have shape \\(3,\\)"):
        gnet.insert_vug_superpore(net, radii_xyz=(1.0, 1.0, 1.0), center=(0.0, 0.0))

    net_fallback, meta_fallback = gnet.insert_vug_superpore(
        net,
        radii_xyz=(0.05, 0.05, 0.05),
        center=(1.0e6, 1.0e6, 1.0e6),
    )
    assert net_fallback.pore_labels["vug"].sum() == 1
    assert meta_fallback["removed_pores"] == 1

    with pytest.raises(RuntimeError, match="zero interface neighbors"):
        gnet.insert_vug_superpore(net, radii_xyz=(1.0e6, 1.0e6, 1.0e6))

    def fake_induced_subnetwork(base_net, _mask):
        return base_net.copy(), np.array([], dtype=int), np.array([], dtype=bool)

    monkeypatch.setattr(gnet, "induced_subnetwork", fake_induced_subnetwork)
    center = tuple(float(v) for v in net.pore_coords[0])
    with pytest.raises(RuntimeError, match="No boundary pores survived"):
        gnet.insert_vug_superpore(net, radii_xyz=(0.05, 0.05, 0.05), center=center)


def test_insert_vug_superpore_2d_error_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = make_cartesian_mesh_network((10, 10), spacing=1.0, thickness=0.5)

    with pytest.raises(ValueError, match="All radii_xy values must be positive"):
        gnet.insert_vug_superpore_2d(net, radii_xy=(0.0, 1.0))
    with pytest.raises(ValueError, match="shape_factor must be positive"):
        gnet.insert_vug_superpore_2d(net, radii_xy=(1.0, 1.0), shape_factor=0.0)

    net_fallback, meta_fallback = gnet.insert_vug_superpore_2d(
        net,
        radii_xy=(0.05, 0.05),
        center_xy=(1.0e6, 1.0e6),
    )
    assert net_fallback.pore_labels["vug"].sum() == 1
    assert meta_fallback["removed_pores"] == 1

    with pytest.raises(RuntimeError, match="zero interface neighbors"):
        gnet.insert_vug_superpore_2d(net, radii_xy=(1.0e6, 1.0e6))

    net_missing = make_cartesian_mesh_network((10, 10), spacing=1.0, thickness=0.5)
    net_missing.pore.pop("radius_inscribed")
    with pytest.raises(KeyError, match="radius_inscribed"):
        gnet.insert_vug_superpore_2d(net_missing, radii_xy=(2.0, 1.5))

    with pytest.raises(ValueError, match="depth must be positive"):
        gnet.insert_vug_superpore_2d(net, radii_xy=(2.0, 1.5), depth=0.0)

    def fake_induced_subnetwork(base_net, _mask):
        return base_net.copy(), np.array([], dtype=int), np.array([], dtype=bool)

    monkeypatch.setattr(gnet, "induced_subnetwork", fake_induced_subnetwork)
    center = (
        float(net.pore_coords[0, 0]),
        float(net.pore_coords[0, 1]),
    )
    with pytest.raises(RuntimeError, match="No interface pores remained"):
        gnet.insert_vug_superpore_2d(net, radii_xy=(0.05, 0.05), center_xy=center)
