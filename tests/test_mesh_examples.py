from __future__ import annotations

import numpy as np

from voids.examples.mesh import make_cartesian_mesh_network
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve


def test_make_cartesian_mesh_network_2d_topology() -> None:
    """Test topology, labels, and sample geometry of a 2D Cartesian mesh."""

    net = make_cartesian_mesh_network((4, 3), spacing=2.0, thickness=1.5)

    assert net.Np == 12
    assert net.Nt == (4 - 1) * 3 + 4 * (3 - 1)
    assert net.pore_coords.shape == (12, 3)
    assert np.allclose(net.pore_coords[:, 2], 0.75)
    assert net.pore_labels["inlet_xmin"].sum() == 3
    assert net.pore_labels["outlet_xmax"].sum() == 3
    assert net.pore_labels["inlet_ymin"].sum() == 4
    assert net.pore_labels["outlet_ymax"].sum() == 4
    assert net.pore_labels["boundary"].sum() == 10
    assert np.all(net.throat["core_length"] > 0.0)
    assert np.isclose(net.sample.length_for_axis("x"), 8.0)
    assert np.isclose(net.sample.area_for_axis("x"), 9.0)


def test_make_cartesian_mesh_network_3d_topology() -> None:
    """Test topology, labels, and sample geometry of a 3D Cartesian mesh."""

    net = make_cartesian_mesh_network((3, 4, 2), spacing=1.5)

    assert net.Np == 24
    assert net.Nt == (3 - 1) * 4 * 2 + 3 * (4 - 1) * 2 + 3 * 4 * (2 - 1)
    assert net.pore_labels["inlet_zmin"].sum() == 12
    assert net.pore_labels["outlet_zmax"].sum() == 12
    assert net.extra["mesh_shape"] == (3, 4, 2)
    assert np.isclose(net.sample.length_for_axis("z"), 3.0)
    assert np.isclose(net.sample.area_for_axis("z"), 27.0)


def test_make_cartesian_mesh_network_supports_singlephase_solve() -> None:
    """Test that the Cartesian mesh example produces the expected pressure field."""

    net = make_cartesian_mesh_network((5, 4), spacing=1.0)
    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )

    x = net.pore_coords[:, 0]
    for xpos in np.unique(x):
        mask = np.isclose(x, xpos)
        assert np.allclose(result.pore_pressure[mask], result.pore_pressure[mask][0])
    expected_column_pressures = 1.0 - np.arange(5, dtype=float) / 4.0
    for idx, xpos in enumerate(sorted(np.unique(x))):
        mask = np.isclose(x, xpos)
        assert np.allclose(result.pore_pressure[mask], expected_column_pressures[idx])

    assert result.total_flow_rate > 0.0
    assert result.permeability is not None and result.permeability["x"] > 0.0
    assert result.mass_balance_error < 1e-12
