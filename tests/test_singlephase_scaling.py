from __future__ import annotations

import numpy as np

from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve


def test_singlephase_scaling_with_pressure_drop(line_network):
    """Test linear flow-rate scaling and permeability invariance with pressure drop."""

    r1 = solve(
        line_network,
        FluidSinglePhase(1.0),
        PressureBC("inlet_xmin", "outlet_xmax", 1.0, 0.0),
        axis="x",
    )
    r2 = solve(
        line_network,
        FluidSinglePhase(1.0),
        PressureBC("inlet_xmin", "outlet_xmax", 2.0, 0.0),
        axis="x",
    )
    assert np.isclose(r2.total_flow_rate / r1.total_flow_rate, 2.0)
    assert np.isclose(r2.permeability["x"], r1.permeability["x"])


def test_singlephase_scaling_with_viscosity(line_network):
    """Test inverse flow-rate scaling and permeability invariance with viscosity."""

    # Since precomputed conductance is used, viscosity won't affect g. Use geometry route instead.
    line_network.throat.pop("hydraulic_conductance")
    line_network.throat["area"] = np.sqrt(8.0 * np.pi) * np.ones(
        2
    )  # Gives g=1 for mu=1 and L=1 via generic model
    r1g = solve(
        line_network,
        FluidSinglePhase(1.0),
        PressureBC("inlet_xmin", "outlet_xmax", 1.0, 0.0),
        axis="x",
    )
    r2g = solve(
        line_network,
        FluidSinglePhase(2.0),
        PressureBC("inlet_xmin", "outlet_xmax", 1.0, 0.0),
        axis="x",
    )
    assert np.isclose(r1g.total_flow_rate / r2g.total_flow_rate, 2.0)
    assert np.isclose(r2g.permeability["x"], r1g.permeability["x"])
