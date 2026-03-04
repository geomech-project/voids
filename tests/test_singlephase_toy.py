from __future__ import annotations

import numpy as np

from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve


def test_singlephase_line_network_solution(line_network):
    """Test the analytic single-phase solution on the line network."""

    r = solve(
        line_network,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )
    assert np.allclose(r.pore_pressure, [1.0, 0.5, 0.0])
    assert np.allclose(r.throat_flux, [0.5, 0.5])
    assert np.isclose(r.total_flow_rate, 0.5)
    assert np.isclose(r.permeability["x"], 0.5)
    assert r.residual_norm < 1e-12
    assert r.mass_balance_error < 1e-12
