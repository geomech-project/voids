from __future__ import annotations

import numpy as np
import pytest

from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve


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


def test_singlephase_pardiso_solver_matches_direct(line_network):
    """Test that PARDISO solver produces identical results to the direct solver."""

    bc = PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0)
    fluid = FluidSinglePhase(viscosity=1.0)

    # Solve with default direct solver
    r_direct = solve(
        line_network,
        fluid=fluid,
        bc=bc,
        axis="x",
        options=SinglePhaseOptions(solver="direct"),
    )

    # Try PARDISO solver
    try:
        r_pardiso = solve(
            line_network,
            fluid=fluid,
            bc=bc,
            axis="x",
            options=SinglePhaseOptions(solver="pardiso"),
        )

        # Results should match to machine precision
        assert np.allclose(r_pardiso.pore_pressure, r_direct.pore_pressure, rtol=1e-12, atol=1e-14)
        assert np.allclose(r_pardiso.throat_flux, r_direct.throat_flux, rtol=1e-12, atol=1e-14)
        assert np.isclose(r_pardiso.total_flow_rate, r_direct.total_flow_rate, rtol=1e-12)
        assert np.isclose(r_pardiso.permeability["x"], r_direct.permeability["x"], rtol=1e-12)
        assert r_pardiso.solver_info["method"] == "pardiso"

    except ImportError as exc:
        # Expected on non-Linux platforms
        assert "pypardiso" in str(exc).lower()
        pytest.skip("PARDISO solver not available on this platform")
