from __future__ import annotations

import json

from voids.examples import make_linear_chain_network
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve


def main() -> None:
    """Run the canonical single-phase demonstration workflow and print JSON.

    Notes
    -----
    The workflow builds the default linear-chain example, solves steady
    single-phase flow along the x-direction, and prints a compact JSON summary
    containing total flow rate, permeability, residual norm, mass-balance error,
    and pore pressures.
    """

    net = make_linear_chain_network()
    result = solve(
        net,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )
    summary = {
        "Q": result.total_flow_rate,
        "Kx": result.permeability["x"] if result.permeability else None,
        "residual_norm": result.residual_norm,
        "mass_balance_error": result.mass_balance_error,
        "p": result.pore_pressure.tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
