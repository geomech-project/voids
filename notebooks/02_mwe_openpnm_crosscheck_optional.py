# %% [markdown]
# # MWE 02 — Optional OpenPNM solver cross-check
#
# This compares `voids` and OpenPNM `StokesFlow` using the *same* throat hydraulic conductance values.
# The notebook now includes both:
# - a constant-viscosity reference case
# - a pressure-coupled water-viscosity case using the new `thermo` backend with cached interpolation
#
# Because thermodynamic backends interpret pressure as **absolute pressure in Pa**, the coupled case
# uses `pin=2e5 Pa`, `pout=1e5 Pa`. The older gauge-only choice `pin=1`, `pout=0` remains fine for
# constant-viscosity checks but is not physically meaningful for `mu(P, T)`.
# Run in `pixi run -e test python -m jupyter lab`.
#

# %%
import numpy as np

from voids.examples import make_linear_chain_network
from voids.physics.singlephase import FluidSinglePhase, PressureBC
from voids.physics.thermo import TabulatedWaterViscosityModel
from voids.benchmarks.crosscheck import (
    crosscheck_singlephase_roundtrip_openpnm_dict,
    crosscheck_singlephase_with_openpnm,
)

# %%
net_constant = make_linear_chain_network()
fluid_constant = FluidSinglePhase(viscosity=1.0)
bc_constant = PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0)

net_thermo = make_linear_chain_network()
net_thermo.throat.pop("hydraulic_conductance")
net_thermo.throat["area"] = np.sqrt(8.0 * np.pi) * np.ones(net_thermo.Nt)
fluid_thermo = FluidSinglePhase(
    viscosity=1.0e-3,
    viscosity_model=TabulatedWaterViscosityModel.from_backend(
        "thermo",
        temperature=298.15,
        pressure_points=128,
    ),
)
bc_thermo = PressureBC("inlet_xmin", "outlet_xmax", pin=2.0e5, pout=1.0e5)

# %%
print("Constant-viscosity dictionary round-trip:")
print(
    crosscheck_singlephase_roundtrip_openpnm_dict(
        net_constant, fluid_constant, bc_constant, axis="x"
    )
)

# %%
try:
    print("\nConstant-viscosity OpenPNM cross-check:")
    s_constant = crosscheck_singlephase_with_openpnm(
        net_constant,
        fluid_constant,
        bc_constant,
        axis="x",
    )
    print(s_constant)
    print(s_constant.details)

    print("\nPressure-coupled thermo/OpenPNM cross-check:")
    s_thermo = crosscheck_singlephase_with_openpnm(
        net_thermo,
        fluid_thermo,
        bc_thermo,
        axis="x",
    )
    print(s_thermo)
    print(s_thermo.details)
except ImportError as exc:
    print(exc)
