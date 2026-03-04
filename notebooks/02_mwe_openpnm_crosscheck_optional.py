# %% [markdown]
# # MWE 02 — Optional OpenPNM solver cross-check
#
# This compares `voids` and OpenPNM `StokesFlow` using the *same* throat hydraulic conductance values.
# Run in `pixi run -e test python -m jupyter lab`.
#

# %%
from voids.examples import make_linear_chain_network
from voids.physics.singlephase import FluidSinglePhase, PressureBC
from voids.benchmarks.crosscheck import (
    crosscheck_singlephase_roundtrip_openpnm_dict,
    crosscheck_singlephase_with_openpnm,
)

# %%
net = make_linear_chain_network()
fluid = FluidSinglePhase(viscosity=1.0)
bc = PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0)

# %%
print(crosscheck_singlephase_roundtrip_openpnm_dict(net, fluid, bc, axis="x"))

# %%
try:
    s = crosscheck_singlephase_with_openpnm(net, fluid, bc, axis="x")
    print(s)
    print(s.details)
except ImportError as exc:
    print(exc)
