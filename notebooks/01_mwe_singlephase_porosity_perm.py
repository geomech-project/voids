# %% [markdown]
# # MWE 01 — Single-phase flow, porosity, and permeability
#
# Minimal `voids` example using a tiny hand-built network. This notebook runs in the `default` environment.
#

# %%
# Minimal visualization fallback (matplotlib) so this notebook remains dependency-light
import matplotlib.pyplot as plt

from voids.examples import make_linear_chain_network
from voids.physics.petrophysics import (
    absolute_porosity,
    effective_porosity,
    connectivity_metrics,
)
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    solve,
)

# %%
net = make_linear_chain_network()
net

# %%
print("phi_abs =", absolute_porosity(net))
print("phi_eff(boundary-connected) =", effective_porosity(net))
print("phi_eff(spanning x) =", effective_porosity(net, axis="x"))
print(connectivity_metrics(net))

# %%
res = solve(
    net,
    fluid=FluidSinglePhase(viscosity=1.0),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
    axis="x",
    options=SinglePhaseOptions(conductance_model="generic_poiseuille", solver="direct"),
)
print("Q =", res.total_flow_rate)
print("Kx =", res.permeability["x"])
print("mass balance error =", res.mass_balance_error)

# %%
p = res.pore_pressure
fig, ax = plt.subplots(figsize=(5, 2.8))
ax.plot(net.pore_coords[:, 0], p, marker="o")
ax.set_xlabel("x")
ax.set_ylabel("pressure")
ax.set_title("Pore pressures")
plt.show()
