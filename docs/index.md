<p align="center">
  <img src="assets/logo.png" alt="voids logo" style="max-width: 600px; width: 100%;">
</p>

# voids

[![Tests](https://github.com/geomech-project/voids/actions/workflows/tests.yml/badge.svg)](https://github.com/geomech-project/voids/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/geomech-project/voids/branch/main/graph/badge.svg)](https://codecov.io/gh/geomech-project/voids)

**`voids`** is a scientific Python package for pore network modeling (PNM) aimed at
research workflows where reproducibility, explicit assumptions, and validation matter.

The current project emphasis is a clean canonical network model, interoperability with
PoreSpy/OpenPNM-style data, and a validated single-phase reference workflow before
expanding to more complex physics.

---

## Why voids?

Pore network modeling research often struggles with reproducibility because the mapping
from segmented images to simulation results involves many implicit choices:

- Which pore and throat size definitions are used?
- How is the bulk volume defined relative to the image?
- What constitutive model is used for hydraulic conductance?

`voids` addresses these concerns by:

- enforcing an explicit, versioned **canonical network schema**
- requiring **provenance metadata** at construction time
- keeping **physics modules** narrowly scoped with documented assumptions
- providing **regression fixtures** to lock numerical results over time

---

## Goals

- Provide a rigorous internal representation of pore-throat networks
- Preserve sample geometry and provenance information needed for reproducible studies
- Support import and normalization of extracted networks from external tools
- Expose well-scoped physics modules with diagnostics and regression tests
- Build confidence on single-phase transport first, then expand toward richer models

---

## Current Scope (v0.1.x)

| Feature | Status |
|---|---|
| Canonical `Network`, `SampleGeometry`, `Provenance` data structures | ✅ |
| Import of PoreSpy/OpenPNM-style dictionaries | ✅ |
| Geometry normalization helpers | ✅ |
| Absolute and effective porosity | ✅ |
| Connectivity metrics | ✅ |
| Single-phase incompressible flow | ✅ |
| Directional permeability estimation | ✅ |
| HDF5 serialization | ✅ |
| Plotly and PyVista visualization | ✅ |
| OpenPNM cross-checks | ✅ |
| Multiphase flow | ❌ Not yet |
| Image segmentation | ❌ Delegated to PoreSpy |

---

## Quick Start

```python
from voids.examples import make_linear_chain_network
from voids.physics.petrophysics import absolute_porosity
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve

net = make_linear_chain_network()

result = solve(
    net,
    fluid=FluidSinglePhase(viscosity=1.0),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
    axis="x",
)

print("phi_abs =", absolute_porosity(net))
print("Q =", result.total_flow_rate)
print("Kx =", result.permeability["x"])
print("mass_balance_error =", result.mass_balance_error)
```

See [Getting Started](getting_started.md) for installation and a full walkthrough.

---

## Status

`voids` is pre-alpha. The codebase is already useful for controlled PNM experiments,
solver validation, and interoperability studies, but it should not be described as a
complete pore-network simulation platform yet.
