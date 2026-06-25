# Getting Started

This page covers package installation and the smallest end-to-end workflow that
exercises the pore-network single-phase solver. For the map-based FVM/FEM and
direct-image LBM backends, see
[Map-Based Single-Phase Solvers](map_based_singlephase_solvers.md) after the
basic installation is working.

---

## Installation

### Install from PyPI

If you want the published package from PyPI:

```bash
pip install voids
```

Package page:
<https://pypi.org/project/voids/>

### Editable pip install

If you prefer a plain Python environment from a local repository checkout
(Python ≥ 3.11):

```bash
python -m pip install -e .
```

Optional extras:

```bash
# Development tools (pytest, ruff, mypy, jupyterlab …)
python -m pip install -e ".[dev]"

# PyVista visualization
python -m pip install -e ".[viz]"

# OpenPNM cross-check tests
python -m pip install -e ".[test]"

# Optional XLB benchmark stack
python -m pip install -e ".[lbm]"

# All extras
python -m pip install -e ".[dev,viz,test,lbm,docs]"
```

There is no `fem` PyPI extra at the moment. Plain pip users must install a
compatible FEniCSx/DOLFINx stack separately before using `voids.fem`.

Repository development, notebook variables, and documentation build commands are
covered in [Development](development.md).

---

## Minimal Example

The simplest way to exercise `voids` is with a synthetic linear-chain network:

```python
from voids.examples import make_linear_chain_network
from voids.physics.petrophysics import absolute_porosity
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve

# Build a small synthetic network
net = make_linear_chain_network()

# Define fluid and boundary conditions
fluid = FluidSinglePhase(viscosity=1.0)
bc = PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0)

# Solve single-phase incompressible flow
result = solve(net, fluid=fluid, bc=bc, axis="x")

# Print results
print("phi_abs               =", absolute_porosity(net))
print("total_flow_rate       =", result.total_flow_rate)
print("permeability x        =", result.permeability["x"])
print("mass_balance_error    =", result.mass_balance_error)
```

Assumptions to keep in mind:

- the default demo network is synthetic and intentionally simple
- `viscosity=1.0` is dimensionless unless you also define consistent physical units
- permeability is only meaningful when the attached `SampleGeometry` is physically meaningful

For the canonical data model, label conventions, and unit expectations behind that
example, see [Concepts and Conventions](concepts.md).

For extracted or imported networks, continue with
[Scientific Workflow](workflow.md) rather than copying the synthetic example verbatim.

### Pressure-Dependent Viscosity

For water-property studies, `voids` can also solve with pressure-dependent viscosity:

```python
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve
from voids.physics.thermo import TabulatedWaterViscosityModel

mu_model = TabulatedWaterViscosityModel.from_backend(
    "thermo",
    temperature=298.15,
    pressure_points=128,
)

result = solve(
    net,
    fluid=FluidSinglePhase(viscosity_model=mu_model),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=2.0e5, pout=1.0e5),
    axis="x",
    options=SinglePhaseOptions(
        conductance_model="valvatne_blunt",
        nonlinear_solver="newton",
        solver="gmres",
        solver_parameters={"preconditioner": "pyamg"},
    ),
)
```

Two assumptions change in this mode:

- pressures must be absolute and positive, typically in Pa
- the nonlinear solve is with respect to the tabulated/interpolated constitutive law,
  not the raw backend callable directly
