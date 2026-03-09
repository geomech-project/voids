# Getting Started

This page covers installation, environment setup, and the smallest end-to-end
workflow that exercises the current single-phase solver.

---

## Installation

### Install from PyPI

If you want the published package from PyPI:

```bash
pip install voids
```

Package page:
<https://pypi.org/project/voids/>

### Recommended: Pixi

The repository is configured for [Pixi](https://pixi.sh) and exposes four primary
environments:

- **`default`**: core runtime + development tools + plotting + PyVista + thermodynamic viscosity backends
- **`test`**: `default` plus test-only dependencies used in the full verification suite
- **`lbm`**: `test` plus the optional XLB direct-image benchmark stack
- **`docs`**: MkDocs, mkdocs-material, and mkdocstrings for building this documentation

```bash
# Install all environments from the lock file
pixi install

# Verify the installation
pixi run -e default python -c "import voids; print(voids.__version__)"
```

Pixi activation sets the following environment variables used by notebooks:

| Variable | Description |
|---|---|
| `VOIDS_PROJECT_ROOT` | Root of the repository |
| `VOIDS_NOTEBOOKS_PATH` | `notebooks/` directory |
| `VOIDS_EXAMPLES_PATH` | `examples/` directory |
| `VOIDS_DATA_PATH` | `examples/data/` directory |

After installation, the fastest sanity check is:

```bash
pixi run examples-singlephase
```

That command exercises the packaged demo workflow and prints a compact JSON summary.

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

---

## Development Commands

Useful Pixi tasks for development:

| Command | Description |
|---|---|
| `pixi run test` | Run the test suite |
| `pixi run test-cov` | Run tests with coverage report |
| `pixi run lint` | Run Ruff linter |
| `pixi run typecheck` | Run MyPy type checker |
| `pixi run precommit` | Run all pre-commit hooks |
| `pixi run notebooks-smoke` | List all paired notebooks |
| `pixi run examples-singlephase` | Run the single-phase workflow entry point |
| `pixi run docs-build` | Build the documentation in the docs environment |
| `pixi run docs-serve` | Serve the documentation locally with live reload |

---

## Building the Docs Locally

To build and preview this documentation locally:

```bash
# Build the docs
pixi run docs-build

# Serve locally with live reload
pixi run docs-serve
```

If you prefer, `pixi run -e docs docs-build` and `pixi run -e docs docs-serve`
also work, but the explicit `-e docs` is redundant because the tasks already enter
the docs environment.

With the current MkDocs configuration, the local preview is served at
<http://127.0.0.1:8000/voids/>.

Alternatively, install the docs extra via pip and use `mkdocs` directly:

```bash
pip install -e ".[docs]"
mkdocs serve
```
