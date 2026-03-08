# Getting Started

This page covers installation, environment setup, and a minimal end-to-end workflow.

---

## Installation

### Recommended: Pixi

The repository is configured for [Pixi](https://pixi.sh) and exposes three environments:

- **`default`**: development + plotting + PyVista
- **`test`**: everything in `default` plus OpenPNM and test-only dependencies
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

### Editable pip install

If you prefer a plain Python environment (Python ≥ 3.11):

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

# All extras
python -m pip install -e ".[dev,viz,test]"
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

---

## Building the Docs Locally

To build and preview this documentation locally using the dedicated Pixi environment:

```bash
# Build the docs
pixi run -e docs docs-build

# Serve locally with live reload
pixi run -e docs docs-serve
```

Then open <http://127.0.0.1:8000> in your browser.

Alternatively, install the docs extra via pip and use `mkdocs` directly:

```bash
pip install -e ".[docs]"
mkdocs serve
```
