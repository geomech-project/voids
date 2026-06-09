# voids Development

This page is for contributors working from a local repository checkout. For
package installation and the minimal user workflow, start with
[Getting Started](getting_started.md).

`voids` uses [Pixi](https://pixi.sh) to keep development, testing, notebook, FEM,
LBM, and documentation dependencies reproducible across machines. The Pixi
configuration is repository infrastructure: it is the recommended way to develop
`voids`, run validation checks, and rebuild the docs.

---

## Pixi Environments

The repository exposes three Pixi environments:

| Environment | Purpose |
|---|---|
| `default` | Main development environment with notebooks, plotting, PyVista, thermodynamic backends, FEniCSx FEM, and XLB LBM support |
| `test` | Test environment with the dependencies used by the verification suite |
| `docs` | Documentation environment with MkDocs, Material for MkDocs, and mkdocstrings |

Install the default environment from the lock file:

```bash
pixi install
```

Verify that the local checkout imports correctly:

```bash
pixi run -e default python -c "import voids; print(voids.__version__)"
```

For a lightweight scientific smoke test:

```bash
pixi run examples-singlephase
```

That command runs the packaged single-phase pore-network example and prints a
compact JSON summary.

---

## Notebook Environment Variables

Pixi activation sets project path variables used by the notebooks:

| Variable | Description |
|---|---|
| `VOIDS_PROJECT_ROOT` | Root of the repository |
| `VOIDS_NOTEBOOKS_PATH` | `notebooks/` directory |
| `VOIDS_EXAMPLES_PATH` | `examples/` directory |
| `VOIDS_DATA_PATH` | `examples/data/` directory |

Register Jupyter kernels once when you want to run notebooks from JupyterLab:

```bash
pixi run register-kernels
```

The registered kernels are:

- `voids-default`
- `voids-test`

---

## Development Checks

Use the repository tasks instead of ad hoc commands whenever possible:

| Command | Description |
|---|---|
| `pixi run test` | Run the test suite |
| `pixi run test-cov` | Run tests with coverage report |
| `pixi run test-lbm` | Run the XLB/LBM-focused tests |
| `pixi run lint` | Run Ruff lint checks |
| `pixi run format-check` | Check Ruff formatting |
| `pixi run typecheck` | Run MyPy type checks |
| `pixi run spec-check` | Run schema-focused checks |
| `pixi run crosscheck-roundtrip` | Run OpenPNM/`voids` round-trip checks |
| `pixi run precommit` | Run all pre-commit hooks |
| `pixi run notebooks-smoke` | List paired notebooks as a quick notebook inventory check |
| `pixi run notebooks-sync` | Synchronize paired notebooks and `py:percent` scripts |
| `pixi run precommit-sync` | Synchronize notebooks and then run pre-commit |

For code that changes numerics, geometry, transport, extraction, or solver
behavior, prefer the narrowest scientifically meaningful check first, then broaden
to the full test and documentation checks as needed.

---

## Documentation Builds

Build and preview the documentation with:

```bash
pixi run docs-build
pixi run docs-serve
```

The tasks enter the `docs` environment internally. With the current MkDocs
configuration, the local preview is served at:

<http://127.0.0.1:8000/voids/>

If you are only checking a plain pip environment, the docs extra can be used
directly:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

---

## Dependency And Version Maintenance

Pixi is the source of truth for the local development environments. The PyPI
package metadata in `pyproject.toml` is synchronized from the Pixi manifest where
the dependency exists in both packaging surfaces.

Use:

```bash
pixi run sync-deps-check
```

to verify synchronization, and:

```bash
pixi run sync-deps
```

to update synchronized dependency ranges.

Version updates are handled with:

```bash
pixi run bump-version <new-version>
```
