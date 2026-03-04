# Contributing

`voids` is a scientific Python project for pore-network modeling. Contributions
should prioritize reproducibility, explicit assumptions, and regression safety
over convenience. New features are useful only if they are testable,
documented, and consistent with the current project scope.

This guide describes the expected development environment and the workflows used
in this repository.

## Development Principles

When contributing code, keep these project constraints in mind:

- prefer physically and numerically explicit implementations over implicit or
  opaque shortcuts
- state modeling assumptions clearly in docstrings, tests, notebooks, or PR text
- treat synthetic or manufactured examples as validation tools, not evidence of
  universal physical correctness
- avoid broad refactors in the same PR as scientific or behavioral changes unless
  they are required to make the result interpretable
- add or update regression tests whenever behavior changes

## Before You Start

1. Open or identify a GitHub issue describing the problem, bug, or proposed change.
2. Create a branch using the repository naming convention:

```text
fb-voids-<issue-number>-<short-branch-name>
```

Example:

```text
fb-voids-123-fix-singlephase-boundary-flux
```

Branch naming guidelines:

- use the GitHub issue number in `<issue-number>`
- keep `<short-branch-name>` lowercase and hyphen-separated
- make the suffix describe the behavior or subsystem being changed
- keep one branch focused on one issue when practical

Important limitation:

- GitHub issue branch creation allows manual branch naming, but this repository
  does not automatically inject the issue number into the branch name; enter the
  format explicitly when creating the branch

## Repository Layout

The most important directories are:

- `src/voids/`: package source code
- `tests/`: automated regression and validation tests
- `notebooks/`: paired Jupyter notebooks and `py:percent` scripts
- `examples/`: example assets and generated data used by workflows
- `scripts/`: repository maintenance scripts such as version bumping
- `.github/workflows/`: CI definitions used on pull requests and pushes

High-level source organization under `src/voids/`:

- `core/`, `graph/`, `geom/`: core data structures and geometry/network utilities
- `physics/`: physical models and solvers
- `io/`: import/export and interoperability helpers
- `visualization/`: optional plotting and rendering integrations
- `examples/`, `workflows/`: reproducible example entry points and workflows

## Supported Development Environments

The repository supports Python `3.11` and `3.12`. The recommended workflow uses
Pixi because CI, notebooks, optional dependencies, and path setup are all defined
there. A plain `pip` editable install is available, but it is less representative
of the full contributor workflow.

### Recommended: Pixi

Install the project environments:

```bash
pixi install
```

This repository defines two main Pixi environments:

- `default`: development tools, plotting support, and PyVista-related dependencies
- `test`: everything in `default`, plus OpenPNM and other test-only scientific dependencies

The package is installed editable in Pixi through:

```text
voids = { path = ".", editable = true }
```

Pixi activation also provides these environment variables:

- `VOIDS_PROJECT_ROOT`
- `VOIDS_NOTEBOOKS_PATH`
- `VOIDS_EXAMPLES_PATH`
- `VOIDS_DATA_PATH`

Those variables matter for notebook and path-resolution workflows, so Pixi is the
most reliable way to reproduce development and CI behavior.

### Fallback: Editable pip install

If you need a plain Python environment:

```bash
python -m pip install -e .
```

For a fuller local setup:

```bash
python -m pip install -e ".[dev,viz,test]"
```

Known limitation of the fallback path:

- notebook, optional-visualization, and interoperability workflows are exercised
  primarily through Pixi, so `pip` setups should be treated as secondary

## Initial Local Setup

After cloning the repository, the usual setup is:

```bash
pixi install
pixi run -e default python -c "import voids; print(voids.__version__)"
```

If you plan to work with notebooks, you can register project kernels:

```bash
pixi run register-kernels
```

This creates:

- `voids (default)`
- `voids (test)`

To remove them later:

```bash
pixi run unregister-kernels
```

## Daily Development Workflow

A typical contribution flow is:

1. Sync your branch with the latest `main`.
2. Implement a focused change tied to one issue.
3. Add or update tests for any behavioral change.
4. Run the relevant local checks.
5. Update documentation, notebooks, or examples if the public behavior changed.
6. Open a pull request using the repository template.

For scientific or numerical changes, do not rely only on a green test suite if
the behavior shift is material. Also explain:

- what physical, numerical, or data-model assumption changed
- why the change is correct or preferable
- what evidence supports the change, for example regression tests, manufactured
  examples, or cross-checks against a known reference workflow

## Code Quality And Validation Commands

The main local commands are defined in `pixi.toml`.

### Core checks

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test
pixi run test-cov
pixi run precommit
```

What they do:

- `pixi run lint`: runs Ruff on `src` and `tests`
- `pixi run format-check`: checks formatting with Ruff formatter
- `pixi run typecheck`: runs MyPy on `src`
- `pixi run test`: runs the full pytest suite quietly
- `pixi run test-cov`: runs pytest with coverage reporting
- `pixi run precommit`: runs all configured pre-commit hooks on the repository

### Targeted checks

Useful narrower commands include:

```bash
pixi run spec-check
pixi run crosscheck-roundtrip
pixi run examples-singlephase
pixi run notebooks-smoke
```

Use them when relevant:

- `spec-check`: validates schema-oriented network behavior
- `crosscheck-roundtrip`: exercises interoperability roundtrips
- `examples-singlephase`: runs the single-phase workflow entry point
- `notebooks-smoke`: verifies notebook discovery in the repository

Pytest can also be run directly for focused work:

```bash
pixi run -e test pytest -q tests/test_singlephase_toy.py
```

or with a keyword filter:

```bash
pixi run -e test pytest -q -k singlephase
```

## What CI Enforces

GitHub Actions currently checks several things on pull requests:

- tests on Linux, macOS, and Windows
- coverage reporting on Linux
- diff coverage on Linux pull requests with a required threshold of `99%`
- pre-commit checks on the PR diff

Practical implication:

- a change that is technically correct but insufficiently covered by tests may
  still fail CI because of the diff-coverage threshold

If your change affects executable lines, plan to add tests early rather than as a
cleanup step at the end.

## Pre-commit And Notebook Hygiene

This repository uses `pre-commit` for both source files and notebooks.

Configured hooks currently include:

- trailing whitespace and end-of-file normalization
- `ruff` and `ruff-format` on source code
- `nb-clean` for notebook cleanup
- `nbqa-black` and `nbqa-ruff` for notebook code cells
- `jupytext --sync` for paired notebook/script synchronization

The notebook policy is important:

- notebooks under `notebooks/` are maintained as paired `.ipynb` and `.py` files
- the `.py` files are part of the authoritative review surface because they diff cleanly
- if you edit one side of a notebook pair, keep the pair synchronized

The configured `jupytext` hook helps enforce this, but contributors should still
verify that both files reflect the intended final state before opening a PR.

## Working With Notebooks

The notebooks in this repository are not just presentation artifacts. They are
part of the reproducible workflow and often demonstrate validation scenarios.

Current notebook set includes workflows such as:

- minimal single-phase porosity and permeability
- optional OpenPNM cross-checks
- PyVista visualization
- manufactured PoreSpy extraction
- synthetic Cartesian mesh-style examples

Environment expectations:

- notebooks that use OpenPNM or PoreSpy should be run in the `test` environment
- lighter examples can run in the `default` environment

When changing notebook-backed workflows:

- keep outputs meaningful but not noisy
- avoid committing accidental state unrelated to the change
- ensure path-dependent code uses the project path helpers or Pixi-provided env vars
- prefer deterministic examples when possible

## Testing Expectations For Scientific Changes

Not every contribution needs the same validation depth, but behavioral changes
should usually include one or more of the following:

- a direct unit test for the changed function or class
- a regression test covering a previous failure mode
- a manufactured example with analytically interpretable behavior
- an interoperability or roundtrip check against an external tool when relevant

Examples of changes that should almost certainly come with tests:

- altered boundary-condition handling
- changes to permeability or flow assembly
- import normalization changes
- serialization format changes
- visualization code that affects saved artifacts or workflow branching

Potentially incorrect assumption to avoid:

- passing tests in a narrow synthetic case does not imply scientific validity over
  the full range of network topologies or geometry regimes used by contributors

If a change has known validity limits, state them explicitly in the code or PR.

## Documentation Expectations

Update documentation when public behavior changes. Depending on the scope, that may
mean one or more of:

- docstrings
- `README.md`
- notebooks
- examples
- issue or PR descriptions clarifying assumptions and limitations

For user-visible behavior changes, do not rely on code alone to communicate intent.

## Issues

Use the issue form in GitHub and provide:

- `Expected Impact`: choose exactly one of `Minor`, `Regular`, or `Major`
- `Overview`: a brief description of the issue, motivation, or observed problem

The issue should be specific enough that another contributor can understand:

- what behavior is wrong, missing, or unclear
- why the issue matters scientifically, numerically, or ergonomically
- what evidence or example demonstrates the problem

## Pull Requests

Use the pull request template and fill in:

- `Overview`: the purpose of the pull request
- `Changes`: what code behavior changed

A good PR description should also summarize:

- why the implementation is scientifically or technically justified
- what tests were added or updated
- what assumptions remain unresolved
- whether notebooks, examples, or documentation were updated

Keep pull requests focused. Small, reviewable changes are preferred over large mixed
PRs that combine scientific changes, formatting churn, and unrelated cleanup.

## Version Updates

Version changes should use the repository script rather than editing files manually:

```bash
pixi run bump-version 0.1.4
```

This updates the authoritative version declarations consistently across project
metadata files.

## If You Are Unsure

If a contribution involves new physics, new data-model assumptions, or a broader
workflow change, prefer over-explaining the rationale in the issue or PR rather
than leaving reviewers to infer it. In a scientific codebase, ambiguity about
assumptions is usually more costly than a longer explanation.
