# Contributing

`voids` is a scientific Python project for pore-network modeling. Contributions
should improve the codebase without weakening reproducibility, numerical clarity, or
scientific traceability.

This guide is intentionally general. It describes how to contribute effectively
without repeating project details that are better treated as the source of truth in
`pixi.toml`, `pyproject.toml`, the docs, tests, and CI configuration.

## Principles

Contributions to `voids` should follow a few simple rules:

- prefer explicit physical and numerical assumptions over hidden convenience logic
- keep changes interpretable, testable, and reviewable
- treat tests, examples, and documentation as part of the scientific result
- avoid mixing unrelated refactors with behavioral or scientific changes unless there
  is a clear reason
- document known limits rather than implying broader validity than the code supports

For scientific or numerical changes, a green test suite is necessary but not always
sufficient. If behavior changes materially, explain why the new behavior is correct
and what evidence supports it.

## Before You Start

Before implementing a change:

1. Make sure the problem or proposal is described in a GitHub issue, discussion, or
   an equivalent reviewable context.
2. Keep one branch focused on one topic when practical.
3. Read the relevant code, tests, and documentation before changing behavior.

If you are proposing new physics, new constitutive assumptions, or a workflow-level
change, include the scientific motivation early so reviewers can judge the scope
correctly.

## Development Environment

The recommended contributor workflow uses [Pixi](https://pixi.sh), because the
repository tasks, optional features, notebooks, and docs are all wired through it.

Typical setup:

```bash
pixi install
pixi run python -c "import voids; print(voids.__version__)"
```

A plain editable `pip` install is possible, but Pixi is the most representative path
for local work and CI parity.

The authoritative dependency and task definitions live in:

- `pixi.toml`
- `pyproject.toml`

If those files and a prose document disagree, treat the configuration files as the
current truth and update the prose.

## Repository Layout

The main directories contributors usually touch are:

- `src/voids/` for package code
- `tests/` for automated validation and regression coverage
- `docs/` for user and developer documentation
- `notebooks/` for paired notebook workflows
- `examples/` for example assets and workflow data
- `scripts/` for project-maintenance utilities

You do not need to memorize the whole tree. Read the local module layout before
changing code, and keep changes close to the subsystem they affect.

## Typical Workflow

A normal contribution usually looks like this:

1. understand the current behavior and identify the exact change
2. implement the smallest coherent code change that solves the problem
3. add or update tests
4. update documentation, examples, or notebooks if public behavior changed
5. run the relevant local checks
6. open a focused pull request with enough context for review

For behavior changes, do not leave the rationale implicit. Reviewers should be able to
see what changed, why it changed, and what evidence supports it.

## Code Expectations

When writing or modifying code:

- preserve readability and scientific intent
- keep interfaces typed and specific where practical
- avoid unnecessary complexity or abstraction
- prefer deterministic behavior in tests and examples
- keep comments and docstrings informative rather than verbose

If a function encodes a modeling assumption, say so in the code or docstring. If a
result is only valid under certain geometric, physical, or numerical conditions, state
that explicitly.

## Testing Expectations

Behavioral changes should usually come with one or more of:

- a unit test for the changed function or class
- a regression test for a previously failing case
- a manufactured or analytically interpretable example
- a cross-check against an external or reference workflow when relevant

Changes that affect scientific results, solver behavior, geometry interpretation,
import/export, or serialization should not be merged without tests unless there is a
clear and documented reason.

Be careful with a common bad assumption:

- passing in one synthetic case does not imply validity across the range of network
  topologies, geometries, or parameter regimes used by the project

If a change has known limits, state them in the test, docstring, issue, or PR.

## Running Checks

The canonical local commands are defined in `pixi.toml`. Common checks include:

- linting
- formatting checks
- type checking
- tests
- coverage
- docs build
- pre-commit hooks

Use the project tasks rather than inventing ad hoc commands when possible. For focused
work, it is also fine to run a specific test file or a narrower subset first.

Before opening a PR, run the checks that are relevant to your change. For example:

- code changes should usually be linted, type-checked, and tested
- docs changes should at least build cleanly
- notebook changes should be synchronized and reviewed in their paired form

## Documentation

Update documentation when public behavior or scientific interpretation changes. That
may include:

- docstrings
- `README.md`
- pages under `docs/`
- examples
- notebooks
- issue or PR text clarifying assumptions and limitations

Documentation in a scientific codebase is not secondary. If users can easily misuse a
feature or misinterpret a result, the docs are incomplete.

## Notebooks

Notebooks in this repository are part of the reproducible workflow, not just
presentation artifacts.

Contributors working on notebooks should keep in mind:

- notebooks under `notebooks/` are maintained as paired `.ipynb` and `.py` files
- the paired `.py` file is the cleaner review surface and should stay synchronized
- notebook outputs should be meaningful and intentional, not accidental state
- path-dependent code should use the project path helpers or the configured
  environment variables instead of hard-coded local paths

If you edit one side of a notebook pair, make sure the pair is synchronized before
opening a PR.

## Pull Requests

A good pull request should make the reviewer’s job easy. It should state:

- what changed
- why the change is needed
- what evidence supports it
- what tests were added or updated
- what assumptions or limitations remain
- whether docs, notebooks, or examples were updated

Keep PRs focused. Small, reviewable changes are preferred over broad mixed PRs that
combine scientific changes, formatting churn, and unrelated cleanup.

## Issues And Discussions

When opening an issue or proposing a change, try to make three things clear:

- what behavior is wrong, missing, or unclear
- why it matters scientifically, numerically, or ergonomically
- what example, evidence, or use case demonstrates the problem

That context is often the difference between a quick, correct review and a long,
ambiguous one.

## Versioning And Project Metadata

If a change requires updating project metadata or the published version, use the
repository utilities and existing project workflow rather than editing scattered files
manually.

The same principle applies more broadly:

- treat repository configuration files as authoritative
- update prose documents when they drift from those files
- avoid duplicating volatile project state in multiple places when a pointer to the
  source of truth is enough

## If You Are Unsure

If you are unsure whether a change is too large, too speculative, or insufficiently
validated, say so explicitly in the issue or PR. In a research codebase, explicit
uncertainty is easier to review than hidden assumptions.
