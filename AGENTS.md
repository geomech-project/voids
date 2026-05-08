# AGENTS.md

Instructions for Codex and other agentic assistants working in this repository.

## Scope

This file applies to the whole repository tree unless a deeper `AGENTS.md`
overrides part of it.

## Project Context

- `voids` is a scientific Python package for pore network modeling (PNM).
- The current scientific emphasis is:
  - canonical network representation,
  - reproducible geometry and provenance handling,
  - validated single-phase transport,
  - image-to-network workflows that make assumptions explicit.
- The repository is a package-first codebase, not a loose script collection.
  Prefer library-quality changes over notebook-local reinvention.
- Multiphase flow is not implemented yet. Do not describe it as available.

## Runtime Portability

- These instructions are intended to work across different PCs and different LLM
  runtimes.
- Do not assume a specific assistant client, shell profile, editor integration,
  or local absolute path.
- If nested `AGENTS.md` handling is unavailable in a given runtime, follow the
  most specific instructions that are visible and say so when it matters.
- Prefer commands that work from the repository root.

## Scientific Rigor (Required)

- Be rigorous in mathematical, physical, and numerical reasoning.
- Double-check dimensional consistency, sign conventions, and geometry/units
  assumptions before finalizing changes.
- Prefer conservative claims:
  - separate what was executed from what was inferred from code inspection,
  - identify assumptions that remain unverified,
  - say how those assumptions could fail and how to test them.
- For extraction, permeability, porosity, and transport claims, treat provenance,
  sample geometry, and boundary conditions as part of the scientific model.

## Public Surface And Copyright Safety

- Do not copy code from non-distributed software, local ignored directories, or
  unpublished external implementations.
- Do not imply that non-distributed external software ships with `voids`.
- Public docs, comments, and examples should describe external comparisons in a
  generic and accurate way unless the user explicitly wants a benchmark page
  that names a specific public reference.
- Prefer reimplementation from:
  - published papers,
  - public documentation,
  - independently observed behavior,
  - the scientific requirements of `voids`.
- Avoid large verbatim quotations from papers or external documentation.

## Repository Boundaries

- Treat ignored and local-only areas as non-public implementation aids unless
  the user explicitly asks to work with them:
  - `refs/`
  - `tmp/`
  - `outputs/`
  - `.pixi/`
  - `site/`
- Do not make tracked source or docs depend on files from ignored directories.
- Do not present local-only artifacts as part of the repository contract.

## Environment Discovery

- Preferred environment manager: Pixi.
- Secondary path: editable `pip install -e .` with optional extras.
- On a fresh machine, first determine what is available before attempting heavy
  runs.
- Preferred discovery order:
  1. inspect `README.md`, `pixi.toml`, and `pyproject.toml`,
  2. check whether `pixi` is available,
  3. if needed, check whether Python can import `voids` and the relevant
     optional stack.
- Do not commit machine-specific activation commands or absolute environment
  paths.

## Canonical Commands

Use the repository tasks before inventing ad hoc alternatives:

- `pixi run examples-singlephase`
- `pixi run test`
- `pixi run test-cov`
- `pixi run test-lbm`
- `pixi run lint`
- `pixi run format-check`
- `pixi run typecheck`
- `pixi run spec-check`
- `pixi run crosscheck-roundtrip`
- `pixi run notebooks-smoke`
- `pixi run docs-build`
- `pixi run docs-serve`
- `pixi run precommit`
- `pixi run precommit-sync`

If only a lightweight check is needed, prefer the narrowest meaningful command
over the full suite.

## Testing, Coverage, And Typing Requirements

- Any change to shipped `voids` behavior should be accompanied by tests or test
  updates that cover the changed behavior.
- Treat test coverage as a hard quality gate for code changes:
  - aim for at least `99%` coverage for the affected `voids` code path,
  - do not accept untested new logic unless the user explicitly accepts the
    verification gap,
  - use `pixi run test-cov` as the authoritative repository coverage check.
- Treat typing as a hard quality gate for added or modified code:
  - do not leave new or changed code with MyPy issues,
  - use `pixi run typecheck` as the authoritative typing check.
- Prefer also running the style checks that match the repository tasks:
  - `pixi run lint`
  - `pixi run format-check`
  - `pixi run precommit`
- When the change touches optional XLB/LBM functionality, prefer the dedicated
  narrow task before escalating:
  - `pixi run test-lbm`

## Code And Design Expectations

- Keep changes minimal, targeted, and scientifically motivated.
- Preserve backward compatibility in public APIs unless the user explicitly asks
  for a breaking change.
- Prefer meaningful variable names over abbreviated names.
- Add comments only where the science-to-code mapping is not obvious.
- Avoid broad refactors when a narrow fix is sufficient.
- Prefer package functions and canonical data structures over notebook-local
  duplicate implementations.

## Core Scientific Model Expectations

- Preserve and use the canonical `Network`, `SampleGeometry`, and `Provenance`
  structures.
- Keep units explicit.
- Do not silently change the meaning of boundary labels, flow axes, pressure
  conventions, or geometry scaling.
- When editing image-based workflows, preserve the separation between:
  - segmentation/preprocessing,
  - extraction/reduction,
  - import/normalization,
  - transport solve,
  - verification/validation.
- When comparing with an external reference, do not assume agreement implies
  correctness; explain whether the comparison is:
  - same-network parity,
  - extraction-workflow comparison,
  - direct-image vs reduced-network comparison,
  - or experiment vs simulation.

## Numerical Verification Policy

For changes that affect numerics, geometry, extraction logic, constitutive
models, or solver behavior:

1. run the cheapest scientifically meaningful check first,
2. report what was and was not executed,
3. compare key invariants or benchmark outputs when feasible.

Examples of meaningful checks:

- targeted `pytest` for the edited module,
- a focused benchmark wrapper,
- `pixi run examples-singlephase`,
- a notebook-source smoke run,
- a docs build for documentation-linked scientific claims.

If execution is not possible, state the verification gap explicitly.

## Documentation Policy

- Keep docs aligned with the actual shipped `voids` surface.
- Do not describe ignored local references or non-distributed external code as
  if they were part of `voids`.
- When adding or changing code, notebooks, examples, datasets, benchmarks, or
  validation workflows based on a scientific reference, add or update the
  corresponding documentation citation in the same change. Prefer the public
  paper, dataset DOI, official documentation, or other citable primary source;
  do not rely on unpublished local files as the only public attribution.
- If a reference informs implementation details but cannot be shipped or quoted
  directly, document the scientific method or public source used, and keep any
  local-only reference material clearly separated from the public repository
  contract.
- If you change figures or diagrams, visually inspect them for readability,
  overlap, and scientific consistency.
- If you change docs navigation or generated docs pages, rebuild with
  `pixi run docs-build`.

## Notebook And Example Policy

- Notebooks under `notebooks/` are paired with `py:percent` scripts and should
  stay synchronized.
- Prefer deterministic workflows over hidden manual tweaks.
- Keep user-editable configuration separate from deterministic generation logic.
- If a notebook or script generates figures during automated runs, prefer a
  non-interactive path that will not block on a GUI backend.

## Data And Benchmark Policy

- Benchmarks and validation studies should remain reproducible from tracked
  inputs when possible.
- Raw experimental or scan data that are intentionally local-only should not be
  treated as guaranteed-available inputs on another machine.
- If a workflow depends on optional data or optional software, say so plainly in
  code comments, docs, and final explanations.

## Final Explanations

When reporting work, summarize:

- what changed,
- what was verified,
- what remains an assumption or limitation,
- and any scientifically important caveat the user should know.
