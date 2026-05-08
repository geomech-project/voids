# AGENTS.md

Instructions for agentic assistants working under `examples/data/`.

## Scope

This file applies to `examples/data/` and all descendants. It extends the root
`AGENTS.md`.

## Data, Benchmarks, And Citations

- Treat files in this tree as scientific data dependencies, benchmark fixtures,
  or generated reference artifacts. Preserve provenance when editing, replacing,
  or adding them.
- If a dataset, benchmark case, image, network file, extracted table, or
  generated reference output is based on a paper, DOI, public dataset, official
  software output, or other citable source, add or update the relevant citation
  in tracked documentation in the same change.
- Prefer updating the nearest documentation first, such as
  `examples/data/README.md`, the corresponding notebook report, or the
  validation/verification page that consumes the data. Add broader references in
  `docs/background.md` when the source supports shipped methods or scientific
  model assumptions.
- Do not present local-only inputs from `refs/`, `tmp/`, `outputs/`, `.pixi/`,
  or `site/` as shipped data sources. If such material guided a comparison,
  document the public method or public reference instead, and state any
  non-public/local dependency as a limitation.
- Keep unit conventions, phase conventions, voxel sizes, sample dimensions, and
  boundary-condition provenance explicit whenever adding or regenerating data
  used by extraction, porosity, permeability, or transport examples.
