# AGENTS.md (docs/)

Additional instructions for `docs/` and its subdirectories.

## Scope

These rules apply to all files under `docs/` and override less specific
guidance when there is a conflict.

## Public Documentation Discipline

- Everything under `docs/` is part of the public-facing explanation of `voids`.
- Do not mention ignored local directories such as `refs/`.
- Do not describe non-distributed external software as if it ships with
  `voids`.
- If a benchmark compares against an external workflow, describe it accurately
  but keep the wording compatible with public distribution and licensing.

## Documentation Style

- Prefer precise, scientific wording over marketing language.
- Avoid overly explicit reference-specific framing in headings and section
  titles. Prefer method, model, or domain language such as "Micro-CT Grayscale
  Calibration In The Literature" instead of titles like "Relation To The
  Petrobras Paper", "Comparison With Paper X", or labels based on local
  reference folders. Cite the paper in the prose and reference list, not in a
  way that makes the section read like a named-company or local-reference note,
  unless the page is explicitly a benchmark or validation page for that source.
- Make assumptions visible:
  - phase convention,
  - geometry convention,
  - pressure convention,
  - boundary-condition convention,
  - scope limits.
- Distinguish clearly between:
  - what `voids` implements directly,
  - what `voids` imports or interoperates with,
  - what remains an upstream preprocessing or external-reference step.

## References And Citations

- Prefer published papers, public datasets, and public package documentation.
- Do not use ignored local files as citations in tracked docs.
- Do not rely on local non-distributed source trees to justify public
  scientific claims.
- If a method in `voids` is only inspired by external work, describe that
  relationship in literature terms rather than code-copy terms.

## Figures And Schematics

- If you create or modify figures, verify them visually.
- Reject figures that have:
  - overlapping text,
  - poor contrast,
  - ambiguous geometry,
  - graph overlays that do not match the depicted pores/throats,
  - unreadable labels on light/dark backgrounds.
- Prefer publication-style readability over decorative complexity.

## Navigation And Build Checks

- After changing docs pages, navigation, or assets, run:
  - `pixi run docs-build`
- For navigation edits, confirm the new grouping improves usability and does not
  over-expose long top-level menus.
- Keep link text stable and descriptive.

## Rendered Notebook Reports

- Pages under `docs/notebook_reports/` are rendered research artifacts tied to
  committed notebook outputs.
- Do not casually hand-edit generated content unless the user explicitly wants a
  targeted docs-side fix.
- If a substantive notebook-report change is needed, prefer updating the source
  notebook or paired script and then regenerating the rendered page when
  feasible.
