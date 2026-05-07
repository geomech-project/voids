# AGENTS.md (notebooks/)

Additional instructions for `notebooks/` and its subdirectories.

## Scope

These rules apply to all files under `notebooks/` and override less specific
guidance when there is a conflict.

## Paired Notebook Discipline

- Each main notebook is paired with a `py:percent` script of the same stem.
- Keep the `.ipynb` and `.py` versions synchronized.
- Preferred sync paths:
  - `python scripts/sync_notebooks.py`
  - `pixi run precommit-sync`
- Avoid making a substantive change in only one of the pair unless the user
  explicitly requests that.

## Reproducibility

- Preserve user-editable configuration cells where possible.
- Put deterministic generation logic in code, not in hidden manual notebook
  state.
- Do not hard-code machine-specific paths, personal directories, or local
  environment assumptions in notebooks.

## Execution Policy

- Prefer running notebook source as a script for smoke checks:
  - `pixi run python notebooks/<name>.py`
- Use `pixi run notebooks-smoke` for a quick inventory check.
- If a notebook is expensive, prefer:
  1. import/syntax or source smoke validation,
  2. a reduced run if the notebook supports it,
  3. a full run only when the user needs the full result.

## Non-Interactive Plotting

- Avoid GUI-blocking plotting during automated runs.
- Prefer headless execution paths, for example by using a non-interactive
  Matplotlib backend when needed.
- If a notebook writes figures to disk, verify those artifacts instead of
  relying on an on-screen window.

## Notebook Outputs

- Treat committed notebook outputs as research artifacts, not as a substitute
  for reasoning about the underlying code.
- If a notebook contains stored outputs that no longer match the source, say so
  explicitly.
- Do not overwrite useful stored outputs in heavyweight notebooks unless the
  task actually requires regeneration.

## External References

- If a notebook compares `voids` against an external reference workflow, keep
  the explanation compatible with public distribution:
  - do not imply that external non-distributed software is bundled,
  - do not depend on ignored local directories,
  - do not copy code from external implementations into tracked notebook
    sources.
