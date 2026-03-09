# External `pnextract` / `pnflow` benchmark data

This directory stores the committed reference dataset used by
`notebooks/15_mwe_external_pnflow_benchmark.ipynb`.

Contents:

- `manifest.csv`: case metadata, voxel size, and relative file paths
- `<case>/void_volume.npy`: exact binary benchmark input (`True` = void)
- `<case>/*_pnflow.prt`, `<case>/*_upscaled.tsv`: saved `pnflow` reports
- `<case>/*_node*.dat`, `<case>/*_link*.dat`: extracted network files written by `pnextract`
- `<case>/input_pnflow.dat`: input deck used for the saved `pnflow` run

These files are versioned so the notebook can compare the current `voids`
workflow against a fixed external reference without requiring `pnextract` or
`pnflow` to be installed.

Scientific note:

- the committed `.npy` volumes are the canonical benchmark inputs
- the saved external outputs correspond to those exact volumes, not to volumes
  regenerated later from the same nominal case parameters
- if the synthetic generator changes in the future, the notebook still compares
  against the same fixed benchmark inputs because it loads these saved volumes
