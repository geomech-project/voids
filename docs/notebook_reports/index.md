# Notebook Reports

This section renders selected notebooks directly into the documentation from their
committed `.ipynb` files. The docs build does **not** execute them again; it uses the
outputs already stored in the notebooks.

That design is intentional:

- the rendered page is a frozen research artifact tied to a committed notebook state
- the docs build remains fast and deterministic
- heavy notebooks do not need to be re-executed just to rebuild the site

The current rendered set focuses on the benchmark and comparison notebooks whose
stored outputs are already publication-style:

- [14 — Shape-Factor Conductance Comparison](14_mwe_shape_factor_conductance_comparison.md)
- [15 — External `pnextract` / `pnflow` Benchmark](15_mwe_external_pnflow_benchmark.md)
- [16 — `Kabs` Benchmark: Constant vs Thermodynamic Viscosity](16_mwe_viscosity_model_kabs_benchmark.md)
- [17 — Solver Options Benchmark](17_mwe_solver_options_benchmark.md)

The complete notebook inventory, including paired `.py` scripts, remains documented in
[Examples](../examples.md).
