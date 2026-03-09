# Examples and Notebooks

The `notebooks/` directory contains paired Jupyter notebooks and `py:percent` scripts
covering the main `voids` workflows, from the smallest synthetic example to image-
based sensitivity studies and OpenPNM benchmarks.

Each notebook is both a tutorial and a regression artifact: if it no longer runs,
some documented scientific workflow has drifted.

---

## Running the Notebooks

With Pixi:

```bash
pixi run register-kernels   # register Jupyter kernels once
jupyter lab                  # open JupyterLab
```

The notebooks rely on environment variables set by Pixi activation
(`VOIDS_PROJECT_ROOT`, `VOIDS_DATA_PATH`, etc.), so they should be launched from
within a Pixi-managed shell.

In practice:

- use `default` for most notebooks, including OpenPNM and thermodynamic-viscosity workflows
- use `test` when you want the full test-suite dependency set around the notebook workflow
- use `lbm` when a notebook depends on the optional XLB direct-image benchmark
- expect the image-based notebooks to be materially heavier than the minimal demos

---

## Choosing A Notebook

| Notebook | Best use |
|---|---|
| `01_mwe_singlephase_porosity_perm` | Learn the minimal solver API |
| `02_mwe_openpnm_crosscheck_optional` | Verify consistency against OpenPNM |
| `03_mwe_pyvista_visualization` | Inspect networks visually in 3-D |
| `04_mwe_manufactured_porespy_extraction` | Start from a controlled synthetic void image |
| `05_mwe_cartesian_mesh_network` | Generate mesh-like reference networks |
| `06_mwe_real_porespy_extraction` | Run a realistic extracted-image workflow |
| `07_mwe_synthetic_vug_case` | Study pruning and disconnected clusters |
| `08` to `11` vug notebooks | Run controlled geometry-sensitivity studies |
| `12_mwe_synthetic_volume_openpnm_benchmark` | Benchmark extracted-volume transport against OpenPNM |
| `13_mwe_synthetic_volume_xlb_benchmark` | Benchmark direct-image LBM transport against extracted-network PNM |
| `14_mwe_shape_factor_conductance_comparison` | Compare conductance closures and shape-factor sensitivity |
| `15_mwe_external_pnflow_benchmark` | Compare `voids` against a committed external `pnextract` / `pnflow` reference |
| `16_mwe_viscosity_model_kabs_benchmark` | Quantify `Kabs` drift under constant vs thermodynamic viscosity |
| `17_mwe_solver_options_benchmark` | Compare direct, Krylov, nonlinear, and preconditioned solver choices |
| `18_mwe_drp317_berea_raw_porosity_perm` | Validate the DRP-317 Berea case against experimental porosity and permeability |
| `19_mwe_drp317_bentheimer_raw_porosity_perm` | Validate the DRP-317 Bentheimer case against experimental porosity and permeability |
| `20_mwe_drp317_banderagray_raw_porosity_perm` | Validate the DRP-317 Bandera Gray case against experimental porosity and permeability |
| `21_mwe_drp317_banderabrown_raw_porosity_perm` | Run the DRP-317 Bandera Brown raw-volume workflow against the Table 1 experimental references |
| `22_mwe_drp317_bereasistergray_raw_porosity_perm` | Run the DRP-317 Berea Sister Gray raw-volume workflow against the Table 1 experimental references |
| `23_mwe_drp317_bereauppergray_raw_porosity_perm` | Run the DRP-317 Berea Upper Gray raw-volume workflow against the Table 1 experimental references |
| `24_mwe_drp317_buffberea_raw_porosity_perm` | Run the DRP-317 Buff Berea raw-volume workflow against the Table 1 experimental references |
| `25_mwe_drp317_castlegate_raw_porosity_perm` | Run the DRP-317 Castlegate raw-volume workflow against the Table 1 experimental references |
| `26_mwe_drp317_kirby_raw_porosity_perm` | Run the DRP-317 Kirby raw-volume workflow against the Table 1 experimental references |
| `27_mwe_drp317_leopard_raw_porosity_perm` | Run the DRP-317 Leopard raw-volume workflow against the Table 1 experimental references |
| `28_mwe_drp317_parker_raw_porosity_perm` | Run the DRP-317 Parker raw-volume workflow against the Table 1 experimental references |
| `29_mwe_drp443_ifn_raw_porosity_perm` | Benchmark DRP-443 IFN fractured-media permeability against SPE 212849 Table 2 (LBM reference) |
| `30_mwe_drp443_dilatedifn_raw_porosity_perm` | Benchmark DRP-443 Dilated IFN fractured-media permeability against SPE 212849 Table 2 (LBM reference) |
| `31_mwe_drp10_estaillades_raw_porosity_perm` | Benchmark DRP-10 Estaillades v2 carbonate permeability against Muljadi et al. (2016) Table 2 (OpenFOAM reference) |

---

## Notebook Overview

### 01 — Minimal Single-Phase Solve

**`01_mwe_singlephase_porosity_perm`**

Demonstrates the canonical single-phase workflow on a small synthetic network:

- build a `Network` from scratch
- compute absolute porosity
- solve incompressible single-phase flow
- extract directional permeability

This is the best starting point for understanding the core API.

---

### 02 — OpenPNM Cross-Check

**`02_mwe_openpnm_crosscheck_optional`**

Imports a PoreSpy/OpenPNM-style dictionary into `voids`, solves the same flow problem
with both `voids` and OpenPNM, and compares permeability estimates.
Requires the `test` Pixi environment (OpenPNM dependency).

---

### 03 — PyVista Visualization

**`03_mwe_pyvista_visualization`**

Shows optional 3-D network rendering using PyVista:

- pore coloring by pressure field
- throat sizing by conductance
- export to interactive HTML

---

### 04 — Manufactured PoreSpy Extraction

**`04_mwe_manufactured_porespy_extraction`**

Creates a deterministic 3D void image, extracts a pore network with PoreSpy, imports
it into `voids`, and serializes the result to HDF5.

---

### 05 — Cartesian Mesh Network

**`05_mwe_cartesian_mesh_network`**

Generates a configurable 2D or 3D mesh-like pore network, solves single-phase flow,
creates a Plotly interactive visualization, and exports to HDF5.

---

### 06 — Real PoreSpy Extraction (Ketton)

**`06_mwe_real_porespy_extraction`**

Applies the full extraction → import → solve → diagnostics pipeline to a real
segmented Ketton carbonate image.

---

### 07 — Synthetic Vug Case

**`07_mwe_synthetic_vug_case`**

Processes a grayscale synthetic vug volume, extracts the network, solves flow, and
compares results with and without pruning isolated void clusters.

---

### 08 — Image-Based Vug Shape Sensitivity (3D)

**`08_mwe_image_based_vug_shape_sensitivity`**

Controlled sensitivity study comparing a baseline network against networks with
spherical or ellipsoidal vugs. Reports porosity, absolute permeability `Kabs`, and
network statistics.

---

### 09 — Image-Based Vug Sensitivity (2D)

**`09_mwe_image_based_vug_sensitivity_2d`**

Simplified 2D counterpart of notebook 08 using circular and elliptical inclusions.
Produces porosity vs. `Kabs` and `K/K0` distributions.

---

### 10 — Lattice-Based Vug Sensitivity (3D)

**`10_mwe_lattice_based_vug_sensitivity`**

Stochastic lattice-based baselines with spherical and ellipsoidal vug insertions.
Reports `Kabs`/porosity sensitivity curves and `K/K0` distributions across multiple
realisations.

---

### 11 — Lattice-Based Vug Sensitivity (2D)

**`11_mwe_lattice_based_vug_sensitivity_2d`**

Simplified 2D lattice counterpart with circular and elliptical vugs, multi-baseline
sensitivity study, and `K/K0` frequency distributions.

---

### 12 — Synthetic Volume OpenPNM Benchmark

**`12_mwe_synthetic_volume_openpnm_benchmark`**

Builds synthetic spanning volumes, derives a synthetic grayscale segmentation,
extracts a network with PoreSpy, and compares resulting `Kabs` predictions between
`voids` and OpenPNM.

This notebook is the closest thing in the current tree to an end-to-end extraction
and solver-comparison benchmark.
The corresponding narrative report is documented in
[Verification & Validation / Verification / OpenPNM Extracted-Network Cross-Check](verification/openpnm.md).

---

### 13 — Synthetic Volume XLB Benchmark

**`13_mwe_synthetic_volume_xlb_benchmark`**

Builds fifteen synthetic segmented spanning volumes, solves them directly with XLB
on the binary image, extracts pore networks with `snow2`, and compares resulting
`Kabs` predictions between XLB and `voids`.

This is the notebook to use when the scientific question is whether the extracted
PNM workflow tracks a higher-fidelity voxel-scale reference closely enough.
It also documents the actual LBM formulation used by the current XLB adapter and
includes the shared pressure-drop mapping used to couple PNM and XLB, explains
why the preferred high-level input is `delta_p` rather than an absolute
pressure level, and includes a full 15-case steady Stokes-limit rerun alongside the standard
benchmark-mode comparison.
The corresponding narrative report is documented in
[Verification & Validation / Verification / XLB Direct-Image Permeability Benchmark](verification/xlb.md).

---

### 15 — External `pnextract` / `pnflow` Benchmark

**`15_mwe_external_pnflow_benchmark`**

Loads a committed reference dataset produced earlier with external `pnextract`
and `pnflow`, reruns the current `voids` workflow on the same exact saved
binary volumes, and compares permeability, porosity, and extracted-network
size.

This is the notebook to use when the scientific question is whether the current
`voids` image-to-network workflow tracks an independent external PNM workflow
closely enough on controlled synthetic cases.
The corresponding narrative report is documented in
[Verification & Validation / Verification / External pnextract / pnflow Benchmark](verification/pnflow.md).

---

### 16 — `Kabs` Benchmark: Constant vs Thermodynamic Viscosity

**`16_mwe_viscosity_model_kabs_benchmark`**

Benchmarks the apparent permeability estimate obtained with:

- a constant viscosity equal to the midpoint viscosity over the pressure interval
- a pressure-dependent thermodynamic viscosity field tabulated from `thermo`

This notebook is the right entry point when the scientific question is whether
pressure-dependent viscosity materially changes `Kabs` for a fixed network geometry.

---

### 17 — Solver Options Benchmark

**`17_mwe_solver_options_benchmark`**

Benchmarks the currently available solver options on both constant-viscosity and
pressure-dependent viscosity cases. It is the right notebook when the question is
numerical rather than physical: which combination of direct solve, Krylov method,
nonlinear strategy, and `pyamg` preconditioner is most effective for a given regime.

---

### 18 — DRP-317 Berea Validation

**`18_mwe_drp317_berea_raw_porosity_perm`**

Runs the current image-to-network workflow on the DRP-317 Berea sandstone volume and
compares porosity and apparent permeability against the experimental values reported
for the same rock sample.

The corresponding narrative report is documented in
[Verification & Validation / Validation / DRP-317 Berea Notebook Report](validation/drp317_berea.md).

---

### 19 — DRP-317 Bentheimer Validation

**`19_mwe_drp317_bentheimer_raw_porosity_perm`**

Runs the same validation workflow for the Bentheimer sandstone case from DRP-317,
including ROI diagnostics, extracted-network porosity, and directional permeability.

The corresponding narrative report is documented in
[Verification & Validation / Validation / DRP-317 Bentheimer Notebook Report](validation/drp317_bentheimer.md).

---

### 20 — DRP-317 Bandera Gray Validation

**`20_mwe_drp317_banderagray_raw_porosity_perm`**

Runs the DRP-317 Bandera Gray validation case, which is currently the most demanding
of the three because the permeability gap to experiment remains much larger than for
Berea and Bentheimer.

The corresponding narrative report is documented in
[Verification & Validation / Validation / DRP-317 Bandera Gray Notebook Report](validation/drp317_banderagray.md).

---

### 21-28 - Additional DRP-317 Sandstones

**`21_mwe_drp317_banderabrown_raw_porosity_perm`**
**`22_mwe_drp317_bereasistergray_raw_porosity_perm`**
**`23_mwe_drp317_bereauppergray_raw_porosity_perm`**
**`24_mwe_drp317_buffberea_raw_porosity_perm`**
**`25_mwe_drp317_castlegate_raw_porosity_perm`**
**`26_mwe_drp317_kirby_raw_porosity_perm`**
**`27_mwe_drp317_leopard_raw_porosity_perm`**
**`28_mwe_drp317_parker_raw_porosity_perm`**

These notebooks extend the existing DRP-317 workflow to the remaining raw binary
volumes under `examples/data/drp-317/`. They keep the same current PNM setup as
notebooks 18-20: ROI-based extraction from the full `1000^3` volume, pressure-
dependent water viscosity, and directional `Kabs` comparison against the Table 1
experimental values from the Scientific Reports paper.

The paper-reference values used by all eleven DRP-317 notebooks are committed in
`examples/data/drp-317/drp317_experimental_references.csv`.

---

### 29-30 - DRP-443 Fractured IFN Cases

**`29_mwe_drp443_ifn_raw_porosity_perm`**
**`30_mwe_drp443_dilatedifn_raw_porosity_perm`**

These notebooks benchmark `voids` on two DRP-443 induced-fracture-network
volumes (`IFN` and `DilatedIFN`) using paper-reference values from SPE
212849 Table 2.

For DRP-443, both workflows are intentionally full-volume only (no ROI
selection/subvolume analysis), matching your fractured-media requirement.

The extracted paper-reference values used by these notebooks are committed in
`examples/data/drp-443/drp443_reference_values.csv`.

The corresponding report is documented in
[Verification & Validation / Verification / DRP-443 Fracture-Network Verification Overview](verification/drp443.md).

---

### 31 - DRP-10 Estaillades v2

**`31_mwe_drp10_estaillades_raw_porosity_perm`**

This notebook benchmarks `voids` on the DRP-10 Estaillades v2 carbonate volume
using porosity and permeability references from Muljadi et al. (2016).

The corresponding report is documented in
[Verification & Validation / Verification / DRP-10 Estaillades Verification Overview](verification/drp10.md).

---

## DRP-317 Data Source

The DRP-317 notebooks use the following sources and should cite them explicitly in
downstream work:

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

For convenience, the Table 1 experimental porosity and permeability values used by the
notebooks are also committed in `examples/data/drp-317/drp317_experimental_references.csv`.

---

## Rendered Notebook Reports

The notebooks below are also rendered directly into the docs from their committed
`.ipynb` outputs, without re-executing them during the docs build:

The DRP-317 experimental-comparison notebooks are documented separately under
[Verification & Validation / Validation](validation/index.md) because those pages
are written as experimental-validation reports rather than as frozen rendered
notebook outputs.

- [Notebook Reports Overview](notebook_reports/index.md)
- [14 — Shape-Factor Conductance Comparison](notebook_reports/14_mwe_shape_factor_conductance_comparison.md)
- [15 — External `pnextract` / `pnflow` Benchmark](notebook_reports/15_mwe_external_pnflow_benchmark.md)
- [16 — `Kabs` Benchmark: Constant vs Thermodynamic Viscosity](notebook_reports/16_mwe_viscosity_model_kabs_benchmark.md)
- [17 — Solver Options Benchmark](notebook_reports/17_mwe_solver_options_benchmark.md)
