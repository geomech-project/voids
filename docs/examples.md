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

- use `default` for most notebooks
- use `test` when a notebook depends on OpenPNM cross-checks
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
[Verification / XLB Direct-Image Permeability Benchmark](verification/xlb.md).
