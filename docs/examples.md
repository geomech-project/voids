# Examples and Notebooks

The `notebooks/` directory contains paired Jupyter notebooks and `py:percent` scripts
that demonstrate all major `voids` workflows.
Each notebook is both a tutorial and a regression artifact: running it to completion
checks that the code path is still functional.

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
