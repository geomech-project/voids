<p align="center">
  <img src="resources/logo/Voids%20logo.png" alt="voids logo" width="800">
</p>

# voids

[![Tests](https://github.com/geomech-project/voids/actions/workflows/tests.yml/badge.svg)](https://github.com/geomech-project/voids/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/geomech-project/voids/branch/main/graph/badge.svg)](https://codecov.io/gh/geomech-project/voids)
[![Supported OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://github.com/geomech-project/voids/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/voids)](https://pypi.org/project/voids/)
[![pip install voids](https://img.shields.io/badge/pip%20install-voids-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/voids/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18937646.svg)](https://doi.org/10.5281/zenodo.18937646)

`voids` is a scientific Python package for digital porous media research. Its main
modeling approach is pore-network modeling (PNM): images, extracted networks,
geometry, provenance, and transport assumptions are kept explicit so permeability
studies can be reproduced and compared.

Alongside PNM, `voids` provides complementary single-phase transport backends:
micro-continuum models with the finite-volume method (FVM) and finite-element
method (FEM), and direct numerical simulation (DNS) with the lattice Boltzmann
method (LBM). These backends make it possible to compare pore-network,
micro-continuum, and voxel-image descriptions of the same digital porous medium.
Interoperability with PoreSpy/OpenPNM-style data remains part of the package
contract.

## Goals

The intended direction of `voids` is:

- make PNM the main modeling path for digital porous media studies
- preserve sample geometry and provenance information needed for reproducible studies
- support import and normalization of extracted networks from external tools
- provide micro-continuum FVM/FEM and DNS LBM backends for single-phase comparison
  and upscaling
- expose well-scoped physics modules with diagnostics and regression tests
- build confidence from validated single-phase transport before adding richer models

This is a research codebase, not a GUI application or a full image-to-network extraction
pipeline. Raw segmentation and extraction are intentionally delegated to upstream tools
such as PoreSpy. The FEM, FVM, and LBM backends are provided for documented
digital-porous-media workflows and should be interpreted through their stated
governing equations, boundary conditions, map/image assumptions, and solver diagnostics.

## Current Scope

The current `v0.1.x` implementation includes:

- canonical `Network`, `SampleGeometry`, and `Provenance` data structures
- import of PoreSpy/OpenPNM-style dictionaries into the canonical model
- geometry normalization helpers for extracted networks, including optional
  external-reservoir boundary augmentation for image-extracted flow benchmarks
- static petrophysics:
  - absolute porosity
  - effective porosity
  - connectivity metrics
- single-phase incompressible flow with directional permeability estimation
- data-adaptive `auto`, OpenPNM size-factor, circular `hagen_poiseuille`, and
  shape-aware `valvatne_blunt_throat` / `valvatne_blunt` conductance closures
- optional PoreSpy/PREGO hydraulic size factors for
  pyramids-and-cuboids conduit transport
- pressure-dependent water viscosity via `thermo` and `CoolProp`
- Picard and damped-Newton nonlinear solves for variable-viscosity problems
- Krylov linear solvers with optional `pyamg` preconditioning
- porosity/permeability map generation and structured field export
- finite-volume TPFA Darcy upscaling on scalar permeability maps
- FEniCSx finite-element Darcy-Darcy and Darcy-Brinkman micro-continuum upscaling
- XLB/JAX direct-image LBM DNS in the Stokes-limit permeability setting
- HDF5 serialization
- optional Plotly and PyVista network visualization
- interoperability cross-checks against OpenPNM
- direct-image XLB/LBM permeability solves and benchmarking
- synthetic and manufactured examples for regression and tutorials

Important boundaries:

- multiphase flow is not implemented yet
- production image acquisition and fully automated "push-button" extraction pipelines are out of scope
- FEM and LBM backends require their external solver stacks to be installed
- map-based and direct-image solver results depend on map closure, boundary
  conditions, resolution, and solver diagnostics
- controlled grayscale preprocessing, segmentation helpers, and `snow2`-based extraction helpers are available in `voids.image`
- synthetic mesh/manufactured examples are controlled validation cases, not realistic rock reconstructions

For a more formal statement of scope and assumptions, see [spec_v0_1.md](spec_v0_1.md).

The rendered documentation is intended to live alongside the repository at
<https://geomech-project.github.io/voids/>.

## Installation

### Install from PyPI

If you want the published package rather than a local editable checkout:

```bash
pip install voids
```

PyPI package page:
<https://pypi.org/project/voids/>

### Editable pip install

If you prefer a plain Python environment from the repository checkout:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev,viz,test,lbm,docs]"
```

Assumptions to keep in mind:

- FEniCSx is not installed by a PyPI extra; plain pip users must provide a
  compatible DOLFINx/FEniCSx installation before using `voids.fem`
- repository development, notebook kernels, and documentation builds are covered
  in [Development](#development)

## Quick Start

```python
from voids.examples import make_linear_chain_network
from voids.physics.petrophysics import absolute_porosity
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve

net = make_linear_chain_network()

result = solve(
    net,
    fluid=FluidSinglePhase(viscosity=1.0),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
    axis="x",
)

print("phi_abs =", absolute_porosity(net))
print("Q =", result.total_flow_rate)
print("Kx =", result.permeability["x"])
print("mass_balance_error =", result.mass_balance_error)
```

There is also a small workflow entry point:

```bash
pixi run examples-singlephase
```

## Examples And Notebooks

The repository includes paired notebooks and `py:percent` scripts under `notebooks/`:

- `01_mwe_singlephase_porosity_perm`
  - minimal single-phase solve, porosity, and permeability
- `02_mwe_openpnm_crosscheck_optional`
  - roundtrip and OpenPNM cross-check workflow
- `03_mwe_pyvista_visualization`
  - optional PyVista-based network rendering
- `04_mwe_manufactured_porespy_extraction`
  - manufactured 3D void image, PoreSpy extraction, import into `voids`, and serialization
- `05_mwe_cartesian_mesh_network`
  - configurable 2D/3D mesh-like pore networks, flow solve, Plotly visualization, and HDF5 export
- `06_mwe_real_porespy_extraction`
  - real segmented Ketton image, PoreSpy extraction, `voids` import, solve, and diagnostics
- `07_mwe_synthetic_vug_case`
  - grayscale synthetic vug volume preprocessing, extraction, solve, and pruning comparison
- `08_mwe_image_based_vug_shape_sensitivity`
  - controlled baseline vs spherical/ellipsoidal vug study with porosity, `Kabs`, and network statistics
- `09_mwe_image_based_vug_sensitivity_2d`
  - simplified 2D image-based baseline vs circular/elliptical vug study with porosity, `Kabs`, and `K/K0` distributions
- `10_mwe_lattice_based_vug_sensitivity`
  - lattice-based stochastic baselines with spherical/ellipsoidal vug insertion, `Kabs`/porosity sensitivity, and `K/K0` distributions
- `11_mwe_lattice_based_vug_sensitivity_2d`
  - simplified 2D lattice counterpart with circular/elliptical vugs, multi-baseline sensitivity, and `K/K0` frequency distributions
- `12_mwe_synthetic_volume_openpnm_benchmark`
  - synthetic spanning volumes, synthetic grayscale segmentation, `snow2` extraction, and `Kabs` cross-checks between `voids` and OpenPNM
- `13_mwe_synthetic_volume_xlb_benchmark`
  - synthetic segmented volumes, direct-image XLB solves, extracted-network `voids` solves, and `Kabs` comparison between voxel-scale LBM and PNM
- `14_mwe_shape_factor_conductance_comparison`
  - synthetic and extracted-network comparison of circular and shape-aware conductance closures, and permeability sensitivity to shape factors
- `15_mwe_external_pnflow_benchmark`
  - committed external `pnextract`/`pnflow` reference cases, including explicit same-network parity on the saved CNM and a separate `snow2` workflow comparison on the original images
- `16_mwe_viscosity_model_kabs_benchmark`
  - benchmark of `Kabs` using constant viscosity versus pressure-dependent thermodynamic viscosity
- `17_mwe_solver_options_benchmark`
  - benchmark of the available linear and nonlinear solver options, including `pyamg`-preconditioned Krylov solves
- `18_mwe_drp317_berea_raw_porosity_perm`
  - DRP-317 Berea validation notebook with `snow2`, PREGO, and native maximal-ball extraction comparisons
- `19_mwe_drp317_bentheimer_raw_porosity_perm`
  - DRP-317 Bentheimer validation notebook with `snow2`, PREGO, and native maximal-ball extraction comparisons
- `20_mwe_drp317_banderagray_raw_porosity_perm`
  - DRP-317 Bandera Gray validation notebook with `snow2`, PREGO, and native maximal-ball extraction comparisons
- `21_mwe_drp317_banderabrown_raw_porosity_perm`
  - DRP-317 Bandera Brown backend-sensitivity notebook against the Table 1 experimental values
- `22_mwe_drp317_bereasistergray_raw_porosity_perm`
  - DRP-317 Berea Sister Gray backend-sensitivity notebook against the Table 1 experimental values
- `23_mwe_drp317_bereauppergray_raw_porosity_perm`
  - DRP-317 Berea Upper Gray backend-sensitivity notebook against the Table 1 experimental values
- `24_mwe_drp317_buffberea_raw_porosity_perm`
  - DRP-317 Buff Berea backend-sensitivity notebook against the Table 1 experimental values
- `25_mwe_drp317_castlegate_raw_porosity_perm`
  - DRP-317 Castlegate backend-sensitivity notebook against the Table 1 experimental values
- `26_mwe_drp317_kirby_raw_porosity_perm`
  - DRP-317 Kirby backend-sensitivity notebook against the Table 1 experimental values
- `27_mwe_drp317_leopard_raw_porosity_perm`
  - DRP-317 Leopard backend-sensitivity notebook against the Table 1 experimental values
- `28_mwe_drp317_parker_raw_porosity_perm`
  - DRP-317 Parker backend-sensitivity notebook against the Table 1 experimental values
- `29_mwe_drp443_ifn_raw_porosity_perm`
  - DRP-443 IFN fractured-media permeability benchmark against SPE 212849 Table 2
- `30_mwe_drp443_dilatedifn_raw_porosity_perm`
  - DRP-443 Dilated IFN fractured-media permeability benchmark against SPE 212849 Table 2
- `31_mwe_drp10_estaillades_raw_porosity_perm`
  - DRP-10 Estaillades v2 carbonate benchmark with native maximal-ball and `snow2` extraction-backend comparisons
- `32_mwe_prego_blobs_backend_comparison`
  - synthetic `256^3` PoreSpy `blobs` comparison of PoreSpy `snow2`, PREGO, and native maximal-ball extraction
- `33_mwe_synthetic_porosity_maps`
  - synthetic binary/grayscale porosity maps, Kozeny-Carman permeability maps, and HDF5 field export
- `34_mwe_macro_micro_synthetic_case`
  - macro/micro synthetic porous-media case for coupled-scale workflow exploration
- `35_mwe_trabecular_bone_morphometry`
  - trabecular-bone RAW segmentation morphometry with bone/marrow phase convention checks
- `36_mwe_trabecular_bone_slice_porosity_permeability_maps`
  - trabecular-bone slice porosity/permeability maps with HDF5 and structured mesh exports
- `37_mwe_trabecular_bone_slice_pore_network`
  - trabecular-bone 3-D ROI pore-network extraction and directional single-phase permeability
- `38_mwe_trabecular_bone_map_resistor_upscaling`
  - trabecular-bone 3-D ROI porosity/permeability map upscaling and continuum/DNS comparison
- `40_mwe_drp317_berea_roi_pnm_comparison`
  - DRP-317 Berea 3-D ROI pore-network result consolidation for map-solver comparison
- `41_mwe_drp317_berea_map_resistor_micro_continuum`
  - DRP-317 Berea 3-D ROI porosity/permeability map upscaling and micro-continuum comparison
- `42_mwe_drp317_berea_block3_same_roi_comparison`
  - DRP-317 Berea same-ROI comparison across pore-network, TPFA, FEniCSx FEM, and XLB/LBM backends
- `43_mwe_drp317_bentheimer_block3_same_roi_comparison`
  - DRP-317 Bentheimer same-ROI comparison across pore-network, TPFA, FEniCSx FEM, and XLB/LBM backends
- `44_mwe_drp317_parker_block3_same_roi_comparison`
  - DRP-317 Parker same-ROI comparison across pore-network, TPFA, FEniCSx FEM, and XLB/LBM backends
- `45_mwe_drp317_lbm_sensitivity`
  - DRP-317 direct-image LBM setup-sensitivity study for the recommended Stokes-limit preset

Example data under `examples/data/` includes a deterministic manufactured void image
and generated artifacts from the extraction, map, and mesh notebooks.

## Verification & Validation

The project documentation now separates two kinds of evidence:

- **Verification**: benchmarks against software references and controlled numerical workflows
- **Validation**: benchmarks against experimental data

Software-verification reports live under [`docs/verification/`](docs/verification/).
Experimental-validation reports for the DRP-317 sandstones live under
[`docs/validation/`](docs/validation/).

### DRP-317 Data Citation

The DRP-317 notebooks and validation reports use the following sources:

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

The full Table 1 sample references used by the DRP-317 notebooks are committed in
[`examples/data/drp-317/drp317_experimental_references.csv`](examples/data/drp-317/drp317_experimental_references.csv).

## Scientific Notes

Several assumptions are deliberate and should be stated explicitly:

- extracted-network predictions depend strongly on upstream segmentation and extraction quality
- imported geometry fields may be incomplete or model-dependent across tools
- map-based continuum predictions depend on the porosity/permeability map closure,
  block size, solver boundary conditions, and representative-volume assumptions
- direct-image LBM predictions depend on voxel resolution, boundary treatment,
  convergence controls, and low-Mach/low-Reynolds diagnostics
- single-phase OpenPNM cross-checks compare solver/assembly consistency, not universal physical truth
- throat visualization may use arithmetic averaging of pore scalars when no throat scalar field is provided; that is a visualization choice, not a constitutive model

If any of those assumptions are inappropriate for a study, the corresponding workflow should
be tightened before using results quantitatively.

## Development

This repository is configured for Pixi and exposes three main environments:

- `default`: core runtime + notebooks + plotting + PyVista + thermodynamic, FEM, and LBM backends
- `test`: core runtime plus test-only dependencies
- `docs`: MkDocs, Material for MkDocs, and mkdocstrings

```bash
pixi install
pixi run -e default python -c "import voids; print(voids.__version__)"
```

Pixi activation also provides project path variables used by notebooks:

- `VOIDS_PROJECT_ROOT`
- `VOIDS_NOTEBOOKS_PATH`
- `VOIDS_EXAMPLES_PATH`
- `VOIDS_DATA_PATH`

Useful development commands:

```bash
pixi run test
pixi run test-cov
pixi run lint
pixi run typecheck
pixi run precommit
pixi run notebooks-smoke
pixi run docs-build
```

Version updates are handled with:

```bash
pixi run bump-version <new-version>
```

## Status

`voids` is still pre-alpha. The codebase is already useful for controlled
single-phase porous-media transport experiments, solver validation, and
interoperability studies, but it should not be described as a complete
porous-media simulation platform yet.

## AI Usage Statement

Starting with `v0.1.7`, `voids` development is aided by AI tools, including
Codex and GitHub Copilot. These tools are used to assist with refactoring,
fast code changes, code review, and documentation writing.

All scientific choices, implementation decisions, and final content remain
under human review and responsibility. This statement is intended as a
transparency measure aligned with current scientific-integrity expectations for
AI-assisted research and software development.

## Institutional Support

`voids` receives institutional support from the
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br),
a research unit of the Ministério da Ciência, Tecnologia e Inovação (MCTI), Brazil.

<p align="center">
  <a href="https://www.gov.br/lncc/pt-br">
    <img src="resources/logo/lncc.svg" alt="LNCC logo" width="420">
  </a>
</p>
