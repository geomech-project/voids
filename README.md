<p align="center">
  <img src="resources/logo/Voids%20logo.png" alt="voids logo" width="800">
</p>

# voids

[![Tests](https://github.com/geomech-project/voids/actions/workflows/tests.yml/badge.svg)](https://github.com/geomech-project/voids/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/geomech-project/voids/branch/main/graph/badge.svg)](https://codecov.io/gh/geomech-project/voids)
[![Supported OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://github.com/geomech-project/voids/actions/workflows/tests.yml)

`voids` is a scientific Python package for pore network modeling (PNM) aimed at
research workflows where reproducibility, explicit assumptions, and validation matter.
The current project emphasis is a clean canonical network model, interoperability with
PoreSpy/OpenPNM-style data, and a validated single-phase reference workflow before
expanding to more complex physics.

## Goals

The intended direction of `voids` is:

- provide a rigorous internal representation of pore-throat networks
- preserve sample geometry and provenance information needed for reproducible studies
- support import and normalization of extracted networks from external tools
- expose well-scoped physics modules with diagnostics and regression tests
- build confidence on single-phase transport first, then expand toward richer models

This is a research codebase, not a GUI application or a full image-to-network extraction
pipeline. Raw segmentation and extraction are intentionally delegated to upstream tools
such as PoreSpy.

## Current Scope

The current `v0.1.x` implementation includes:

- canonical `Network`, `SampleGeometry`, and `Provenance` data structures
- import of PoreSpy/OpenPNM-style dictionaries into the canonical model
- geometry normalization helpers for extracted networks
- static petrophysics:
  - absolute porosity
  - effective porosity
  - connectivity metrics
- single-phase incompressible flow with directional permeability estimation
- HDF5 serialization
- optional Plotly and PyVista network visualization
- interoperability cross-checks against OpenPNM
- optional direct-image permeability benchmarking against XLB
- synthetic and manufactured examples for regression and tutorials

Important boundaries:

- multiphase flow is not implemented yet
- production image acquisition and fully automated "push-button" extraction pipelines are out of scope
- controlled grayscale preprocessing, segmentation helpers, and `snow2`-based extraction helpers are available in `voids.image`
- synthetic mesh/manufactured examples are controlled validation cases, not realistic rock reconstructions

For a more formal statement of scope and assumptions, see [spec_v0_1.md](spec_v0_1.md).

The rendered documentation is intended to live alongside the repository at
<https://geomech-project.github.io/voids/>.

## Installation

### Recommended: Pixi

This repository is configured for Pixi and exposes four main environments:

- `default`: core library + notebooks + plotting + PyVista + OpenPNM
- `test`: everything in `default` plus test-only dependencies
- `lbm`: test environment plus the optional XLB stack
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

### Editable pip install

If you prefer a plain Python environment:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev,viz,test,docs,lbm]"
```

Assumption to keep in mind: the notebooks are exercised primarily through the Pixi
environments, so the most reliable setup is still Pixi.

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
  - synthetic and extracted-network comparison of conductance closures (`generic_poiseuille`, `valvatne_blunt_throat`, and `valvatne_blunt`) and permeability sensitivity to shape factors
- `15_mwe_external_pnflow_benchmark`
  - committed external `pnextract`/`pnflow` reference cases compared against the current `voids` extraction + solve workflow

Example data under `examples/data/` includes a deterministic manufactured void image and
generated artifacts from the extraction/mesh notebooks.

## Scientific Notes

Several assumptions are deliberate and should be stated explicitly:

- extracted-network predictions depend strongly on upstream segmentation and extraction quality
- imported geometry fields may be incomplete or model-dependent across tools
- single-phase OpenPNM cross-checks compare solver/assembly consistency, not universal physical truth
- throat visualization may use arithmetic averaging of pore scalars when no throat scalar field is provided; that is a visualization choice, not a constitutive model

If any of those assumptions are inappropriate for a study, the corresponding workflow should
be tightened before using results quantitatively.

## Development

Useful commands:

```bash
pixi run test
pixi run test-cov
pixi run lint
pixi run typecheck
pixi run precommit
pixi run notebooks-smoke
```

Version updates are handled with:

```bash
pixi run bump-version 0.1.4
```

## Status

`voids` is still pre-alpha. The codebase is already useful for controlled PNM experiments,
solver validation, and interoperability studies, but it should not be described as a
complete pore-network simulation platform yet.

## Institutional Support

`voids` receives institutional support from the
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br),
a research unit of the Ministério da Ciência, Tecnologia e Inovação (MCTI), Brazil.

<p align="center">
  <a href="https://www.gov.br/lncc/pt-br">
    <img src="resources/logo/lncc-mcti.svg" alt="LNCC (MCTI) logo" width="820">
  </a>
</p>
