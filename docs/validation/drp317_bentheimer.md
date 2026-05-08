# DRP-317 Bentheimer Notebook Report

Notebook: `19_mwe_drp317_bentheimer_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `Bentheimer_2d25um_binary.raw`
- ROI size: `(300, 300, 300)` voxels
- Selected ROI origin: `(0, 0, 700)`
- ROI porosity: `26.75%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Primary reported conductance model: `generic_poiseuille`
- Conductance-model audit: `generic_poiseuille`, `hagen_poiseuille`,
  `valvatne_blunt`, and `auto` on the same extracted networks
- PoreSpy/PREGO boundary and transport geometry: external-reservoir helper pores
  with generated pyramids-and-cuboids hydraulic size factors available to `auto`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 22.64 |
| Full-image porosity [%] | 26.72 |
| ROI porosity [%] | 26.75 |
| Experimental permeability [mD] | 386.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 27.57 | 409.55 | 495.34 | 485.67 | 465.11 | 20.49 | 3179 | 5193 |
| PREGO | 26.53 | 779.42 | 926.15 | 920.10 | 877.84 | 127.42 | 2119 | 4926 |
| Native maximal-ball | 26.53 | 218.22 | 315.43 | 330.66 | 292.38 | -24.25 | 1126 | 2130 |

![Bentheimer directional permeability](../assets/validation/drp317_bentheimer_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.27 | 0.361 |
| PREGO | 4.65 | 0.181 |
| Native maximal-ball | 3.78 | 0.219 |

## Conductance-Model Audit

![Bentheimer conductance-model audit](../assets/validation/drp317_bentheimer_conductance_audit.png)

| Backend | generic [mD] | Hagen-Poiseuille [mD] | Valvatne-Blunt [mD] | auto [mD] |
|---|---:|---:|---:|---:|
| PoreSpy snow2 | 465.11 | 1951.71 | 1283.64 | 743.38 |
| PREGO | 877.84 | 2009.74 | 1562.90 | 1431.73 |
| Native maximal-ball | 292.38 | 12167.39 | 3558.18 | 3558.18 |

## Interpretation

For `Bentheimer`, the closest aggregate permeability in this rerun is
from `Native maximal-ball` with a relative permeability error of
`-24.25%`. The spread between the
largest and smallest primary backend aggregate permeability is about `3.00`x,
which makes extraction sensitivity a material part of this sample's validation
result.

The conductance audit is the sharper diagnostic: on the same extracted networks,
`hagen_poiseuille`, `valvatne_blunt`, and `auto` all increase the Bentheimer
permeability relative to the primary `generic_poiseuille` baseline. For this ROI,
the PREGO overestimate is therefore not fixed by switching to the generated
PoreSpy/OpenPNM-style size factors; the reduced geometry and conduit assumptions
remain the dominant modeling uncertainty.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
