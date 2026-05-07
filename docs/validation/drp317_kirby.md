# DRP-317 Kirby Notebook Report

Notebook: `26_mwe_drp317_kirby_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `Kirby_2d25um_binary.raw`
- ROI size: `(300, 300, 300)` voxels
- Selected ROI origin: `(700, 0, 700)`
- ROI porosity: `21.47%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 19.95 |
| Full-image porosity [%] | 21.49 |
| ROI porosity [%] | 21.47 |
| Experimental permeability [mD] | 62.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 22.00 | 119.23 | 122.98 | 138.04 | 127.01 | 104.85 | 4261 | 7372 |
| PREGO | 21.30 | 200.48 | 220.60 | 231.52 | 217.91 | 251.48 | 2588 | 5913 |
| Native maximal-ball | 21.30 | 62.41 | 61.79 | 66.87 | 63.73 | 2.79 | 2150 | 3975 |

![Kirby directional permeability](../assets/validation/drp317_kirby_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.46 | 0.271 |
| PREGO | 4.57 | 0.070 |
| Native maximal-ball | 3.70 | 0.181 |

## Interpretation

For `Kirby`, the closest aggregate permeability in this rerun is
from `Native maximal-ball` with a relative permeability error of
`2.79%`. The spread between the
largest and smallest backend aggregate permeability is about `3.42`x,
which makes extraction sensitivity a material part of this sample's validation
result.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
