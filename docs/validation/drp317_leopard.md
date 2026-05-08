# DRP-317 Leopard Notebook Report

Notebook: `27_mwe_drp317_leopard_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `Leopard_2d25um_binary.raw`
- ROI size: `(300, 300, 300)` voxels
- Selected ROI origin: `(0, 0, 0)`
- ROI porosity: `19.50%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 20.22 |
| Full-image porosity [%] | 19.50 |
| ROI porosity [%] | 19.50 |
| Experimental permeability [mD] | 327.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 20.01 | 475.08 | 259.61 | 131.55 | 321.66 | -1.63 | 1929 | 3158 |
| PREGO | 19.30 | 757.98 | 555.62 | 355.88 | 580.20 | 77.43 | 1049 | 2350 |
| Native maximal-ball | 19.30 | 331.38 | 174.04 | 74.78 | 220.38 | -32.61 | 820 | 1390 |

![Leopard directional permeability](../assets/validation/drp317_leopard_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.27 | 0.338 |
| PREGO | 4.48 | 0.087 |
| Native maximal-ball | 3.39 | 0.229 |

## Interpretation

For `Leopard`, the closest aggregate permeability in this rerun is
from `PoreSpy snow2` with a relative permeability error of
`-1.63%`. The spread between the
largest and smallest backend aggregate permeability is about `2.63`x,
which makes extraction sensitivity a material part of this sample's validation
result.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
