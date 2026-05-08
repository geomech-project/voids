# DRP-317 Buff Berea Notebook Report

Notebook: `24_mwe_drp317_buffberea_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `BB_2d25um_binary.raw`
- ROI size: `(300, 300, 300)` voxels
- Selected ROI origin: `(700, 700, 350)`
- ROI porosity: `22.82%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 24.02 |
| Full-image porosity [%] | 22.71 |
| ROI porosity [%] | 22.82 |
| Experimental permeability [mD] | 275.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 23.40 | 390.63 | 314.67 | 206.72 | 313.23 | 13.90 | 2359 | 4143 |
| PREGO | 22.56 | 771.22 | 543.23 | 414.35 | 594.86 | 116.31 | 1373 | 3235 |
| Native maximal-ball | 22.56 | 200.81 | 298.64 | 110.56 | 217.36 | -20.96 | 998 | 1773 |

![Buff Berea directional permeability](../assets/validation/drp317_buff_berea_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.51 | 0.320 |
| PREGO | 4.71 | 0.077 |
| Native maximal-ball | 3.55 | 0.202 |

## Interpretation

For `Buff Berea`, the closest aggregate permeability in this rerun is
from `PoreSpy snow2` with a relative permeability error of
`13.90%`. The spread between the
largest and smallest backend aggregate permeability is about `2.74`x,
which makes extraction sensitivity a material part of this sample's validation
result.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
