# DRP-317 Berea Sister Gray Notebook Report

Notebook: `22_mwe_drp317_bereasistergray_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `BSG_2d25um_binary.raw`
- ROI size: `(300, 300, 300)` voxels
- Selected ROI origin: `(350, 0, 0)`
- ROI porosity: `19.84%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 19.07 |
| Full-image porosity [%] | 19.79 |
| ROI porosity [%] | 19.84 |
| Experimental permeability [mD] | 80.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 20.20 | 122.98 | 122.61 | 135.06 | 127.01 | 58.77 | 3472 | 6026 |
| PREGO | 19.55 | 198.12 | 213.49 | 211.86 | 207.94 | 159.92 | 2063 | 4679 |
| Native maximal-ball | 19.55 | 65.06 | 61.00 | 68.00 | 64.75 | -19.06 | 1717 | 3058 |

![Berea Sister Gray directional permeability](../assets/validation/drp317_berea_sister_gray_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.47 | 0.276 |
| PREGO | 4.54 | 0.075 |
| Native maximal-ball | 3.56 | 0.196 |

## Interpretation

For `Berea Sister Gray`, the closest aggregate permeability in this rerun is
from `Native maximal-ball` with a relative permeability error of
`-19.06%`. The spread between the
largest and smallest backend aggregate permeability is about `3.21`x,
which makes extraction sensitivity a material part of this sample's validation
result.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
