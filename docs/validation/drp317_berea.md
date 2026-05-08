# DRP-317 Berea Notebook Report

Notebook: `18_mwe_drp317_berea_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>


## Current Setup

- Raw volume: `Berea_2d25um_binary.raw`
- ROI size: `(256, 256, 256)` voxels
- Selected ROI origin: `(0, 744, 0)`
- ROI porosity: `21.32%`
- Extraction backends: `porespy`, `prego`, `native_maximal_ball`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 18.96 |
| Full-image porosity [%] | 21.67 |
| ROI porosity [%] | 21.32 |
| Experimental permeability [mD] | 121.0 |

| Backend | Network phi [%] | Kx [mD] | Ky [mD] | Kz [mD] | RMS K [mD] | Rel. K error [%] | Np | Nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PoreSpy snow2 | 21.82 | 144.91 | 134.71 | 140.13 | 139.98 | 15.69 | 2062 | 3373 |
| PREGO | 20.96 | 258.69 | 227.28 | 247.77 | 244.93 | 102.42 | 1185 | 2566 |
| Native maximal-ball | 20.96 | 90.55 | 73.50 | 74.99 | 80.05 | -33.84 | 963 | 1651 |

![Berea directional permeability](../assets/validation/drp317_berea_directional.png)

## Network Statistics Snapshot

| Backend | Mean coordination | Dead-end pore fraction |
|---|---:|---:|
| PoreSpy snow2 | 3.27 | 0.312 |
| PREGO | 4.33 | 0.079 |
| Native maximal-ball | 3.43 | 0.214 |

## Interpretation

For `Berea`, the closest aggregate permeability in this rerun is
from `PoreSpy snow2` with a relative permeability error of
`15.69%`. The spread between the
largest and smallest backend aggregate permeability is about `3.06`x,
which makes extraction sensitivity a material part of this sample's validation
result.

This is a reduced-network comparison against a laboratory-scale experimental
reference. The numbers depend on the selected ROI, segmentation convention,
boundary labeling, network reduction, and conductance closure; they should not be
read as a direct voxel-scale flow simulation.
