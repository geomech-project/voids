# DRP-317 Bandera Brown Notebook Report

Notebook: `21_mwe_drp317_banderabrown_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

## Current Setup

- Raw volume: `BanderaBrown_2d25um_binary.raw`
- ROI size: `300 x 300 x 300` voxels
- Selected ROI origin: `(700, 350, 350)`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 24.11 |
| Full-image porosity [%] | 21.18 |
| ROI porosity [%] | 21.13 |
| Network absolute porosity [%] | 21.25 |
| Experimental permeability [mD] | 63.0 |
| Kx [mD] | 65.33 |
| Ky [mD] | 58.30 |
| Kz [mD] | 65.45 |
| Arithmetic mean permeability [mD] | 63.03 |
| Quadratic-mean permeability [mD] | 63.12 |
| Relative quadratic-mean error [%] | 0.18 |

![Bandera Brown directional permeability](../assets/validation/drp317_bandera_brown_directional.png)

## Interpretation

The current workflow predicts `Bandera Brown` with a quadratic-mean permeability error of
`0.18%` relative to the Table 1 experimental reference.
This case should be interpreted together with the cross-sample summary in
[DRP-317 sandstone validation overview](drp317.md).
