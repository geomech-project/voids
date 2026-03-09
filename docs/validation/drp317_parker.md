# DRP-317 Parker Notebook Report

Notebook: `28_mwe_drp317_parker_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

## Current Setup

- Raw volume: `Parker_2d25um_binary.raw`
- ROI size: `300 x 300 x 300` voxels
- Selected ROI origin: `(700, 350, 0)`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 14.77 |
| Full-image porosity [%] | 13.65 |
| ROI porosity [%] | 13.68 |
| Network absolute porosity [%] | 12.90 |
| Experimental permeability [mD] | 10.0 |
| Kx [mD] | 20.51 |
| Ky [mD] | 18.17 |
| Kz [mD] | 22.40 |
| Arithmetic mean permeability [mD] | 20.36 |
| Quadratic-mean permeability [mD] | 20.43 |
| Relative quadratic-mean error [%] | 104.32 |

![Parker directional permeability](../assets/validation/drp317_parker_directional.png)

## Interpretation

The current workflow predicts `Parker` with a quadratic-mean permeability error of
`104.32%` relative to the Table 1 experimental reference.
This case should be interpreted together with the cross-sample summary in
[DRP-317 sandstone validation overview](drp317.md).
