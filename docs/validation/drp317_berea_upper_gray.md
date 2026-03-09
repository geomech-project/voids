# DRP-317 Berea Upper Gray Notebook Report

Notebook: `23_mwe_drp317_bereauppergray_raw_porosity_perm`

## Sources

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

## Current Setup

- Raw volume: `BUG_2d25um_binary.raw`
- ROI size: `300 x 300 x 300` voxels
- Selected ROI origin: `(0, 700, 0)`
- Conductance model: `generic_poiseuille`
- Viscosity model: tabulated water viscosity from `thermo`, `298.15 K`
- Boundary pressures: `pout = 5.0 MPa`, `pin = pout + 10 kPa/m * L`

## Key Results

| Quantity | Value |
|---|---:|
| Experimental porosity [%] | 18.56 |
| Full-image porosity [%] | 19.65 |
| ROI porosity [%] | 19.63 |
| Network absolute porosity [%] | 19.91 |
| Experimental permeability [mD] | 86.0 |
| Kx [mD] | 78.48 |
| Ky [mD] | 149.86 |
| Kz [mD] | 101.71 |
| Arithmetic mean permeability [mD] | 110.02 |
| Quadratic-mean permeability [mD] | 113.96 |
| Relative quadratic-mean error [%] | 32.52 |

![Berea Upper Gray directional permeability](../assets/validation/drp317_berea_upper_gray_directional.png)

## Interpretation

The current workflow predicts `Berea Upper Gray` with a quadratic-mean permeability error of
`32.52%` relative to the Table 1 experimental reference.
This case should be interpreted together with the cross-sample summary in
[DRP-317 sandstone validation overview](drp317.md).
