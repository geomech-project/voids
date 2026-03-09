# Verification & Validation

This section separates two different kinds of evidence used in `voids`:

- **Verification**: benchmarks against software references, manufactured cases,
  or controlled numerical workflows
- **Validation**: benchmarks against experimental data

That distinction matters. A software cross-check can show that `voids` is
numerically consistent with a reference implementation, while an experimental
comparison asks whether the present workflow predicts a measured physical
quantity closely enough for the intended scientific use.

## Current Structure

### Verification

The software-verification studies live under [Verification](software.md):

- [OpenPNM extracted-network cross-check](openpnm.md)
- [External `pnextract` / `pnflow` benchmark](pnflow.md)
- [XLB direct-image permeability benchmark](xlb.md)
- [DRP-443 fracture-network verification overview](drp443.md)
- [DRP-10 Estaillades verification overview](drp10.md)

### Validation

The experimental-validation studies live under [Validation](../validation/index.md):

- [DRP-317 sandstone validation overview](../validation/drp317.md)
- [DRP-317 Parker notebook report](../validation/drp317_parker.md)
- [DRP-317 Kirby notebook report](../validation/drp317_kirby.md)
- [DRP-317 Bandera Brown notebook report](../validation/drp317_bandera_brown.md)
- [DRP-317 Berea Sister Gray notebook report](../validation/drp317_berea_sister_gray.md)
- [DRP-317 Berea Upper Gray notebook report](../validation/drp317_berea_upper_gray.md)
- [DRP-317 Berea notebook report](../validation/drp317_berea.md)
- [DRP-317 Castlegate notebook report](../validation/drp317_castlegate.md)
- [DRP-317 Buff Berea notebook report](../validation/drp317_buff_berea.md)
- [DRP-317 Leopard notebook report](../validation/drp317_leopard.md)
- [DRP-317 Bentheimer notebook report](../validation/drp317_bentheimer.md)
- [DRP-317 Bandera Gray notebook report](../validation/drp317_banderagray.md)

## What Each Side Answers

| Category | Typical reference | Main question | Expected agreement |
|---|---|---|---|
| Verification | OpenPNM, `pnflow`, XLB/LBM, OpenFOAM-based paper references, manufactured cases | Is the implementation consistent with a software or numerical reference? | Exact to moderate, depending on shared assumptions |
| Validation | Experimental porosity and permeability data | Does the current workflow predict the measured physical response closely enough? | Case-dependent; mismatch often reflects extraction and constitutive-model limits |

## DRP-317 Source Citations

The current validation set uses:

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>

The reproducible software-verification notebook artifacts are:

- `notebooks/12_mwe_synthetic_volume_openpnm_benchmark.ipynb`
- `notebooks/15_mwe_external_pnflow_benchmark.ipynb`
- `notebooks/13_mwe_synthetic_volume_xlb_benchmark.ipynb`
- `notebooks/29_mwe_drp443_ifn_raw_porosity_perm.ipynb`
- `notebooks/30_mwe_drp443_dilatedifn_raw_porosity_perm.ipynb`
- `notebooks/31_mwe_drp10_estaillades_raw_porosity_perm.ipynb`
