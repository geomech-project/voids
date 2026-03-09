# Experimental Validation

This section collects benchmarks against experimental data rather than against
other software.

In `voids`, validation asks a different question from software verification:

- software verification checks whether the implementation is numerically
  consistent with a reference workflow
- experimental validation checks whether the full image-to-network workflow
  predicts measured porosity or permeability closely enough to be scientifically
  useful

That means a validation mismatch is not automatically a software bug. It can
come from any part of the workflow, including:

- grayscale preprocessing and segmentation assumptions
- ROI selection and representativeness
- network extraction topology
- pore/throat geometry assignment
- hydraulic conductance closure
- the reduction from a voxel image to a pore-throat graph

Current validation studies:

- [DRP-317 sandstone validation overview](drp317.md)
- [DRP-317 Parker notebook report](drp317_parker.md)
- [DRP-317 Kirby notebook report](drp317_kirby.md)
- [DRP-317 Bandera Brown notebook report](drp317_bandera_brown.md)
- [DRP-317 Berea Sister Gray notebook report](drp317_berea_sister_gray.md)
- [DRP-317 Berea Upper Gray notebook report](drp317_berea_upper_gray.md)
- [DRP-317 Berea notebook report](drp317_berea.md)
- [DRP-317 Castlegate notebook report](drp317_castlegate.md)
- [DRP-317 Buff Berea notebook report](drp317_buff_berea.md)
- [DRP-317 Leopard notebook report](drp317_leopard.md)
- [DRP-317 Bentheimer notebook report](drp317_bentheimer.md)
- [DRP-317 Bandera Gray notebook report](drp317_banderagray.md)

The DRP-317 pages use these cited sources:

- Dataset: Neumann, R., ANDREETA, M., Lucas-Oliveira, E. (2020, October 7).
  *11 Sandstones: raw, filtered and segmented data* [Dataset].
  Digital Porous Media Portal. <https://www.doi.org/10.17612/f4h1-w124>
- Experimental reference paper: Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E.,
  Barbalho, H., Trevizan, W. A., Bonagamba, T. J., & Steiner, M. B. (2021).
  *High accuracy capillary network representation in digital rock reveals permeability scaling functions*.
  *Scientific Reports, 11*, 11370. <https://doi.org/10.1038/s41598-021-90090-0>
