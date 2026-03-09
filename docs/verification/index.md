# Verification

This section collects controlled verification studies for `voids`.

The intent is to answer questions such as:

- does `voids` reproduce known manufactured or benchmark behavior?
- how closely does an extracted-network `voids` workflow track an independent
  reference discretization on the same geometry?
- which discrepancies are likely numerical bugs, and which are expected model
  differences?

Verification is not the same as claiming universal physical validity. In this
project, the purpose is narrower and more useful: make assumptions explicit,
quantify disagreement on controlled cases, and keep those checks reproducible as
the code evolves.

Current studies:

- [OpenPNM extracted-network cross-check](openpnm.md)
  This report verifies that `voids` agrees with OpenPNM in the
  machine-precision regime when both solve the same extracted network with the
  same throat conductances and the same pressure boundary conditions.

- [External `pnextract` / `pnflow` benchmark](pnflow.md)
  This report compares `voids` against a fixed external reference dataset
  generated with `pnextract` and `pnflow`, so the mismatch includes extraction
  and constitutive-model differences rather than just solver differences.

- [XLB direct-image permeability benchmark](xlb.md)
  This report includes the implemented LBM formulation, the permeability
  mapping used in `voids`, the shared pressure-drop coupling used between PNM
  and XLB, and a steady Stokes-limit interpretation of the same XLB operator.

The three current studies answer different questions:

| Reference | Geometry seen by reference | Main question | Expected agreement |
|---|---|---|---|
| OpenPNM | Same extracted network as `voids` | Are export/import, BC handling, and solver assembly consistent? | Machine precision |
| `pnextract` + `pnflow` | Independently extracted pore network | How different is the current `voids` image-to-network workflow from an external PNM workflow? | Moderate mismatch is expected |
| XLB | Original voxel image | How different is extracted-network PNM from a direct-image voxel-scale reference? | Larger morphology-dependent mismatch is expected |

The corresponding reproducible notebook artifacts are:

- `notebooks/12_mwe_synthetic_volume_openpnm_benchmark.ipynb`
- `notebooks/15_mwe_external_pnflow_benchmark.ipynb`
- `notebooks/13_mwe_synthetic_volume_xlb_benchmark.ipynb`
