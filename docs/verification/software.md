# Software Verification

This page collects the software-facing benchmark reports used to verify `voids`.

The intent is to answer questions such as:

- does `voids` reproduce known manufactured or benchmark behavior?
- how closely does an extracted-network `voids` workflow track an independent
  reference discretization on the same geometry?
- which discrepancies are likely numerical bugs, and which are expected model
  differences?

Verification is narrower than validation. It does not claim universal physical
truth; it tests whether the implementation behaves consistently against a
software or numerical reference under controlled assumptions.

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

- [DRP-443 fracture-network verification overview](drp443.md)
  This report benchmarks DRP-443 against paper values reported from an LBM
  workflow, so it is treated as numerical-reference verification.

- [DRP-10 Estaillades verification overview](drp10.md)
  This report benchmarks DRP-10 against paper values reported from an OpenFOAM
  workflow, so it is treated as numerical-reference verification.

The current studies answer different questions:

| Reference | Geometry seen by reference | Main question | Expected agreement |
|---|---|---|---|
| OpenPNM | Same extracted network as `voids` | Are export/import, BC handling, and solver assembly consistent? | Machine precision |
| `pnextract` + `pnflow` | Independently extracted pore network | How different is the current `voids` image-to-network workflow from an external PNM workflow? | Moderate mismatch is expected |
| XLB | Original voxel image | How different is extracted-network PNM from a direct-image voxel-scale reference? | Larger morphology-dependent mismatch is expected |
| DRP-443 paper reference | Published LBM simulation outputs | Does the current workflow reproduce paper-scale directional permeability trends on fractured-media volumes? | Moderate mismatch is expected |
| DRP-10 paper reference | Published OpenFOAM simulation outputs | Does the current workflow reproduce paper porosity/permeability on the Estaillades volume? | Moderate mismatch is expected |
