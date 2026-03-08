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

- [XLB direct-image permeability benchmark](xlb.md)
  This report includes the implemented LBM formulation, the permeability
  mapping used in `voids`, the shared pressure-drop coupling used between PNM
  and XLB, and a steady Stokes-limit interpretation of the same XLB operator.

The corresponding reproducible notebook artifact is:

- `notebooks/13_mwe_synthetic_volume_xlb_benchmark.ipynb`
