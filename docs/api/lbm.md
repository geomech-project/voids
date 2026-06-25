# Lattice Boltzmann

The `voids.lbm` sub-package provides package-facing LBM namespaces for
direct-image single-phase flow. The current Stokes-limit implementation wraps
the optional XLB adapter owned by `voids.lbm.singlephase.xlb`.

For the binary image convention, lattice pressure relation, convergence metric,
permeability conversion, and guidance on tuning LBM runs for a new sample, see
[Map-Based Single-Phase Solvers](../map_based_singlephase_solvers.md).

---

## XLB Backend

::: voids.lbm.singlephase.xlb

---

## Stokes-Limit XLB Backend

::: voids.lbm.singlephase.stokes
