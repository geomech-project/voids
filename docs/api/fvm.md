# Finite Volumes

The `voids.fvm` sub-package provides finite-volume single-phase solvers on
regular coefficient maps.

For the TPFA balance equations, transmissibilities, boundary conditions, and
permeability reporting convention, see
[Map-Based Single-Phase Solvers](../map_based_singlephase_solvers.md).

---

## TPFA Darcy Flow

::: voids.fvm.singlephase.tpfa

`solve_tpfa` exposes the same sparse linear solver choices used elsewhere in
`voids`: direct SciPy sparse LU, PARDISO where available, CG, and GMRES. The
large image-map comparison notebooks use CG with PyAMG preconditioning when the
map system is symmetric positive definite and large enough that a direct solve
is not the preferred baseline.

---

## Upscaling

::: voids.fvm.singlephase.upscaling
