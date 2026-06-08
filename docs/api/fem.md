# Finite Elements

The `voids.fem` sub-package provides optional FEniCSx-backed finite-element
single-phase solvers for porosity/permeability maps. These APIs require a
compatible DOLFINx installation, such as the Pixi `fem` feature in this
repository. The PyPI package does not install FEniCSx automatically.

The current single-phase FEM backends report effective permeability from the
computed outlet flux and Darcy's law. They are numerical upscaling tools, not
experimental validation claims by themselves.

For the governing equations, boundary conditions, spaces, stabilization terms,
and permeability reporting convention, see
[Map-Based Single-Phase Solvers](../map_based_singlephase_solvers.md).

---

## Common Types

::: voids.fem.singlephase

---

## Taylor-Hood Backends

::: voids.fem.singlephase.taylorhood

---

## USFEM Backends

::: voids.fem.singlephase.usfem

---

## Upscaling

::: voids.fem.singlephase.upscaling
