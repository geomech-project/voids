# Finite Elements

The `voids.fem` sub-package provides optional FEniCSx-backed finite-element
single-phase solvers for porosity/permeability maps. These APIs require a
compatible DOLFINx installation, such as the Pixi `fem` feature in this
repository. The PyPI package does not install FEniCSx automatically.

The current single-phase FEM backends report effective permeability from the
computed outlet flux and Darcy's law. They are numerical upscaling tools, not
experimental validation claims by themselves.

The default `FEniCSSolverOptions` use PETSc LU with MUMPS. For high-contrast
mixed Darcy-Brinkman maps, avoid treating PETSc's built-in LU backend as a
scientific fallback unless it has been checked against a robust factorization
package on the same problem class; it can return nonphysical permeabilities for
these saddle-point systems.

The USFEM backend is especially sensitive to the external factorization package
and workspace settings on larger 3-D maps. Record the PETSc options from the
result metadata with any reported USFEM permeability.

!!! warning "Thread environment for FEM solves"

    For robust FEniCSx/PETSc direct solves, pin BLAS/OpenMP thread variables
    before Python imports NumPy, SciPy, PETSc, or DOLFINx. `voids.fem` applies
    conservative defaults when those variables are unset, but it does not
    override user-provided values. See the detailed warning in
    [Map-Based Single-Phase Solvers](../map_based_singlephase_solvers.md#fem-map-problem).

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
