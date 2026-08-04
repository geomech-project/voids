# Mesh Export

The `voids.mesh` sub-package converts regular porosity and permeability maps
into structured mesh files for downstream continuum workflows, including
quadrilateral/triangular 2-D exports and hexahedral/tetrahedral 3-D exports.

These helpers preserve the map grid and cell ordering; they do not generate a
boundary-conforming mesh of the original segmented pore/bone interface. For the
map definitions, schemes, Kozeny-Carman closure, export assumptions, and
solver-facing caveats, see [Porosity Maps](../porosity_maps.md).

The optional Gmsh helpers build in-memory CAD models and preserve physical tags
when converting to DOLFINx. They require both the Gmsh executable and Python
bindings; the centered-vug workflow is described in
[FEM Manufactured-Solution Verification](../verification/mms.md#centered-vug-benchmark).

## API

::: voids.mesh
