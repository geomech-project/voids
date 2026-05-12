# Mesh Export

The `voids.mesh` sub-package converts regular porosity and permeability maps
into structured mesh files for downstream continuum workflows, including
quadrilateral/triangular 2-D exports and hexahedral/tetrahedral 3-D exports.

These helpers preserve the map grid and cell ordering; they do not generate a
boundary-conforming mesh of the original segmented pore/bone interface. For the
map definitions, schemes, Kozeny-Carman closure, export assumptions, and
solver-facing caveats, see [Porosity Maps](../porosity_maps.md).

## API

::: voids.mesh
