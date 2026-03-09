# Concepts and Conventions

This page connects the user-facing workflows in `voids` to the underlying data
model and numerical assumptions.
If you already know the installation steps but still need to understand what the
library expects as input and what its outputs mean, this is the right place to start.

---

## The Core Objects

`voids` is built around three primary records:

| Object | Role |
|---|---|
| `Network` | Topology, coordinates, pore/throat arrays, labels, and extra metadata |
| `SampleGeometry` | Bulk volume, sample lengths, cross-sections, voxel size, and unit metadata |
| `Provenance` | Information about where the network came from and how it was processed |

The top-level import surface reflects that split:

```python
from voids import Network, Provenance, SampleGeometry
```

The important design point is that geometry, transport, and provenance are kept
explicit rather than hidden inside a single opaque object.

---

## What A `Network` Means

At the implementation level, a pore network is treated as a graph

\[
G = (V, E),
\]

with pores as vertices and throats as edges.

The canonical `Network` object stores that graph through:

- `throat_conns`: integer array of shape `(Nt, 2)`
- `pore_coords`: floating-point array of shape `(Np, 3)`
- `pore`: dictionary of pore-wise arrays
- `throat`: dictionary of throat-wise arrays
- `pore_labels`: dictionary of boolean pore masks
- `throat_labels`: dictionary of boolean throat masks

This means that most fields are not hard-coded into the class definition.
Instead, they are arrays attached by name, such as:

- `pore.volume`
- `pore.region_volume`
- `throat.length`
- `throat.diameter_inscribed`
- `throat.hydraulic_conductance`

That makes the schema flexible, but it also means array semantics matter: if a
field is attached under a misleading name or with inconsistent units, the solver
cannot recover the scientific intent automatically.

---

## Hydraulic Geometry Fields

For the single-phase conductance models, the most important geometric fields are:

- `throat.length`
- `throat.area`
- `throat.perimeter`
- `throat.shape_factor`
- `throat.radius_inscribed` or `throat.diameter_inscribed`

and, for the full pore-throat-pore conduit model,

- `pore.area`
- `pore.perimeter`
- `pore.shape_factor`
- `pore.radius_inscribed` or `pore.diameter_inscribed`
- `throat.pore1_length`
- `throat.core_length`
- `throat.pore2_length`

The guiding interpretation is:

- `generic_poiseuille` assumes circular throats
- `valvatne_blunt_throat` treats each throat as an equivalent non-circular duct
- `valvatne_blunt` treats each connection as three resistors in series: pore1,
  throat core, pore2

If `shape_factor` is missing but enough surrogate geometry exists, `voids` may
derive it from `A / P^2` or from the equivalent relation between area and
inscribed size.
If conduit sub-lengths are absent, `valvatne_blunt` falls back to the throat-only
shape-aware model.

The richer theory and derivation are documented in
[Theoretical Background](background.md#single-phase-hydraulic-conductance).

---

## Fluid And Solver Semantics

The single-phase solver accepts two scientifically distinct fluid modes:

- `FluidSinglePhase(viscosity=...)` for a constant-viscosity solve
- `FluidSinglePhase(viscosity_model=...)` for a pressure-dependent thermodynamic solve

The first mode keeps the pressure problem linear. The second mode makes conductance a
function of pressure through the viscosity field and therefore activates a nonlinear
solve (`picard` or `newton`).

Three practical conventions matter:

1. constant viscosity can be dimensionless in toy problems if the full workflow is
   treated consistently
2. thermodynamic viscosity requires positive absolute pressures, typically in Pa
3. the reported permeability still uses a scalar reference viscosity, even when the
   solved flow field uses a spatially varying viscosity

The richer physical and numerical interpretation is documented in
[Theoretical Background](background.md#pressure-dependent-viscosity) and
[Theoretical Background](background.md#nonlinear-single-phase-solve).

---

## Pore And Throat Array Semantics

The most important convention is simple:

- pore arrays must be indexed by pore id
- throat arrays must be indexed by throat id
- label arrays must be boolean masks with the same leading dimension as the
  corresponding pore or throat family

For example:

```python
net.pore["volume"].shape == (net.Np,)
net.throat["length"].shape == (net.Nt,)
net.pore_labels["inlet_xmin"].shape == (net.Np,)
```

Multi-column arrays are allowed when the first dimension still matches the pore
or throat count. Coordinates are the main example:

```python
net.pore_coords.shape == (net.Np, 3)
```

Two-dimensional imported coordinates are embedded into 3-D as `(x, y, 0)` during
import so that downstream code can assume a uniform coordinate shape.

---

## Labels And Boundary Conventions

Boundary conditions in `voids` are label-driven.
The single-phase solver does not infer inlet and outlet sets from geometry at solve
time; it expects them to exist as pore labels.

The canonical Cartesian boundary names are:

| Axis | Inlet label | Outlet label |
|---|---|---|
| `x` | `inlet_xmin` | `outlet_xmax` |
| `y` | `inlet_ymin` | `outlet_ymax` |
| `z` | `inlet_zmin` | `outlet_zmax` |

During import, common aliases such as `left/right`, `front/back`, and `bottom/top`
can be mirrored onto those canonical names.

This convention matters in three places:

1. applying Dirichlet boundary conditions
2. defining axis-spanning connected components
3. computing directional permeability

If the labels are geometrically wrong, the resulting solve can still look stable
while corresponding to the wrong physical experiment.

---

## Units And Scaling

`voids` assumes that arrays entering the canonical model are already in a
self-consistent system of physical units.

There are two common cases:

1. purely synthetic or dimensionless examples used for testing
2. image-based or imported networks scaled into physical units, typically SI

For image-based workflows, the most important rule is:

!!! warning "Scale once"
    Common geometric arrays should usually be converted from voxel units to
    physical units exactly once.
    Applying `scale_porespy_geometry` twice is just as problematic as forgetting
    to apply it at all.

`SampleGeometry` stores the sample-scale metadata needed to convert local throat
fluxes into sample-scale properties:

- `bulk_volume`
- `lengths`
- `cross_sections`
- `voxel_size`
- `units`

Without physically meaningful `SampleGeometry`, permeability values may be
numerically reproducible but physically uninterpretable.

---

## Volume Conventions

Porosity calculations in `voids` deliberately distinguish two cases.

### `pore.region_volume` Available

When `pore.region_volume` exists, it is interpreted as a disjoint partition of the
segmented void domain.
In that case, `voids` uses those pore-region volumes directly and does not add
`throat.volume`, because conduit-style throat volumes may overlap the pore-region
partition and cause substantial double-counting.

### Only `pore.volume` And `throat.volume` Available

When the segmented-region partition is unavailable, `voids` falls back to summing
pore and throat volumes.
That is often acceptable for synthetic examples or conduit-based models, but it is
not interchangeable with voxel-partition volume bookkeeping.

The distinction is scientifically important: two networks can have identical graph
topology and still produce different porosity values depending on how volume fields
were constructed upstream.

---

## Active Solve Domain

The single-phase solver excludes connected components that do not touch any fixed-
pressure pore.
Those components form floating pressure blocks and would otherwise make the system
singular or under-determined.

Practically, this means:

- the solve is performed on the induced subnetwork connected to at least one
  Dirichlet pore
- pressures and fluxes outside that active domain are reported as `nan`
- disconnected void space may still contribute to absolute porosity, depending on
  the selected metric

This is one of the key reasons to inspect connectivity before interpreting transport
results.

---

## Serialization Model

The HDF5 interchange format mirrors the conceptual split in the Python objects:

| HDF5 path | Meaning |
|---|---|
| `/meta` | schema version and provenance |
| `/sample` | sample geometry payload |
| `/network/pore` | pore coordinates and pore-wise arrays |
| `/network/throat` | throat connectivity and throat-wise arrays |
| `/labels` | pore and throat label masks |
| root attribute `extra` | auxiliary JSON-compatible metadata |

This structure is intentionally explicit.
The goal is not maximal compactness, but auditability and predictable roundtrips.

---

## Recommended Mental Model

For most studies, the safest interpretation is:

1. upstream tools generate or extract a candidate network
2. `voids` normalizes that network into a canonical representation
3. `SampleGeometry` and `Provenance` make the physical and procedural assumptions explicit
4. analysis and transport are performed on that explicit record

In other words, `voids` is not trying to hide modeling choices.
It is trying to keep them inspectable.

---

## Where To Go Next

- Use [Getting Started](getting_started.md) for installation and the minimal synthetic example.
- Use [Scientific Workflow](workflow.md) for an end-to-end imported-network workflow.
- Use [Theoretical Background](background.md) for the governing equations and transport assumptions.
- Use [API Reference](api/index.md) when you already know the concept and need the exact callable interface.
