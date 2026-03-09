# Theoretical Background

This page gives a concise overview of the theoretical foundations behind `voids`.
Each section corresponds to a scope boundary or a specific physics module in the code.

---

## Pore Network Modeling

A pore network model (PNM) represents a porous medium as a graph

\[
G = (V, E),
\]

where **pores** are the vertices \(V\) (representing void chambers) and
**throats** are the edges \(E\) (representing the narrow channels connecting pores).
Each pore \(i\) is associated with a centroid coordinate \(\mathbf{x}_i\) and geometric
descriptors such as an inscribed diameter or equivalent sphere radius.
Each throat \(k\) connecting pores \(i\) and \(j\) is described by a length \(L_k\)
and a characteristic diameter \(d_k\).

!!! note "Scope boundary"
    `voids` delegates image segmentation and pore/throat extraction to upstream tools
    such as [PoreSpy](https://porespy.org). The canonical representation only enters
    once a `Network` object is constructed or imported.

### Canonical Representation

Within `voids`, the graph is represented explicitly by:

- pore coordinates `pore_coords`
- throat connectivity `throat_conns`
- pore-wise and throat-wise property arrays
- boolean pore and throat labels
- sample-scale geometry and provenance metadata

This separation is not just a software design choice.
It reflects the fact that topology, constitutive geometry, sample geometry, and
workflow provenance play different scientific roles and should be inspected separately.

---

## Hydraulic Conductance

The hydraulic conductance \(g_k\) of throat \(k\) relates volumetric flow rate to
pressure difference:

\[
Q_k = g_k \, (p_i - p_j).
\]

For Hagen–Poiseuille flow in a circular conduit of diameter \(d_k\), length \(L_k\),
and fluid viscosity \(\mu\):

\[
g_k = \frac{\pi d_k^4}{128 \, \mu \, L_k}.
\]

`voids` uses this *generic Poiseuille* model as its default conductance model.
The constitutive choice is explicit: it is stored as
`SinglePhaseOptions.conductance_model` and documented at the call site.

### Equivalent Duct Idea

Image-extracted pores and throats are not literal cylinders.
They are irregular cross-sections that must be reduced to a small number of
geometric descriptors before a pore-network solver can evaluate conductance.

The Valvatne-Blunt style approach used in `voids` starts from the dimensionless
shape factor

\[
G = \frac{A}{P^2},
\]

where \(A\) is the cross-sectional area and \(P\) is the wetted perimeter.
This is the same invariant proposed by Mason and Morrow for irregular triangular
ducts and then used in the Imperial College generalized pore-network workflow.

![Equivalent duct classes from shape factor](assets/valvatne_equivalent_shape_factor.png)

*Equivalent duct concept used by the shape-factor model. The figure is reproduced from
Valvatne, P. H. (2004),* Predictive pore-scale modelling of multiphase flow, *PhD
thesis, Department of Earth Science and Engineering, Imperial College London,
Figure 3.2.*

The key idea is not that the real pore is literally triangular, square, or circular.
Instead, an irregular cross-section is replaced by an **equivalent duct** with the
same shape factor and the same inscribed radius \(r\).
For the equivalent duct family used here,

\[
A = \frac{r^2}{4G},
\]

which is exact for the circle, square, and triangle classes used by `pnflow`.
This relation is implemented directly in
[`voids.geom.hydraulic`] [voids.geom.hydraulic].

For the canonical classes:

- circle: \(G = 1/(4\pi)\)
- square: \(G = 1/16\)
- triangle: \(0 < G \le \sqrt{3}/36\)

Two cautions matter scientifically:

1. Many distinct irregular cross-sections can share the same \((r, G)\), so this is
   a reduced constitutive representation, not a full geometric reconstruction.
2. The square-versus-circle split used in the current implementation follows the
   Imperial College `pnflow` reference code: values with
   \(\sqrt{3}/36 < G < 0.07\) are treated as square-like, and values with
   \(G \ge 0.07\) are treated as circular-like.
   That threshold is a code-level modeling convention rather than a clean theorem
   stated in the primary papers.

### Shape-Aware Single-Phase Conductance

For image-based or otherwise non-circular ducts, `voids` provides a
shape-factor-aware `valvatne_blunt` closure. It classifies each segment as
triangular, square, or circular from its shape factor \(G\), then uses

\[
g_k = \frac{k(G)\, G\, A_k^2}{\mu\, L_k},
\]

with \(k = 3/5\) for triangles, \(k = 0.5623\) for squares, and \(k = 1/2\) for
circles.

The circular case reduces to the usual Hagen-Poiseuille conductance since
\(G = 1/(4\pi)\) implies

\[
g = \frac{1}{2}\frac{GA^2}{\mu L}
  = \frac{\pi r^4}{8 \mu L}.
\]

The coefficients \(3/5\) and \(0.5623\) are the single-phase noncircular
conductance constants cited by Valvatne and Blunt to Patzek and Silin (2001).
This is worth stating explicitly because Patzek's 2001 SPE paper on simulator
verification is a different reference and does **not** provide the one-phase
duct conductance derivation.

### Conduit Decomposition

The full `valvatne_blunt` model in `voids` does not treat a throat as one uniform
resistor from pore center to pore center.
Instead, when conduit sub-lengths are available, each connection is decomposed into:

- a pore-body contribution from pore \(i\) center to a conduit split point on the pore side
- the throat core contribution
- a pore-body contribution from the opposite split point to pore \(j\) center

![Pore-throat-pore conduit decomposition](assets/valvatne_conduit_series_model.svg)

This schematic is adapted from Valvatne's Figure 3.11. The split points correspond
to the pore-throat partition used to define \(L_{p,i}\), \(L_t\), and \(L_{p,j}\).
In the Imperial workflow these lengths are stored explicitly, but the split point
should be read as a conduit bookkeeping location for resistance partitioning, not
as an independently resolved geometric surface.

For one segment \(s\), the hydraulic resistance is

\[
R_s = \frac{\mu L_s}{k_s G_s A_s^2},
\]

so the conduit resistance is

\[
R_{ij} = R_i + R_t + R_j,
\]

and the equivalent conductance is

\[
g_{ij} = \frac{1}{R_{ij}}.
\]

This is exactly the series law implemented in the code through a harmonic
combination of the segment conductances.
In the literature this same idea is often written with length-weighted
conductances between pore centers; the `voids` implementation is algebraically
equivalent, but stores the already length-normalized segment conductances.

There is also no requirement that a pore and its adjacent throat have the same
shape class.
In the current implementation, every pore body and every throat carries its own
element-wise geometry arrays, so a connection can, for example, use a
square-like pore segment on one side, a circular-like throat core, and a
triangular-like pore segment on the other side.

### Model Names In `voids`

`voids` currently exposes three relevant single-phase closures:

- `generic_poiseuille`: circular throat-only model
- `valvatne_blunt_throat`: shape-aware throat-only model
- `valvatne_blunt`: shape-aware pore-throat-pore conduit model, falling back to
  `valvatne_blunt_throat` if conduit sub-lengths are not available

This distinction matters in practice.
A circular `valvatne_blunt_throat` case collapses to the generic Poiseuille model,
but the full `valvatne_blunt` model can still differ because pore-body resistance is
included explicitly.

### Circular Limit Of Throat-Only Versus Full Conduit Models

The circular comparison is subtle and easy to misread.

If a throat is circular and the solver uses only the throat geometry, then

\[
g_{\text{throat-only}} = \frac{\pi r_t^4}{8 \mu L_t},
\]

which is exactly the generic Poiseuille law.
This is why `valvatne_blunt_throat` and `generic_poiseuille` agree in the circular
limit when they are given the same throat radius and the same throat length.

The full conduit model is a different object.
For a circular pore-throat-pore conduit,

\[
\frac{1}{g_{ij}}
= \frac{8\mu}{\pi}
\left(
\frac{L_i}{r_i^4}
+ \frac{L_t}{r_t^4}
+ \frac{L_j}{r_j^4}
\right).
\]

Two limiting cases are useful:

1. If \(r_i = r_t = r_j\) and \(L_i + L_t + L_j = L\), then the full conduit model
   collapses to the same Poiseuille conductance over the total center-to-center
   length \(L\).
2. If the pore radii differ from the throat radius, then the full conduit model
   generally differs from `generic_poiseuille`, even though all three segments are
   circular.

In most realistic networks the pore bodies are larger than the throat core, so the
pore-body resistances are smaller than the throat-core resistance.
In that common case the full conduit model is usually **more conductive** than a
single-throat Poiseuille model that applies the throat radius over the entire
center-to-center length.
However, this is a geometric consequence, not a universal ordering theorem.

### Relation To `pnflow` And `pnextract`

The current `voids` implementation is now close to the Imperial College
single-phase conductance closure, but not yet identical to the entire
`pnextract` → `pnflow` workflow.

What matches well:

- the shape-class logic: triangle, square, circle
- the single-phase coefficients \(3/5\), \(0.5623\), and \(1/2\)
- the harmonic pore1-throat-pore2 series combination when conduit sub-lengths are
  available
- independent pore and throat shape factors at the element level

What can still differ materially:

- `pnextract` geometry generation itself
- `pnextract` shape-factor repair heuristics during export
- availability of conduit sub-lengths in the imported network
- pore shape-factor construction upstream
- boundary and reservoir conventions between codes

In particular, the reference `pnextract` code computes throat shape factors as

\[
G = \frac{r^2}{4A},
\]

then applies additional extraction-stage repairs for problematic values, including:

- clipping or modifying very large throat shape factors
- replacing very small throat shape factors with randomized admissible values
- assigning pore shape factor as a throat-area-weighted average of neighboring
  throat shape factors

`voids` now provides these heuristics through the `imperial_export`
geometry-repair mode in the PoreSpy importer, and the image-extraction workflow enables that
mode by default. Therefore:

- if you provide `voids` with a network already carrying `pnextract`-like
  `shape_factor` and conduit-length fields, or if you extract through the
  default image workflow, the single-phase hydraulic closure should be close to
  `pnflow`
- if the network comes from a different extractor, such as a PoreSpy-based
  workflow, `Kabs` can still differ substantially because topology and geometry
  are already different before the conductance model is even applied

### Implementation Boundary

The present implementation is rigorous for **single-phase** conductance closure,
but it is not yet the full multiphase polygonal model from `pnflow`.
In particular, `voids` does **not** yet evolve:

- corner half-angle distributions
- wetting films in corners
- oil or water layers with separate phase conductances
- capillary-entry and displacement events tied to polygonal corner occupancy

Those are the natural next steps if the project later extends the current
single-phase closure toward the full Valvatne-Blunt multiphase framework.

### References For This Section

- Mason, G., & Morrow, N. R. (1991). Capillary behavior of a perfectly wetting
  liquid in irregular triangular tubes. *Journal of Colloid and Interface Science*,
  141(1), 262-274.
- Patzek, T. W., & Silin, D. B. (2001). Shape factor and hydraulic conductance in
  noncircular capillaries I. One-phase creeping flow. *Journal of Colloid and
  Interface Science*, 236(2), 295-304.
- Valvatne, P. H. (2004). *Predictive pore-scale modelling of multiphase flow*.
  PhD thesis, Department of Earth Science and Engineering, Imperial College London.
- Valvatne, P. H., & Blunt, M. J. (2004). Predictive pore-scale modeling of
  two-phase flow in mixed wet media. *Water Resources Research*, 40(7).

---

## Single-Phase Incompressible Flow

### Governing Equations

At steady state, mass conservation at each free pore \(i\) requires

\[
\sum_{j \in \mathcal{N}(i)} g_{ij} \, (p_i - p_j) = 0,
\]

where \(\mathcal{N}(i)\) is the set of pores connected to \(i\).

Collecting all free-pore equations yields a linear system

\[
\mathbf{A} \, \mathbf{p} = \mathbf{b},
\]

where \(\mathbf{A}\) is the weighted graph Laplacian assembled from throat
conductances, \(\mathbf{p}\) is the unknown pressure vector, and \(\mathbf{b}\)
encodes the Dirichlet boundary contributions.

In expanded graph form, the diagonal and off-diagonal entries satisfy

\[
A_{ii} = \sum_{j \in \mathcal{N}(i)} g_{ij},
\qquad
A_{ij} = -g_{ij} \quad (i \neq j),
\]

for free pores in the active solve domain.

### Boundary Conditions

Fixed pressures are imposed at inlet and outlet pore sets via Dirichlet row/column
elimination:

- \(p_{\text{inlet}} = p_{\text{in}}\)
- \(p_{\text{outlet}} = p_{\text{out}}\)

The current solver uses label-driven Dirichlet conditions.
In practice, that means the physical experiment is defined partly by geometry and
partly by the pore labels attached to the network.

### Active Solve Domain

Connected components that do not touch any fixed-pressure pore are excluded from the
linear solve.
Those components form floating-pressure blocks and would otherwise leave the system
singular or under-determined.

As a result:

- the solve is performed on an induced active subnetwork
- pore pressures outside that subnetwork are reported as `nan`
- throat fluxes outside that subnetwork are reported as `nan`

This is numerically intentional, not an implementation accident.

### Permeability Estimation

Once the pressure field is solved, the total volumetric flow rate \(Q\) is computed
by summing fluxes across a cut perpendicular to the flow direction.

Darcy's law at the sample scale gives the absolute permeability:

\[
K = \frac{Q \, \mu \, L}{A \, \Delta p},
\]

where \(L\) is the sample length along the flow axis, \(A\) is the cross-sectional
area, and \(\Delta p = p_{\text{in}} - p_{\text{out}}\) is the imposed pressure
difference.

The sample geometry (lengths, cross-sections) is stored in the `SampleGeometry`
object attached to the `Network`.

!!! note "Interpretation boundary"
    The permeability reported by `voids` is an apparent sample-scale permeability
    consistent with the supplied network, constitutive conductance model, sample
    lengths, cross-sections, and boundary labels.
    Agreement with another solver or another code path does not by itself validate
    the upstream segmentation or extraction.

---

## Porosity Definitions

### Absolute Porosity

Absolute porosity is the ratio of total void volume to bulk sample volume:

\[
\phi_{\text{abs}} = \frac{V_{\text{void}}}{V_{\text{bulk}}}.
\]

`voids` accumulates \(V_{\text{void}}\) from pore and throat volume fields.
When `pore.region_volume` is available (a disjoint voxel-space partition), it is
used directly and throat volumes are **not** added to avoid double-counting.

### Effective Porosity

Effective porosity considers only the connected void space that participates in
macroscopic transport:

\[
\phi_{\text{eff}} = \frac{V_{\text{connected}}}{V_{\text{bulk}}}.
\]

Two selection rules are available:

1. **Axis-spanning**: include only pores belonging to connected components that
   span both the inlet and outlet labels along the requested axis.
2. **Boundary-connected**: include any component touching an `inlet_*`, `outlet_*`,
   or `boundary` pore label.

The distinction matters when disconnected void clusters are present.
Absolute porosity may remain high even when the transport-relevant connected volume
is much smaller.

---

## Graph Connectivity

`voids` uses standard connected-component analysis on the throat graph to identify
isolated clusters, spanning components, and boundary-connected subsets.
The implementation wraps `scipy.sparse.csgraph.connected_components`.

For transport interpretation, connectivity is not just a descriptive graph property.
It directly controls which parts of the network can participate in sample-scale flow.

Key metrics reported by [`connectivity_metrics`][voids.physics.petrophysics.connectivity_metrics]:

| Metric | Description |
|---|---|
| `n_components` | Total number of connected components |
| `n_isolated_pores` | Number of single-pore components |
| `spanning_components` | Components connecting inlet to outlet |
| `connected_fraction` | Fraction of pores in the largest component |

---

## Assumptions and Limitations

The following assumptions are deliberate and should be stated in any study using `voids`:

1. **Segmentation quality**: extracted-network predictions depend strongly on upstream
   segmentation and extraction quality. `voids` does not validate or correct the
   input geometry.
2. **Incomplete geometry fields**: imported geometry fields may be incomplete or
   model-dependent across tools (PoreSpy, OpenPNM, etc.).
3. **Constitutive models**: `generic_poiseuille` remains a circular simplification.
   The `valvatne_blunt` model improves single-phase conductance for non-circular
   ducts, but full multiphase corner and layer physics are still not implemented.
4. **Scope of OpenPNM cross-checks**: solver/assembly consistency is compared, not
   universal physical truth.
5. **Throat visualization**: pore scalar fields may be arithmetically averaged onto
   throats for visualization; this is a display choice, not a constitutive model.
6. **Sample metadata dependence**: permeability interpretation depends on the
   correctness of sample lengths and cross-sectional areas supplied in
   `SampleGeometry`.

If any of these assumptions are inappropriate for a given study, the corresponding
workflow should be tightened before using results quantitatively.
