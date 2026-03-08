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

### Boundary Conditions

Fixed pressures are imposed at inlet and outlet pore sets via Dirichlet row/column
elimination:

- \(p_{\text{inlet}} = p_{\text{in}}\)
- \(p_{\text{outlet}} = p_{\text{out}}\)

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

---

## Graph Connectivity

`voids` uses standard connected-component analysis on the throat graph to identify
isolated clusters, spanning components, and boundary-connected subsets.
The implementation wraps `scipy.sparse.csgraph.connected_components`.

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
3. **Constitutive models**: single-phase Poiseuille conductance is a simplification.
   More accurate conductance models for non-circular cross-sections are not yet
   implemented.
4. **Scope of OpenPNM cross-checks**: solver/assembly consistency is compared, not
   universal physical truth.
5. **Throat visualization**: pore scalar fields may be arithmetically averaged onto
   throats for visualization; this is a display choice, not a constitutive model.

If any of these assumptions are inappropriate for a given study, the corresponding
workflow should be tightened before using results quantitatively.
