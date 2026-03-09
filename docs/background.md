# Theoretical Background

This page summarizes the current physical and numerical model implemented in `voids`.
The emphasis is on what the code actually solves today, not on the full space of pore-
network models described in the literature.

The main scientific boundary is simple:

- `voids` is currently a single-phase pore-network code
- hydraulic conductance can be constant-viscosity or pressure-dependent
- geometry can be circular, shape-aware throat-only, or shape-aware pore-throat-pore
- thermodynamic viscosity is available through tabulated `thermo` and `CoolProp` backends

If a study needs corner films, capillary entry, wettability hysteresis, or dynamic
multiphase occupancy, those physics are still outside the present scope.

---

## Pore-Network Representation

A pore network is represented as a graph

\[
\mathcal{G} = (V, E),
\]

where pores are vertices \(V\) and throats are edges \(E\). For a throat \(t\) that
connects pores \(i\) and \(j\), `voids` stores:

- the pore coordinates \(\mathbf{x}_i\), \(\mathbf{x}_j\)
- the connectivity pair \((i, j)\)
- pore-wise and throat-wise geometric arrays
- boundary labels used to define the macroscopic experiment
- sample-scale geometry needed for Darcy-scale reporting

This matters scientifically because topology, conduit geometry, and sample geometry
play different roles:

- topology controls connectivity and admissible flow paths
- local geometry controls conductance
- sample geometry controls the conversion from total flow rate to apparent permeability

`voids` therefore keeps those records explicit in `Network`, `SampleGeometry`, and
`Provenance` instead of collapsing them into one opaque container.

---

## Single-Phase Hydraulic Conductance

### Throat Flux Law

The local constitutive law used throughout `voids` is

\[
q_t = g_t \, (p_i - p_j),
\]

where \(q_t\) is the volumetric flow rate through throat \(t\), \(g_t\) is the
hydraulic conductance assigned to that throat or conduit, and \(p_i - p_j\) is the
pressure drop between its end pores.

The scientific question is therefore how \(g_t\) is modeled from geometry and
viscosity.

### Generic Poiseuille Model

The fallback model is the circular Poiseuille conductance

\[
g_t = \frac{\pi r_t^4}{8 \mu L_t}
    = \frac{\pi d_t^4}{128 \mu L_t},
\]

where \(r_t\) and \(d_t\) are throat radius and diameter, \(L_t\) is throat length,
and \(\mu\) is the dynamic viscosity.

This is exposed in `voids` as `generic_poiseuille`.

It is the correct law for creeping flow in a circular tube, but it is only an
equivalent-duct approximation for image-extracted throats with irregular shape.
That simplification is often acceptable for controlled baselines and regression tests,
but it is not a faithful geometric closure for angular or highly non-circular ducts.

### Shape Factor and Equivalent Ducts

For shape-aware closures, `voids` uses the dimensionless shape factor

\[
G = \frac{A}{P^2},
\]

where \(A\) is cross-sectional area and \(P\) is wetted perimeter. For the circle,
square, and equilateral triangle:

\[
G_{\mathrm{circle}} = \frac{1}{4 \pi}, \qquad
G_{\mathrm{square}} = \frac{1}{16}, \qquad
G_{\mathrm{triangle}} = \frac{\sqrt{3}}{36}.
\]

In the equivalent-duct construction used in the Imperial College workflow, the
cross-section is represented by an admissible duct family with the same inscribed
radius \(r\) and shape factor \(G\). For that family,

\[
A = \frac{r^2}{4G}.
\]

This relation is exact for the canonical circle, square, and triangle classes used by
the Valvatne-Blunt style model.

![Equivalent duct classes from shape factor](assets/valvatne_equivalent_shape_factor.png)

Two cautions are important:

1. The pair \((r, G)\) is a reduced constitutive representation, not a full
   reconstruction of the original cross-section.
2. The square/circle transition used in `voids` follows the historical `pnflow`
   implementation convention. It is a pragmatic modeling rule, not a universal theorem.

### Valvatne-Blunt Throat-Only Model

The throat-only shape-aware model classifies the throat from its shape factor and uses

\[
g_t = \frac{k(G_t)\, G_t\, A_t^2}{\mu_t L_t},
\]

with the single-phase coefficients

\[
k =
\begin{cases}
3/5      & \text{triangular ducts}, \\
0.5623   & \text{square-like ducts}, \\
1/2      & \text{circular ducts}.
\end{cases}
\]

In `voids` this closure is exposed as `valvatne_blunt_throat`.

The circular limit is internally consistent. If \(G = 1/(4\pi)\), then

\[
\frac{1}{2}\frac{GA^2}{\mu L}
= \frac{1}{2}\frac{1}{4\pi}\frac{(\pi r^2)^2}{\mu L}
= \frac{\pi r^4}{8 \mu L},
\]

which recovers the usual Poiseuille conductance.

### Valvatne-Blunt Conduit Model

When conduit sub-lengths are available, `voids` uses a pore-throat-pore series model.
Each connection is decomposed into three segments:

- pore-1 segment of length \(L_{p,1}\)
- throat-core segment of length \(L_t\)
- pore-2 segment of length \(L_{p,2}\)

![Pore-throat-pore conduit decomposition](assets/valvatne_conduit_series_model.svg)

For a segment \(s\), the resistance is

\[
R_s = \frac{\mu_s L_s}{k_s G_s A_s^2},
\]

and the conduit resistance is

\[
R_{ij} = R_{p,1} + R_t + R_{p,2}.
\]

Therefore the equivalent conductance is

\[
g_{ij} = \frac{1}{R_{ij}}.
\]

Equivalently, if each segment conductance is computed first,

\[
\frac{1}{g_{ij}} = \frac{1}{g_{p,1}} + \frac{1}{g_t} + \frac{1}{g_{p,2}},
\]

which is the harmonic series combination implemented in `voids`.

This closure is exposed as `valvatne_blunt`. The older name
`valvatne_blunt_baseline` remains as a backward-compatible alias, not as a separate
physical model.

### Fallback Hierarchy

The model-selection logic used by `voids` is:

1. If `throat.hydraulic_conductance` is already present, trust it.
2. Else, if conduit lengths and shape data are available, use `valvatne_blunt`.
3. Else, if throat-only shape data are available, use `valvatne_blunt_throat`.
4. Else, fall back to `generic_poiseuille`.

That hierarchy is scientifically deliberate. It preserves richer geometric
information when available, while keeping the solver usable on incomplete networks.

---

## Pressure-Dependent Viscosity

### Constant-Viscosity Mode

The simplest fluid model is

\[
\mu(\mathbf{x}) = \mu_0,
\]

with one constant dynamic viscosity applied everywhere. This remains the right choice
for:

- dimensionless toy problems
- permeability comparisons where viscosity variation is negligible
- geometry-focused benchmarks where constitutive complexity would only add noise

### Thermodynamic Backend Mode

`voids` also supports pressure-dependent water viscosity through tabulated backend
calls at fixed temperature:

\[
\mu = \mu(P, T).
\]

Two backend families are currently supported:

- `thermo`, through the `thermo.Chemical(...).ViscosityLiquid` interface
- `CoolProp`, through `CoolProp.CoolProp.PropsSI`

At the code level, the constitutive query is not solved directly at every pore during
every iteration. Instead, for a given boundary-pressure interval
\([P_{\min}, P_{\max}]\) and temperature \(T\), `voids` first tabulates the backend
response on a pressure grid:

\[
\{(P_n, \mu_n)\}_{n=1}^{N}, \qquad
\mu_n = \mu_{\mathrm{backend}}(P_n, T).
\]

The tabulated law is then replaced by a clipped piecewise-cubic Hermite interpolant
(PCHIP):

\[
\hat{\mu}(P; T) = \mathcal{I}_{\mathrm{PCHIP}}
\left(
\operatorname{clip}(P, P_{\min}, P_{\max})
\right).
\]

This matters for two reasons:

1. it avoids repeated expensive backend calls during the nonlinear solve
2. it provides a differentiable constitutive law with an explicit derivative
   \(d\hat{\mu}/dP\)

Outside the tabulated interval, pressure is clipped to the interval bounds rather than
extrapolated. Consequently, the effective derivative is zero outside the tabulated
range.

### Pore and Throat Viscosity Fields

The current solver evaluates viscosity at:

- pore centers for pore-body segments
- midpoint throat pressures for throat-core segments

For a throat connecting pores \(i\) and \(j\),

\[
P_t = \frac{P_i + P_j}{2}.
\]

Then the local viscosities are

\[
\mu_{p,i} = \hat{\mu}(P_i; T), \qquad
\mu_t = \hat{\mu}(P_t; T).
\]

The same midpoint rule is used for the throat derivative:

\[
\frac{\partial \mu_t}{\partial P_i}
= \frac{\partial \mu_t}{\partial P_j}
= \frac{1}{2}\frac{d\hat{\mu}}{dP}(P_t; T).
\]

This is not the only admissible closure, but it is consistent with the current local
pressure representation used in the conduit model.

### Absolute Pressure Requirement

Thermodynamic backends are queried in absolute pressure units. Therefore, unlike a
constant-viscosity solve, the pair \((P_{\mathrm{in}}, P_{\mathrm{out}})\) is not
only a pressure drop; its absolute level matters because the constitutive law depends
on pressure itself.

In practice, this means

- `pin=1.0, pout=0.0` is fine for dimensionless constant-viscosity tests
- positive absolute pressures in Pa are required for thermodynamic viscosity solves

---

## Nonlinear Single-Phase Solve

### Governing Balance

For each free pore \(i\), steady incompressible mass conservation requires

\[
R_i(\mathbf{p}) =
\sum_{j \in \mathcal{N}(i)} q_{ij}(\mathbf{p})
= 0,
\]

with

\[
q_{ij}(\mathbf{p}) = g_{ij}(\mathbf{p}) (p_i - p_j).
\]

If viscosity is constant, then \(g_{ij}\) is constant and the residual is linear in
\(\mathbf{p}\). The usual weighted graph-Laplacian system follows:

\[
\mathbf{A}\mathbf{p} = \mathbf{b},
\]

where, for free pores,

\[
A_{ii} = \sum_{j \in \mathcal{N}(i)} g_{ij},
\qquad
A_{ij} = -g_{ij}.
\]

If viscosity depends on pressure, then \(g_{ij}\) depends on \(\mathbf{p}\) through
\(\mu(\mathbf{p})\), and the problem becomes nonlinear.

### Picard Iteration

The Picard strategy used in `voids` is:

1. guess a pressure field
2. evaluate pore and throat viscosities from that field
3. rebuild the conductance field
4. solve the resulting linear pressure problem
5. repeat until the pressure update is small

In symbols, with iterate \(k\),

\[
\mathbf{A}(\mathbf{p}^{(k)}) \, \mathbf{p}^{(k+1)} = \mathbf{b}(\mathbf{p}^{(k)}).
\]

Picard is robust and remains available as a fallback, but its convergence rate is
typically linear.

### Newton Linearization

The current Newton path in `voids` differentiates the tabulated constitutive law and
assembles the pore-balance Jacobian explicitly.

For one throat \(t = (i,j)\),

\[
q_t = g_t(p_i, p_j) (p_i - p_j).
\]

The local derivatives are

\[
\frac{\partial q_t}{\partial p_i}
= g_t + (p_i - p_j)\frac{\partial g_t}{\partial p_i},
\qquad
\frac{\partial q_t}{\partial p_j}
= -g_t + (p_i - p_j)\frac{\partial g_t}{\partial p_j}.
\]

The pore-balance Jacobian is then assembled from those throat contributions.
This is close to a full constitutive Newton method for the problem actually being
solved, with one important caveat:

!!! note "Important modeling point"
    The Jacobian is exact for the tabulated/interpolated constitutive law
    \(\hat{\mu}(P;T)\), not for the original backend callable itself.
    That is a deliberate approximation layer introduced for speed and numerical
    smoothness.

The Newton step \(\delta \mathbf{p}\) solves

\[
\mathbf{J}(\mathbf{p}^{(k)}) \, \delta \mathbf{p}
= -\mathbf{R}(\mathbf{p}^{(k)}),
\]

followed by a damped update

\[
\mathbf{p}^{(k+1)} = \mathbf{p}^{(k)} + \alpha \, \delta \mathbf{p},
\qquad 0 < \alpha \le 1,
\]

with backtracking if the residual does not decrease.

### Boundary Conditions and Active Domain

Dirichlet boundary conditions are imposed on labeled pore sets:

\[
p = p_{\mathrm{in}} \text{ on inlet pores}, \qquad
p = p_{\mathrm{out}} \text{ on outlet pores}.
\]

Connected components that do not touch any Dirichlet pore are excluded from the active
solve domain. This avoids singular floating-pressure blocks. Pressures and fluxes on
those excluded components are reported as `nan` in the returned result.

---

## Linear Solver Options

The inner linear systems used by the constant-viscosity solve, Picard updates, and
Newton steps can be solved with:

- a sparse direct solve
- conjugate gradients (`cg`)
- GMRES (`gmres`)

`voids` also supports optional algebraic multigrid preconditioning through `pyamg`.
That is a linear algebra acceleration, not a separate physical model.

The most defensible rule of thumb is:

- use `cg` plus `pyamg` for constant-viscosity pressure solves when the system is
  close to symmetric positive definite
- use `gmres` plus `pyamg` for Newton inner solves when pressure-dependent viscosity
  makes the Jacobian less symmetric

The actual speedup is geometry- and conditioning-dependent, so it should be treated as
an empirical numerical option rather than as a universal guarantee.

---

## Darcy-Scale Permeability

After solving the pore pressures, `voids` computes throat fluxes and sums the net
inlet flow rate \(Q\). The reported apparent permeability is then obtained from
Darcy's law:

\[
K = \frac{|Q| \, \mu_{\mathrm{ref}} \, L}{A \, |\Delta P|},
\]

where

- \(L\) is the sample length along the chosen axis
- \(A\) is the sample cross-sectional area normal to that axis
- \(\Delta P = P_{\mathrm{in}} - P_{\mathrm{out}}\)
- \(\mu_{\mathrm{ref}}\) is the scalar reporting viscosity

The reporting convention is:

- use `fluid.viscosity` directly if the user supplied a constant reference viscosity
- otherwise, for thermodynamic viscosity, use the midpoint viscosity over the imposed
  pressure interval

This is a reporting choice. It does not change the solved nonlinear flow field, but it
does affect the permeability value reported from that field.

---

## Porosity and Connectivity

### Absolute Porosity

Absolute porosity is

\[
\phi_{\mathrm{abs}} = \frac{V_{\mathrm{void}}}{V_{\mathrm{bulk}}}.
\]

If `pore.region_volume` is available, `voids` treats it as a disjoint partition of the
void domain and uses it directly. Otherwise, it falls back to summing pore and throat
volumes. Those two bookkeeping conventions are not interchangeable and can differ
materially on extracted networks.

### Effective Porosity

Effective porosity is

\[
\phi_{\mathrm{eff}} = \frac{V_{\mathrm{connected}}}{V_{\mathrm{bulk}}},
\]

where the connected volume may be defined either by axis-spanning components or by
boundary-connected components, depending on the selected mode.

### Connectivity Metrics

Connectivity is not only a graph-theoretic descriptor here. It directly controls:

- which pores contribute to effective porosity
- which components are admitted into the active pressure solve
- whether a reported permeability corresponds to a genuinely spanning flow path

---

## Assumptions and Limitations

The main assumptions that should be stated explicitly in any study using `voids` are:

1. Upstream segmentation and extraction quality dominate the scientific credibility of
   the imported network.
2. Shape-factor closures are equivalent-duct models, not reconstructions of the full
   cross-sectional geometry.
3. Pressure-dependent viscosity is currently pressure-only at fixed temperature during
   a given solve; density/compressibility coupling is not modeled.
4. The thermodynamic nonlinear solve is exact only for the tabulated constitutive law,
   not for the raw backend callable.
5. Apparent permeability depends on the correctness of `SampleGeometry.lengths` and
   `SampleGeometry.cross_sections`.
6. Multiphase polygonal corner physics from the full Imperial College `pnflow`
   framework is not implemented yet.

If any of these assumptions are not acceptable for a given study, the workflow needs
to be tightened before the resulting permeability should be interpreted quantitatively.

---

## References

- Mason, G., and N. R. Morrow (1991). Capillary behavior of a perfectly wetting
  liquid in irregular triangular tubes. *Journal of Colloid and Interface Science*,
  141(1), 262-274.
- Patzek, T. W., and D. B. Silin (2001). Shape factor and hydraulic conductance in
  noncircular capillaries I. One-phase creeping flow. *Journal of Colloid and
  Interface Science*, 236(2), 295-304.
- Valvatne, P. H. (2004). *Predictive pore-scale modelling of multiphase flow*.
  PhD thesis, Imperial College London.
- Valvatne, P. H., and M. J. Blunt (2004). Predictive pore-scale modeling of
  two-phase flow in mixed wet media. *Water Resources Research*, 40(7).
- Blunt, M. J., et al. (2013). Pore-scale imaging and modelling. *Advances in Water
  Resources*, 51, 197-216.
- `thermo` project documentation: <https://thermo.readthedocs.io/>
- CoolProp documentation: <https://coolprop.org/>
