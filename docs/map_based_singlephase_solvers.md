# Map-Based Single-Phase Solvers

This page documents the map-based single-phase solvers introduced in
`voids.fvm`, `voids.fem`, and `voids.lbm`. These methods operate directly on
regular image-derived coefficient maps or binary images, rather than on a pore
network extracted from the image.

The methods are useful for direct-image upscaling and for comparing reduced
pore-network predictions against continuum or voxel-scale references. They are
not automatically more accurate than a pore-network model: their interpretation
depends on the segmentation, coefficient closure, grid resolution, boundary
conditions, and representative-volume behavior.

---

## Shared Geometry and Reporting Convention

The map solvers assume a two- or three-dimensional regular grid. For an input
map with shape \(N_0 \times N_1\) or \(N_0 \times N_1 \times N_2\) and cell size
\(\Delta x_i\), the physical domain length in axis \(i\) is

\[
L_i = N_i \Delta x_i .
\]

For a flow direction \(i\), the transverse cross-sectional area is

\[
A_i = \prod_{j \ne i} N_j \Delta x_j .
\]

In two-dimensional calculations, this area is interpreted per unit
out-of-plane thickness.

All map upscaling methods report the apparent permeability from Darcy's law:

\[
K_i =
\frac{Q_i \, \mu \, L_i}{A_i \, \Delta p},
\qquad
\Delta p = p_{\mathrm{in}} - p_{\mathrm{out}} > 0 ,
\]

where \(Q_i\) is the total outlet flux across the maximum-coordinate face normal
to axis \(i\). This reporting convention is shared across TPFA, FEM, and the
XLB/LBM direct-image adapter.

---

## TPFA Darcy Finite Volume

### Continuous Model

The TPFA backend in `voids.fvm.singlephase.tpfa` solves the scalar Darcy problem
on a cell-wise permeability map \(K(\mathbf{x})\):

\[
\mathbf{u} = -\frac{K}{\mu}\nabla p,
\qquad
\nabla \cdot \mathbf{u} = 0 .
\]

The boundary conditions are:

\[
p = p_{\mathrm{in}} \quad \text{on the inlet face},
\qquad
p = p_{\mathrm{out}} \quad \text{on the outlet face},
\]

and no-flow conditions on all transverse faces:

\[
\mathbf{u}\cdot\mathbf{n}=0 .
\]

### Discrete Unknowns

The unknown is one pressure \(p_c\) per map cell \(c\). The finite-volume balance
for each cell is

\[
\sum_{f \in \partial c} F_f = 0 ,
\]

where \(F_f\) is positive when flow leaves the cell.

For an interior face between cells \(c\) and \(d\), the transmissibility is

\[
T_{cd} =
\frac{A_f}{\mu d_{cd}} K_f,
\qquad
K_f =
\frac{2K_cK_d}{K_c + K_d}.
\]

Here \(A_f\) is the face area and \(d_{cd}\) is the distance between cell
centers. The harmonic face permeability \(K_f\) is the conservative choice for
piecewise constant permeability with normal flow across a face. If either
adjacent cell has zero permeability, the face transmissibility is zero.

The interior-face flux is

\[
F_{c \to d} = T_{cd}(p_c - p_d).
\]

For a Dirichlet pressure boundary at a half-cell distance from the adjacent cell
center,

\[
T_b = \frac{A_f}{\mu(\Delta x_i/2)}K_c,
\qquad
F_{c \to b}=T_b(p_c-p_b).
\]

The assembled sparse linear system is the standard cell-centered TPFA system on
an orthogonal Cartesian grid. It is most appropriate for scalar or grid-aligned
permeability. It is not an MPFA scheme and does not reconstruct multi-point
fluxes for full-tensor permeability or strongly non-orthogonal grids.

### Main Failure Modes

- Disconnected zero-permeability regions can produce a singular pressure
  system.
- A small permeability floor may be numerically useful, but it is also a
  physical modeling assumption and should be reported.
- For strongly anisotropic tensor permeability, TPFA is generally not the right
  discretization.

---

## FEM Map Problem

The FEM backends operate on a `FEMMapProblem`, which contains a scalar
permeability map \(K\), an optional porosity map \(\phi\), and a dynamic
viscosity \(\mu\). The code constructs piecewise constant coefficients:

\[
\gamma = \frac{\mu}{\max(K, K_{\min})},
\qquad
\nu_{\mathrm{eff}} =
\frac{\mu}{\max(\phi,\phi_{\min})}.
\]

The permeability floor \(K_{\min}\) prevents infinite Darcy drag, and the
porosity floor \(\phi_{\min}\) prevents singular effective viscosity. These
floors are numerical and physical modeling choices.

The regular map is meshed as a simplicial domain:

- two-dimensional maps use triangles,
- three-dimensional maps use tetrahedra.

The coefficient maps are sampled at DG0 cell-center locations on that mesh.

The FEM pressure conditions are imposed as pressure traction terms on the inlet
and outlet faces:

\[
\ell(\mathbf{v}) =
- \int_{\Gamma_{\mathrm{in}}} p_{\mathrm{in}}\mathbf{v}\cdot\mathbf{n}\,ds
- \int_{\Gamma_{\mathrm{out}}} p_{\mathrm{out}}\mathbf{v}\cdot\mathbf{n}\,ds .
\]

Transverse side walls impose only zero normal velocity. They do not impose full
no-slip tangential velocity. The pressure field is determined up to a constant,
so the implementation applies one pressure gauge degree of freedom during the
linear solve and then subtracts the volume-mean pressure for the returned
pressure field. The computed velocity and outlet flux are driven by the imposed
pressure drop.

The default PETSc configuration uses direct LU factorization with MUMPS in the
Pixi `fem` stack. Solver options are exposed through `FEniCSSolverOptions`.

---

## Taylor-Hood Darcy-Darcy

The Taylor-Hood Darcy-Darcy backend is a mixed FEM comparison model. It solves

\[
\gamma \mathbf{u} + \nabla p = \mathbf{0},
\qquad
\nabla \cdot \mathbf{u} = 0 .
\]

The weak form is: find \((\mathbf{u},p)\in V_h\times Q_h\) such that

\[
\int_\Omega \gamma \mathbf{u}\cdot\mathbf{v}\,dx
- \int_\Omega p \nabla\cdot\mathbf{v}\,dx
+ \int_\Omega q \nabla\cdot\mathbf{u}\,dx
= \ell(\mathbf{v})
\]

for all test functions \((\mathbf{v},q)\).

The finite-element spaces are Taylor-Hood:

\[
V_h = [\mathrm{CG}_2]^d,
\qquad
Q_h = \mathrm{CG}_1 .
\]

This backend is called "Darcy-Darcy" in the comparison notebooks because it uses
a Darcy drag law everywhere, with a spatially varying permeability map. It does
not include a Brinkman viscous diffusion term.

---

## Taylor-Hood Brinkman

The Taylor-Hood Brinkman backend solves a Darcy-Brinkman micro-continuum model:

\[
-\nabla\cdot(\nu_{\mathrm{eff}}\nabla\mathbf{u})
+ \gamma\mathbf{u}
+ \nabla p
= \mathbf{0},
\qquad
\nabla\cdot\mathbf{u}=0 .
\]

The weak form used in `voids` is

\[
\int_\Omega
\nu_{\mathrm{eff}}\nabla\mathbf{u}:\nabla\mathbf{v}\,dx
+ \int_\Omega \gamma\mathbf{u}\cdot\mathbf{v}\,dx
- \int_\Omega p\nabla\cdot\mathbf{v}\,dx
+ \int_\Omega q\nabla\cdot\mathbf{u}\,dx
= \ell(\mathbf{v}).
\]

The implemented viscous term uses the full gradient
\(\nabla\mathbf{u}:\nabla\mathbf{v}\), not the symmetric strain-rate tensor. The
spaces are again

\[
V_h = [\mathrm{CG}_2]^d,
\qquad
Q_h = \mathrm{CG}_1 .
\]

This is the closest FEniCSx analogue of the Taylor-Hood Brinkman calculations
used in the exploratory notebooks. The result should be interpreted as a
map-based micro-continuum upscaling estimate, not as a pore-network solve.

---

## Stabilized USFEM Brinkman

The USFEM implementation follows the unusual stabilized finite element lineage:
the original scalar advective-reactive-diffusive formulation by Franca and
Valentin, the generalized-Stokes extension by Barrenechea and Valentin, and the
recent locally conservative low-order Brinkman/vug formulation by Pacazuca,
Valentin, and Volpatto. The implementation in `voids` currently covers the
stabilized Brinkman solve and reports the raw FEM velocity field; the local
RT0-style conservative velocity recovery described in the vug reference is a
separate postprocessing step and is not implemented yet.

The USFEM backend uses equal-order-like low-order spaces with a discontinuous
pressure:

\[
V_h = [\mathrm{CG}_1]^d,
\qquad
Q_h = \mathrm{DG}_1 .
\]

It starts from the same Brinkman bilinear form and adds two stabilization terms:

\[
a_{\mathrm{USFEM}}
= a_{\mathrm{Brinkman}}
+ \sum_{f\in\mathcal{F}_{\mathrm{int}}}
\int_f \tau_f \llbracket p\rrbracket\llbracket q\rrbracket\,ds
- \sum_{T\in\mathcal{T}_h}
\int_T \tau_T \mathbf{R}_u(\mathbf{u},p)
\cdot\mathbf{R}_v(\mathbf{v},q)\,dx .
\]

The momentum residuals implemented in the code are

\[
\mathbf{R}_u(\mathbf{u},p)
=
\gamma\mathbf{u}+\nabla p
-\nu_{\mathrm{eff}}\nabla\cdot(\nabla\mathbf{u}),
\]

and

\[
\mathbf{R}_v(\mathbf{v},q)
=
\gamma\mathbf{v}-\nabla q
-\nu_{\mathrm{eff}}\nabla\cdot(\nabla\mathbf{v}).
\]

For a cell diameter \(h_T\), the cell stabilization coefficient is

\[
\tau_T =
\frac{\alpha_\tau h_T^2}
{\gamma h_T^2 \max(1,\mathrm{Pe}_T) + 4\nu_{\mathrm{eff}}/m_T},
\]

with

\[
\mathrm{Pe}_T =
\frac{4\nu_{\mathrm{eff}}}{\gamma h_T^2 m_T}.
\]

If \(\gamma\le 0\), the implementation uses the viscous limiting denominator
\(4\nu_{\mathrm{eff}}/m_T\). The exposed parameters are:

- `tau_factor` for \(\alpha_\tau\),
- `m_t` for \(m_T\), defaulting to \(1/3\),
- `alpha_edge` for the pressure-jump coefficient scale.

For an interior face with averaged face diameter \(h_f\),

\[
\nu_{\max} = \max(\nu_{\mathrm{eff}}^+,\nu_{\mathrm{eff}}^-),
\qquad
\gamma_{\max} = \max(\gamma^+,\gamma^-,0),
\]

\[
\alpha_f =
\sqrt{\frac{\gamma_{\max}h_f^2}{\nu_{\max}}}.
\]

The face coefficient is

\[
\tau_f = \alpha_{\mathrm{edge}}
\frac{h_f}{\nu_{\max}\alpha_f^2}
\left(
1 - \frac{2}{\alpha_f}\tanh\frac{\alpha_f}{2}
\right),
\qquad \alpha_f > 10^{-12}.
\]

For very small \(\alpha_f\), the limiting expression is used:

\[
\tau_f =
\alpha_{\mathrm{edge}}\frac{h_f}{12\nu_{\max}}.
\]

The pressure-jump term is important because the pressure space is discontinuous.
The residual term controls the Darcy-Brinkman momentum residual on each cell.
Changing these stabilization parameters changes the numerical method and should
be reported in any comparison table.

---

## XLB/LBM Direct-Image Stokes-Limit Solver

The LBM namespace `voids.lbm.singlephase.xlb` owns the direct-image XLB adapter.
The convenience function `voids.lbm.singlephase.stokes.solve_binary_volume_stokes`
uses the same backend with conservative steady creeping-flow defaults.

### Binary Image Convention

The adapter expects a binary segmented image with

\[
\text{void}=1,
\qquad
\text{solid}=0 .
\]

The selected flow axis is moved to the leading array dimension. Optional
fluid-buffer cells are added before the inlet and after the outlet. The inlet
and outlet pressure conditions are imposed only on void voxels at the reservoir
faces. Solid voxels and transverse side walls are assigned halfway bounce-back
conditions.

### Lattice Pressure and BGK Relaxation

The current adapter uses XLB's incompressible Navier-Stokes stepper. In the
Stokes-limit preset, the same stepper is run with a smaller pressure drop and
tighter steady-state controls; it is therefore a low-Mach, low-Reynolds
interpretation, not a separate analytical Stokes discretization.

The isothermal lattice pressure relation is

\[
p_{\mathrm{lu}} = c_s^2\rho_{\mathrm{lu}},
\qquad
c_s^2 = \frac{1}{3}.
\]

The public XLB options accept either lattice pressures
\(p_{\mathrm{in,lu}},p_{\mathrm{out,lu}}\), a lattice pressure drop, or legacy
density inputs. The BGK relaxation parameter is

\[
\omega =
\frac{1}{3\nu_{\mathrm{lu}} + 1/2},
\]

where \(\nu_{\mathrm{lu}}\) is the lattice kinematic viscosity.

### Convergence Diagnostic

During the run, the adapter computes the superficial axial velocity profile over
planes normal to the flow axis. The scalar convergence metric is the relative
change in mean superficial velocity:

\[
\epsilon_U =
\frac{|U^{(n)} - U^{(n-m)}|}
{\max(|U^{(n-m)}|,10^{-30})}.
\]

The run is marked converged when this metric falls below `steady_rtol` after
`min_steps`. If the run reaches `max_steps` first, the result is returned with a
warning because the permeability may be biased.

### Permeability Conversion

The direct-image LBM estimate uses the lattice Darcy relation

\[
K_{\mathrm{lu}}
=
\frac{\nu_{\mathrm{lu}} U_{\mathrm{lu}} L_{\mathrm{lu}}}
{\Delta p_{\mathrm{lu}}},
\]

where \(U_{\mathrm{lu}}\) is the superficial velocity and \(L_{\mathrm{lu}}\) is
the sample length in voxels. The physical permeability is then

\[
K_{\mathrm{phys}} = K_{\mathrm{lu}}\Delta x^2 .
\]

Equivalently, because \(L_{\mathrm{phys}}=L_{\mathrm{lu}}\Delta x\), the code
computes

\[
K_{\mathrm{phys}}
=
\frac{\nu_{\mathrm{lu}} U_{\mathrm{lu}} L_{\mathrm{phys}}\Delta x}
{\Delta p_{\mathrm{lu}}}.
\]

The result also records maximum lattice Mach number and a voxel-scale Reynolds
diagnostic:

\[
\mathrm{Ma}_{\max} = \frac{|\mathbf{u}|_{\max}}{c_s},
\qquad
\mathrm{Re}_{\Delta x,\max}
=
\frac{|\mathbf{u}|_{\max}}{\nu_{\mathrm{lu}}}.
\]

These diagnostics are essential when the run is interpreted as a creeping-flow
reference.

### Physical Pressure Coupling for Benchmarks

The benchmark wrapper `voids.benchmarks.xlb.benchmark_segmented_volume_with_xlb`
maps a shared physical pressure drop into lattice units before calling the LBM
solver. With physical voxel size \(\Delta x\), physical density \(\rho\), and
physical dynamic viscosity \(\mu\),

\[
\nu_{\mathrm{phys}}=\frac{\mu}{\rho},
\qquad
\Delta t_{\mathrm{phys}}
=
\frac{\nu_{\mathrm{lu}}\Delta x^2}{\nu_{\mathrm{phys}}},
\]

and

\[
\Delta p_{\mathrm{lu}}
=
\Delta p_{\mathrm{phys}}
\frac{\Delta t_{\mathrm{phys}}^2}{\rho\,\Delta x^2}.
\]

This coupling is used only by the benchmark layer so that the pore-network solve
and the direct-image XLB solve represent the same physical pressure drop.

---

## Choosing Between the Methods

| Method | Input | Main Unknowns | Strength | Main Caveat |
|---|---|---|---|---|
| TPFA Darcy | scalar permeability map | cell pressure | fast conservative Darcy upscaling | scalar/grid-aligned permeability only |
| Taylor-Hood Darcy-Darcy | permeability map | velocity and pressure | mixed FEM comparison to Darcy map flow | no Brinkman diffusion |
| Taylor-Hood Brinkman | porosity and permeability maps | velocity and pressure | stable higher-order Brinkman reference | more expensive, map closure still controls accuracy |
| USFEM Brinkman | porosity and permeability maps | velocity and DG pressure | low-order stabilized Brinkman comparison | stabilization parameters affect results |
| XLB/LBM Stokes limit | binary image | lattice distribution functions | direct-image voxel-scale reference | expensive; must monitor Mach, Reynolds, and convergence |

For scientific reporting, always record:

- input image or map provenance,
- block/coarsening size used to build porosity and permeability maps,
- \(K_{\min}\), \(\phi_{\min}\), and any closure parameters,
- flow axis and pressure drop,
- side-wall and inlet/outlet boundary assumptions,
- solver backend and options,
- convergence diagnostics and runtime.

---

## References and Public Lineage

The Darcy-Brinkman and micro-continuum terminology used here follows the
standard Brinkman extension of Darcy flow and later pore-scale micro-continuum
formulations for image-based porous media simulation. The stabilization and
coefficient choices implemented in `voids` are documented above as package
behavior; for bibliographic details, see the reference list in
[Theoretical Background](background.md).

The three USFEM-specific references used for the stabilized formulations are:

- Franca, L. P., and Valentin, F. (2000). On an improved unusual stabilized
  finite element method for the advective-reactive-diffusive equation.
  *Computer Methods in Applied Mechanics and Engineering*, 190(13-14),
  1785-1800. <https://doi.org/10.1016/S0045-7825(00)00190-0>
- Barrenechea, G. R., and Valentin, F. (2002). An unusual stabilized finite
  element method for a generalized Stokes problem. *Numerische Mathematik*,
  92, 653-677. <https://doi.org/10.1007/s002110100371>
- Pacazuca, J. F., Valentin, F., and Volpatto, D. (2026). A Locally Conservative
  Low-Order Stabilized Mixed Finite Element Method for the Brinkman Problem in
  Highly Heterogeneous Porous Media. InterPore 2026 poster.
  <https://doi.org/10.13140/RG.2.2.23699.23840>
