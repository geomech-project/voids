# Map-Based Single-Phase Solvers

This page documents the map-based single-phase solvers introduced in
`voids.fvm`, `voids.fem`, and `voids.lbm`. These methods operate directly on
regular image-derived coefficient maps or binary images, rather than on a pore
network extracted from the image.

The methods are useful for direct-image upscaling and for comparing
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

### Linear Solver Controls

`solve_tpfa` accepts the same sparse solver controls as
`voids.linalg.solve.solve_linear_system`. A direct sparse solve is the default
for small maps and reproducibility tests. For larger 3-D maps, the comparison
notebooks use conjugate gradients with PyAMG preconditioning, for example
`solver_method="cg"` with `{"preconditioner": "pyamg"}`. This changes only the
linear algebra backend, not the TPFA balance equations above, but the tolerance,
preconditioner family, and residual should be reported with performance and
permeability results.

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
Pixi `fem` stack. Solver options are exposed through `FEniCSSolverOptions`, and
the effective PETSc options are stored in the returned result metadata. On
high-contrast mixed Darcy-Brinkman systems, robust external factorization
packages such as MUMPS or SuperLU_DIST should be preferred for scientific
comparisons. PETSc's built-in LU backend can be useful for small diagnostics,
but it should not be accepted as a fallback for permeability estimates unless it
has been checked against the same formulation with a trusted factorization
package.

!!! warning "FEniCSx and direct-solver thread settings"

    Sparse direct solvers used through PETSc can become slow or unstable when
    the FEniCSx process inherits aggressive BLAS/OpenMP threading. For
    reproducible FEM runs, especially three-dimensional USFEM or Taylor-Hood
    Brinkman solves, pin the numerical thread environment before Python imports
    NumPy, SciPy, PETSc, or DOLFINx:

    ```bash
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    ```

    `voids.fem` applies these conservative defaults on import when the
    variables are unset, but it deliberately does not override user-provided
    values. In scripts, set them in the shell before launching Python; in
    notebooks, start Jupyter from an environment with these values or restart
    the kernel before importing numerical packages. Record the PETSc
    factorization backend and these thread settings with any reported FEM
    permeability values.

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

The result should be interpreted as a map-based micro-continuum upscaling
estimate, not as a pore-network solve.

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
\begin{aligned}
a_{\mathrm{USFEM}}
&= a_{\mathrm{Brinkman}}
+ \sum_{f\in\mathcal{F}_{\mathrm{int}}}
\int_f \tau_f [\![p]\!][\![q]\!]\,ds \\
&\quad
- \sum_{T\in\mathcal{T}_h}
\int_T \tau_T \mathbf{R}_u(\mathbf{u},p)
\cdot\mathbf{R}_v(\mathbf{v},q)\,dx .
\end{aligned}
\]

Here \([\![\cdot]\!]\) denotes the jump across an interior face.

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

USFEM systems can be more sensitive to sparse direct solver workspace,
threading, and factorization details than the Taylor-Hood systems. For large
heterogeneous three-dimensional maps, treat unsuccessful factorization, fallback
to a different package, or unusually small/negative permeabilities as a
numerical diagnostic, not as a physical prediction. A full-size USFEM row should
therefore be reported only when the exact solver backend, thread settings,
workspace options, and convergence/failure diagnostics are recorded.

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

### Recommended Stokes-Limit Controls

For permeability estimates intended to approximate steady creeping flow,
`XLBOptions.steady_stokes_defaults()` currently uses

| Option | Value |
|---|---:|
| `lattice_viscosity` | `0.10` |
| `pressure_drop_lattice` | `6.667e-5` |
| `inlet_outlet_buffer_cells` | `12` |
| `max_steps` | `8000` |
| `min_steps` | `1200` |
| `check_interval` | `100` |
| `steady_rtol` | `1.0e-4` |

These values were selected from the DRP-317 same-ROI sensitivity study as a
conservative default for the current BGK, pressure-BC, halfway-bounce-back XLB
adapter. The pressure-drop sweep was Darcy-linear to within about 0.5 % on the
representative axes; increasing the reservoir from 6 to 12 cells changed the
estimate by about 1-2 %, while increasing it to 18 cells changed it by less than
another 1 %. Varying the BGK lattice viscosity changed the permeability by about
4-5 %, so \(\nu_{\mathrm{lu}}=0.10\) is retained as a middle-of-range numerical
choice rather than fitted to experiment.

The validation study is documented in
[DRP-317 LBM default sensitivity](validation/drp317_lbm_sensitivity.md). The
important caveat is that this preset improves numerical defensibility but does
not calibrate the direct-image LBM result to a bulk experimental permeability.

### Meaning of the LBM Options

LBM permeability estimates are sensitive to unit conversion, pressure driving,
wall treatment, and stopping criteria. The options below should therefore be
treated as part of the numerical model and recorded with every result.

| Option | Meaning | What To Monitor | Typical Tuning |
|---|---|---|---|
| `formulation` | Interpretation label for the XLB run. `steady_stokes_limit` still uses the incompressible Navier-Stokes LBM stepper, but with conservative low-Mach controls. | `max_mach_lattice`, `reynolds_voxel_max`, pressure-drop linearity. | Use `XLBOptions.steady_stokes_defaults()` for permeability studies unless deliberately testing the generic transient preset. |
| `pressure_drop_lattice` | Imposed pressure difference in lattice pressure units, \(\Delta p_{\mathrm{lu}}\). If explicit inlet/outlet pressures are not given, it is applied relative to `reference_density_lattice`. | Permeability invariance under half/double pressure drop; maximum Mach number; voxel Reynolds diagnostic. | Decrease it if permeability changes with driving or Mach/Re become too large. Do not increase it to make the run faster unless Darcy-linearity is rechecked. |
| `pressure_inlet_lattice`, `pressure_outlet_lattice` | Direct pressure boundary values in lattice units. They override the pressure-drop construction when both are supplied. | Positive pressures and the resolved \(\Delta p_{\mathrm{lu}}\). | Prefer these only when reproducing a specific lattice setup or when benchmark coupling supplies an explicit physical pressure conversion. |
| `rho_inlet`, `rho_outlet` | Legacy density-based boundary inputs. They are converted with \(p_{\mathrm{lu}}=c_s^2\rho_{\mathrm{lu}}\). | Consistency with any pressure inputs. | Prefer pressure inputs for new studies; densities are mainly backward-compatible aliases. |
| `reference_density_lattice` | Baseline density used to construct the outlet pressure when only `pressure_drop_lattice` is supplied. | Positive resolved pressures and a small density jump. | Keep the default for ordinary studies. Changing this is a pressure gauge choice unless it affects numerical positivity. |
| `lattice_viscosity` | Kinematic viscosity in lattice units, \(\nu_{\mathrm{lu}}\). It sets the BGK relaxation \(\omega=(3\nu_{\mathrm{lu}}+1/2)^{-1}\). | Sensitivity of \(K\) to relaxation time; stability; velocity field smoothness near voxel walls. | Use `0.10` as the default. Sweep values such as `0.05`, `0.10`, and `1/6` to estimate BGK/wall-location sensitivity. Do not fit this value to experiment. |
| `inlet_outlet_buffer_cells` | Number of fully fluid reservoir layers inserted before the sample inlet and after the sample outlet. | Change in \(K\) as the buffer is increased; pressure and velocity fields near inlet/outlet planes. | Increase from 6 to 12 to 18 or 24 until \(K\) is stable within the study tolerance. Larger buffers cost memory and runtime. |
| `max_steps` | Hard cap on LBM time steps. | `converged`, `n_steps`, `convergence_metric`, and whether \(K\) is still drifting. | Increase if the run stops at `max_steps`; report non-converged estimates as numerical diagnostics, not final predictions. |
| `min_steps` | Minimum time steps before the steady-state criterion is allowed to stop the run. | Early stopping and transient pressure/velocity development. | Increase for larger domains or slow axes so the convergence check does not accept a premature transient plateau. |
| `check_interval` | Number of time steps between convergence checks. | Runtime overhead versus resolution of the convergence history. | Keep near 100 for most studies; reduce only when diagnosing short runs. |
| `steady_rtol` | Relative tolerance on the change in mean superficial velocity. | Final convergence metric and sensitivity of \(K\) to tighter tolerances. | Use `1e-4` as a default; test `5e-5` or `1e-5` for a publication-quality sensitivity check. |
| `precision_policy` | XLB precision policy, currently passed to the JAX backend. | Reproducibility, memory use, and whether small pressure drops remain resolved. | Use the default unless testing precision sensitivity. Higher precision may improve robustness but increases memory and runtime. |
| `collision_model` | Collision operator label passed to XLB. The current default is BGK. | Sensitivity of \(K\) to wall treatment and relaxation model. | Keep fixed for validation rows. If XLB exposes TRT/MRT in the installed stack, compare them as a model study rather than a silent default change. |
| `streaming_scheme` | XLB streaming scheme label. The current default is pull streaming. | Reproducibility with the backend and collision model. | Keep fixed unless comparing LBM discretization choices explicitly. |
| `backend` | XLB compute backend. The current `voids` adapter supports `jax`. | Backend version, CPU/GPU device, and precision behavior. | Backend changes should affect performance more than physics, but should still be recorded for reproducibility. |

The pressure drop is small in lattice units by design. The current Stokes-limit
default corresponds to a density jump of \(2\times10^{-4}\), well below the
adapter's warning threshold for weakly compressible pressure forcing. This is a
numerical sanity condition, not a physical density contrast in the rock.

### Tuning Workflow for a New Sample

For a new segmented image, start by treating the LBM row as a DNS-style
reference calculation whose setup must be verified. A useful sequence is:

1. Verify the image convention and connectivity. The input must be `void=1`,
   `solid=0`; the pore space should connect the selected inlet and outlet
   faces, and isolated artifacts should be documented rather than silently
   edited away.
2. Run one representative axis with `XLBOptions.steady_stokes_defaults()`.
   Record permeability, porosity, ROI origin and shape, voxel size, `n_steps`,
   `converged`, `convergence_metric`, `max_mach_lattice`, and
   `reynolds_voxel_max`.
3. Test pressure-drop linearity. Repeat the same run with half and double
   `pressure_drop_lattice`. A Stokes/Darcy estimate should give nearly the
   same permeability. If the value changes materially, reduce the pressure
   drop and inspect the pressure and velocity fields.
4. Test reservoir length. Repeat the run with 6, 12, 18, and, if feasible, 24
   `inlet_outlet_buffer_cells`. Choose the smallest buffer for which the
   permeability and near-boundary fields are stable at the tolerance required
   by the study.
5. Test steady-state tolerance. Tighten `steady_rtol`, increase `min_steps`,
   and increase `max_steps`. The final permeability should not depend strongly
   on the stopping rule.
6. Test BGK relaxation sensitivity. Sweep `lattice_viscosity` across a small
   physically reasonable numerical range, for example 0.05, 0.10, and 1/6. The
   spread is a numerical uncertainty estimate for the current BGK,
   halfway-bounce-back, voxel-staircase setup.
7. Repeat across all flow axes and, when possible, across multiple ROIs. A
   single small ROI can overrepresent a connected high-permeability channel or
   miss lower-permeability portions of the sample.
8. Compare against independent upscaling methods and experiment only after the
   numerical checks pass. Keep directional permeabilities separate from scalar
   experimental bulk values unless a stated averaging rule is used.

For example, a small pressure and buffer sweep can be scripted from the package
API:

```python
from voids.lbm.singlephase.stokes import solve_binary_volume_stokes, steady_stokes_options

base = steady_stokes_options()
configs = {
    "default": base,
    "half_dp": steady_stokes_options(
        pressure_drop_lattice=0.5 * base.pressure_drop_lattice,
    ),
    "double_dp": steady_stokes_options(
        pressure_drop_lattice=2.0 * base.pressure_drop_lattice,
    ),
    "buffer_18": steady_stokes_options(inlet_outlet_buffer_cells=18),
}

for label, options in configs.items():
    result = solve_binary_volume_stokes(
        phases,
        voxel_size=voxel_size,
        flow_axis="x",
        options=options,
    )
    print(label, result.permeability, result.converged, result.convergence_metric)
```

### Diagnostics to Report

A defensible LBM permeability table should include more than the final \(K\).
At minimum, report:

- sample name, segmentation convention, ROI origin, ROI shape, and voxel size,
- flow axis, porosity, pressure drop, resolved inlet/outlet lattice pressures,
  and equivalent density jump,
- `lattice_viscosity`, \(\omega\), collision model, streaming scheme, precision
  policy, backend, and backend version,
- `inlet_outlet_buffer_cells`, `n_steps`, `max_steps`, `min_steps`,
  `steady_rtol`, `converged`, and `convergence_metric`,
- maximum Mach number and voxel-scale Reynolds diagnostic,
- wall time and device information when comparing performance,
- pressure and velocity field slices, and full field exports when the fields
  will be inspected downstream.

The field plots are not decorative. They can reveal boundary-condition
artifacts, channelized flow through a tiny connected throat, disconnected
regions, or sign/axis mistakes that a scalar permeability alone can hide.

### What Can Improve LBM Results

If an LBM result disagrees with experiment, the first response should be a
controlled numerical and image-representativeness study, not parameter fitting.
The most useful improvements are usually:

- larger and more representative ROIs, or multiple ROIs with uncertainty bars;
- finer voxel resolution when the pore-wall geometry is under-resolved;
- careful segmentation review, especially near narrow throats and clay or
  microporous regions that may be unresolved by the binary image;
- longer inlet/outlet reservoirs or alternative boundary treatments when
  pressure and velocity fields show inlet/outlet artifacts;
- lower pressure driving if pressure-drop linearity fails;
- longer runs and tighter convergence tolerances when the superficial velocity
  is still drifting;
- precision and device studies when the selected pressure drop is very small or
  the domain is large;
- future TRT/MRT collision-model comparisons if the installed XLB stack exposes
  those models robustly.

The `lattice_viscosity`, collision model, buffer length, and pressure drop can
all change the numerical answer, but they have different scientific meanings.
Only pressure-drop and convergence changes are expected to disappear in the
well-resolved creeping-flow limit. Relaxation-model, wall-treatment, and
voxel-resolution sensitivities are part of the discretization uncertainty and
should be reported as such.

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
`min_steps`. If the run reaches `max_steps` first, the result is returned with
`converged=False` and `voids.lbm.singlephase.XLBConvergenceWarning` is emitted
because the permeability may be biased. Users who need strict production runs
can promote this warning to an exception with Python's `warnings` module.

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
