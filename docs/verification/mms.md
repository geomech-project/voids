# Finite-Element Manufactured-Solution Verification

`voids.examples.mms` provides reproducible two- and three-dimensional
manufactured solutions for the constant-coefficient Brinkman problem,

\[
-\nu\Delta\mathbf{u}+\gamma\mathbf{u}+\nabla p=\mathbf{f},
\qquad
\nabla\cdot\mathbf{u}=0
\quad\text{in }(0,1)^d.
\]

The user supplies an exact velocity and pressure; `voids` differentiates the
UFL expressions to manufacture

\[
\mathbf{f}
=-\nu\Delta\mathbf{u}_{\mathrm{exact}}
+\gamma\mathbf{u}_{\mathrm{exact}}
+\nabla p_{\mathrm{exact}}.
\]

The exact velocity is imposed on the complete boundary. Since incompressible
pressure is determined only up to a constant, the reported pressure error is
aligned by subtracting the domain mean of \(p_h-p_{\mathrm{exact}}\).

This is **software verification**, not validation. Recovering the expected
orders detects weak-form, forcing, boundary-condition, finite-element-space,
and stabilization regressions for these controlled problems. It does not show
that a heterogeneous map closure or a particular porous sample is physically
accurate.

## Quick refinement study

```python
from voids.examples.mms import boundary_layer_case_2d, run_mms_convergence
from voids.fem.singlephase import FEniCSSolverOptions

study = run_mms_convergence(
    boundary_layer_case_2d(viscosity=0.1),
    method="usfem_p1dg0",
    resolutions=(8, 16, 32),
    options=FEniCSSolverOptions.superlu_direct(),
    keep_solution=False,
)

for row in study.as_dicts():
    print(row)

study.assert_expected_rates(absolute_tolerance=0.35)
```

The runner supports:

| Method name | Velocity space | Pressure space | Nominal smooth rates \((L^2_u,H^1_u,L^2_p)\) |
|---|---:|---:|---:|
| `taylor_hood` | \([\mathrm{CG}_2]^d\) | \(\mathrm{CG}_1\) | \((3,2,2)\) |
| `usfem_p1dg0` | \([\mathrm{CG}_1]^d\) | \(\mathrm{DG}_0\) | \((2,1,1)\) |
| `usfem_p1dg1` | \([\mathrm{CG}_1]^d\) | \(\mathrm{DG}_1\) | \((2,1,1)\) |

### Stabilization controls and the P1/DG0 high-drag limit

The reaction--diffusion facet coefficient is parameter-free. Its
\(\alpha_f=\sqrt{\gamma_f h_f^2/\nu_f}\) is a nondimensional physical scale,
not the user `alpha_edge` multiplier. `alpha_edge` therefore affects only the
classic and shifted laws. The 3D `face3d` coefficient is likewise fixed by its
reference-face subproblem. A nondefault `alpha_edge` with either
parameter-free law is ignored with a runtime warning.

`tau_factor=0` disables only the cell residual term; it does not disable the
pressure-jump term required by the discontinuous pressure space. For P1/DG0,
the cell contribution changes the reaction coefficient to

\[
\gamma_{\mathrm{eff}}=\gamma(1-\gamma\tau_K).
\]

This explains the high-contrast failure observed in the earlier exploratory
scripts: when \(\gamma\tau_K\) is near one, the method nearly cancels the
physical matrix drag. `tau_gamma_cap=theta`, with \(0<\theta<1\), provides the
explicit bound
\(\gamma_{\mathrm{eff}}\ge(1-\theta)\gamma>0\). The cap is opt-in because it
changes the method; use a mesh-refinement and reference-solution comparison to
justify the selected \(\theta\).

For consecutive mesh levels, the observed order is

\[
r_i =
\frac{\log(e_{i-1}/e_i)}
     {\log(h_{i-1}/h_i)},
\qquad h_i=\frac{1}{n_i}.
\]

The programmed order check is one-sided and uses only the finest pair. It is
an asymptotic diagnostic, not a proof: a coarse sequence, an unresolved
boundary layer, algebraic solver error, or inadequate quadrature can make a
correct method fail it. Conversely, a superconvergent coarse-grid slope is not
rejected.

## Replicate the supplied presentation results

Nominal rates alone are too weak for a presentation replication: a different
case or stabilization can recover the same integer slope. The shipped
replication profiles therefore fix the exact solution, \(\nu\), \(\gamma\),
finite-element pair, facet law, reference-face refinement, complete mesh
sequence, finest-pair rates, and reported finest-mesh values.

```python
from voids.examples.mms import run_presentation_mms
from voids.fem.singlephase import FEniCSSolverOptions

replication = run_presentation_mms(
    "2d_boundary_layer_p1dg1",
    options=FEniCSSolverOptions.superlu_direct(),
)
replication.assert_matches()

for row in replication.comparison.as_dicts():
    print(row)
```

`compare_mms_with_presentation` can also check an already-computed
`MMSConvergenceResult`. It refuses to compare results from a different case,
coefficient pair, method, facet law, or finest mesh pair. This prevents a
coarse trend from being labeled as a reproduction of a finer reported run.
The 2D profiles use the physical interior-edge length for \(h_F\). The 3D
classic, shifted, and reaction-diffusion profiles use the report's
representative triangular-face diameter \(\sqrt{2}/n\). These conventions are
stored as `facet_size_mode` metadata because replacing \(h_F\) by the
neighboring cell diameter measurably changes pressure and divergence even when
the velocity convergence rate looks correct.

The two-dimensional profiles transcribe these supplied report values:

| Profile | Finest pair | Finest raw errors \((L^2_u,H^1_u,L^2_p,\|\nabla\cdot u_h\|)\) | Reported finest-pair rates |
|---|---:|---:|---:|
| `2d_boundary_layer_p1dg0` | \(128\to256\) | \((1.384\,10^{-3},1.119,9.821\,10^{-4},2.041\,10^{-3})\) | \((1.96,0.97,1.02)\) for \(L^2_u,H^1_u,L^2_p\) |
| `2d_boundary_layer_p1dg1` | \(64\to128\) | \((5.577\,10^{-3},2.191,1.085\,10^{-2},6.246\,10^{-2})\) | \(1.866\) for \(L^2_u\), \(1.019\) for \(L^2_p\) |

The supported linear three-dimensional profiles all use
\((4,6,8,10,12,16,20)^3\), \(\nu=10^{-2}\), \(\gamma=1\), and compare the
reported \(16^3\to20^3\) rates:

| Pair | Facet law | \(L^2_u\) | \(H^1_u\) | \(L^2_p\) | raw divergence |
|---|---|---:|---:|---:|---:|
| P1/DG0 | classic | 1.724 | 1.478 | 1.340 | 1.471 |
| P1/DG0 | shifted | 1.044 | 0.875 | 1.202 | 0.801 |
| P1/DG0 | reaction-diffusion | 1.620 | 1.381 | 1.313 | 1.369 |
| P1/DG0 | face3d | 1.968 | 1.082 | 1.174 | 1.020 |
| P1/DG1 | classic | 2.033 | 0.990 | 1.787 | 0.980 |
| P1/DG1 | shifted | 2.074 | 0.991 | 1.890 | 0.976 |
| P1/DG1 | reaction-diffusion | 2.030 | 0.990 | 1.795 | 0.980 |
| P1/DG1 | face3d | 1.976 | 1.012 | 1.099 | 0.919 |

Rate comparisons use an absolute tolerance of \(0.01\). Reported 2D
finest-mesh errors use a 0.5% relative tolerance. These tolerances accommodate
the rounding in the supplied tables while remaining tight enough to detect a
facet-size or formulation change; they are regression gates, not uncertainty
estimates or error bars.

The report also contains higher-order P2/DG2 and P3/DG3 rows, a
degree-aware `face3d-op` experiment, and Raviart--Thomas recovered-velocity
diagnostics. They are not claimed by the current profiles because those
discretizations and conservative recovery are not yet public `voids` FEM
capabilities. Raw and recovered divergence must not be conflated.

## Built-in 2D case

`boundary_layer_case_2d` defines

\[
g(s)=s-
\frac{\exp((s-1)/\nu)-\exp(-1/\nu)}
     {1-\exp(-1/\nu)},
\qquad
\mathbf{u}=(g(y),g(x)),
\qquad
p=x-y.
\]

The velocity is divergence-free because \(u_1\) depends only on \(y\) and
\(u_2\) only on \(x\). The reference value \(\nu=10^{-2}\) produces a thin
exponential boundary layer. A sequence ending at \(64^2\) can still be
pre-asymptotic; the 2D notebook exposes a full sequence through \(128^2\).
The default notebook uses \(\nu=10^{-1}\) so that its quick \(8^2\), \(16^2\),
and \(32^2\) run resolves the layer.

## Built-in 3D case

`bubble_case_3d` uses

\[
B=x^2(1-x)^2y^2(1-y)^2z^2(1-z)^2,
\qquad
\phi=32B(1+x+2y+3z),
\]

\[
\mathbf{u}=
\begin{bmatrix}
3\phi_y-2\phi_z\\
\phi_z-3\phi_x\\
2\phi_x-\phi_y
\end{bmatrix},
\qquad
p=\sin(2\pi x)\sin(\pi y)\sin(\pi z).
\]

Cancellation of mixed derivatives makes \(\nabla\cdot\mathbf{u}=0\), and the
boundary-bubble factor makes the velocity vanish on every cube face.

For the 3D MMS runner, `facet_law="auto"` selects `face3d`. This option computes
a scalar pressure-jump coefficient from the reference-face problem

\[
-\Delta_{\widehat F}\eta+\alpha^2\eta=1
\quad\text{in }\widehat F,
\qquad
\eta=0
\quad\text{on }\partial\widehat F,
\]

where \(\widehat F\) is a right reference triangle,
\(\alpha^2=\gamma s^2/\nu\), \(s=1/n\), and
\(h_f=\sqrt{2}s\). The coefficient is

\[
\tau_f =
\frac{s^2}{\nu h_f}
\frac{1}{|\widehat F|}
\int_{\widehat F}\eta\,d\widehat F .
\]

`face_refinement` controls the small continuous-P1 reference solve and is
recorded in result metadata. This is a numerically verified candidate
three-dimensional extension for the structured MMS meshes; it is not exposed
as a general heterogeneous-map facet law and should not be interpreted as an
established tetrahedral stability theorem.

## Centered-vug benchmark

The same module also provides a pressure-driven centered circular/spherical
vug benchmark. It is **not an MMS case** because no exact flow field is known.
Its purpose is a like-for-like formulation and geometry check:

\[
\gamma_{\mathrm{matrix}}=10^7,\qquad
\gamma_{\mathrm{vug}}=1,\qquad
\nu=10^{-2},\qquad
p_L=1,\quad p_R=-1,
\]

on the unit square or cube with centered vug radius \(0.25\).

Unlike the full-Dirichlet-velocity MMS problem, the vug problem uses natural
pressure-traction data at the inlet and outlet. Those data fix the discrete
pressure level, so no point gauge is imposed during the solve. The returned
pressure field is shifted to zero volume mean only after solving for convenient
comparison and plotting.

`mesh_representation="body_fitted"` uses Gmsh to fragment the domain and
preserve separate matrix, vug, outer-boundary, and internal-interface physical
tags in DOLFINx. `mesh_representation="structured"` instead classifies map
cells by their centers and is useful for representation-sensitivity checks.
The body-fitted benchmark defaults to the reaction-diffusion face law in 2D
and the shifted face law in 3D. It uses `facet_size_mode="facet_measure"`:
physical edge length in 2D and \(\sqrt{|F|}\) in 3D. The 3D quantity is
dimensionally consistent but is not an exact face diameter for a general
triangle.

The low-order P1/DG0 high-contrast row is intentionally retained as a negative
diagnostic: its stabilized volumetric term can nearly cancel matrix drag and
produce a nonphysical flux. A small divergence error would not rescue that
result. Use `tau_gamma_cap` or `tau_factor=0` only as explicit sensitivity
branches and compare them with a stable reference; neither is an automatic
correction. The P1/DG1 and Taylor-Hood rows are the meaningful baseline flux
comparison.
Report-scale 3D comparisons require materially finer meshes than the quick
notebook defaults.

Two report-scale 3D profiles are available through
`run_presentation_vug`: `3d_centered_vug_p1dg1` targets
\(Q_R=2.413\times10^{-7}\), while `3d_centered_vug_taylor_hood` targets
\(Q_R=2.42167\times10^{-7}\). Both use radius \(0.25\) and target mesh size
\(\sqrt{3}/30\). The flux tolerance is 1%; the represented-volume tolerance is
3% because Gmsh versions can produce different, scientifically equivalent
body-fitted tetrahedralizations. Exact tetrahedron count is therefore reported
as provenance but is not a cross-version pass criterion.

## Physical 2D centered-vug upscaling family

`CenteredVugFlowCase2D` turns image-scale metadata and physical coefficients
into a body-fitted, pressure-driven 2D flow case. The default physical
configuration is a square represented by \(500^2\) pixels of width
\(15\,\mu\mathrm{m}\), hence \(L=7.5\,\mathrm{mm}\), with matrix porosity
\(\phi_m=0.2\), matrix permeability \(K_m=200\,\mathrm{mD}\), and a configurable
finite Darcy-Darcy vug closure \(K_v=10^{-8}\,\mathrm{m}^2\). For a requested
vug area fraction \(f_v\), the centered circle radius is

\[
r=L\sqrt{\frac{f_v}{\pi}}.
\]

The strict contained-circle limit is \(f_v<\pi/4\). A zero fraction is a
matrix-only case and does not create an internal Gmsh interface.

`run_centered_vug_flow_case` provides two continuous Taylor-Hood
\([\mathrm{CG}_2]^2\times\mathrm{CG}_1\) branches. The Darcy-Brinkman branch
uses

\[
\nu_{\mathrm{eff}}=
\begin{cases}
\mu/\phi_m,&\text{matrix},\\
\mu,&\text{vug},
\end{cases}
\qquad
\gamma_B=
\begin{cases}
\mu/K_m,&\text{matrix},\\
0,&\text{vug}.
\end{cases}
\]

This zero vug reaction is the drag-free prescription of the layered-domain
Darcy-Brinkman model. The Darcy-Darcy branch omits diffusion and instead uses

\[
\gamma_D=
\begin{cases}
\mu/K_m,&\text{matrix},\\
\mu/K_v,&\text{vug},
\end{cases}
\]

before adding the residual-based variational multiscale (VMS) term

\[
\frac12\left(
-\gamma_D\mathbf v+\nabla q,\,
\tau_M(\gamma_D\mathbf u+\nabla p)
\right),
\qquad
\tau_M=\min\left(\frac{C_Mh^2}{\mu},\frac{C_M}{\gamma_D}\right),
\quad C_M=1.
\]

The solve is nondimensionalized with the matrix Darcy velocity scale and
converted back to SI velocity, pressure, discharge per unit out-of-plane depth,
and permeability. Natural pressure traction is applied at left/right, while
the top and bottom impose zero normal velocity and natural tangential traction.
The defaults are \(p_L=1\,\mathrm{Pa}\) and \(p_R=0\,\mathrm{Pa}\), hence
\(\Delta p=1\,\mathrm{Pa}\). The returned pressure is shifted to zero domain
mean after the solve.

\(K_v\) is a finite high-permeability closure used only by Darcy-Darcy, not an
intrinsic material measurement of an open cavity. Darcy-Brinkman sets
\(\gamma_B=0\) in the vug directly and does not use \(K_v\) there. Changing
\(L\), the Darcy-Darcy \(K_v\), or the pressure boundary convention changes
the modeled physical problem and must be reported with any permeability curve.

The tutorial uses a nearly uniform body-fitted mesh with nominal resolution
100. At the default physical scale its target triangle diameter is about
\(106\,\mu\mathrm{m}\), whereas the matrix Brinkman screening length
\(\sqrt{K_m/\phi_m}\) is about \(1\,\mu\mathrm{m}\). The notebook therefore
checks mesh sensitivity of the integral \(K_{\mathrm{eff}}\) output but does
not claim pointwise resolution of the matrix-vug interface layer.

## Executable notebooks

- [`48_mwe_fem_mms_2d.ipynb`](https://github.com/geomech-project/voids/blob/main/notebooks/48_mwe_fem_mms_2d.ipynb)
  runs every shipped formulation, asserts rates, and plots error curves.
- [`49_mwe_fem_mms_3d.ipynb`](https://github.com/geomech-project/voids/blob/main/notebooks/49_mwe_fem_mms_3d.ipynb)
  verifies the three-dimensional exact case and reference-face coefficient.
- [`50_mwe_fem_centered_vug_benchmark.ipynb`](https://github.com/geomech-project/voids/blob/main/notebooks/50_mwe_fem_centered_vug_benchmark.ipynb)
  generates body-fitted 2D/3D vugs, compares outlet fluxes, and contrasts the
  unsafe default P1/DG0 high-drag branch with explicit \(\tau_K=0\) and
  \(\gamma\tau_K\le0.5\) sensitivity runs.
- [`51_mwe_fem_mms_presentation_replication.ipynb`](https://github.com/geomech-project/voids/blob/main/notebooks/51_mwe_fem_mms_presentation_replication.ipynb)
  runs named target-versus-computed regression profiles. It reproduces the 2D
  equal-order row by default and exposes explicit switches for report-scale 3D
  MMS and vug runs.
- [`53_mwe_body_fitted_2d_centered_vug_upscaling.ipynb`](https://github.com/geomech-project/voids/blob/main/notebooks/53_mwe_body_fitted_2d_centered_vug_upscaling.ipynb)
  configures the physical matrix/vug family, solves both P2/P1 models, plots
  every pressure and velocity field, evaluates separate \(p\), \(u_x\), and
  \(u_y\) profiles on \(y/L=0.5\), checks the 70% case under mesh refinement,
  compares \(K_{\mathrm{eff}}/K_m\), and writes XDMF/HDF5 ParaView files.

The notebooks generate plots from the live solve instead of embedding a static
convergence figure that could become inconsistent with the current solver.

## Scientific lineage

The stabilized generalized-Stokes formulation follows:

- Barrenechea, G. R., and Valentin, F. (2002). An unusual stabilized finite
  element method for a generalized Stokes problem. *Numerische Mathematik*,
  92, 653-677. <https://doi.org/10.1007/s002110100371>
- Pacazuca, J. F., Valentin, F., and Volpatto, D. (2026). A Locally Conservative
  Low-Order Stabilized Mixed Finite Element Method for the Brinkman Problem in
  Highly Heterogeneous Porous Media. InterPore 2026 poster.
  <https://doi.org/10.13140/RG.2.2.23699.23840>

The physical vug-upscaling and stabilized Darcy comparison also follow:

- Golfier, F., Lasseux, D., and Quintard, M. (2015). Investigation of the
  effective permeability of vuggy or fractured porous media from a
  Darcy-Brinkman approach. *Computational Geosciences*, 19, 63-78.
  <https://doi.org/10.1007/s10596-014-9448-5>
- Masud, A. (2007). A stabilized mixed finite element method for Darcy-Stokes
  flow. *International Journal for Numerical Methods in Fluids*, 54(6-8),
  665-681. <https://doi.org/10.1002/fld.1508>

The exact fields, mesh sequences, automatic forcing, error norms, reported
target values, and verification assertions above are tracked `voids`
benchmark definitions. Local development reports are not runtime inputs and
are not part of the public package contract.
