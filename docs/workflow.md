# Scientific Workflow

This page describes a practical, reproducible workflow for using `voids` in a
research setting.
The emphasis is not just "getting a permeability number", but also recording the
assumptions that make that number interpretable.

For the underlying object model and naming conventions used here, read
[Concepts and Conventions](concepts.md) alongside this page.

---

## Recommended Sequence

For image-based or externally extracted networks, the most defensible order is:

1. define physical units and sample-scale geometry
2. normalize imported geometry into consistent units
3. infer or validate boundary labels
4. attach provenance metadata before solving
5. check porosity and connectivity before transport
6. solve flow and record diagnostics
7. cross-check or serialize the final network

If one of those steps is skipped, the result may still be numerically stable while
being physically ambiguous.

---

## Typical Imported-Network Workflow

```python
from voids import Provenance, SampleGeometry
from voids.benchmarks.crosscheck import crosscheck_singlephase_roundtrip_openpnm_dict
from voids.io.porespy import (
    ensure_cartesian_boundary_labels,
    from_porespy,
    scale_porespy_geometry,
)
from voids.physics.petrophysics import (
    absolute_porosity,
    connectivity_metrics,
    effective_porosity,
)
from voids.physics.singlephase import FluidSinglePhase, PressureBC, solve

voxel_size = 5e-6
nx, ny, nz = 256, 256, 256

raw = ...  # PoreSpy/OpenPNM-style mapping with pore.coords and throat.conns

scaled = scale_porespy_geometry(raw, voxel_size=voxel_size)
labeled = ensure_cartesian_boundary_labels(scaled, axes=("x",), tol_fraction=0.05)

sample = SampleGeometry(
    voxel_size=voxel_size,
    bulk_shape_voxels=(nx, ny, nz),
    lengths={"x": nx * voxel_size},
    cross_sections={"x": ny * nz * voxel_size**2},
    units={"length": "m", "pressure": "Pa"},
)

provenance = Provenance(
    source_kind="porespy",
    extraction_method="snow2",
    voxel_size_original=voxel_size,
    segmentation_notes="Record thresholding and cleanup choices here.",
)

net = from_porespy(labeled, sample=sample, provenance=provenance)

phi_abs = absolute_porosity(net)
phi_eff = effective_porosity(net, axis="x")
conn = connectivity_metrics(net)

result = solve(
    net,
    fluid=FluidSinglePhase(viscosity=1.0e-3),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
    axis="x",
)

crosscheck = crosscheck_singlephase_roundtrip_openpnm_dict(
    net,
    fluid=FluidSinglePhase(viscosity=1.0e-3),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
    axis="x",
)

print(phi_abs, phi_eff)
print(conn)
print(result.permeability["x"], result.mass_balance_error)
print(crosscheck.permeability_rel_diff, crosscheck.total_flow_rel_diff)
```

For most image-based permeability studies, a constant viscosity such as `1.0e-3 Pa s`
remains the defensible first choice. It isolates geometry and topology effects and
keeps the pressure solve linear.

---

## What To Check Before Solving

Use the following checks before interpreting any transport result quantitatively:

| Check | Why it matters |
|---|---|
| `SampleGeometry.lengths` and `cross_sections` are populated | Darcy-scale permeability depends directly on these values |
| `voxel_size` reflects the real acquisition scale | A unit error propagates into all geometric quantities |
| inlet/outlet labels match the intended flow axis | A mislabeled boundary can produce a plausible but wrong solution |
| porosity is in a physically credible range | Large discrepancies often indicate double counting or scaling errors |
| connectivity is inspected explicitly | Disconnected void clusters can inflate pore volume without contributing to flow |

---

## Common Failure Modes

### Unit Drift

If the imported dictionary is still in voxel units while `SampleGeometry` is in SI
units, the resulting permeability will be inconsistent by powers of voxel size.
`scale_porespy_geometry` should usually be applied exactly once.

### Boundary Heuristics

`ensure_cartesian_boundary_labels` is convenient, but it assumes the sample axes are
aligned with the coordinate frame and that a simple geometric tolerance is adequate.
That assumption can be incorrect for cropped, rotated, or irregular domains.

### Apparent Agreement

A low solver residual does not guarantee the imported network is scientifically valid.
It only means the linear system was solved consistently for the supplied graph and
geometry.

### Cross-Check Interpretation

The OpenPNM-style roundtrip cross-check tests representation consistency, not ground
truth. Agreement after roundtripping is reassuring, but it does not validate the
segmentation or extraction itself.

### Thermodynamic Viscosity

If the scientific question genuinely depends on \(\mu(P, T)\), the workflow changes in
two important ways:

1. boundary pressures must be specified as positive absolute pressures
2. the flow solve becomes nonlinear because conductance depends on the evolving
   pressure field

A representative pattern is:

```python
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve
from voids.physics.thermo import TabulatedWaterViscosityModel

mu_model = TabulatedWaterViscosityModel.from_backend(
    "thermo",
    temperature=298.15,
    pressure_points=128,
)

result = solve(
    net,
    fluid=FluidSinglePhase(viscosity_model=mu_model),
    bc=PressureBC("inlet_xmin", "outlet_xmax", pin=8.0e6, pout=5.0e6),
    axis="x",
    options=SinglePhaseOptions(
        conductance_model="valvatne_blunt",
        nonlinear_solver="newton",
        solver="gmres",
        solver_parameters={"preconditioner": "pyamg"},
    ),
)
```

This mode should be justified, not used automatically. If the expected pressure
variation in viscosity is negligible over the imposed pressure window, constant
viscosity is scientifically cleaner and numerically cheaper.

---

## Where To Go Next

- Use [Getting Started](getting_started.md) for installation and the minimal synthetic example.
- Use [Concepts and Conventions](concepts.md) for the canonical schema, unit semantics, and label conventions.
- Use [Examples](examples.md) to pick a notebook matching your workflow.
- Use [Theoretical Background](background.md) when you need the governing equations and assumptions.
- Use [API Reference](api/index.md) for the exact callable interfaces.
