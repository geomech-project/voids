# voids — Technical Specification v0.1 (Draft)

## Status
Draft specification for **v0.1** of the `voids` scientific Python package.

## Project identity

### Name
**`voids`**

### Mission
`voids` is a **scientific Python package** for pore network modeling (PNM), designed for:

- **research reproducibility**
- **PoreSpy interoperability**
- **validation-first development**
- **backend portability** (NumPy/SciPy first, optional PyTorch/JAX later)
- phased expansion from **single-phase** to **quasi-static multiphase**

---

## Scope for v0.1

### Included
- Network import/normalization from **PoreSpy-derived outputs**
- Internal canonical `Network` data model
- Static petrophysics:
  - absolute porosity
  - effective porosity (connectivity-based; explicit definitions)
  - connectivity descriptors
- Single-phase incompressible flow in a pore network
- Absolute permeability estimation (directional; tensor estimate optional helper)
- Serialization + provenance metadata
- Validation and regression test suite

### Excluded
- GUI / desktop app / web app
- Full multiphase engine (Phase 2+)
- Full autodiff through discrete invasion events
- Network extraction implementation from raw images (delegated to PoreSpy)
- End-user workflow manager / project database

---

## Scientific assumptions, boundaries, and risks

### Core assumptions
1. **Extraction is externalized**
   - Image segmentation and network extraction are upstream (e.g., `porespy`).
   - `voids` operates on extracted network data and metadata.

2. **Single-phase validation is foundational**
   - Geometry + connectivity + single-phase flow are validated before multiphase.

3. **Backend portability is a design goal, not day-1 optimization**
   - v0.1 targets correctness and testability with NumPy/SciPy.

### Risks
1. **Extraction uncertainty contaminates transport predictions**
   - Mitigation: preserve provenance and extraction metadata.

2. **Incomplete geometry fields from extraction pipelines**
   - Mitigation: normalization layer with explicit fallbacks and warnings.

3. **Premature optimization for autodiff/GPU**
   - Mitigation: reference backend first; backend abstraction kept minimal.

---

## Package architecture (planned)

```text
voids/
  pyproject.toml
  pixi.toml
  README.md
  LICENSE
  src/voids/
    __init__.py
    version.py

    core/
      network.py
      sample.py
      provenance.py
      enums.py
      typing.py
      validation.py

    io/
      porespy.py
      hdf5.py
      jsonmeta.py

    graph/
      connectivity.py
      boundary.py
      metrics.py
      incidence.py

    geom/
      derived.py
      shape_factors.py
      hydraulic.py

    linalg/
      assemble.py
      bc.py
      solve.py
      diagnostics.py
      backends.py

    physics/
      petrophysics.py
      singlephase.py
      transport.py

    workflows/
      summarize.py
      run_singlephase.py

    utils/
      logging.py
      checks.py
      seeds.py

  tests/
    test_schema_network.py
    test_porespy_import.py
    test_connectivity.py
    test_porosity.py
    test_singlephase_toy.py
    test_singlephase_scaling.py
    test_bc.py
    test_serialization.py

  docs/
    index.md
    spec_v0_1.md
    examples/
```

---

## Design principles

### 1) Array-first internal representation
Use vectorized/columnar arrays (not object-per-pore) for:
- sparse matrix assembly
- serialization
- backend portability
- performance

### 2) Normalized canonical schema
Imported PoreSpy/OpenPNM-style dictionaries are converted into `voids.core.Network`.

### 3) Validation-first
Every physics module exposes diagnostics (residuals, mass balance, schema checks).

### 4) Provenance is mandatory-ish
Extraction details affect predictions and must be tracked.

---

## Core data model specification

## `Network` (canonical internal representation)

`Network` is the single source of truth for topology + geometry + labels + metadata.

### Required fields
- `Np: int` — number of pores
- `Nt: int` — number of throats
- `throat_conns: int[Nt, 2]`
  - each row is `(i, j)` with `0 <= i,j < Np`
  - no self-connections in v0.1 (`i != j`)
- `pore_coords: float[Np, 3]`
  - 2D inputs are stored with `z = 0`

### Strongly recommended geometric fields
- `pore_volume: float[Np]`
- `throat_volume: float[Nt]`
- `throat_length: float[Nt]`
- `throat_area: float[Nt]` *(or enough geometry to derive conductance)*
- `pore_area: float[Np]` *(optional in v0.1)*
- `pore_shape_factor: float[Np]` *(optional in v0.1)*
- `throat_shape_factor: float[Nt]` *(optional in v0.1)*
- `pore_diameter_inscribed: float[Np]` *(optional)*
- `throat_diameter_inscribed: float[Nt]` *(optional)*

### Optional fields for composite conductance models
- `throat_pore1_length: float[Nt]`
- `throat_core_length: float[Nt]`
- `throat_pore2_length: float[Nt]`
- or a generic segment representation sufficient to compose pore–throat–pore resistance

### Boundary labeling
- `pore_labels: dict[str, bool[Np]]`
  - examples: `inlet_xmin`, `outlet_xmax`, `boundary`, `internal`
- `throat_labels: dict[str, bool[Nt]]` *(optional)*

### Metadata and supporting objects
- `sample: SampleGeometry`
- `provenance: Provenance`
- `schema_version: str`
- `extra: dict[str, Any]` *(optional spillover for imported fields not yet normalized)*

---

## `SampleGeometry` schema

Defines sample-scale geometry required for porosity and permeability.

### Fields
- `voxel_size: float | tuple[float, float, float] | None`
- `bulk_shape_voxels: tuple[int, int, int] | None`
- `bulk_volume: float | None`
- `lengths: dict[str, float]`
  - e.g. `{"x": Lx, "y": Ly, "z": Lz}`
- `cross_sections: dict[str, float]`
  - e.g. `{"x": Ayz, "y": Axz, "z": Axy}`
- `axis_map: dict[str, str] | None`
- `units: dict[str, str]`
  - e.g. `{"length": "m", "pressure": "Pa", "viscosity": "Pa*s"}`

### Rules
- `bulk_volume` must be provided **or** derivable from voxel shape × voxel size **or** lengths.
- Directional permeability requires `lengths[axis]` and `cross_sections[axis]`.

---

## `Provenance` schema

Tracks extraction/segmentation context for reproducibility.

### Fields
- `source_kind: str` (`"porespy"`, `"openpnm"`, `"custom"`, ...)
- `source_version: str | None`
- `extraction_method: str | None` (e.g., `"snow2"`)
- `segmentation_notes: str | None`
- `voxel_size_original: float | tuple[float, float, float] | None`
- `image_hash: str | None`
- `preprocessing_hash: str | None`
- `random_seed: int | None`
- `created_at: str` (ISO timestamp)
- `user_notes: dict[str, Any] | None`

### Rationale
This supports scientific traceability and helps diagnose differences caused by extraction choices.

---

## Validation rules (`voids.core.validation.validate_network`)

### Topology checks
- `throat_conns.shape == (Nt, 2)`
- indices in valid range
- no NaNs in `pore_coords`
- no self-loops in v0.1
- duplicated throats: warning by default (configurable)

### Geometry checks
- positive lengths where present
- nonnegative volumes and areas
- finite numeric arrays
- soft validation for shape-factor bounds (warn unless model requires strict bounds)

### Label checks
- label mask lengths match `Np` / `Nt`
- inlet/outlet overlaps warn (or error for incompatible BC type)

### Sample consistency checks
- porosity requires void volume + bulk volume (direct or derivable)
- permeability along axis requires `L` and `A` for axis

---

## Import and normalization API

### `voids.io.porespy.from_porespy(network_dict, *, sample=None, provenance=None, strict=True) -> Network`

Normalize a PoreSpy-style network dictionary into the `Network` schema.

### Behavior
- Maps known keys to canonical names
- Preserves unknown fields in `Network.extra` (optional)
- Derives selected fields when possible
- Raises on missing required topology fields if `strict=True`
- Emits warnings when recommended geometry fields are unavailable

### Non-goal (v0.1)
- No raw image segmentation or network extraction

---

## Petrophysics API (v0.1)

### `voids.physics.petrophysics.absolute_porosity(net: Network) -> float`

Definition:
\[
\phi_{\mathrm{abs}} = \frac{V_{\mathrm{void}}}{V_{\mathrm{bulk}}}
\]

Where:
- \(V_{\mathrm{void}} = \sum V_p + \sum V_t\) by default
- optional safeguards may avoid double counting if importer provides already partitioned volumes

### `voids.physics.petrophysics.effective_porosity(net: Network, axis: str | None = None, *, mode: str = "spanning") -> float`

Explicit modes (to avoid ambiguity):
- `"connected_boundary"`: connected to any external boundary label
- `"spanning"`: connected component spans inlet→outlet along specified axis

### `voids.physics.petrophysics.connectivity_metrics(net: Network) -> ConnectivitySummary`

Returns metrics such as:
- component count
- giant component fraction
- isolated pore fraction
- dead-end fraction
- coordination distribution stats
- spanning status per axis

---

## Single-phase flow API (v0.1)

### Data classes

#### `FluidSinglePhase`
- `viscosity: float`
- `density: float | None = None` *(future gravity support)*

#### `PressureBC`
- `inlet_label: str`
- `outlet_label: str`
- `pin: float`
- `pout: float`

#### `BodyForce`
- `gravity_vector: tuple[float, float, float] | None`
- `enabled: bool = False`

#### `SinglePhaseOptions`
- `conductance_model: str = "generic_poiseuille"`
- `solver: str = "direct"` (`"direct"`, `"cg"`, `"gmres"`)
- `check_mass_balance: bool = True`
- `gravity: BodyForce | None = None`
- `regularization: float | None = None`

#### `SinglePhaseResult`
- `pore_pressure: float[Np]`
- `throat_flux: float[Nt]`
- `throat_conductance: float[Nt]`
- `total_flow_rate: float`
- `permeability: dict[str, float] | None`
- `residual_norm: float`
- `mass_balance_error: float`
- `solver_info: dict[str, Any]`

---

### `voids.physics.singlephase.solve(...) -> SinglePhaseResult`

Proposed signature:

```python
solve(
    net: Network,
    fluid: FluidSinglePhase,
    bc: PressureBC,
    *,
    axis: str,
    options: SinglePhaseOptions | None = None,
) -> SinglePhaseResult
```

### Mathematical model

For each internal pore \(i\):
\[
\sum_{j \in \mathcal{N}(i)} q_{ij} = 0
\]

Throat flux:
\[
q_{ij} = g_{ij}(F_i - F_j)
\]

Potential interface (gravity-compatible):
\[
F = P - \rho g h
\]

Composite conductance relation (when supported by geometry model):
\[
\frac{1}{g_{ij}} = \frac{1}{g_i} + \frac{1}{g_t} + \frac{1}{g_j}
\]

### Sparse assembly
Assemble pore-level system \(A x = b\):
- diagonal entries: sum of connected conductances
- off-diagonals: negative conductances
- BC imposition:
  - v0.1 default: Dirichlet via row/column modification (documented and tested)
  - elimination kept as future option

### Postprocessing
- per-throat fluxes
- inlet total flow \(Q\)
- mass-balance diagnostics
- directional permeability

Permeability (magnitude convention):
\[
K = \frac{|Q| \mu L}{A \Delta P}
\]

---

## Conductance model interface specification

### Protocol (conceptual)
```python
class ConductanceModel(Protocol):
    name: str
    required_fields: tuple[str, ...]
    def throat_conductance(self, net: Network, fluid: FluidSinglePhase) -> ArrayLike: ...
```

### v0.1 planned models

#### 1) `generic_poiseuille` (robust fallback)
- Prioritizes broad compatibility and validation
- Uses available throat geometry (e.g., area, diameter/hydraulic radius, length)
- Emits explicit warnings when assumptions/fallbacks are used

#### 2) `valvatne_blunt` (scientific default when fields exist)
- Uses shape-factor-informed conductance and composite pore–throat–pore resistance
- Activated when required geometry fields are available
- Fails loudly (with actionable error) if required fields are missing
- `valvatne_blunt_baseline` may remain as a backward-compatible alias, but it is not
  a distinct physical model

### Policy
- `generic_poiseuille` is the **fallback implementation**
- `valvatne_blunt` is the **preferred scientific model** when supported by imported geometry

---

## Linear algebra backend abstraction

### Goal
Keep physics code independent from specific sparse backends.

### `voids.linalg.backends`
Defines a minimal interface used by `voids.linalg.solve`:
- sparse matrix creation (COO/CSR)
- direct solve
- iterative solve (`cg`, `gmres`)
- vector norms / diagnostics

### v0.1 backend
- `scipy` direct and Krylov solves
- optional `pyamg` preconditioning for iterative solves

### Future optional backends
- `torch` (limited sparse support initially; selected kernels first)
- `jax` (experimental sparse path; optional research backend)

### Design constraint
`voids.physics.*` should not import SciPy directly.

---

## Boundary conditions (v0.1)

### Supported BC type
- **Dirichlet pressure BC** across two pore label sets

### Requirements
- inlet and outlet labels must exist
- each label must contain at least one pore
- inlet and outlet sets must be disjoint (error otherwise)

### Future BCs
- Neumann flux BC
- mixed BCs
- periodic BCs
- gravity/potential BC variants

---

## Serialization specification (HDF5)

### Required HDF5 layout (draft)

```text
/meta/schema_version
/meta/package_version
/meta/provenance/...
/sample/...
/network/pore/coords
/network/pore/volume
/network/throat/conns
/network/throat/volume
/network/throat/length
...
/labels/pore/<label_name>
/labels/throat/<label_name>   # optional
```

### Requirements
- Preserve dtypes where practical
- Store units and schema version
- Save provenance and solver configuration summaries
- Roundtrip load/save tested

---

## Validation and testing specification

### Test philosophy
Validation is a first-class deliverable.

### Categories

#### 1) Schema and import
- missing required keys
- shape mismatches
- dtype coercion behavior
- warnings on partial geometry

#### 2) Graph/connectivity
- component detection on toy graphs
- spanning component detection by axis
- dead-end and isolated pore identification

#### 3) Porosity
- exact porosity for toy networks with known volumes
- effective vs absolute porosity consistency
- directional effective porosity checks

#### 4) Single-phase solver correctness
- hand-solvable toy networks (2–5 pores)
- symmetry invariance tests
- scaling laws:
  - \(Q \propto \Delta P\)
  - \(Q \propto 1/\mu\)
- permeability invariant to \(\Delta P\)

#### 5) Diagnostics
- mass balance error below tolerance
- residual norms below tolerance
- disconnected-labeled network singularity detection path tested

#### 6) Serialization
- save/load roundtrip consistency
- metadata preservation
- deterministic summaries for fixed inputs (within tolerances)

### Initial tolerances (draft)
- scalar toy tests: `rtol=1e-10`, `atol=1e-12`
- direct solver residual norm: `< 1e-10`
- iterative solver residual norm: `< 1e-8`
- mass balance relative error: `< 1e-8`

---

## Reproducibility specification

Every persisted workflow result should store:
- `voids` version
- schema version
- provenance
- random seed (if applicable)
- solver options
- conductance model identifier

`SinglePhaseResult` should be serializable (HDF5 or JSON summary + arrays).

---

## Backend portability and autodiff strategy (v0.1 policy)

### What is designed for now
- clean separation of:
  - data model
  - conductance kernels
  - linear algebra
  - physics postprocessing

### What is *not* promised in v0.1
- end-to-end differentiability through discrete multiphase event engines

### Intended future path
- differentiable conductance and single-phase kernels
- calibration/inverse modules
- optional PyTorch/JAX backends for selected operations

---

## Pixi environment policy (high level)

`pixi` manages:
- reproducible dev/runtime environments
- feature-specific dependencies
- common tasks (test, lint, typecheck, examples)

`pyproject.toml` remains the source for:
- build metadata
- package metadata
- Python packaging configuration

---

## Open decisions (locked for v0.1 draft)

These are the current defaults adopted in this spec:

1. **Conductance baseline**
   - Implement both interfaces
   - `generic_poiseuille` = fallback default
   - `valvatne_blunt` = preferred scientific model when fields exist

2. **Gravity**
   - API supports potential-based extension
   - gravity disabled by default in v0.1 examples/tests

3. **Effective porosity**
   - Support both `"connected_boundary"` and `"spanning"` definitions explicitly
   - never overload a single ambiguous metric name

---

## Immediate next coding deliverable (after this spec)
- Thin package scaffold (`src/voids/...`)
- Core dataclasses (`Network`, `SampleGeometry`, `Provenance`)
- Network validation module
- PoreSpy import normalizer stub
- Tests for schema and validation
