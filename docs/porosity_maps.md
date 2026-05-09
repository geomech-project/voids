# Porosity Maps

This page documents the continuum porosity-map workflow implemented in
`voids.image.porosity`.
It is separate from pore-network porosity: here the output is a regular grid of
cell-average porosity values intended for continuum, FEM, finite-volume, or
external solver workflows.

The minimal demonstration is
`notebooks/33_mwe_synthetic_porosity_maps`, which builds porosity maps from a
synthetic PoreSpy `blobs` binary image and a derived toy grayscale image.

---

## What `block_shape` Means

`block_shape` defines how many fine image voxels are averaged into one porosity-map
cell.

For a 3-D image with shape

\[
(n_0, n_1, n_2),
\]

and

\[
\texttt{block\_shape} = (b_0, b_1, b_2),
\]

the output porosity map has shape

\[
\left(
\left\lfloor \frac{n_0}{b_0} \right\rfloor,
\left\lfloor \frac{n_1}{b_1} \right\rfloor,
\left\lfloor \frac{n_2}{b_2} \right\rfloor
\right).
\]

When `strict=True`, which is the default, every image dimension must be exactly
divisible by the corresponding block dimension.
For example:

```python
shape = (300, 300, 300)
block_shape = (10, 10, 10)
```

gives a porosity map with shape:

\[
(30, 30, 30).
\]

Each coarse porosity cell is the average over one \(10 \times 10 \times 10\)
voxel block.
If the fine voxel size is \(40\,\mu\mathrm{m}\), then the porosity-map cell size is:

\[
(10, 10, 10) \times 40\,\mu\mathrm{m}
=
(400, 400, 400)\,\mu\mathrm{m}.
\]

In code:

```python
porosity = porosity_map_from_binary(
    image,
    block_shape=(10, 10, 10),
    voxel_size=40.0e-6,
)

porosity.shape       # (30, 30, 30)
porosity.cell_size   # (4.0e-4, 4.0e-4, 4.0e-4) in meters
```

!!! note "Axis convention"
    `block_shape` follows the NumPy array axis order of the input image.
    `voids` does not silently reinterpret array axes as geological or scanner axes.
    If a dataset uses a different physical axis order, transpose or document that
    convention before exporting fields to another solver.

---

## Binary-Image Porosity

For a segmented binary image, the input is interpreted as a void mask.
By default:

- `True` or `1` means void,
- `False` or `0` means solid.

Let \(V_{ijk}\) be the fine-grid void indicator:

\[
V_{ijk}
=
\begin{cases}
1, & \text{if voxel }(i,j,k)\text{ is void}, \\
0, & \text{if voxel }(i,j,k)\text{ is solid}.
\end{cases}
\]

For porosity-map cell \((I,J,K)\), with block shape
\((b_0,b_1,b_2)\), the local porosity is:

\[
\phi^{\mathrm{bin}}_{IJK}
=
\frac{1}{b_0 b_1 b_2}
\sum_{i=I b_0}^{(I+1)b_0 - 1}
\sum_{j=J b_1}^{(J+1)b_1 - 1}
\sum_{k=K b_2}^{(K+1)b_2 - 1}
V_{ijk}.
\]

So the binary porosity map is exactly a local void-volume fraction on the
regular image grid.

If `image_is_void=False`, the input binary image is interpreted as a solid mask
and inverted before applying the same formula.

### Conservation Check

If the image shape is exactly divisible by `block_shape`, the mean of the local
porosity map equals the global void fraction:

\[
\overline{\phi}^{\mathrm{bin}}
=
\frac{1}{N_c}\sum_{IJK}\phi^{\mathrm{bin}}_{IJK}
=
\frac{1}{N_v}\sum_{ijk}V_{ijk},
\]

where \(N_c\) is the number of coarse cells and \(N_v\) is the number of fine
voxels.

This is one of the most important synthetic verification checks.

---

## Grayscale-Image Porosity

For a grayscale image, `voids` first maps each fine voxel intensity to a porosity
value using a two-point linear calibration.

The calibration inputs are:

| Parameter | Meaning |
|---|---|
| `solid_gray` | grayscale value assigned to `background_porosity` |
| `pore_gray` | grayscale value assigned to porosity \(1\) |
| `background_porosity` | porosity floor at the solid endpoint |

Let \(G_{ijk}\) be the grayscale value of voxel \((i,j,k)\), \(G_s\) be
`solid_gray`, \(G_p\) be `pore_gray`, and \(\phi_b\) be `background_porosity`.
The voxel-scale porosity is:

\[
\phi^{\mathrm{vox}}_{ijk}
=
\phi_b
+
(1-\phi_b)
\frac{G_{ijk} - G_s}{G_p - G_s}.
\]

By default, values are clipped to:

\[
\phi^{\mathrm{vox}}_{ijk} \in [\phi_b, 1].
\]

Then the coarse grayscale porosity map is a block average of those voxel
porosities:

\[
\phi^{\mathrm{gray}}_{IJK}
=
\frac{1}{b_0 b_1 b_2}
\sum_{i=I b_0}^{(I+1)b_0 - 1}
\sum_{j=J b_1}^{(J+1)b_1 - 1}
\sum_{k=K b_2}^{(K+1)b_2 - 1}
\phi^{\mathrm{vox}}_{ijk}.
\]

### Dark-Pore And Bright-Pore Images

The formula does not require pores to be darker than solid.
It only requires `solid_gray` and `pore_gray` to be distinct.

For a dark-pore image:

\[
G_p < G_s.
\]

For a bright-pore image:

\[
G_p > G_s.
\]

Both cases are handled by the same denominator \(G_p - G_s\).

### Important Assumption

The grayscale path assumes that a linear grayscale interpolation is a defensible
proxy for local porosity between the two calibration endpoints.
That may be incorrect when the image has severe beam hardening, ring artifacts,
mixed mineral phases, uncorrected scanner drift, or a nonlinear intensity-density
relationship.
For real micro-CT data, calibration must be treated as part of the physical model,
not as a plotting choice.

---

## Physical Volumes

For a porosity map with cell size

\[
(\Delta x, \Delta y, \Delta z),
\]

the bulk volume of one porosity cell is:

\[
\Delta V = \Delta x \Delta y \Delta z.
\]

The total represented bulk volume is:

\[
V_{\mathrm{bulk}}
=
N_c \Delta V.
\]

The implied pore volume is:

\[
V_{\mathrm{void}}
=
\sum_{IJK}\phi_{IJK}\Delta V.
\]

The mean porosity is therefore:

\[
\overline{\phi}
=
\frac{V_{\mathrm{void}}}{V_{\mathrm{bulk}}}.
\]

This relation is used by `PorosityMap.bulk_volume`,
`PorosityMap.void_volume`, and `PorosityMap.mean_porosity`.

!!! warning "2-D maps"
    The same implementation accepts 2-D arrays, where the product of `cell_size`
    is an area. If the target solver expects a 3-D volume, supply a 3-D image or
    define an explicit thickness outside this 2-D helper.

---

## Exported HDF5 Files

`save_porosity_map_hdf5` writes:

- `/porosity`: the cell-average porosity array,
- root attribute `schema_version`,
- root attribute `metadata`, encoded as JSON.

The metadata includes:

- porosity-map shape,
- cell size,
- origin,
- units,
- source/calibration metadata such as `block_shape`, `solid_gray`, `pore_gray`,
  and `background_porosity`.

This export is intentionally solver-neutral.
It is a stable field file, not a full FEM mesh.
Solver-specific exporters should define cell ordering, mesh topology, boundary
patches, and field naming explicitly.

---

## Synthetic Verification Plan

Synthetic cases are useful because the expected porosity is known before running
the code.
They should be used before interpreting real scanner-derived fields.

### 1. All-Void And All-Solid Images

For an all-void binary image:

\[
V_{ijk}=1
\quad\Rightarrow\quad
\phi_{IJK}=1
\quad\text{for all cells}.
\]

For an all-solid binary image:

\[
V_{ijk}=0
\quad\Rightarrow\quad
\phi_{IJK}=0
\quad\text{for all cells}.
\]

These cases verify phase polarity and the averaging formula.

### 2. Known Block Fractions

Construct a small image where each block has a manually known number of void
voxels.
For example, a \(2 \times 2\) block with three void voxels must give:

\[
\phi = \frac{3}{4}.
\]

This verifies that `block_shape` is applied in the expected axis order.

### 3. Global Conservation

For a binary image whose shape is divisible by `block_shape`, verify:

\[
\mathrm{mean}(\phi_{\mathrm{map}})
=
\mathrm{mean}(V_{\mathrm{image}}).
\]

This check is robust and should hold for random synthetic images, PoreSpy
`blobs`, checkerboards, and hand-built masks.

### 4. Grayscale Endpoint Calibration

Use a tiny grayscale image containing only the calibration endpoints:

\[
G=G_p \Rightarrow \phi=1,
\]

\[
G=G_s \Rightarrow \phi=\phi_b.
\]

The midpoint should give:

\[
G=\frac{G_p+G_s}{2}
\Rightarrow
\phi=\phi_b+\frac{1-\phi_b}{2}.
\]

This verifies the linear calibration independently of block averaging.

### 5. Binary-To-Grayscale Consistency

Generate a grayscale image directly from a binary image using exactly the two
calibration endpoint intensities and no blur or noise:

\[
G_{ijk}
=
\begin{cases}
G_p, & V_{ijk}=1,\\
G_s, & V_{ijk}=0.
\end{cases}
\]

If `background_porosity=0`, then:

\[
\phi^{\mathrm{gray}}_{IJK}
=
\phi^{\mathrm{bin}}_{IJK}.
\]

If `background_porosity > 0`, then:

\[
\phi^{\mathrm{gray}}_{IJK}
=
\phi_b + (1-\phi_b)\phi^{\mathrm{bin}}_{IJK}.
\]

This is the cleanest synthetic test connecting the binary and grayscale
workflows.

### 6. HDF5 Round Trip

After export and import, verify:

\[
\phi_{\mathrm{loaded}} = \phi_{\mathrm{original}},
\]

and check that `cell_size`, `origin`, `units`, and calibration metadata are
unchanged.

This verifies the interchange format, not the physical calibration.

### 7. PoreSpy Blob Sanity Check

For a PoreSpy `blobs` realization with target porosity \(\phi_t\), check:

\[
\mathrm{mean}(V_{\mathrm{image}}) \approx \phi_t,
\]

then check conservation:

\[
\mathrm{mean}(\phi_{\mathrm{map}})
=
\mathrm{mean}(V_{\mathrm{image}}).
\]

This verifies that the workflow preserves the realized synthetic image porosity.
It does not prove that the morphology is realistic for a particular rock.

---

## What This Does Not Validate

These calculations validate the mechanics of local porosity mapping.
They do not, by themselves, validate:

- scanner grayscale calibration,
- beam-hardening correction,
- mineral-density interpretation,
- unresolved microporosity,
- permeability closure,
- connectivity or flow behavior,
- or agreement with laboratory porosity.

For real datasets, a defensible validation sequence should include:

1. record voxel size, image crop, support mask, and preprocessing,
2. document `solid_gray`, `pore_gray`, and `background_porosity`,
3. compare mean porosity against laboratory porosity or a trusted image-derived
   reference,
4. inspect slices of the grayscale image, binary segmentation, and porosity map,
5. test sensitivity to the calibration endpoints and block size,
6. only then export the field to an external continuum solver.

---

## Relation To Soulaine Micro-Continuum Papers

The `voids` porosity-map representation is compatible with the micro-continuum
porosity fields used by Soulaine and co-workers, but it is not yet a full
implementation of their Darcy-Brinkman or Darcy-Brinkman-Stokes model.

Two references in `refs/soulaine/` are especially relevant:

- Soulaine et al. (2016), *The Impact of Sub-Resolution Porosity of X-ray
  Microtomography Images on the Permeability*, DOI:
  <https://doi.org/10.1007/s11242-016-0690-2>
- Soulaine and Tchelepi (2016), *Micro-continuum Approach for Pore-Scale
  Simulation of Subsurface Processes*, DOI:
  <https://doi.org/10.1007/s11242-016-0701-3>

In those papers, the central image-derived field is a local void fraction:

\[
\epsilon_f(\mathbf{x}) \in [0, 1],
\]

or, in the sub-resolution porosity paper,

\[
\epsilon_{\mathrm{micro}}(\mathbf{x}) \in [0, 1].
\]

Cells with \(\epsilon_f=1\) represent fully resolved free-flow regions, cells with
\(\epsilon_f=0\) represent solid or no-flow regions, and intermediate values
represent porous control volumes whose sub-cell pore structure is not resolved.
The flow model then uses a single-domain Darcy-Brinkman-type equation, with a
permeability or inverse-permeability field coupled to porosity.

The `voids` porosity map stores the same type of mathematical object:

\[
\phi_{IJK} \in [0, 1],
\]

a cell-average void fraction on a regular grid.
Therefore, a `PorosityMap.values` array can be interpreted as a candidate
\(\epsilon_f\) field for a micro-continuum solver if the control-volume size,
axis order, and physical units are aligned with that solver.

The important difference is what `voids` currently does with the field.

| Aspect | `voids` porosity map | Soulaine micro-continuum model |
|---|---|---|
| Main field | Cell-average porosity \(\phi\) | Void fraction \(\epsilon_f\) or microporosity \(\epsilon_{\mathrm{micro}}\) |
| Binary image route | Block-average of a resolved void indicator | Compatible with a filtered control-volume porosity |
| Grayscale route | Linear grayscale-to-porosity calibration, then block average | Compatible only if calibration matches the image-processing assumptions |
| Flow equations | Not solved by this feature | Darcy-Brinkman or Darcy-Brinkman-Stokes single-domain solve |
| Permeability closure | Not yet included | Required, often \(k=k(\epsilon)\), for example Kozeny-Carman-style |
| Free-flow handling | Stored only as \(\phi=1\) cells | Solver switches toward Stokes/Navier-Stokes behavior |
| Porous/no-flow handling | Stored only as \(\phi=0\) or \(0<\phi<1\) cells | Solver gives high resistance or no flow depending on closure |

### Same Interpretation When `block_shape=(1, 1, 1)`

If the input image voxel is the intended micro-continuum control volume, then:

```python
porosity = porosity_map_from_binary(
    image,
    block_shape=(1, 1, 1),
    voxel_size=voxel_size,
)
```

produces one porosity-map cell per image voxel.
For a binary image, each value is either 0 or 1.
For a calibrated grayscale image, each value can be intermediate and may be used
as a sub-resolution porosity estimate.

This is closest to the voxel-wise porosity-field interpretation in the
sub-resolution porosity paper.

### Filtered Interpretation When `block_shape > 1`

When `block_shape` is larger than one, `voids` is not merely copying voxel labels.
It is applying a spatial filter:

\[
\phi_{IJK}
=
\frac{1}{|B_{IJK}|}
\sum_{(i,j,k)\in B_{IJK}}\phi^{\mathrm{vox}}_{ijk}.
\]

This is compatible with the general micro-continuum idea of averaging below a
chosen cutoff length.
However, the physical interpretation changes:

- intermediate \(\phi\) values may represent resolved pores and grains mixed by
  coarse graining,
- not necessarily true sub-resolution microporosity inside each original voxel,
- so the permeability closure should be chosen for the chosen filter scale, not
  blindly copied from a voxel-scale model.

### What Would Be Needed For Full Soulaine-Style Compatibility

To move from a compatible porosity field to a Soulaine-style solver input, the
next pieces are:

1. a documented \(\phi \mapsto k\) or \(\phi \mapsto k^{-1}\) closure,
2. optional lower/upper clamps for \(\phi=0\) and \(\phi=1\) depending on the
   external solver formulation,
3. explicit cell ordering and axis convention for the target solver,
4. solver-specific export, for example OpenFOAM scalar fields for porosity and
   permeability,
5. validation against known synthetic cases before real micro-CT data.

So the safest statement is:

!!! summary
    `voids` now computes a porosity field that is mathematically compatible with
    the \(\epsilon_f\) field used in Soulaine-style micro-continuum models.
    It does not yet implement the associated Darcy-Brinkman-Stokes equations or
    the required porosity-permeability closure.
