# Image Processing

The `voids.image` sub-package provides utilities for segmented image processing,
connectivity analysis, and pore network extraction used in vug sensitivity studies.

---

## Maximal-Ball Extraction

::: voids.image.maximal_ball

---

## PREGO Extraction

::: voids.image.prego

---

## Porosity Maps

For the calculation steps, `block_shape` interpretation, and synthetic
verification plan, see [Porosity Maps](../porosity_maps.md).

::: voids.image.porosity

---

## Morphometry

The morphometry helpers compute local-thickness diameter maps for binary 2D/3D
phase images. They are intended for phase-domain measurements where the
reported quantity is a local **diameter**, not a pore-network radius.

### Local-Thickness Definition

For a phase domain \(\Omega\) represented by a binary mask, the local thickness
at a phase voxel centered at \(x\) is

\[
T(x)
= 2 \max_{c,r}
\left\{
r \;:\;
x \in B(c,r)
\;\mathrm{and}\;
B(c,r) \subseteq \Omega
\right\},
\]

where \(B(c,r)\) is a disk in 2D or a sphere in 3D. In words, each phase voxel
is assigned the diameter of the largest sphere that both fits inside the phase
and contains that voxel.

This is a model-independent local-thickness definition for a selected binary
phase. `voids` does not infer the physical meaning of an image label; the
caller must pass the phase mask whose local diameter should be measured.

### Calculation Implemented In `voids`

The public workflow is:

```python
from voids.image import local_thickness_analysis

phase_result = local_thickness_analysis(
    phase_mask,
    voxel_size=2.086,
    units="um",
    label="phase",
)

complement_result = local_thickness_analysis(
    complementary_phase_mask,
    voxel_size=2.086,
    units="um",
    label="complementary phase",
)
```

The calculation has five explicit steps:

1. Validate the phase image.
   The input must be a 2D or 3D boolean/binary image. `True` marks the phase
   whose local thickness is being measured.
2. Validate the voxel spacing.
   `voxel_size` may be a scalar or a sequence with one value per image axis,
   but all entries must be equal. The current implementation is intentionally
   isotropic because the fitted objects are disks/spheres in physical space.
3. Optionally validate a precomputed distance map.
   If supplied, `distance_map` must have the same shape as the phase mask, be
   finite and nonnegative, and be expressed in **voxel units**.
4. Call PoreSpy's local-thickness filter:

   \[
   R = \mathrm{local\_thickness}
   (M, D, \mathrm{method}, \mathrm{sizes}, \mathrm{smooth}, \mathrm{approx}),
   \]

   where \(M\) is the binary phase mask and \(D\) is the optional Euclidean
   distance map. The returned field \(R\) is interpreted by `voids` as a
   local radius field in voxel units.
5. Convert the radius field to a physical diameter map:

   \[
   T_i =
   \begin{cases}
   2 h R_i, & M_i = 1, \\
   0, & M_i = 0,
   \end{cases}
   \]

   where \(h\) is the isotropic voxel edge length in `units`.

The factor of two is deliberate: the returned `thickness_map` stores diameters.
For example, if PoreSpy labels a phase voxel with radius \(R_i = 4\) voxels and
the voxel size is \(h = 2.086\,\mu\mathrm{m}\), then `voids` reports

\[
T_i = 2 \times 2.086 \times 4 = 16.688\,\mu\mathrm{m}.
\]

### Backend Method Choices

`local_thickness_map` forwards the algorithm controls to
`porespy.filters.local_thickness`.

| Method | Practical meaning |
|---|---|
| `"dt"` | Distance-transform based erosion/dilation over sampled radii. This is the default because it is practical for moderately large 3D images. |
| `"conv"` | FFT-convolution based erosion/dilation over sampled radii. |
| `"bf"` | Brute-force sphere insertion. This is conceptually direct but can be expensive. |
| `"imj"` | ImageJ-style sphere insertion with a reduced set of insertion sites. This is useful when matching ImageJ-style local-thickness workflows is more important than runtime. |

For `"dt"` and `"conv"`, `sizes` controls the sampled radii. A scalar requests
that many radii spanning the distance-transform range. A sequence uses the
given radii directly. `None` uses all unique distance-transform values, which
is more detailed but can be much slower and more memory-intensive.

`smooth=True` asks PoreSpy to remove protrusions from the generated sphere
faces. `approx` is only used by the `"imj"` method; `approx=True` is faster but
can sacrifice voxel-by-voxel agreement with the more exact path.

### Summary Statistics

`local_thickness_analysis` returns a `LocalThicknessResult`:

- `thickness_map`: the full diameter field in physical units, with zeros
  outside the measured phase;
- `summary`: a `LocalThicknessSummary` computed only over phase voxels.

For phase voxels \(\{i: M_i = 1\}\), the summary stores:

\[
\bar{T}
=
\frac{1}{N}
\sum_{i:M_i=1} T_i,
\qquad
\sigma_T
=
\sqrt{
\frac{1}{N}
\sum_{i:M_i=1}(T_i-\bar{T})^2
},
\]

plus the 10th, 50th, and 90th percentiles and the maximum. The standard
deviation is NumPy's default population standard deviation (`ddof=0`). Empty
phase masks return an all-zero map and `NaN` summary statistics.

### Assumptions And Limitations

- The input is already segmented. These functions do not threshold grayscale
  images or decide the physical meaning of image labels.
- Voxel spacing must be isotropic. Anisotropic local thickness would require
  ellipsoid-aware handling or resampling before analysis.
- Boundary behavior is inherited from the input mask and PoreSpy backend.
  `voids` does not add periodic padding or special exterior-boundary
  corrections. If ROI boundary effects matter, pad, crop, or document the
  boundary convention before interpreting edge-adjacent values.
- Derived morphometric quantities are not computed by this API. Downstream
  analyses may combine local-thickness summaries with other image statistics,
  but should state the additional formula and assumptions used.
- Agreement with another local-thickness implementation is method-dependent. In
  particular, `"imj"` is closer to an ImageJ-style sphere-insertion workflow,
  while `"dt"` is the practical default used for routine analysis.

### Verification In This Package

The morphometry tests exercise the `voids` behavior around the external
backend rather than trying to reproduce PoreSpy's full algorithm. In
`tests/test_image_morphometry.py`, the checks verify that:

- PoreSpy is imported lazily, so importing `voids.image.morphometry` does not
  immediately require the optional image-analysis stack;
- PoreSpy's radius-like output is converted to a physical diameter map by
  \(T = 2hR\);
- summary statistics are computed only over the selected phase voxels;
- empty phases return zero maps and `NaN` summaries instead of failing
  ambiguously;
- nonbinary masks, anisotropic voxel spacing, invalid distance maps, and invalid
  summary inputs fail loudly.

These tests validate the `voids` API contract and unit conversion. They do not
constitute an independent scientific validation of PoreSpy or any other
local-thickness implementation.

### References

- Hildebrand and Ruegsegger (1997), *A new method for the model-independent
  assessment of thickness in three-dimensional images*.
  <https://doi.org/10.1046/j.1365-2818.1997.1340694.x>
- PoreSpy local-thickness filter documentation:
  <https://porespy.org/examples/filters/reference/local_thickness.html>

::: voids.image.morphometry

---

## Network Extraction

::: voids.image.network_extraction

---

## Segmentation

::: voids.image.segmentation

---

## Image Connectivity

::: voids.image.connectivity
