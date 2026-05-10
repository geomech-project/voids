# Local Thickness Morphometry

This page documents the local-thickness calculation used by
`voids.image.morphometry`.
It covers the scientific definition, unit conversion, algorithm choices, and
interpretation limits. The callable API reference remains in
[Image Processing](api/image.md).

---

## Phase-Domain Definition

Local thickness is defined on a selected binary phase, not on a whole grayscale
image and not on an inferred material label.
Let \(\Omega\) be the physical domain occupied by the selected phase. For a
phase voxel centered at \(x\), the local thickness is

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

where \(B(c,r)\) is a disk in 2D or a sphere in 3D.
Each phase voxel is assigned the diameter of the largest disk or sphere that
both fits inside the phase and contains that voxel.

Two points are important for interpretation:

- the result is a local diameter field, not a pore-network radius;
- the result depends on which binary phase mask is passed to the function.

For a binary mask \(M_i\), `True` marks the phase being measured. The
complementary phase can be analyzed by calling the same routine on `~M`.
`voids` does not decide what those phases mean physically.

---

## Calculation Path In `voids`

`voids` delegates sphere insertion or distance-transform filtering to
PoreSpy, then applies the `voids` API contract around validation, units, and
summary statistics.

```text
binary phase mask M
        |
        v
validate binary 2D/3D mask and isotropic voxel size h
        |
        v
optional Euclidean distance map D in voxel units
        |
        v
porespy.filters.local_thickness(...)
        |
        v
radius-like field R in voxel units
        |
        v
voids diameter map T = 2 h R, masked to M
        |
        v
phase-only summary statistics
```

The explicit steps are:

**Step 1: validate the phase image.**
The input must be a 2D or 3D boolean/binary image. `True` marks the phase whose
local thickness is being measured.

**Step 2: validate the voxel spacing.**
`voxel_size` may be a scalar or a sequence with one value per image axis, but
all entries must be equal. The current implementation is intentionally
isotropic because the fitted objects are disks or spheres in physical space.

**Step 3: optionally validate a precomputed distance map.**
If supplied, `distance_map` must have the same shape as the phase mask, be
finite and nonnegative, and be expressed in voxel units.

**Step 4: call PoreSpy's local-thickness filter.**

\[
R = \operatorname{local\_thickness}
(M, D, \mathrm{method}, \mathrm{sizes}, \mathrm{smooth}, \mathrm{approx})
\]

Here \(M\) is the selected binary phase mask and \(D\) is the optional Euclidean
distance map.

**Step 5: convert the returned radius-like field to a physical diameter map.**

\[
T_i =
\begin{cases}
2 h R_i, & M_i = 1, \\
0, & M_i = 0,
\end{cases}
\]

Here \(h\) is the isotropic voxel edge length in the requested units.

The factor of two is deliberate. If the backend labels a phase voxel with
\(R_i = 4\) voxels and the voxel edge length is
\(h = 2.086\,\mu\mathrm{m}\), then `voids` reports

\[
T_i = 2 \times 2.086 \times 4 = 16.688\,\mu\mathrm{m}.
\]

---

## Backend Method Choices

`local_thickness_map` forwards the algorithm controls to
`porespy.filters.local_thickness`.

| Method | Practical meaning |
|---|---|
| `"dt"` | Distance-transform based erosion/dilation over sampled radii. This is the default because it is practical for moderately large 3D images. |
| `"conv"` | FFT-convolution based erosion/dilation over sampled radii. |
| `"bf"` | Brute-force sphere insertion. This is conceptually direct but can be expensive. |
| `"imj"` | ImageJ-style sphere insertion with a reduced set of insertion sites. This is useful when matching ImageJ-style local-thickness workflows is more important than runtime. |

For `"dt"` and `"conv"`, `sizes` controls the sampled radii.
A scalar requests that many radii spanning the distance-transform range.
A sequence uses the supplied radii directly.
`None` uses all unique distance-transform values, which is more detailed but
can be much slower and more memory-intensive.

`smooth=True` asks PoreSpy to remove protrusions from the generated sphere
faces. `approx` is only used by the `"imj"` method; `approx=True` is faster but
can sacrifice voxel-by-voxel agreement with the more exact path.

---

## Summary Statistics

`local_thickness_analysis` returns a `LocalThicknessResult` with:

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

plus the 10th, 50th, and 90th percentiles and the maximum.
The standard deviation is NumPy's default population standard deviation
(`ddof=0`).
Empty phase masks return an all-zero map and `NaN` summary statistics.

---

## Assumptions And Limitations

- The input is already segmented. These routines do not threshold grayscale
  images or decide the physical meaning of image labels.
- Voxel spacing must be isotropic. Anisotropic local thickness would require
  ellipsoid-aware handling or resampling before analysis.
- Boundary behavior is inherited from the input mask and PoreSpy backend.
  `voids` does not add periodic padding or special exterior-boundary
  corrections. If ROI boundary effects matter, pad, crop, or document the
  boundary convention before interpreting edge-adjacent values.
- Derived application-specific morphometric quantities are not computed by this
  API. Downstream analyses may combine local-thickness summaries with other
  image statistics, but should state the additional formula and assumptions
  used.
- Agreement with another local-thickness implementation is method-dependent.
  In particular, `"imj"` is closer to an ImageJ-style sphere-insertion workflow,
  while `"dt"` is the practical default used for routine analysis.

---

## Verification In This Package

The morphometry tests exercise the `voids` behavior around the external backend
rather than trying to reproduce PoreSpy's full algorithm. In
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

---

## References

- Hildebrand and Ruegsegger (1997), *A new method for the model-independent
  assessment of thickness in three-dimensional images*.
  <https://doi.org/10.1046/j.1365-2818.1997.1340694.x>
- PoreSpy local-thickness filter documentation:
  <https://porespy.org/examples/filters/reference/local_thickness.html>
