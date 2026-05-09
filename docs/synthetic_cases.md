# Synthetic Image Cases

`voids.generators.porous_image` provides package-level helpers for constructing
controlled 2-D and 3-D binary porous images. These are intended for testing,
algorithm development, and reproducible demonstrations before moving to
scanner-derived data.

The central convention is:

- `True` means void,
- `False` means solid or matrix,
- all shapes follow NumPy axis order.

---

## Macro/Micro Pore Images

The macro/micro generator builds a two-porosity synthetic image from two PoreSpy
`blobs` fields:

1. generate a resolved macropore image,
2. generate a finer-scale micropore score field,
3. keep micropores only inside the solid matrix of the macropore image,
4. combine the two masks.

```python
from voids.generators import generate_macro_micro_blobs_matrix

case = generate_macro_micro_blobs_matrix(
    shape=(160, 160, 160),
    macro_porosity=0.25,
    matrix_microporosity=0.12,
    macro_blobiness=1.2,
    micropore_blobiness=8.0,
    axis_index=0,
    seed_start=2026,
    max_tries=50,
)

void = case.void
macro = case.macro_void
micropores = case.micropore_void
```

The invariant is:

\[
\Omega_{\mathrm{void}}
=
\Omega_{\mathrm{macro}}
\cup
\Omega_{\mathrm{micro}},
\qquad
\Omega_{\mathrm{micro}}
\subset
\Omega_{\mathrm{matrix}}.
\]

The reported total porosity is:

\[
\phi_{\mathrm{total}}
=
\phi_{\mathrm{macro}}
+
(1-\phi_{\mathrm{macro}})
\phi_{\mathrm{micro|matrix}},
\]

where \(\phi_{\mathrm{micro|matrix}}\) is the micropore fraction measured only in
the matrix region of the macro image.

The `micropore_blobiness` parameter controls the feature scale of the small
pores. In PoreSpy `blobs`, larger `blobiness` values produce smaller features,
so a typical macro/micro setup uses:

- lower `macro_blobiness` for broader connected macropores,
- higher `micropore_blobiness` for smaller pores in the matrix.

!!! warning "Synthetic interpretation"
    The micropore mask is a controlled synthetic model, not a calibrated
    unresolved-porosity law. It is useful for algorithm sensitivity studies, but
    the values should not be interpreted as a measured carbonate microporosity
    field without calibration.

---

## Vug Insertions

Existing helpers can add idealized circular, elliptical, spherical, or
ellipsoidal vugs to a binary matrix:

```python
from voids.generators import insert_spherical_vug

with_vug, vug_mask = insert_spherical_vug(
    case.void,
    radius_vox=18,
    center=(80, 80, 80),
)
```

The operation is a boolean union, so existing pores remain void.

---

## Export And Import

Synthetic cases are ordinary `voids` image volumes once generated. Export and
import them through the dedicated
[Image Volume And Surface Mesh I/O](api/io.md#image-volume-and-surface-mesh-io)
section, which documents `VolumeData`, supported voxel formats, STL/OBJ surface
exports, sidecar metadata, and explicit voxel-size handling.

A typical synthetic-case export wraps the generated void mask with physical
resolution metadata before writing files:

```python
from voids.io import VolumeData, save_volume_bundle

case_data = VolumeData(
    with_vug,
    voxel_size=(40.0e-6, 40.0e-6, 40.0e-6),
    units={"length": "m"},
    metadata=case.metadata,
)

written = save_volume_bundle(
    case_data,
    "outputs/synthetic_case",
    stem="macro_micro_vug",
    formats=("raw", "npy", "h5", "nc", "tiff", "stl", "obj"),
)
```

Read the exported data with `load_volume_data` when physical resolution matters:

```python
from voids.io import load_volume_data

volume_data = load_volume_data("outputs/synthetic_case/macro_micro_vug.tiff")
```
