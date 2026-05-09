# I/O

The `voids.io` sub-package handles serialization and import/export for canonical
networks, PoreSpy interoperability, OpenPNM interoperability, and image-volume
cases.

---

## Image Volume And Surface Mesh I/O

`voids.io.volume` provides the image-volume import/export surface used by the
synthetic image workflows. The central object is `VolumeData`, which couples a
2-D or 3-D image array to its physical voxel spacing, length units, and
provenance metadata.

![Image volume and surface mesh IO workflow](../assets/io/volume_io_workflow.png)

Use `VolumeData` when the image is intended to leave Python or feed a
continuum/FEM workflow:

```python
from voids.io import VolumeData, save_volume_bundle

case_data = VolumeData(
    values=void_image,
    voxel_size=(40.0e-6, 40.0e-6, 40.0e-6),
    units={"length": "m"},
    metadata={"case": "macro_micro_vug"},
)

written = save_volume_bundle(
    case_data,
    "outputs/synthetic_case",
    stem="macro_micro_vug",
    formats=("raw", "npy", "h5", "nc", "tiff", "stl", "obj"),
)
```

### Supported Formats

| Format | Kind | Metadata handling |
|---|---|---|
| `.raw` | voxel field | Written with a `.raw.json` sidecar containing shape, dtype, voxel size, units, and provenance metadata |
| `.npy` | voxel field | NumPy-native array plus `.npy.json` sidecar for voxel size, units, and provenance metadata |
| `.h5` | voxel field | HDF5 dataset `/volume` plus JSON metadata attributes |
| `.nc` | voxel field | Basic netCDF variable `volume` plus metadata attributes |
| `.tif`, `.tiff` | voxel field | TIFF stack plus `.tif.json` or `.tiff.json` sidecar for voxel size, units, and provenance metadata |
| `.stl` | surface mesh | 3-D binary interface extracted by marching cubes using `voxel_size` as physical spacing |
| `.obj` | surface mesh | 3-D binary interface extracted by marching cubes using `voxel_size` as physical spacing |

STL and OBJ exports require a 3-D binary volume containing both void and solid
voxels. They represent the void/solid interface as a triangular surface, not the
full voxel field.

### Loading Voxel Volumes

Use `load_volume` when only the array is needed:

```python
from voids.io import load_volume

volume = load_volume("outputs/synthetic_case/macro_micro_vug.h5")
```

Use `load_volume_data` when physical resolution matters for porosity maps,
permeability maps, surface exports, or external FEM/continuum solvers:

```python
from voids.io import load_volume_data

volume_data = load_volume_data("outputs/synthetic_case/macro_micro_vug.tiff")
```

TIFF files may contain some resolution tags in particular software workflows,
but they should not be treated as a reliable source of 3-D voxel spacing. If the
TIFF was not written by `voids` with its JSON sidecar, pass the voxel size
explicitly:

```python
external_scan = load_volume_data(
    "micro_ct_stack.tiff",
    voxel_size=(40.0e-6, 40.0e-6, 40.0e-6),
    units={"length": "m"},
)
```

Raw binary files have no self-describing shape, dtype, or voxel resolution. If
the `voids` sidecar is absent, provide shape, dtype, and voxel size explicitly
when those quantities matter:

```python
volume_data = load_volume_data(
    "macro_micro_vug.raw",
    shape=(160, 160, 160),
    dtype="uint8",
    voxel_size=(40.0e-6, 40.0e-6, 40.0e-6),
    units={"length": "m"},
)
```

### Loading Surface Meshes

Surface meshes can be read back with:

```python
from voids.io import load_surface_mesh

mesh = load_surface_mesh("outputs/synthetic_case/macro_micro_vug.obj")
```

Surface files are geometric interchange files. They do not replace the voxel
field when voxel-wise phase information is needed.

::: voids.io.volume

---

## HDF5

::: voids.io.hdf5

---

## PoreSpy Import

::: voids.io.porespy

---

## OpenPNM Interoperability

::: voids.io.openpnm
