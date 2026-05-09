# API Reference

`voids` is organized into focused modules, each with a clearly scoped responsibility.

---

## Module Overview

| Module | Description |
|---|---|
| [`voids.core`](core.md) | Network, SampleGeometry, and Provenance data structures |
| [`voids.physics`](physics.md) | Petrophysics and single-phase flow solver |
| [`voids.geom`](geom.md) | Geometry helpers and characteristic-size normalization |
| [`voids.graph`](graph.md) | Graph algorithms: connectivity and metrics |
| [`voids.linalg`](linalg.md) | Linear-algebra assembly, solvers, and diagnostics |
| [`voids.io`](io.md) | HDF5, PoreSpy, OpenPNM, image-volume, and surface-mesh I/O |
| [`voids.generators`](generators.md) | Synthetic and mesh-based network generators |
| [`voids.examples`](examples.md) | Deterministic synthetic networks and images for testing/demos |
| [`voids.image`](image.md) | Image processing and connectivity helpers |
| [`voids.visualization`](visualization.md) | Plotly and PyVista network rendering |
| [`voids.simulators`](simulators.md) | Ready-to-run simulation entry points |
| [`voids.benchmarks`](benchmarks.md) | Verification and validation benchmark utilities |

---

## Common Tasks

If you already know what you want to do, these are the main entry points:

| Task | Primary API |
|---|---|
| Create a minimal synthetic network | `voids.examples.make_linear_chain_network` |
| Create a structured mesh-like network | `voids.examples.make_cartesian_mesh_network` |
| Scale an imported network from voxel units | `voids.io.porespy.scale_porespy_geometry` |
| Infer Cartesian boundary labels | `voids.io.porespy.ensure_cartesian_boundary_labels` |
| Import a PoreSpy/OpenPNM-style dictionary | `voids.io.porespy.from_porespy` |
| Export to an OpenPNM-style dictionary or object | `voids.io.openpnm.to_openpnm_dict`, `voids.io.openpnm.to_openpnm_network` |
| Export or reload an image volume | `voids.io.VolumeData`, `voids.io.save_volume_bundle`, `voids.io.load_volume_data` |
| Extract a PREGO-style network from a segmented image | `voids.image.extract_prego_network_dict` |
| Extract a native maximal-ball network from a segmented image | `voids.image.extract_maximal_ball_network_dict` |
| Compute porosity and connectivity diagnostics | `voids.physics.petrophysics` |
| Solve single-phase flow | `voids.physics.singlephase.solve` |
| Save and reload a canonical network | `voids.io.hdf5.save_hdf5`, `voids.io.hdf5.load_hdf5` |
| Render a network interactively | `voids.visualization.plotly`, `voids.visualization.pyvista` |
| Cross-check a workflow against OpenPNM conventions | `voids.benchmarks.crosscheck` |

---

## Public Top-Level Imports

The main `voids` package re-exports the three primary data structures:

```python
from voids import Network, SampleGeometry, Provenance
```

The package version is available as:

```python
import voids
print(voids.__version__)
```

For the data model and interpretation behind these APIs, see
[Concepts and Conventions](../concepts.md) and
[Theoretical Background](../background.md).
