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
| [`voids.io`](io.md) | HDF5, PoreSpy, and OpenPNM I/O |
| [`voids.generators`](generators.md) | Synthetic and mesh-based network generators |
| [`voids.examples`](examples.md) | Deterministic synthetic networks and images for testing/demos |
| [`voids.image`](image.md) | Image processing and connectivity helpers |
| [`voids.visualization`](visualization.md) | Plotly and PyVista network rendering |
| [`voids.simulators`](simulators.md) | Ready-to-run simulation entry points |
| [`voids.benchmarks`](benchmarks.md) | Cross-check and validation utilities |

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
