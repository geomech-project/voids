# Visualization

The `voids.visualization` sub-package provides network rendering via Plotly and
PyVista, plus reusable scalar/vector field plotting and ParaView export helpers
for map-based validation workflows.

!!! note "PyVista dependency"
    PyVista is installed as a core dependency of `voids` and is available by
    default when you install the package.

---

## Plotly

::: voids.visualization.plotly

---

## PyVista

::: voids.visualization.pyvista

---

## Field Plots And Exports

The field helpers are intended for pressure and velocity diagnostics generated
by finite-volume, finite-element, and lattice-Boltzmann single-phase workflows.
Velocity midplane plots draw velocity magnitude as the scalar background and
in-plane quiver arrows as the vector overlay. Structured vector fields can be
written as VTU/VTK cell data, and DOLFINx functions can be written as XDMF/HDF5
after interpolation to first-order visualization spaces for ParaView
compatibility. For pressure diagnostics, `reference_pressure_to_outlet` removes
arbitrary pressure gauges by shifting the field so the outlet layer has a
prescribed reference pressure, while preserving all pressure differences.

::: voids.visualization.fields
