"""Optional visualization utilities (PyVista- and Plotly-backed)."""

from voids.visualization.fields import (
    plot_scalar_midplanes,
    plot_vector_midplanes,
    reference_pressure_to_outlet,
    reconstruct_tpfa_cell_velocity,
    sample_dolfinx_function_on_grid,
    vector_magnitude,
    write_dolfinx_function_xdmf,
    write_structured_vector_field,
)
from voids.visualization.plotly import plot_network_plotly
from voids.visualization.pyvista import network_to_pyvista_polydata, plot_network_pyvista

__all__ = [
    "network_to_pyvista_polydata",
    "plot_network_pyvista",
    "plot_network_plotly",
    "plot_scalar_midplanes",
    "plot_vector_midplanes",
    "reference_pressure_to_outlet",
    "reconstruct_tpfa_cell_velocity",
    "sample_dolfinx_function_on_grid",
    "vector_magnitude",
    "write_dolfinx_function_xdmf",
    "write_structured_vector_field",
]
