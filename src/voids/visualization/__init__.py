"""Optional visualization utilities (PyVista- and Plotly-backed)."""

from __future__ import annotations

from voids.visualization.fields import (
    plot_scalar_midplanes,
    plot_vector_midplanes,
    reference_pressure_to_outlet,
    reconstruct_tpfa_cell_velocity,
    sample_dolfinx_function_at_points,
    sample_dolfinx_function_on_grid,
    vector_magnitude,
    write_dolfinx_function_xdmf,
    write_fem_result_xdmf,
    write_structured_vector_field,
)
from voids.visualization.plotly import plot_network_plotly

_PYVISTA_EXPORTS = {"network_to_pyvista_polydata", "plot_network_pyvista"}


def __getattr__(name: str) -> object:
    """Load PyVista-backed exports only when callers request them."""

    if name in _PYVISTA_EXPORTS:
        from voids.visualization.pyvista import (
            network_to_pyvista_polydata,
            plot_network_pyvista,
        )

        values = {
            "network_to_pyvista_polydata": network_to_pyvista_polydata,
            "plot_network_pyvista": plot_network_pyvista,
        }
        value = values[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "network_to_pyvista_polydata",
    "plot_network_pyvista",
    "plot_network_plotly",
    "plot_scalar_midplanes",
    "plot_vector_midplanes",
    "reference_pressure_to_outlet",
    "reconstruct_tpfa_cell_velocity",
    "sample_dolfinx_function_at_points",
    "sample_dolfinx_function_on_grid",
    "vector_magnitude",
    "write_dolfinx_function_xdmf",
    "write_fem_result_xdmf",
    "write_structured_vector_field",
]
