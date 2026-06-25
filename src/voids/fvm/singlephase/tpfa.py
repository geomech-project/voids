from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any, cast
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning

from voids.linalg.solve import SolverParameters, solve_linear_system

from voids.image.porosity import PermeabilityMap


_AXIS_NAMES = ("x", "y", "z")


@dataclass(slots=True)
class TPFAResult:
    """Result of a cell-centered two-point flux approximation solve.

    Parameters
    ----------
    pressure :
        Cell-centered pressure field. It has the same shape as the input
        permeability field.
    flow_axis :
        Axis along which the pressure drop was imposed.
    permeability :
        Effective permeability inferred from the outlet flow rate through
        Darcy's law.
    flow_rate :
        Total outlet flow rate. For 2-D maps this is the flow rate per unit
        out-of-plane thickness.
    inlet_flow_rate, outlet_flow_rate :
        Boundary flow rates computed at the inlet and outlet faces. Their
        agreement is a finite-volume mass-balance diagnostic.
    mass_balance_error :
        Absolute difference between inlet and outlet flow rates normalized by
        the larger boundary flow magnitude.
    """

    pressure: np.ndarray
    flow_axis: str
    permeability: float
    flow_rate: float
    inlet_flow_rate: float
    outlet_flow_rate: float
    mass_balance_error: float
    pressure_inlet: float
    pressure_outlet: float
    viscosity: float
    domain_length: float
    cross_section_area: float
    cell_size: tuple[float, ...]
    matrix_nnz: int
    solve_seconds: float
    solver_method: str
    solver_info: dict[str, Any] = field(default_factory=dict)
    residual_relative: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_permeability_array(
    permeability: PermeabilityMap | np.ndarray,
    *,
    cell_size: float | Sequence[float] | None,
) -> tuple[np.ndarray, tuple[float, ...], dict[str, Any]]:
    if isinstance(permeability, PermeabilityMap):
        values = np.asarray(permeability.values, dtype=float)
        size = tuple(float(v) for v in cast(tuple[float, ...], permeability.cell_size))
        metadata = dict(permeability.metadata)
    else:
        values = np.asarray(permeability, dtype=float)
        if values.ndim not in {2, 3}:
            raise ValueError("permeability must be a 2D or 3D field")
        if cell_size is None:
            size = (1.0,) * values.ndim
        elif isinstance(cell_size, Sequence) and not isinstance(cell_size, str):
            size = tuple(float(v) for v in cell_size)
        else:
            size = (float(cell_size),) * values.ndim
        metadata = {}

    if values.ndim not in {2, 3}:
        raise ValueError("permeability must be a 2D or 3D field")
    if len(size) != values.ndim:
        raise ValueError("cell_size dimensionality must match permeability.ndim")
    if any(v <= 0.0 or not np.isfinite(v) for v in size):
        raise ValueError("cell_size values must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("permeability must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("permeability must be non-negative")
    return values, size, metadata


def _axis_index(axis: str, ndim: int) -> int:
    if axis not in _AXIS_NAMES[:ndim]:
        raise ValueError(f"flow_axis must be one of {_AXIS_NAMES[:ndim]}, got {axis!r}")
    return _AXIS_NAMES.index(axis)


def _harmonic_face_permeability(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float(2.0 * left * right / (left + right))


def _face_area(cell_size: tuple[float, ...], axis_index: int) -> float:
    return float(np.prod([v for i, v in enumerate(cell_size) if i != axis_index]))


def _domain_length(shape: tuple[int, ...], cell_size: tuple[float, ...], axis_index: int) -> float:
    return float(shape[axis_index] * cell_size[axis_index])


def _cross_section_area(
    shape: tuple[int, ...], cell_size: tuple[float, ...], axis_index: int
) -> float:
    return float(np.prod([shape[i] * cell_size[i] for i in range(len(shape)) if i != axis_index]))


def solve_tpfa(
    permeability: PermeabilityMap | np.ndarray,
    *,
    flow_axis: str = "x",
    viscosity: float = 1.0,
    pressure_inlet: float = 1.0,
    pressure_outlet: float = 0.0,
    cell_size: float | Sequence[float] | None = None,
    solver_method: str = "direct",
    solver_parameters: SolverParameters | None = None,
) -> TPFAResult:
    """Solve Darcy flow on a regular permeability map with TPFA.

    The discrete unknown is one pressure value per map cell. Internal face
    transmissibilities use the harmonic mean of adjacent permeability values,
    while inlet and outlet Dirichlet pressures are imposed half a cell from the
    adjacent cell center. All transverse boundaries are no-flow boundaries.

    Parameters
    ----------
    permeability :
        Cell-wise scalar permeability map. Zero values are treated as
        impermeable. Completely isolated zero-transmissibility cells may make
        the pressure system singular; use a small permeability floor before
        calling this solver if the map contains solid-like cells.
    flow_axis :
        Axis along which ``pressure_inlet > pressure_outlet`` is imposed.
    viscosity :
        Dynamic viscosity multiplying Darcy resistance.
    pressure_inlet, pressure_outlet :
        Dirichlet pressure values imposed on the minimum and maximum faces of
        ``flow_axis``.
    cell_size :
        Physical cell size used when ``permeability`` is an array rather than a
        :class:`~voids.image.porosity.PermeabilityMap`.
    solver_method :
        Sparse linear solver backend passed to
        :func:`voids.linalg.solve.solve_linear_system`. Supported values include
        ``"direct"``, ``"pardiso"``, ``"cg"``, and ``"gmres"``.
    solver_parameters :
        Optional backend-specific controls. For example,
        ``{"rtol": 1e-10, "preconditioner": "pyamg"}`` uses a PyAMG
        preconditioner with SciPy CG, matching the larger notebook comparisons.
    """

    values, size, metadata = _as_permeability_array(permeability, cell_size=cell_size)
    if viscosity <= 0.0 or not np.isfinite(viscosity):
        raise ValueError("viscosity must be positive and finite")
    if not np.isfinite(pressure_inlet) or not np.isfinite(pressure_outlet):
        raise ValueError("pressure values must be finite")
    if pressure_inlet <= pressure_outlet:
        raise ValueError("pressure_inlet must be greater than pressure_outlet")

    shape = values.shape
    ndim = values.ndim
    axis = _axis_index(flow_axis, ndim)
    n_cells = int(values.size)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n_cells, dtype=float)
    diagonal = np.zeros(n_cells, dtype=float)

    def flat(index: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(index, shape, order="C"))

    for index in np.ndindex(shape):
        row = flat(index)
        k_cell = float(values[index])

        if index[axis] == 0 and k_cell > 0.0:
            transmissibility = k_cell * _face_area(size, axis) / (viscosity * (size[axis] / 2.0))
            diagonal[row] += transmissibility
            rhs[row] += transmissibility * float(pressure_inlet)
        if index[axis] == shape[axis] - 1 and k_cell > 0.0:
            transmissibility = k_cell * _face_area(size, axis) / (viscosity * (size[axis] / 2.0))
            diagonal[row] += transmissibility
            rhs[row] += transmissibility * float(pressure_outlet)

        for direction in range(ndim):
            neighbor_index = list(index)
            neighbor_index[direction] += 1
            if neighbor_index[direction] >= shape[direction]:
                continue
            neighbor = tuple(neighbor_index)
            neighbor_row = flat(neighbor)
            k_face = _harmonic_face_permeability(k_cell, float(values[neighbor]))
            if k_face <= 0.0:
                continue
            transmissibility = k_face * _face_area(size, direction) / (viscosity * size[direction])
            diagonal[row] += transmissibility
            diagonal[neighbor_row] += transmissibility
            rows.extend((row, neighbor_row))
            cols.extend((neighbor_row, row))
            data.extend((-transmissibility, -transmissibility))

    rows.extend(range(n_cells))
    cols.extend(range(n_cells))
    data.extend(diagonal.tolist())
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            umfpack = import_module("scikits.umfpack")
            umfpack_warning_type = cast(type[Warning], umfpack.UmfpackWarning)
        except ImportError:  # pragma: no cover - depends on optional solver package

            class FallbackUmfpackWarning(Warning):
                """Fallback warning type used when UMFPACK is not installed."""

            umfpack_warning_type = FallbackUmfpackWarning

        warnings.simplefilter("error", umfpack_warning_type)
        try:
            pressure_vector, solver_info = solve_linear_system(
                matrix,
                rhs,
                method=solver_method,
                solver_parameters=solver_parameters,
            )
        except (MatrixRankWarning, umfpack_warning_type) as exc:
            raise RuntimeError(
                "TPFA pressure system is singular. Check for disconnected zero-permeability "
                "regions or apply a physically justified permeability floor."
            ) from exc
    solve_seconds = perf_counter() - start
    if int(solver_info.get("info", 0)) != 0:
        raise RuntimeError(f"TPFA linear solve did not converge: {solver_info}")

    if not np.all(np.isfinite(pressure_vector)):
        raise RuntimeError("TPFA solve produced non-finite pressures")
    residual = np.asarray(matrix @ pressure_vector - rhs, dtype=float)
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs))
    residual_relative = residual_norm / max(rhs_norm, 1.0e-300)
    pressure = pressure_vector.reshape(shape, order="C")

    inlet_flow = 0.0
    outlet_flow = 0.0
    inlet_selector: list[slice | int] = [slice(None)] * ndim
    outlet_selector: list[slice | int] = [slice(None)] * ndim
    inlet_selector[axis] = 0
    outlet_selector[axis] = shape[axis] - 1
    for index in np.ndindex(values[tuple(inlet_selector)].shape):
        full_index = list(index)
        full_index.insert(axis, 0)
        idx = tuple(full_index)
        k_cell = float(values[idx])
        if k_cell > 0.0:
            transmissibility = k_cell * _face_area(size, axis) / (viscosity * (size[axis] / 2.0))
            inlet_flow += transmissibility * (float(pressure_inlet) - float(pressure[idx]))
    for index in np.ndindex(values[tuple(outlet_selector)].shape):
        full_index = list(index)
        full_index.insert(axis, shape[axis] - 1)
        idx = tuple(full_index)
        k_cell = float(values[idx])
        if k_cell > 0.0:
            transmissibility = k_cell * _face_area(size, axis) / (viscosity * (size[axis] / 2.0))
            outlet_flow += transmissibility * (float(pressure[idx]) - float(pressure_outlet))

    pressure_drop = float(pressure_inlet) - float(pressure_outlet)
    length = _domain_length(shape, size, axis)
    area = _cross_section_area(shape, size, axis)
    permeability_eff = float(outlet_flow * viscosity * length / (area * pressure_drop))
    balance_scale = max(abs(inlet_flow), abs(outlet_flow), 1.0e-300)
    mass_balance_error = float(abs(inlet_flow - outlet_flow) / balance_scale)

    return TPFAResult(
        pressure=pressure,
        flow_axis=flow_axis,
        permeability=permeability_eff,
        flow_rate=float(outlet_flow),
        inlet_flow_rate=float(inlet_flow),
        outlet_flow_rate=float(outlet_flow),
        mass_balance_error=mass_balance_error,
        pressure_inlet=float(pressure_inlet),
        pressure_outlet=float(pressure_outlet),
        viscosity=float(viscosity),
        domain_length=length,
        cross_section_area=area,
        cell_size=size,
        matrix_nnz=int(matrix.nnz),
        solve_seconds=float(solve_seconds),
        solver_method=str(solver_info.get("method", solver_method)),
        solver_info=dict(solver_info),
        residual_relative=residual_relative,
        metadata=metadata,
    )


__all__ = ["TPFAResult", "solve_tpfa"]
