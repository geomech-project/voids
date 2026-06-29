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


def _assemble_tpfa_system(
    values: np.ndarray,
    *,
    cell_size: tuple[float, ...],
    flow_axis_index: int,
    viscosity: float,
    pressure_inlet: float,
    pressure_outlet: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble the TPFA pressure matrix using vectorized face operations."""

    shape = values.shape
    ndim = values.ndim
    n_cells = int(values.size)
    flat_ids = np.arange(n_cells, dtype=np.int64).reshape(shape, order="C")
    diagonal = np.zeros(n_cells, dtype=float)
    rhs = np.zeros(n_cells, dtype=float)

    boundary_factor = _face_area(cell_size, flow_axis_index) / (
        viscosity * (cell_size[flow_axis_index] / 2.0)
    )
    inlet_selector: list[slice | int] = [slice(None)] * ndim
    outlet_selector: list[slice | int] = [slice(None)] * ndim
    inlet_selector[flow_axis_index] = 0
    outlet_selector[flow_axis_index] = shape[flow_axis_index] - 1
    inlet = tuple(inlet_selector)
    outlet = tuple(outlet_selector)

    inlet_ids = flat_ids[inlet].ravel(order="C")
    inlet_transmissibility = (
        np.asarray(values[inlet], dtype=float).ravel(order="C") * boundary_factor
    )
    inlet_active = inlet_transmissibility > 0.0
    diagonal[inlet_ids[inlet_active]] += inlet_transmissibility[inlet_active]
    rhs[inlet_ids[inlet_active]] += inlet_transmissibility[inlet_active] * float(pressure_inlet)

    outlet_ids = flat_ids[outlet].ravel(order="C")
    outlet_transmissibility = (
        np.asarray(values[outlet], dtype=float).ravel(order="C") * boundary_factor
    )
    outlet_active = outlet_transmissibility > 0.0
    diagonal[outlet_ids[outlet_active]] += outlet_transmissibility[outlet_active]
    rhs[outlet_ids[outlet_active]] += outlet_transmissibility[outlet_active] * float(
        pressure_outlet
    )

    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []

    for direction in range(ndim):
        left_selector = [slice(None)] * ndim
        right_selector = [slice(None)] * ndim
        left_selector[direction] = slice(0, -1)
        right_selector[direction] = slice(1, None)
        left = tuple(left_selector)
        right = tuple(right_selector)

        left_values = np.asarray(values[left], dtype=float)
        right_values = np.asarray(values[right], dtype=float)
        active = (left_values > 0.0) & (right_values > 0.0)
        if not np.any(active):
            continue

        left_active = left_values[active]
        right_active = right_values[active]
        face_permeability = 2.0 * left_active * right_active / (left_active + right_active)
        transmissibility = (
            face_permeability
            * _face_area(cell_size, direction)
            / (viscosity * cell_size[direction])
        )

        left_ids = flat_ids[left][active]
        right_ids = flat_ids[right][active]
        np.add.at(diagonal, left_ids, transmissibility)
        np.add.at(diagonal, right_ids, transmissibility)
        row_blocks.extend((left_ids, right_ids))
        col_blocks.extend((right_ids, left_ids))
        data_blocks.extend((-transmissibility, -transmissibility))

    row_blocks.append(np.arange(n_cells, dtype=np.int64))
    col_blocks.append(np.arange(n_cells, dtype=np.int64))
    data_blocks.append(diagonal)
    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    data = np.concatenate(data_blocks)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    return matrix, rhs


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
    matrix, rhs = _assemble_tpfa_system(
        values,
        cell_size=size,
        flow_axis_index=axis,
        viscosity=float(viscosity),
        pressure_inlet=float(pressure_inlet),
        pressure_outlet=float(pressure_outlet),
    )

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

    inlet_selector: list[slice | int] = [slice(None)] * ndim
    outlet_selector: list[slice | int] = [slice(None)] * ndim
    inlet_selector[axis] = 0
    outlet_selector[axis] = shape[axis] - 1
    boundary_factor = _face_area(size, axis) / (viscosity * (size[axis] / 2.0))
    inlet = tuple(inlet_selector)
    outlet = tuple(outlet_selector)
    inlet_transmissibility = values[inlet] * boundary_factor
    outlet_transmissibility = values[outlet] * boundary_factor
    inlet_flow = float(np.sum(inlet_transmissibility * (float(pressure_inlet) - pressure[inlet])))
    outlet_flow = float(
        np.sum(outlet_transmissibility * (pressure[outlet] - float(pressure_outlet)))
    )

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
