from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np

from voids.image.porosity import PermeabilityMap, PorosityMap


_AXIS_NAMES = ("x", "y", "z")
_MIN_MARKER = {"x": 1, "y": 3, "z": 5}
_MAX_MARKER = {"x": 2, "y": 4, "z": 6}


@dataclass(slots=True)
class FEniCSSolverOptions:
    """PETSc solver controls for FEniCSx linear problems.

    Defaults use a direct LU solve with MUMPS, matching the Pixi ``fem``
    feature stack. Override ``petsc_options`` when a different PETSc
    installation or preconditioned iterative strategy is desired.
    """

    petsc_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "pc_factor_shift_type": "nonzero",
            "ksp_error_if_not_converged": True,
        }
    )
    petsc_options_prefix: str = "voids_fem_"


@dataclass(slots=True)
class FEMMapProblem:
    """Porosity/permeability coefficient maps for FEM single-phase solves.

    Parameters
    ----------
    permeability_map :
        Scalar cell-wise permeability map.
    porosity_map :
        Optional porosity map on the same grid. Brinkman solves use this field
        in ``nu_eff = mu / max(phi, porosity_floor)``. Darcy-only comparison
        solves do not use it.
    viscosity :
        Dynamic viscosity ``mu``.
    porosity_floor :
        Lower bound used only in the Brinkman effective-viscosity coefficient.
    permeability_floor :
        Lower bound used in ``gamma = mu / max(K, permeability_floor)``.
    """

    permeability_map: PermeabilityMap
    porosity_map: PorosityMap | None = None
    viscosity: float = 1.0
    porosity_floor: float = 1.0e-6
    permeability_floor: float = 1.0e-30

    def __post_init__(self) -> None:
        if self.viscosity <= 0.0 or not np.isfinite(self.viscosity):
            raise ValueError("viscosity must be positive and finite")
        if self.porosity_floor <= 0.0 or not np.isfinite(self.porosity_floor):
            raise ValueError("porosity_floor must be positive and finite")
        if self.permeability_floor <= 0.0 or not np.isfinite(self.permeability_floor):
            raise ValueError("permeability_floor must be positive and finite")
        if self.permeability_map.ndim not in {2, 3}:
            raise ValueError("permeability_map must be 2D or 3D")
        if self.porosity_map is not None:
            if self.porosity_map.shape != self.permeability_map.shape:
                raise ValueError("porosity_map and permeability_map must have the same shape")
            porosity_cell_size = tuple(
                float(v) for v in cast(tuple[float, ...], self.porosity_map.cell_size)
            )
            if porosity_cell_size != _cell_size_tuple(self.permeability_map):
                raise ValueError("porosity_map and permeability_map must have the same cell_size")


@dataclass(slots=True)
class FEMSinglePhaseResult:
    """Finite-element single-phase flow result."""

    method: str
    formulation: str
    flow_axis: str
    permeability: float
    flow_rate: float
    pressure_inlet: float
    pressure_outlet: float
    pressure_drop: float
    viscosity: float
    domain_length: float
    cross_section_area: float
    solve_seconds: float
    velocity: Any
    pressure: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _DolfinxAPI:
    MPI: Any
    basix_ufl: Any
    fem: Any
    mesh: Any
    petsc: Any
    ufl: Any


@dataclass(slots=True)
class _FEMContext:
    api: _DolfinxAPI
    mesh: Any
    ds: Any
    dx: Any
    dS: Any
    normal: Any
    coefficients: dict[str, Any]
    domain_length: float
    cross_section_area: float


def _require_dolfinx() -> _DolfinxAPI:
    try:
        import basix.ufl as basix_ufl
        from dolfinx import fem, mesh
        import dolfinx.fem.petsc as petsc
        from mpi4py import MPI
        import ufl
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "FEniCSx FEM backends require DOLFINx. Use the Pixi 'fem' feature/environment "
            "or install a compatible fenics-dolfinx stack before calling voids.fem."
        ) from exc
    return _DolfinxAPI(
        MPI=MPI,
        basix_ufl=basix_ufl,
        fem=fem,
        mesh=mesh,
        petsc=petsc,
        ufl=ufl,
    )


def _axis_index(axis: str, ndim: int) -> int:
    if axis not in _AXIS_NAMES[:ndim]:
        raise ValueError(f"flow_axis must be one of {_AXIS_NAMES[:ndim]}, got {axis!r}")
    return _AXIS_NAMES.index(axis)


def _cell_size_tuple(permeability_map: PermeabilityMap) -> tuple[float, ...]:
    return tuple(float(v) for v in cast(tuple[float, ...], permeability_map.cell_size))


def _origin_tuple(permeability_map: PermeabilityMap) -> tuple[float, ...]:
    return tuple(float(v) for v in cast(tuple[float, ...], permeability_map.origin))


def _close_coordinate(values: Any, coordinate: float, *, atol: float) -> np.ndarray:
    return np.isclose(np.asarray(values, dtype=float), float(coordinate), atol=float(atol))


def _match_point(values: Any, point: np.ndarray, *, ndim: int, atol: float) -> np.ndarray:
    coords = np.asarray(values[:ndim], dtype=float).T
    matches = np.all(np.isclose(coords, point, atol=float(atol)), axis=1)
    return np.asarray(matches, dtype=bool)


def _domain_length(shape: tuple[int, ...], cell_size: tuple[float, ...], axis_index: int) -> float:
    return float(shape[axis_index] * cell_size[axis_index])


def _cross_section_area(
    shape: tuple[int, ...], cell_size: tuple[float, ...], axis_index: int
) -> float:
    return float(np.prod([shape[i] * cell_size[i] for i in range(len(shape)) if i != axis_index]))


def _create_box_mesh(api: _DolfinxAPI, problem: FEMMapProblem) -> Any:
    shape = problem.permeability_map.shape
    cell_size = _cell_size_tuple(problem.permeability_map)
    origin = _origin_tuple(problem.permeability_map)
    upper = tuple(origin[i] + shape[i] * cell_size[i] for i in range(len(shape)))
    if len(shape) == 2:
        return api.mesh.create_rectangle(
            api.MPI.COMM_WORLD,
            [origin, upper],
            list(shape),
            cell_type=api.mesh.CellType.triangle,
        )
    return api.mesh.create_box(
        api.MPI.COMM_WORLD,
        [origin, upper],
        list(shape),
        cell_type=api.mesh.CellType.tetrahedron,
    )


def _facet_tags(api: _DolfinxAPI, domain: Any, problem: FEMMapProblem) -> Any:
    ndim = problem.permeability_map.ndim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    origin = np.asarray(_origin_tuple(problem.permeability_map), dtype=float)
    shape = np.asarray(problem.permeability_map.shape, dtype=float)
    cell_size = np.asarray(_cell_size_tuple(problem.permeability_map), dtype=float)
    upper = origin + shape * cell_size
    extent = float(np.max(upper - origin))
    atol = max(extent * 1.0e-10, float(np.finfo(float).eps))

    facets: list[np.ndarray] = []
    markers: list[np.ndarray] = []
    marker_pairs = ((1, 2), (3, 4), (5, 6))
    for axis in range(ndim):
        low_marker, high_marker = marker_pairs[axis]
        lower_coordinate = float(origin[axis])
        upper_coordinate = float(upper[axis])
        low = api.mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x, axis=axis, lower_coordinate=lower_coordinate: _close_coordinate(
                x[axis], lower_coordinate, atol=atol
            ),
        )
        high = api.mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x, axis=axis, upper_coordinate=upper_coordinate: _close_coordinate(
                x[axis], upper_coordinate, atol=atol
            ),
        )
        facets.extend((low, high))
        markers.extend(
            (
                np.full(low.shape, low_marker, dtype=np.int32),
                np.full(high.shape, high_marker, dtype=np.int32),
            )
        )
    facet_array = np.hstack(facets).astype(np.int32)
    marker_array = np.hstack(markers).astype(np.int32)
    order = np.argsort(facet_array)
    return api.mesh.meshtags(domain, fdim, facet_array[order], marker_array[order])


def _cell_values_from_map(
    api: _DolfinxAPI,
    space: Any,
    values: np.ndarray,
    problem: FEMMapProblem,
) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates(), dtype=float)[:, : values.ndim]
    origin = np.asarray(_origin_tuple(problem.permeability_map), dtype=float)
    cell_size = np.asarray(_cell_size_tuple(problem.permeability_map), dtype=float)
    indices = np.floor((coords - origin) / cell_size).astype(int)
    for axis in range(values.ndim):
        indices[:, axis] = np.clip(indices[:, axis], 0, values.shape[axis] - 1)
    if values.ndim == 2:
        return np.asarray(values[indices[:, 0], indices[:, 1]], dtype=float)
    return np.asarray(values[indices[:, 0], indices[:, 1], indices[:, 2]], dtype=float)


def _dg0_function(api: _DolfinxAPI, domain: Any, values: np.ndarray, *, name: str) -> Any:
    space = api.fem.functionspace(
        domain,
        api.basix_ufl.element("DG", domain.basix_cell(), 0),
    )
    field = api.fem.Function(space)
    field.name = name
    field.x.array[:] = np.asarray(values, dtype=float)[: field.x.array.size]
    field.x.scatter_forward()
    return field


def _build_context(problem: FEMMapProblem, *, flow_axis: str) -> _FEMContext:
    api = _require_dolfinx()
    axis = _axis_index(flow_axis, problem.permeability_map.ndim)
    domain = _create_box_mesh(api, problem)
    tags = _facet_tags(api, domain, problem)
    dx = api.ufl.Measure("dx", domain=domain)
    ds = api.ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = api.ufl.Measure("dS", domain=domain)
    normal = api.ufl.FacetNormal(domain)

    dg0 = api.fem.functionspace(
        domain,
        api.basix_ufl.element("DG", domain.basix_cell(), 0),
    )
    permeability_raw = _cell_values_from_map(api, dg0, problem.permeability_map.values, problem)
    permeability = np.maximum(permeability_raw, float(problem.permeability_floor))
    if problem.porosity_map is None:
        porosity_raw = np.ones_like(permeability)
    else:
        porosity_raw = _cell_values_from_map(api, dg0, problem.porosity_map.values, problem)
    porosity = np.maximum(porosity_raw, float(problem.porosity_floor))

    gamma = _dg0_function(
        api,
        domain,
        float(problem.viscosity) / permeability,
        name="Darcy drag mu / K",
    )
    nu_eff = _dg0_function(
        api,
        domain,
        float(problem.viscosity) / porosity,
        name="Brinkman effective viscosity mu / phi",
    )

    shape = problem.permeability_map.shape
    size = _cell_size_tuple(problem.permeability_map)
    return _FEMContext(
        api=api,
        mesh=domain,
        ds=ds,
        dx=dx,
        dS=dS,
        normal=normal,
        coefficients={
            "gamma": gamma,
            "nu_eff": nu_eff,
            "permeability_values": permeability,
            "porosity_values": porosity,
        },
        domain_length=_domain_length(shape, size, axis),
        cross_section_area=_cross_section_area(shape, size, axis),
    )


def _mixed_space(
    api: _DolfinxAPI, domain: Any, *, velocity_degree: int, pressure_family: str
) -> Any:
    velocity_element = api.basix_ufl.element(
        "Lagrange",
        domain.basix_cell(),
        velocity_degree,
        shape=(domain.geometry.dim,),
    )
    pressure_element = api.basix_ufl.element(pressure_family, domain.basix_cell(), 1)
    return api.fem.functionspace(
        domain,
        api.basix_ufl.mixed_element([velocity_element, pressure_element]),
    )


def _side_wall_bcs(context: _FEMContext, mixed_space: Any, *, flow_axis: str) -> list[Any]:
    axis = _axis_index(flow_axis, context.mesh.geometry.dim)
    local_origin = np.min(context.mesh.geometry.x[:, : context.mesh.geometry.dim], axis=0)
    local_upper = np.max(context.mesh.geometry.x[:, : context.mesh.geometry.dim], axis=0)
    problem_origin = np.asarray(
        context.mesh.comm.allreduce(local_origin, op=context.api.MPI.MIN),
        dtype=float,
    )
    problem_upper = np.asarray(
        context.mesh.comm.allreduce(local_upper, op=context.api.MPI.MAX),
        dtype=float,
    )
    extent = float(np.max(problem_upper - problem_origin))
    atol = max(extent * 1.0e-10, float(np.finfo(float).eps))
    bcs: list[Any] = []
    for side_axis in range(context.mesh.geometry.dim):
        if side_axis == axis:
            continue
        component_space = mixed_space.sub(0).sub(side_axis)
        collapsed, _ = component_space.collapse()
        zero = context.api.fem.Function(collapsed)
        zero.x.array[:] = 0.0
        for coordinate in (float(problem_origin[side_axis]), float(problem_upper[side_axis])):
            dofs = context.api.fem.locate_dofs_geometrical(
                (component_space, collapsed),
                lambda x, side_axis=side_axis, coordinate=coordinate: _close_coordinate(
                    x[side_axis], coordinate, atol=atol
                ),
            )
            bcs.append(context.api.fem.dirichletbc(zero, dofs, component_space))
    return bcs


def _pressure_gauge_bc(context: _FEMContext, mixed_space: Any) -> Any:
    pressure_space = mixed_space.sub(1)
    collapsed, _ = pressure_space.collapse()
    zero = context.api.fem.Function(collapsed)
    zero.x.array[:] = 0.0

    local_origin = np.min(context.mesh.geometry.x[:, : context.mesh.geometry.dim], axis=0)
    problem_origin = np.asarray(
        context.mesh.comm.allreduce(local_origin, op=context.api.MPI.MIN),
        dtype=float,
    )
    extent = float(
        context.mesh.comm.allreduce(
            np.max(context.mesh.geometry.x[:, : context.mesh.geometry.dim]) - np.min(local_origin),
            op=context.api.MPI.MAX,
        )
    )
    atol = max(extent * 1.0e-10, float(np.finfo(float).eps))
    dofs = context.api.fem.locate_dofs_geometrical(
        (pressure_space, collapsed),
        lambda x: _match_point(x, problem_origin, ndim=context.mesh.geometry.dim, atol=atol),
    )
    if dofs[0].size > 1:
        dofs = [dofs[0][:1], dofs[1][:1]]
    return context.api.fem.dirichletbc(zero, dofs, pressure_space)


def _pressure_boundary_load(
    context: _FEMContext,
    test_velocity: Any,
    *,
    flow_axis: str,
    pressure_inlet: float,
    pressure_outlet: float,
) -> Any:
    ufl = context.api.ufl
    n = context.normal
    return -context.api.fem.Constant(context.mesh, float(pressure_inlet)) * ufl.dot(
        test_velocity, n
    ) * context.ds(_MIN_MARKER[flow_axis]) - context.api.fem.Constant(
        context.mesh, float(pressure_outlet)
    ) * ufl.dot(test_velocity, n) * context.ds(_MAX_MARKER[flow_axis])


def _assemble_scalar(context: _FEMContext, expression: Any) -> float:
    local = context.api.fem.assemble_scalar(context.api.fem.form(expression))
    return float(context.mesh.comm.allreduce(local, op=context.api.MPI.SUM))


def _solve_mixed_problem(
    context: _FEMContext,
    *,
    form: Any,
    rhs: Any,
    bcs: list[Any],
    options: FEniCSSolverOptions | None,
    prefix_suffix: str,
) -> tuple[Any, float]:
    solver_options = options or FEniCSSolverOptions()
    start = perf_counter()
    problem = context.api.petsc.LinearProblem(
        form,
        rhs,
        bcs=bcs,
        petsc_options_prefix=f"{solver_options.petsc_options_prefix}{prefix_suffix}_",
        petsc_options=dict(solver_options.petsc_options),
    )
    solution = problem.solve()
    return solution, perf_counter() - start


def _collapse_solution(solution: Any) -> tuple[Any, Any]:
    velocity = solution.sub(0).collapse()
    pressure = solution.sub(1).collapse()
    velocity.name = "velocity"
    pressure.name = "pressure"
    return velocity, pressure


def _zero_mean_pressure(context: _FEMContext, pressure: Any) -> Any:
    volume = _assemble_scalar(context, 1.0 * context.dx)
    mean_value = _assemble_scalar(context, pressure * context.dx) / volume
    pressure.x.array[:] -= mean_value
    pressure.x.scatter_forward()
    return pressure


def _result_from_solution(
    context: _FEMContext,
    solution: Any,
    *,
    method: str,
    formulation: str,
    flow_axis: str,
    pressure_inlet: float,
    pressure_outlet: float,
    viscosity: float,
    solve_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> FEMSinglePhaseResult:
    velocity, pressure = _collapse_solution(solution)
    pressure = _zero_mean_pressure(context, pressure)
    flow_rate = _assemble_scalar(
        context,
        context.api.ufl.dot(velocity, context.normal) * context.ds(_MAX_MARKER[flow_axis]),
    )
    pressure_drop = float(pressure_inlet) - float(pressure_outlet)
    permeability = float(
        flow_rate * viscosity * context.domain_length / (context.cross_section_area * pressure_drop)
    )
    return FEMSinglePhaseResult(
        method=method,
        formulation=formulation,
        flow_axis=flow_axis,
        permeability=permeability,
        flow_rate=float(flow_rate),
        pressure_inlet=float(pressure_inlet),
        pressure_outlet=float(pressure_outlet),
        pressure_drop=pressure_drop,
        viscosity=float(viscosity),
        domain_length=context.domain_length,
        cross_section_area=context.cross_section_area,
        solve_seconds=float(solve_seconds),
        velocity=velocity,
        pressure=pressure,
        metadata=dict(metadata or {}),
    )


def _validate_pressure_drop(pressure_inlet: float, pressure_outlet: float) -> None:
    if not np.isfinite(pressure_inlet) or not np.isfinite(pressure_outlet):
        raise ValueError("pressure values must be finite")
    if pressure_inlet <= pressure_outlet:
        raise ValueError("pressure_inlet must be greater than pressure_outlet")


def _solve_with_form_builder(
    problem: FEMMapProblem,
    *,
    flow_axis: str,
    pressure_inlet: float,
    pressure_outlet: float,
    options: FEniCSSolverOptions | None,
    velocity_degree: int,
    pressure_family: str,
    method: str,
    formulation: str,
    prefix_suffix: str,
    form_builder: Callable[[_FEMContext, Any, Any, Any, Any], Any],
) -> FEMSinglePhaseResult:
    _validate_pressure_drop(pressure_inlet, pressure_outlet)
    context = _build_context(problem, flow_axis=flow_axis)
    W = _mixed_space(
        context.api,
        context.mesh,
        velocity_degree=velocity_degree,
        pressure_family=pressure_family,
    )
    u, p = context.api.ufl.TrialFunctions(W)
    v, q = context.api.ufl.TestFunctions(W)
    form = form_builder(context, u, p, v, q)
    rhs = _pressure_boundary_load(
        context,
        v,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
    )
    bcs = _side_wall_bcs(context, W, flow_axis=flow_axis)
    bcs.append(_pressure_gauge_bc(context, W))
    solution, solve_seconds = _solve_mixed_problem(
        context,
        form=form,
        rhs=rhs,
        bcs=bcs,
        options=options,
        prefix_suffix=prefix_suffix,
    )
    return _result_from_solution(
        context,
        solution,
        method=method,
        formulation=formulation,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        viscosity=problem.viscosity,
        solve_seconds=solve_seconds,
        metadata={
            "velocity_degree": velocity_degree,
            "pressure_family": pressure_family,
            "porosity_floor": problem.porosity_floor,
            "permeability_floor": problem.permeability_floor,
        },
    )


__all__ = [
    "FEMMapProblem",
    "FEMSinglePhaseResult",
    "FEniCSSolverOptions",
    "_build_context",
    "_solve_with_form_builder",
]
