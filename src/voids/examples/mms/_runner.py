from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, cast
import warnings

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from voids.examples.mms._core import (
    BrinkmanMMSCase,
    ConvergenceExpectation,
    MMSConvergenceLevel,
    MMSConvergenceResult,
    MMSDiscreteSolution,
    MMSFacetLaw,
    MMSFacetSizeMode,
    MMSMethod,
)
from voids.fem.singlephase._common import (
    FEMMapProblem,
    FEniCSSolverOptions,
    _assemble_scalar,
    _build_context,
    _collapse_solution,
    _mixed_space,
    _pressure_gauge_bc,
    _require_dolfinx_core,
    _require_dolfinx_petsc,
    _resolve_fem_linear_system_dtype,
    _resolve_linear_backend,
    _solve_mixed_problem,
    _solve_mixed_problem_serial_direct,
)
from voids.fem.singlephase.usfem import (
    USFEMFacetLaw,
    _usfem_stabilization_terms,
)
from voids.image.porosity import PermeabilityMap, PorosityMap


@dataclass(frozen=True, slots=True)
class _MMSMethodSpec:
    velocity_degree: int
    pressure_family: str
    pressure_degree: int
    expected_rates: ConvergenceExpectation


_METHOD_SPECS: dict[MMSMethod, _MMSMethodSpec] = {
    "taylor_hood": _MMSMethodSpec(
        velocity_degree=2,
        pressure_family="Lagrange",
        pressure_degree=1,
        expected_rates=ConvergenceExpectation(
            velocity_l2=3.0,
            velocity_h1=2.0,
            pressure_l2=2.0,
        ),
    ),
    "usfem_p1dg0": _MMSMethodSpec(
        velocity_degree=1,
        pressure_family="DG",
        pressure_degree=0,
        expected_rates=ConvergenceExpectation(
            velocity_l2=2.0,
            velocity_h1=1.0,
            pressure_l2=1.0,
        ),
    ),
    "usfem_p1dg1": _MMSMethodSpec(
        velocity_degree=1,
        pressure_family="DG",
        pressure_degree=1,
        expected_rates=ConvergenceExpectation(
            velocity_l2=2.0,
            velocity_h1=1.0,
            pressure_l2=1.0,
        ),
    ),
}


def available_mms_methods() -> tuple[MMSMethod, ...]:
    """Return the finite-element formulations supported by the MMS runner."""

    return tuple(_METHOD_SPECS)


def observed_rate(
    previous_h: float,
    previous_error: float,
    current_h: float,
    current_error: float,
) -> float:
    """Return the two-level observed order ``log(e0/e1) / log(h0/h1)``."""

    values = (previous_h, previous_error, current_h, current_error)
    if any(value <= 0.0 or not np.isfinite(value) for value in values):
        return float("nan")
    if previous_h <= current_h:
        raise ValueError("previous_h must be greater than current_h")
    return float(np.log(previous_error / current_error) / np.log(previous_h / current_h))


@lru_cache(maxsize=128)
def _reference_face_average(
    alpha_squared: float,
    face_refinement: int,
) -> float:
    node_index: dict[tuple[int, int], int] = {}
    nodes: list[tuple[float, float]] = []
    for i in range(face_refinement + 1):
        for j in range(face_refinement + 1 - i):
            node_index[(i, j)] = len(nodes)
            nodes.append((i / face_refinement, j / face_refinement))

    triangles: list[tuple[int, int, int]] = []
    for i in range(face_refinement):
        for j in range(face_refinement - i):
            triangles.append(
                (
                    node_index[(i, j)],
                    node_index[(i + 1, j)],
                    node_index[(i, j + 1)],
                )
            )
            if i + j <= face_refinement - 2:
                triangles.append(
                    (
                        node_index[(i + 1, j)],
                        node_index[(i + 1, j + 1)],
                        node_index[(i, j + 1)],
                    )
                )

    matrix = lil_matrix((len(nodes), len(nodes)), dtype=float)
    load = np.zeros(len(nodes), dtype=float)
    for triangle in triangles:
        coordinates = np.asarray([nodes[index] for index in triangle], dtype=float)
        twice_area = abs(
            np.linalg.det(
                np.array(
                    [
                        coordinates[1] - coordinates[0],
                        coordinates[2] - coordinates[0],
                    ]
                )
            )
        )
        area = 0.5 * twice_area
        b = (
            np.array(
                [
                    coordinates[1, 1] - coordinates[2, 1],
                    coordinates[2, 1] - coordinates[0, 1],
                    coordinates[0, 1] - coordinates[1, 1],
                ]
            )
            / twice_area
        )
        c = (
            np.array(
                [
                    coordinates[2, 0] - coordinates[1, 0],
                    coordinates[0, 0] - coordinates[2, 0],
                    coordinates[1, 0] - coordinates[0, 0],
                ]
            )
            / twice_area
        )
        stiffness = area * (np.outer(b, b) + np.outer(c, c))
        mass = (
            area
            / 12.0
            * np.array(
                [
                    [2.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 2.0],
                ]
            )
        )
        local_matrix = stiffness + alpha_squared * mass
        for local_row, global_row in enumerate(triangle):
            load[global_row] += area / 3.0
            for local_column, global_column in enumerate(triangle):
                matrix[global_row, global_column] += local_matrix[
                    local_row,
                    local_column,
                ]

    interior = np.asarray(
        [
            index
            for (i, j), index in node_index.items()
            if i > 0 and j > 0 and i + j < face_refinement
        ],
        dtype=np.int32,
    )
    solution = np.zeros(len(nodes), dtype=float)
    solution[interior] = spsolve(
        matrix.tocsr()[interior][:, interior],
        load[interior],
    )
    return float(np.dot(load, solution) / 0.5)


def face3d_pressure_jump_coefficient(
    *,
    viscosity: float,
    reaction: float,
    resolution: int,
    face_refinement: int = 24,
) -> float:
    """Compute the scalar triangular-face subscale coefficient.

    The reference problem is solved with continuous piecewise-linear finite
    elements on a uniformly refined right triangle. ``resolution`` is the
    unit-cube subdivision count, giving representative physical face diameter
    ``sqrt(2) / resolution``.
    """

    if viscosity <= 0.0 or not np.isfinite(viscosity):
        raise ValueError("viscosity must be positive and finite")
    if reaction < 0.0 or not np.isfinite(reaction):
        raise ValueError("reaction must be nonnegative and finite")
    if resolution < 1:
        raise ValueError("resolution must be positive")
    if face_refinement < 2:
        raise ValueError("face_refinement must be at least 2")
    scale = 1.0 / resolution
    face_diameter = np.sqrt(2.0) * scale
    alpha_squared = reaction * scale * scale / viscosity
    average = _reference_face_average(alpha_squared, face_refinement)
    return float(scale * scale / (viscosity * face_diameter) * average)


def _validate_resolutions(resolutions: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in resolutions)
    if len(normalized) < 2:
        raise ValueError("resolutions must contain at least two mesh levels")
    if any(value < 1 for value in normalized):
        raise ValueError("all resolutions must be positive")
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError("resolutions must be strictly increasing")
    return normalized


def _mesh_problem(case: BrinkmanMMSCase, resolution: int) -> FEMMapProblem:
    shape = (resolution,) * case.dimension
    cell_size = (1.0 / resolution,) * case.dimension
    return FEMMapProblem(
        permeability_map=PermeabilityMap(
            np.ones(shape, dtype=float),
            cell_size=cell_size,
        ),
        porosity_map=PorosityMap(
            np.ones(shape, dtype=float),
            cell_size=cell_size,
        ),
        viscosity=case.viscosity,
    )


def _exact_velocity_bc(context: Any, mixed_space: Any, exact_velocity: Any) -> Any:
    fem = context.api.fem
    velocity_subspace = mixed_space.sub(0)
    velocity_space, _ = velocity_subspace.collapse()
    boundary_velocity = fem.Function(velocity_space)
    expression = fem.Expression(
        exact_velocity,
        velocity_space.element.interpolation_points,
    )
    boundary_velocity.interpolate(expression)
    fdim = context.mesh.topology.dim - 1
    facets = context.api.mesh.locate_entities_boundary(
        context.mesh,
        fdim,
        lambda x: np.ones(x.shape[1], dtype=bool),
    )
    dofs = fem.locate_dofs_topological(
        (velocity_subspace, velocity_space),
        fdim,
        facets,
    )
    return fem.dirichletbc(boundary_velocity, dofs, velocity_subspace)


def _safe_sqrt(value: float, *, name: str) -> float:
    tolerance = 1.0e-12
    if value < -tolerance:
        raise RuntimeError(f"assembled squared {name} norm is negative: {value}")
    return float(np.sqrt(max(value, 0.0)))


def _solve_level(
    case: BrinkmanMMSCase,
    method: MMSMethod,
    resolution: int,
    *,
    options: FEniCSSolverOptions | None,
    tau_factor: float,
    tau_gamma_cap: float | None,
    m_t: float,
    alpha_edge: float,
    facet_law: MMSFacetLaw,
    facet_size_mode: MMSFacetSizeMode,
    face_refinement: int,
) -> tuple[MMSConvergenceLevel, MMSDiscreteSolution, dict[str, Any]]:
    spec = _METHOD_SPECS[method]
    solver_options = options or FEniCSSolverOptions()
    requested_dtype = _resolve_fem_linear_system_dtype(solver_options.linear_system_dtype)
    api = _require_dolfinx_core()
    selected_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_backend == "petsc" and requested_dtype != "float64":
        raise ValueError("linear_system_dtype='float32' is not supported by the PETSc MMS path")
    if selected_backend == "petsc":
        api = _require_dolfinx_petsc(api)

    context = _build_context(_mesh_problem(case, resolution), flow_axis="x", api=api)
    mixed_space = _mixed_space(
        context.api,
        context.mesh,
        velocity_degree=spec.velocity_degree,
        pressure_family=spec.pressure_family,
        pressure_degree=spec.pressure_degree,
    )
    u, p = context.api.ufl.TrialFunctions(mixed_space)
    v, q = context.api.ufl.TestFunctions(mixed_space)
    exact_velocity, exact_pressure = case.ufl_solution(context.api.ufl, context.mesh)
    ufl = context.api.ufl
    dx = context.dx
    gamma = context.api.fem.Constant(context.mesh, case.reaction)
    nu = context.api.fem.Constant(context.mesh, case.viscosity)
    forcing = (
        -nu * ufl.div(ufl.grad(exact_velocity)) + gamma * exact_velocity + ufl.grad(exact_pressure)
    )

    if method == "taylor_hood":
        form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            + ufl.inner(gamma * u, v) * dx
            - p * ufl.div(v) * dx
            + q * ufl.div(u) * dx
        )
        rhs = ufl.inner(forcing, v) * dx
    else:
        standard_facet_law: USFEMFacetLaw = (
            "reaction_diffusion"
            if facet_law in {"auto", "face3d"}
            else cast(USFEMFacetLaw, facet_law)
        )
        if facet_size_mode == "cell_diameter":
            facet_size = None
        elif facet_size_mode == "facet_diameter":
            if case.dimension != 2:
                raise ValueError("facet_size_mode='facet_diameter' is currently defined only in 2D")
            facet_size = ufl.avg(ufl.FacetArea(context.mesh))
        else:
            representative_size = (
                1.0 / resolution if case.dimension == 2 else np.sqrt(2.0) / resolution
            )
            facet_size = context.api.fem.Constant(
                context.mesh,
                representative_size,
            )
        gamma_stab, nu_stab, tau, tau_f = _usfem_stabilization_terms(
            context,
            tau_factor=tau_factor,
            m_t=m_t,
            alpha_edge=alpha_edge,
            facet_law=standard_facet_law,
            gamma=cast(Any, gamma),
            nu_eff=cast(Any, nu),
            facet_size=cast(Any, facet_size),
            tau_gamma_cap=tau_gamma_cap,
        )
        if facet_law == "face3d":
            if case.dimension != 3:
                raise ValueError("facet_law='face3d' is defined only for 3D MMS cases")
            tau_f = context.api.fem.Constant(
                context.mesh,
                face3d_pressure_jump_coefficient(
                    viscosity=case.viscosity,
                    reaction=case.reaction,
                    resolution=resolution,
                    face_refinement=face_refinement,
                ),
            )
        residual_u = gamma_stab * u + ufl.grad(p) - nu_stab * ufl.div(ufl.grad(u))
        residual_vq = gamma_stab * v - ufl.grad(q) - nu_stab * ufl.div(ufl.grad(v))
        form = (
            nu_stab * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            + ufl.inner(gamma_stab * u, v) * dx
            - p * ufl.div(v) * dx
            + q * ufl.div(u) * dx
            + tau_f * ufl.jump(p) * ufl.jump(q) * context.dS
            - tau * ufl.inner(residual_u, residual_vq) * dx
        )
        rhs = ufl.inner(forcing, v) * dx - tau * ufl.inner(forcing, residual_vq) * dx

    bcs = [
        _exact_velocity_bc(context, mixed_space, exact_velocity),
        _pressure_gauge_bc(context, mixed_space),
    ]
    if selected_backend == "petsc":
        solution, solve_seconds, solver_metadata = _solve_mixed_problem(
            context,
            form=form,
            rhs=rhs,
            bcs=bcs,
            options=solver_options,
            prefix_suffix=f"mms_{method}_{case.dimension}d_n{resolution}",
        )
    else:
        solution, solve_seconds, solver_metadata = _solve_mixed_problem_serial_direct(
            context,
            mixed_space=mixed_space,
            form=form,
            rhs=rhs,
            bcs=bcs,
            linear_backend=cast(Any, selected_backend),
            linear_system_dtype=requested_dtype,
            superlu_controls=solver_options.superlu_controls,
            umfpack_controls=solver_options.umfpack_controls,
            nvmath_cudss_controls=solver_options.nvmath_cudss_controls,
        )
    velocity, pressure = _collapse_solution(solution)

    velocity_error = velocity - exact_velocity
    velocity_l2_squared = _assemble_scalar(
        context,
        ufl.inner(velocity_error, velocity_error) * dx,
    )
    velocity_h1_squared = velocity_l2_squared + _assemble_scalar(
        context,
        ufl.inner(ufl.grad(velocity_error), ufl.grad(velocity_error)) * dx,
    )
    volume = _assemble_scalar(context, 1.0 * dx)
    pressure_difference = pressure - exact_pressure
    pressure_difference_mean = _assemble_scalar(context, pressure_difference * dx) / volume
    pressure_error = pressure_difference - pressure_difference_mean
    pressure_l2_squared = _assemble_scalar(
        context,
        pressure_error * pressure_error * dx,
    )
    divergence_l2_squared = _assemble_scalar(
        context,
        ufl.div(velocity) * ufl.div(velocity) * dx,
    )

    topology = context.mesh.topology
    num_cells = int(topology.index_map(topology.dim).size_global)
    index_map = mixed_space.dofmap.index_map
    num_dofs = int(index_map.size_global * mixed_space.dofmap.index_map_bs)
    level = MMSConvergenceLevel(
        resolution=resolution,
        h=1.0 / resolution,
        num_cells=num_cells,
        num_dofs=num_dofs,
        solve_seconds=solve_seconds,
        velocity_l2_error=_safe_sqrt(velocity_l2_squared, name="velocity L2"),
        velocity_h1_error=_safe_sqrt(velocity_h1_squared, name="velocity H1"),
        pressure_l2_error=_safe_sqrt(pressure_l2_squared, name="pressure L2"),
        divergence_l2=_safe_sqrt(divergence_l2_squared, name="divergence L2"),
    )
    discrete_solution = MMSDiscreteSolution(
        mesh=context.mesh,
        velocity=velocity,
        pressure=pressure,
    )
    metadata = {
        "linear_backend": selected_backend,
        "linear_system_dtype": requested_dtype,
        "velocity_degree": spec.velocity_degree,
        "pressure_family": spec.pressure_family,
        "pressure_degree": spec.pressure_degree,
        "tau_factor": tau_factor if method != "taylor_hood" else None,
        "tau_gamma_cap": tau_gamma_cap if method != "taylor_hood" else None,
        "m_t": m_t if method != "taylor_hood" else None,
        "alpha_edge": alpha_edge if method != "taylor_hood" else None,
        "alpha_edge_active": (
            facet_law in {"classic", "shifted"} if method != "taylor_hood" else None
        ),
        "facet_law": facet_law if method != "taylor_hood" else None,
        "facet_size_mode": facet_size_mode if method != "taylor_hood" else None,
        "face_refinement": (
            face_refinement if method != "taylor_hood" and facet_law == "face3d" else None
        ),
        **solver_metadata,
    }
    return level, discrete_solution, metadata


def _rates(previous: MMSConvergenceLevel, current: MMSConvergenceLevel) -> dict[str, float]:
    return {
        name: observed_rate(
            previous.h,
            previous.errors()[name],
            current.h,
            current.errors()[name],
        )
        for name in previous.errors()
    }


def run_mms_convergence(
    case: BrinkmanMMSCase,
    *,
    method: MMSMethod = "taylor_hood",
    resolutions: Sequence[int] = (4, 8, 16),
    options: FEniCSSolverOptions | None = None,
    tau_factor: float = 1.0,
    tau_gamma_cap: float | None = None,
    m_t: float = 1.0 / 3.0,
    alpha_edge: float = 1.0,
    facet_law: MMSFacetLaw = "auto",
    facet_size_mode: MMSFacetSizeMode = "cell_diameter",
    face_refinement: int = 24,
    keep_solution: bool = True,
    callback: Callable[[MMSConvergenceLevel], None] | None = None,
) -> MMSConvergenceResult:
    """Run a structured-mesh Brinkman MMS refinement study.

    Parameters
    ----------
    case :
        Exact Brinkman solution. The body force is derived automatically.
    method :
        ``"taylor_hood"`` for CG2 x CG1, ``"usfem_p1dg0"`` for CG1 x
        DG0, or ``"usfem_p1dg1"`` for CG1 x DG1.
    resolutions :
        Strictly increasing numbers of subdivisions per coordinate direction.
        The reported refinement parameter is ``h = 1 / resolution``.
    options :
        Linear solver controls shared by all levels.
    tau_factor, tau_gamma_cap, m_t, alpha_edge, facet_law, facet_size_mode, face_refinement :
        USFEM stabilization controls. ``facet_size_mode="cell_diameter"``
        preserves the generic solver convention, ``"facet_diameter"`` uses
        physical edge length in 2D, and ``"representative"`` uses ``1 / n`` in
        2D or ``sqrt(2) / n`` in 3D. Set ``tau_factor=0`` to disable the cell
        term. ``tau_gamma_cap`` optionally bounds ``gamma * tau_K``. They are
        ignored by Taylor-Hood.
    keep_solution :
        Retain the finest DOLFINx velocity and pressure fields for plotting.
    callback :
        Optional callable invoked after each completed level.

    Returns
    -------
    MMSConvergenceResult
        Errors, pairwise observed rates, nominal expected rates, solver
        metadata, and optionally the finest fields.
    """

    if method not in _METHOD_SPECS:
        supported = ", ".join(_METHOD_SPECS)
        raise ValueError(f"method must be one of {supported}")
    normalized_resolutions = _validate_resolutions(resolutions)
    if tau_factor < 0.0 or not np.isfinite(tau_factor):
        raise ValueError("tau_factor must be nonnegative and finite")
    if tau_gamma_cap is not None and (
        tau_gamma_cap <= 0.0 or tau_gamma_cap >= 1.0 or not np.isfinite(tau_gamma_cap)
    ):
        raise ValueError("tau_gamma_cap must satisfy 0 < tau_gamma_cap < 1")
    if m_t <= 0.0 or not np.isfinite(m_t):
        raise ValueError("m_t must be positive and finite")
    if alpha_edge <= 0.0 or not np.isfinite(alpha_edge):
        raise ValueError("alpha_edge must be positive and finite")
    if facet_law not in {
        "auto",
        "classic",
        "reaction_diffusion",
        "shifted",
        "face3d",
    }:
        raise ValueError(
            "facet_law must be one of 'auto', 'classic', 'reaction_diffusion', "
            "'shifted', or 'face3d'"
        )
    if facet_size_mode not in {
        "cell_diameter",
        "facet_diameter",
        "representative",
    }:
        raise ValueError(
            "facet_size_mode must be one of 'cell_diameter', 'facet_diameter', or 'representative'"
        )
    if face_refinement < 2:
        raise ValueError("face_refinement must be at least 2")
    resolved_facet_law: MMSFacetLaw = (
        "face3d" if facet_law == "auto" and case.dimension == 3 else facet_law
    )
    if resolved_facet_law == "auto":
        resolved_facet_law = "reaction_diffusion"
    if (
        method != "taylor_hood"
        and resolved_facet_law in {"reaction_diffusion", "face3d"}
        and not np.isclose(alpha_edge, 1.0)
    ):
        warnings.warn(
            f"alpha_edge is ignored by the parameter-free {resolved_facet_law} facet law",
            RuntimeWarning,
            stacklevel=2,
        )

    levels: list[MMSConvergenceLevel] = []
    finest_solution: MMSDiscreteSolution | None = None
    metadata: dict[str, Any] = {}
    for resolution in normalized_resolutions:
        level, discrete_solution, metadata = _solve_level(
            case,
            method,
            resolution,
            options=options,
            tau_factor=tau_factor,
            tau_gamma_cap=tau_gamma_cap,
            m_t=m_t,
            alpha_edge=alpha_edge,
            facet_law=resolved_facet_law,
            facet_size_mode=facet_size_mode,
            face_refinement=face_refinement,
        )
        if levels:
            level = replace(level, rates=_rates(levels[-1], level))
        levels.append(level)
        finest_solution = discrete_solution
        if callback is not None:
            callback(level)

    return MMSConvergenceResult(
        case=case,
        method=method,
        levels=tuple(levels),
        expected_rates=_METHOD_SPECS[method].expected_rates,
        finest_solution=finest_solution if keep_solution else None,
        metadata=metadata,
    )


__all__ = [
    "available_mms_methods",
    "face3d_pressure_jump_coefficient",
    "observed_rate",
    "run_mms_convergence",
]
