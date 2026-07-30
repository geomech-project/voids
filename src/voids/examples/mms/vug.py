from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
import warnings

import numpy as np

from voids.examples.mms._core import MMSMethod
from voids.fem.singlephase import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    USFEMFacetLaw,
    USFEMFacetSizeMode,
    solve_brinkman_taylor_hood,
    solve_brinkman_usfem,
)
from voids.fem.singlephase._common import (
    _FEMContext,
    _mixed_space,
    _pressure_boundary_load,
    _require_dolfinx_core,
    _require_dolfinx_petsc,
    _resolve_fem_linear_system_dtype,
    _resolve_linear_backend,
    _result_from_solution,
    _side_wall_bcs,
    _solve_mixed_problem,
    _solve_mixed_problem_serial_direct,
)
from voids.fem.singlephase.usfem import (
    _facet_size_expression,
    _usfem_stabilization_terms,
    _validate_usfem_controls,
)
from voids.image.porosity import PermeabilityMap, PorosityMap
from voids.mesh.gmsh import (
    add_physical_group,
    axis_aligned_boundary_entities,
    configure_uniform_mesh_size,
    generate_dolfinx_gmsh_mesh,
)


VugMeshRepresentation = Literal["body_fitted", "structured"]

_MATRIX_TAG = 1
_VUG_TAG = 2
_FACET_MARKERS = {
    "x_min": 1,
    "x_max": 2,
    "y_min": 3,
    "y_max": 4,
    "z_min": 5,
    "z_max": 6,
}
_VUG_INTERFACE_TAG = 7


def _centered_vug_p1dg0_uncapped_tau_gamma(
    benchmark: CenteredVugBenchmark,
    *,
    tau_factor: float,
    m_t: float,
) -> float:
    h_squared = benchmark.target_mesh_size**2
    pe_t = 4.0 * benchmark.viscosity / (benchmark.matrix_drag * h_squared * float(m_t))
    denominator = benchmark.matrix_drag * h_squared * max(
        1.0, pe_t
    ) + 4.0 * benchmark.viscosity / float(m_t)
    return float(float(tau_factor) * benchmark.matrix_drag * h_squared / denominator)


@dataclass(frozen=True, slots=True)
class CenteredVugBenchmark:
    """Centered circular/spherical vug benchmark on the unit domain.

    The defaults reproduce the documented physical configuration: radius
    ``0.25``, matrix drag ``1e7``, vug drag ``1``, viscosity ``1e-2``, and
    pressure values ``1`` and ``-1``. ``mesh_representation="body_fitted"``
    uses Gmsh physical cell tags. The ``"structured"`` option classifies
    coefficient-map cells by their centers and is useful as a portable
    representation-sensitivity comparison.

    ``resolution`` is the number of nominal elements per coordinate direction.
    For body-fitted meshes, the Gmsh target size is
    ``sqrt(dimension) / resolution``, matching the mesh-size convention used in
    the reference vug studies.
    """

    dimension: Literal[2, 3] = 2
    resolution: int = 32
    radius: float = 0.25
    viscosity: float = 1.0e-2
    matrix_drag: float = 1.0e7
    vug_drag: float = 1.0
    pressure_inlet: float = 1.0
    pressure_outlet: float = -1.0
    mesh_representation: VugMeshRepresentation = "body_fitted"
    matrix_effective_viscosity: float | None = None
    vug_effective_viscosity: float | None = None

    def __post_init__(self) -> None:
        if self.dimension not in {2, 3}:
            raise ValueError("dimension must be either 2 or 3")
        if self.resolution < 2:
            raise ValueError("resolution must be at least 2")
        if not 0.0 <= self.radius < 0.5:
            raise ValueError("radius must lie between 0 (inclusive) and 0.5")
        for name in ("viscosity", "matrix_drag"):
            value = float(getattr(self, name))
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        if self.vug_drag < 0.0 or not np.isfinite(self.vug_drag):
            raise ValueError("vug_drag must be non-negative and finite")
        for name in ("matrix_effective_viscosity", "vug_effective_viscosity"):
            value = getattr(self, name)
            if value is not None and (value <= 0.0 or not np.isfinite(value)):
                raise ValueError(f"{name} must be positive and finite when provided")
        if (
            not np.isfinite(self.pressure_inlet)
            or not np.isfinite(self.pressure_outlet)
            or self.pressure_inlet <= self.pressure_outlet
        ):
            raise ValueError("pressure_inlet must be finite and greater than pressure_outlet")
        if self.mesh_representation not in {"body_fitted", "structured"}:
            raise ValueError("mesh_representation must be either 'body_fitted' or 'structured'")

    @property
    def target_mesh_size(self) -> float:
        """Return the Gmsh target element diameter."""

        return float(np.sqrt(self.dimension) / self.resolution)

    @property
    def matrix_nu(self) -> float:
        """Return the matrix coefficient multiplying the Brinkman gradient term."""

        if self.matrix_effective_viscosity is None:
            return float(self.viscosity)
        return float(self.matrix_effective_viscosity)

    @property
    def vug_nu(self) -> float:
        """Return the vug coefficient multiplying the Brinkman gradient term."""

        if self.vug_effective_viscosity is None:
            return float(self.viscosity)
        return float(self.vug_effective_viscosity)

    def vug_mask(self) -> np.ndarray:
        """Return the structured cell-center classification of the vug."""

        coordinates = (np.arange(self.resolution, dtype=float) + 0.5) / self.resolution
        grids = np.meshgrid(
            *((coordinates,) * self.dimension),
            indexing="ij",
        )
        radius_squared = sum((grid - 0.5) ** 2 for grid in grids)
        return np.asarray(radius_squared <= self.radius**2, dtype=bool)

    @property
    def represented_fraction(self) -> float:
        """Return the structured cell-volume fraction classified as vug."""

        return float(np.mean(self.vug_mask()))

    @property
    def analytic_fraction(self) -> float:
        """Return the exact circular area or spherical volume fraction."""

        if self.dimension == 2:
            return float(np.pi * self.radius**2)
        return float((4.0 / 3.0) * np.pi * self.radius**3)

    def make_problem(self) -> FEMMapProblem:
        """Build the structured constant-porosity coefficient-map problem."""

        mask = self.vug_mask()
        drag = np.where(mask, self.vug_drag, self.matrix_drag)
        permeability = np.divide(
            self.viscosity,
            drag,
            out=np.full_like(drag, np.inf, dtype=float),
            where=drag > 0.0,
        )
        if not np.all(np.isfinite(permeability)):
            raise ValueError(
                "structured vug benchmarks require positive vug_drag; "
                "zero drag is supported only by the body-fitted formulation"
            )
        effective_viscosity = np.where(mask, self.vug_nu, self.matrix_nu)
        porosity = self.viscosity / effective_viscosity
        cell_size = (1.0 / self.resolution,) * self.dimension
        return FEMMapProblem(
            permeability_map=PermeabilityMap(
                permeability,
                cell_size=cell_size,
            ),
            porosity_map=PorosityMap(
                porosity,
                cell_size=cell_size,
            ),
            viscosity=self.viscosity,
        )


@dataclass(slots=True)
class BodyFittedVugMesh:
    """DOLFINx mesh and physical tags generated for a centered vug."""

    mesh: Any
    cell_tags: Any
    facet_tags: Any
    physical_groups: dict[str, Any]


def _classify_volume_entities(gmsh: Any, benchmark: CenteredVugBenchmark) -> tuple[int, int]:
    entities = [tag for _, tag in gmsh.model.getEntities(benchmark.dimension)]
    if len(entities) != 2:
        raise RuntimeError(
            "Gmsh fragment should produce exactly two matrix/vug entities; "
            f"received {len(entities)}"
        )
    target_vug_measure = benchmark.analytic_fraction
    vug_tag = min(
        entities,
        key=lambda tag: abs(
            float(gmsh.model.occ.getMass(benchmark.dimension, tag)) - target_vug_measure
        ),
    )
    matrix_tag = next(tag for tag in entities if tag != vug_tag)
    return matrix_tag, vug_tag


def make_body_fitted_centered_vug_mesh(
    benchmark: CenteredVugBenchmark,
) -> BodyFittedVugMesh:
    """Generate a tagged body-fitted centered-vug mesh with Gmsh."""

    if benchmark.mesh_representation != "body_fitted":
        raise ValueError(
            "make_body_fitted_centered_vug_mesh requires mesh_representation='body_fitted'"
        )

    def build_model(gmsh: Any) -> None:
        if benchmark.dimension == 2:
            outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        else:
            outer = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        if benchmark.radius > 0.0:
            if benchmark.dimension == 2:
                vug = gmsh.model.occ.addDisk(
                    0.5,
                    0.5,
                    0.0,
                    benchmark.radius,
                    benchmark.radius,
                )
            else:
                vug = gmsh.model.occ.addSphere(
                    0.5,
                    0.5,
                    0.5,
                    benchmark.radius,
                )
            gmsh.model.occ.fragment(
                [(benchmark.dimension, outer)],
                [(benchmark.dimension, vug)],
            )
        gmsh.model.occ.synchronize()
        if benchmark.radius > 0.0:
            matrix_entity, vug_entity = _classify_volume_entities(gmsh, benchmark)
        else:
            volume_entities = gmsh.model.getEntities(benchmark.dimension)
            if len(volume_entities) != 1:
                raise RuntimeError(
                    "matrix-only Gmsh model should contain exactly one volume entity"
                )
            matrix_entity = int(volume_entities[0][1])
            vug_entity = None
        add_physical_group(
            gmsh,
            benchmark.dimension,
            [matrix_entity],
            tag=_MATRIX_TAG,
            name="matrix",
        )
        if vug_entity is not None:
            add_physical_group(
                gmsh,
                benchmark.dimension,
                [vug_entity],
                tag=_VUG_TAG,
                name="vug",
            )
        outer_groups, interface = axis_aligned_boundary_entities(
            gmsh,
            benchmark.dimension,
            tolerance=1.0e-6,
        )
        missing = [name for name, entities in outer_groups.items() if not entities]
        if missing:
            raise RuntimeError(f"Gmsh did not create the expected outer facets: {missing}")
        if benchmark.radius > 0.0 and not interface:
            raise RuntimeError("Gmsh did not create a vug interface")
        if benchmark.radius == 0.0 and interface:
            raise RuntimeError("matrix-only Gmsh model unexpectedly contains interior facets")
        fdim = benchmark.dimension - 1
        for name, entities in outer_groups.items():
            marker = _FACET_MARKERS[name]
            add_physical_group(
                gmsh,
                fdim,
                entities,
                tag=marker,
                name=name,
            )
        if interface:
            add_physical_group(
                gmsh,
                fdim,
                interface,
                tag=_VUG_INTERFACE_TAG,
                name="vug_interface",
            )
        configure_uniform_mesh_size(gmsh, benchmark.target_mesh_size)
        gmsh.model.mesh.generate(benchmark.dimension)

    mesh_data = generate_dolfinx_gmsh_mesh(
        build_model,
        name=f"voids_centered_vug_{benchmark.dimension}d",
        geometric_dimension=benchmark.dimension,
    )

    if mesh_data.cell_tags is None or mesh_data.facet_tags is None:
        raise RuntimeError("Gmsh import did not preserve the required physical tags")
    return BodyFittedVugMesh(
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags,
        physical_groups=dict(mesh_data.physical_groups),
    )


def _tagged_dg0_function(
    api: Any,
    mesh: Any,
    cell_tags: Any,
    *,
    matrix_value: float,
    vug_value: float,
    name: str,
) -> Any:
    space = api.fem.functionspace(
        mesh,
        api.basix_ufl.element("DG", mesh.basix_cell(), 0),
    )
    function = api.fem.Function(space)
    function.name = name
    tdim = mesh.topology.dim
    for tag, value in (
        (_MATRIX_TAG, matrix_value),
        (_VUG_TAG, vug_value),
    ):
        cells = cell_tags.find(tag)
        dofs = api.fem.locate_dofs_topological(space, tdim, cells)
        function.x.array[dofs] = value
    function.x.scatter_forward()
    return function


def _body_fitted_context(
    benchmark: CenteredVugBenchmark,
    *,
    api: Any,
) -> tuple[_FEMContext, float, int]:
    tagged_mesh = make_body_fitted_centered_vug_mesh(benchmark)
    mesh = tagged_mesh.mesh
    ufl = api.ufl
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_data=tagged_mesh.cell_tags,
    )
    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=tagged_mesh.facet_tags,
    )
    context = _FEMContext(
        api=api,
        mesh=mesh,
        ds=ds,
        dx=dx,
        dS=ufl.Measure("dS", domain=mesh),
        normal=ufl.FacetNormal(mesh),
        coefficients={
            "gamma": _tagged_dg0_function(
                api,
                mesh,
                tagged_mesh.cell_tags,
                matrix_value=benchmark.matrix_drag,
                vug_value=benchmark.vug_drag,
                name="Darcy drag gamma",
            ),
            "nu_eff": _tagged_dg0_function(
                api,
                mesh,
                tagged_mesh.cell_tags,
                matrix_value=benchmark.matrix_nu,
                vug_value=benchmark.vug_nu,
                name="Brinkman effective viscosity",
            ),
        },
        domain_length=1.0,
        cross_section_area=1.0,
    )
    local_total = api.fem.assemble_scalar(api.fem.form(1.0 * dx))
    local_vug = api.fem.assemble_scalar(api.fem.form(1.0 * dx(_VUG_TAG)))
    total = float(mesh.comm.allreduce(local_total, op=api.MPI.SUM))
    vug_measure = float(mesh.comm.allreduce(local_vug, op=api.MPI.SUM))
    tdim = mesh.topology.dim
    num_cells = int(mesh.topology.index_map(tdim).size_global)
    return context, vug_measure / total, num_cells


def _run_body_fitted(
    benchmark: CenteredVugBenchmark,
    *,
    method: MMSMethod,
    options: FEniCSSolverOptions | None,
    tau_factor: float,
    tau_gamma_cap: float | None,
    m_t: float,
    alpha_edge: float,
    facet_law: USFEMFacetLaw,
    facet_size_mode: USFEMFacetSizeMode,
) -> FEMSinglePhaseResult:
    solver_options = options or FEniCSSolverOptions()
    requested_dtype = _resolve_fem_linear_system_dtype(solver_options.linear_system_dtype)
    api = _require_dolfinx_core()
    selected_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_backend == "petsc" and requested_dtype != "float64":
        raise ValueError("linear_system_dtype='float32' is not supported by the PETSc vug path")
    if selected_backend == "petsc":
        api = _require_dolfinx_petsc(api)
    context, represented_fraction, num_cells = _body_fitted_context(
        benchmark,
        api=api,
    )
    if method == "taylor_hood":
        velocity_degree = 2
        pressure_degree = 1
        pressure_family = "Lagrange"
    elif method == "usfem_p1dg0":
        velocity_degree = 1
        pressure_degree = 0
        pressure_family = "DG"
    else:
        velocity_degree = 1
        pressure_degree = 1
        pressure_family = "DG"
    mixed_space = _mixed_space(
        context.api,
        context.mesh,
        velocity_degree=velocity_degree,
        pressure_family=pressure_family,
        pressure_degree=pressure_degree,
    )
    u, p = context.api.ufl.TrialFunctions(mixed_space)
    v, q = context.api.ufl.TestFunctions(mixed_space)
    ufl = context.api.ufl
    gamma = context.coefficients["gamma"]
    nu = context.coefficients["nu_eff"]
    if method == "taylor_hood":
        form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
            + ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
        )
    else:
        gamma, nu, tau, tau_f = _usfem_stabilization_terms(
            context,
            tau_factor=tau_factor,
            m_t=m_t,
            alpha_edge=alpha_edge,
            facet_law=facet_law,
            facet_size=_facet_size_expression(context, facet_size_mode),
            tau_gamma_cap=tau_gamma_cap,
        )
        residual_u = gamma * u + ufl.grad(p) - nu * ufl.div(ufl.grad(u))
        residual_vq = gamma * v - ufl.grad(q) - nu * ufl.div(ufl.grad(v))
        form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
            + ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
            + tau_f * ufl.jump(p) * ufl.jump(q) * context.dS
            - tau * ufl.inner(residual_u, residual_vq) * context.dx
        )
    rhs = _pressure_boundary_load(
        context,
        v,
        flow_axis="x",
        pressure_inlet=benchmark.pressure_inlet,
        pressure_outlet=benchmark.pressure_outlet,
    )
    bcs = _side_wall_bcs(context, mixed_space, flow_axis="x")
    if selected_backend == "petsc":
        solution, solve_seconds, solver_metadata = _solve_mixed_problem(
            context,
            form=form,
            rhs=rhs,
            bcs=bcs,
            options=solver_options,
            prefix_suffix=(f"centered_vug_{benchmark.dimension}d_{method}_n{benchmark.resolution}"),
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
    result = _result_from_solution(
        context,
        solution,
        method=(
            "Darcy-Brinkman Taylor-Hood CG2 x CG1"
            if method == "taylor_hood"
            else f"Darcy-Brinkman USFEM CG1 x DG{pressure_degree}"
        ),
        formulation=(
            "brinkman_taylor_hood_p2p1"
            if method == "taylor_hood"
            else f"brinkman_usfem_p1dg{pressure_degree}"
        ),
        flow_axis="x",
        pressure_inlet=benchmark.pressure_inlet,
        pressure_outlet=benchmark.pressure_outlet,
        viscosity=benchmark.viscosity,
        solve_seconds=solve_seconds,
        metadata={
            "linear_backend": selected_backend,
            "linear_system_dtype": requested_dtype,
            "velocity_degree": velocity_degree,
            "pressure_family": pressure_family,
            "pressure_degree": pressure_degree,
            "num_cells": num_cells,
            "pressure_constraint": "natural_traction",
            "returned_pressure_normalization": "zero_mean",
            "facet_law": facet_law if method != "taylor_hood" else None,
            "tau_factor": tau_factor if method != "taylor_hood" else None,
            "tau_gamma_cap": tau_gamma_cap if method != "taylor_hood" else None,
            "m_t": m_t if method != "taylor_hood" else None,
            "alpha_edge": alpha_edge if method != "taylor_hood" else None,
            "alpha_edge_active": (
                facet_law in {"classic", "shifted"} if method != "taylor_hood" else None
            ),
            "facet_size_mode": (facet_size_mode if method != "taylor_hood" else None),
            "represented_vug_fraction": represented_fraction,
            **solver_metadata,
        },
    )
    return result


def run_centered_vug_benchmark(
    benchmark: CenteredVugBenchmark,
    *,
    method: MMSMethod = "taylor_hood",
    options: FEniCSSolverOptions | None = None,
    tau_factor: float = 1.0,
    tau_gamma_cap: float | None = None,
    m_t: float = 1.0 / 3.0,
    alpha_edge: float = 1.0,
    facet_law: USFEMFacetLaw | None = None,
    facet_size_mode: USFEMFacetSizeMode = "facet_measure",
) -> FEMSinglePhaseResult:
    """Run the pressure-driven centered-vug FEM benchmark.

    Body-fitted USFEM defaults to the reaction-diffusion facet law in 2D and
    the shifted law in 3D, matching the documented benchmark choices. Pass
    ``facet_law`` explicitly whenever method-to-method comparisons need one
    common stabilization. The default facet size is the physical edge length
    in 2D and the square root of facet area in 3D, matching the body-fitted
    benchmark implementation. The latter is only a measure-based length, not
    an exact triangular-face diameter.
    """

    if method not in {"taylor_hood", "usfem_p1dg0", "usfem_p1dg1"}:
        raise ValueError("method must be one of 'taylor_hood', 'usfem_p1dg0', or 'usfem_p1dg1'")
    resolved_facet_law: USFEMFacetLaw = facet_law or (
        "reaction_diffusion" if benchmark.dimension == 2 else "shifted"
    )
    _validate_usfem_controls(
        tau_factor=tau_factor,
        m_t=m_t,
        alpha_edge=alpha_edge,
        facet_law=resolved_facet_law,
        facet_size_mode=facet_size_mode,
        tau_gamma_cap=tau_gamma_cap,
    )
    uncapped_max_tau_gamma = (
        _centered_vug_p1dg0_uncapped_tau_gamma(
            benchmark,
            tau_factor=tau_factor,
            m_t=m_t,
        )
        if method == "usfem_p1dg0"
        else None
    )
    if (
        benchmark.mesh_representation == "body_fitted"
        and uncapped_max_tau_gamma is not None
        and tau_factor > 0.0
        and tau_gamma_cap is None
        and uncapped_max_tau_gamma >= 0.9
    ):
        warnings.warn(
            "The body-fitted CG1 x DG0 vug case has estimated "
            f"max(gamma * tau_K)={uncapped_max_tau_gamma:.3g}; the cell "
            "term can nearly cancel matrix drag. Set tau_gamma_cap below 1 "
            "or use tau_factor=0 as an explicit sensitivity branch.",
            RuntimeWarning,
            stacklevel=2,
        )
    if benchmark.mesh_representation == "body_fitted":
        result = _run_body_fitted(
            benchmark,
            method=method,
            options=options,
            tau_factor=tau_factor,
            tau_gamma_cap=tau_gamma_cap,
            m_t=m_t,
            alpha_edge=alpha_edge,
            facet_law=resolved_facet_law,
            facet_size_mode=facet_size_mode,
        )
        represented_fraction = float(result.metadata["represented_vug_fraction"])
    else:
        problem = benchmark.make_problem()
        if method == "taylor_hood":
            result = solve_brinkman_taylor_hood(
                problem,
                flow_axis="x",
                pressure_inlet=benchmark.pressure_inlet,
                pressure_outlet=benchmark.pressure_outlet,
                options=options,
            )
        elif method in {"usfem_p1dg0", "usfem_p1dg1"}:
            pressure_degree: Literal[0, 1] = 0 if method == "usfem_p1dg0" else 1
            result = solve_brinkman_usfem(
                problem,
                pressure_degree=pressure_degree,
                tau_factor=tau_factor,
                tau_gamma_cap=tau_gamma_cap,
                m_t=m_t,
                alpha_edge=alpha_edge,
                facet_law=resolved_facet_law,
                facet_size_mode=facet_size_mode,
                flow_axis="x",
                pressure_inlet=benchmark.pressure_inlet,
                pressure_outlet=benchmark.pressure_outlet,
                options=options,
            )
        represented_fraction = benchmark.represented_fraction
    result.metadata.update(
        {
            "benchmark": "centered_vug",
            "geometry_representation": benchmark.mesh_representation,
            "dimension": benchmark.dimension,
            "resolution": benchmark.resolution,
            "target_mesh_size": benchmark.target_mesh_size,
            "radius": benchmark.radius,
            "analytic_vug_fraction": benchmark.analytic_fraction,
            "represented_vug_fraction": represented_fraction,
            "matrix_drag": benchmark.matrix_drag,
            "vug_drag": benchmark.vug_drag,
            "matrix_effective_viscosity": benchmark.matrix_nu,
            "vug_effective_viscosity": benchmark.vug_nu,
            "p1dg0_uncapped_max_tau_gamma": uncapped_max_tau_gamma,
        }
    )
    return result


__all__ = [
    "BodyFittedVugMesh",
    "CenteredVugBenchmark",
    "VugMeshRepresentation",
    "make_body_fitted_centered_vug_mesh",
    "run_centered_vug_benchmark",
]
