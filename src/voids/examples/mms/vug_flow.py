from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from voids.examples.mms.vug import (
    CenteredVugBenchmark,
    _body_fitted_context,
)
from voids.fem.singlephase import FEMSinglePhaseResult, FEniCSSolverOptions
from voids.fem.singlephase._common import (
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


M2_PER_MILLIDARCY = 9.869233e-16
CenteredVugFlowModel = Literal["darcy_brinkman", "darcy_darcy"]


@dataclass(frozen=True, slots=True)
class CenteredVugFlowCase2D:
    """Physical 2D centered-vug case solved on a body-fitted unit-square mesh.

    The continuum domain size is derived from ``image_shape`` and
    ``voxel_size_m``. Gmsh coordinates are normalized to the unit square for
    numerical conditioning; the solver nondimensionalizes the equations and
    converts velocity, pressure, flux, and permeability back to SI units.

    The Darcy--Brinkman branch follows the layered-domain model: its reaction
    coefficient is ``mu / K_matrix`` in the matrix and exactly zero in the
    vug. ``vug_permeability_m2`` is used only by the Darcy--Darcy branch as a
    finite high-permeability closure. It is not an intrinsic permeability
    measurement of an open cavity.
    """

    area_fraction: float
    image_shape: tuple[int, int] = (500, 500)
    voxel_size_m: float = 15.0e-6
    matrix_porosity: float = 0.2
    matrix_permeability_md: float = 200.0
    vug_permeability_m2: float = 1.0e-8
    dynamic_viscosity_pa_s: float = 1.0e-3
    pressure_inlet_pa: float = 1.0
    pressure_outlet_pa: float = 0.0
    mesh_resolution: int = 100

    def __post_init__(self) -> None:
        if len(self.image_shape) != 2 or self.image_shape[0] != self.image_shape[1]:
            raise ValueError("image_shape must describe a square 2D image")
        if self.image_shape[0] < 2:
            raise ValueError("image_shape values must be at least 2")
        if self.voxel_size_m <= 0.0 or not np.isfinite(self.voxel_size_m):
            raise ValueError("voxel_size_m must be positive and finite")
        maximum_fraction = float(np.pi / 4.0)
        if (
            self.area_fraction < 0.0
            or self.area_fraction >= maximum_fraction
            or not np.isfinite(self.area_fraction)
        ):
            raise ValueError("area_fraction must lie between 0 (inclusive) and pi/4 (exclusive)")
        for name in (
            "matrix_porosity",
            "matrix_permeability_md",
            "vug_permeability_m2",
            "dynamic_viscosity_pa_s",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        if self.matrix_porosity > 1.0:
            raise ValueError("matrix_porosity must not exceed 1")
        if (
            not np.isfinite(self.pressure_inlet_pa)
            or not np.isfinite(self.pressure_outlet_pa)
            or self.pressure_inlet_pa <= self.pressure_outlet_pa
        ):
            raise ValueError("pressure_inlet_pa must be finite and greater than pressure_outlet_pa")
        if self.mesh_resolution < 8:
            raise ValueError("mesh_resolution must be at least 8")

    @property
    def side_length_m(self) -> float:
        """Return the square side length derived from the image metadata."""

        return float(self.image_shape[0] * self.voxel_size_m)

    @property
    def radius_fraction(self) -> float:
        """Return the circle radius divided by the square side length."""

        return float(np.sqrt(self.area_fraction / np.pi))

    @property
    def radius_m(self) -> float:
        """Return the physical centered-vug radius."""

        return float(self.radius_fraction * self.side_length_m)

    @property
    def matrix_permeability_m2(self) -> float:
        """Return the matrix permeability converted from mD to square metres."""

        return float(self.matrix_permeability_md * M2_PER_MILLIDARCY)

    @property
    def matrix_screening_length_m(self) -> float:
        """Return the matrix Brinkman screening length ``sqrt(K / phi)``."""

        return float(np.sqrt(self.matrix_permeability_m2 / self.matrix_porosity))

    @property
    def permeability_contrast(self) -> float:
        """Return the Darcy--Darcy closure contrast ``K_vug / K_matrix``."""

        return float(self.vug_permeability_m2 / self.matrix_permeability_m2)

    @property
    def pressure_drop_pa(self) -> float:
        """Return the applied pressure drop."""

        return float(self.pressure_inlet_pa - self.pressure_outlet_pa)

    @property
    def base_target_mesh_size_fraction(self) -> float:
        """Return the far-field target diameter divided by the side length."""

        return float(np.sqrt(2.0) / self.mesh_resolution)

    def make_benchmark(
        self,
        *,
        model: CenteredVugFlowModel = "darcy_brinkman",
    ) -> CenteredVugBenchmark:
        """Return the model-specific nondimensional mesh/coefficient definition."""

        if model not in {"darcy_brinkman", "darcy_darcy"}:
            raise ValueError("model must be either 'darcy_brinkman' or 'darcy_darcy'")
        darcy_number = self.matrix_permeability_m2 / self.side_length_m**2
        vug_drag = (
            0.0
            if model == "darcy_brinkman"
            else self.matrix_permeability_m2 / self.vug_permeability_m2
        )
        return CenteredVugBenchmark(
            dimension=2,
            resolution=self.mesh_resolution,
            radius=self.radius_fraction,
            viscosity=darcy_number,
            matrix_drag=1.0,
            vug_drag=vug_drag,
            pressure_inlet=1.0,
            pressure_outlet=0.0,
            mesh_representation="body_fitted",
            matrix_effective_viscosity=darcy_number / self.matrix_porosity,
            vug_effective_viscosity=darcy_number,
        )


def run_centered_vug_flow_case(
    case: CenteredVugFlowCase2D,
    *,
    model: CenteredVugFlowModel,
    options: FEniCSSolverOptions | None = None,
    vms_constant: float = 1.0,
) -> FEMSinglePhaseResult:
    """Solve one physical centered-vug case with Taylor--Hood P2/P1 fields.

    ``model="darcy_brinkman"`` uses piecewise Brinkman viscosity
    ``mu/phi_matrix`` in the matrix and ``mu`` in the vug, with reaction
    ``mu/K_matrix`` in the matrix and zero in the vug.
    ``model="darcy_darcy"`` omits viscous diffusion, uses the configured finite
    vug permeability, and applies residual-based VMS stabilization with
    continuous P2 velocity and P1 pressure.
    """

    if model not in {"darcy_brinkman", "darcy_darcy"}:
        raise ValueError("model must be either 'darcy_brinkman' or 'darcy_darcy'")
    if vms_constant <= 0.0 or not np.isfinite(vms_constant):
        raise ValueError("vms_constant must be positive and finite")

    solver_options = options or FEniCSSolverOptions()
    requested_dtype = _resolve_fem_linear_system_dtype(solver_options.linear_system_dtype)
    api = _require_dolfinx_core()
    selected_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_backend == "petsc" and requested_dtype != "float64":
        raise ValueError("linear_system_dtype='float32' is not supported by the PETSc vug path")
    if selected_backend == "petsc":
        api = _require_dolfinx_petsc(api)

    benchmark = case.make_benchmark(model=model)
    context, represented_fraction, num_cells = _body_fitted_context(benchmark, api=api)
    mixed_space = _mixed_space(
        context.api,
        context.mesh,
        velocity_degree=2,
        pressure_family="Lagrange",
        pressure_degree=1,
    )
    u, p = context.api.ufl.TrialFunctions(mixed_space)
    v, q = context.api.ufl.TestFunctions(mixed_space)
    ufl = context.api.ufl
    gamma = context.coefficients["gamma"]
    if model == "darcy_brinkman":
        nu = context.coefficients["nu_eff"]
        form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * context.dx
            + ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
        )
        method = "Darcy-Brinkman Taylor-Hood CG2 x CG1"
        formulation = "centered_vug_darcy_brinkman_taylor_hood_p2p1"
    else:
        cell_diameter = ufl.CellDiameter(context.mesh)
        darcy_number = case.matrix_permeability_m2 / case.side_length_m**2
        tau_vms = ufl.min_value(
            vms_constant * cell_diameter**2 / darcy_number,
            vms_constant / gamma,
        )
        form = (
            ufl.inner(gamma * u, v) * context.dx
            - p * ufl.div(v) * context.dx
            + q * ufl.div(u) * context.dx
            + 0.5
            * ufl.inner(
                -gamma * v + ufl.grad(q),
                tau_vms * (gamma * u + ufl.grad(p)),
            )
            * context.dx
        )
        method = "Darcy-Darcy VMS Taylor-Hood CG2 x CG1"
        formulation = "centered_vug_darcy_darcy_vms_taylor_hood_p2p1"

    rhs = _pressure_boundary_load(
        context,
        v,
        flow_axis="x",
        pressure_inlet=1.0,
        pressure_outlet=0.0,
    )
    bcs = _side_wall_bcs(context, mixed_space, flow_axis="x")
    prefix_suffix = (
        f"centered_vug_flow_{model}_f{case.area_fraction:.3f}_n{case.mesh_resolution}"
    ).replace(".", "p")
    if selected_backend == "petsc":
        solution, solve_seconds, solver_metadata = _solve_mixed_problem(
            context,
            form=form,
            rhs=rhs,
            bcs=bcs,
            options=solver_options,
            prefix_suffix=prefix_suffix,
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

    velocity_scale = (
        case.pressure_drop_pa
        * case.matrix_permeability_m2
        / (case.dynamic_viscosity_pa_s * case.side_length_m)
    )
    context.domain_length = case.side_length_m
    context.cross_section_area = 1.0
    result = _result_from_solution(
        context,
        solution,
        method=method,
        formulation=formulation,
        flow_axis="x",
        pressure_inlet=case.pressure_inlet_pa,
        pressure_outlet=case.pressure_outlet_pa,
        viscosity=case.dynamic_viscosity_pa_s,
        solve_seconds=solve_seconds,
        velocity_scale=velocity_scale,
        pressure_scale=case.pressure_drop_pa,
        metadata={
            "benchmark": "physical_centered_vug_flow_2d",
            "model": model,
            "linear_backend": selected_backend,
            "linear_system_dtype": requested_dtype,
            "velocity_degree": 2,
            "pressure_family": "Lagrange",
            "pressure_degree": 1,
            "pressure_constraint": "natural_traction",
            "returned_pressure_normalization": "zero_mean",
            "num_cells": num_cells,
            "image_shape": case.image_shape,
            "voxel_size_m": case.voxel_size_m,
            "side_length_m": case.side_length_m,
            "requested_vug_area_fraction": case.area_fraction,
            "represented_vug_area_fraction": represented_fraction,
            "vug_radius_m": case.radius_m,
            "matrix_porosity": case.matrix_porosity,
            "matrix_permeability_m2": case.matrix_permeability_m2,
            "matrix_permeability_md": case.matrix_permeability_md,
            "configured_darcy_darcy_vug_permeability_m2": (case.vug_permeability_m2),
            "vug_permeability_m2": (case.vug_permeability_m2 if model == "darcy_darcy" else None),
            "permeability_contrast": (
                case.permeability_contrast if model == "darcy_darcy" else None
            ),
            "matrix_drag_coefficient_dimensionless": benchmark.matrix_drag,
            "vug_drag_coefficient_dimensionless": benchmark.vug_drag,
            "vug_drag_pa_s_per_m2": (
                0.0
                if model == "darcy_brinkman"
                else case.dynamic_viscosity_pa_s / case.vug_permeability_m2
            ),
            "matrix_screening_length_m": case.matrix_screening_length_m,
            "base_target_mesh_size_m": (case.base_target_mesh_size_fraction * case.side_length_m),
            "mesh_size_policy": "nearly_uniform_body_fitted",
            "nondimensionalization": "matrix_darcy_velocity",
            "velocity_scale_m_per_s": velocity_scale,
            "pressure_scale_pa": case.pressure_drop_pa,
            "vms_constant": vms_constant if model == "darcy_darcy" else None,
            **solver_metadata,
        },
    )
    # The solve integrates velocity over a normalized outlet of length one.
    # Convert that mean velocity to physical 2D discharge per unit out-of-plane
    # depth while retaining the already computed physical permeability.
    result.flow_rate *= case.side_length_m
    result.cross_section_area = case.side_length_m
    return result


__all__ = [
    "CenteredVugFlowCase2D",
    "CenteredVugFlowModel",
    "M2_PER_MILLIDARCY",
    "run_centered_vug_flow_case",
]
