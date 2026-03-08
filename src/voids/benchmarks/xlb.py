from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from voids.image.network_extraction import (
    NetworkExtractionResult,
    extract_spanning_pore_network,
    infer_sample_axes,
)
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    SinglePhaseResult,
    solve,
)


def _as_binary_volume(phases: np.ndarray) -> np.ndarray:
    """Validate and normalize a binary segmented volume."""

    arr = np.asarray(phases)
    if arr.ndim not in {2, 3}:
        raise ValueError("phases must be a 2D or 3D binary segmented volume")

    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0, 1, False, True))):
        raise ValueError("phases must be binary with void=1 and solid=0")
    return np.asarray(arr, dtype=int)


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0e-30)
    return abs(a - b) / denom


def _axis_to_index(axis: str, ndim: int) -> int:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError(f"flow_axis must be one of {sorted(axis_map)}, got {axis!r}")
    axis_index = axis_map[axis]
    if axis_index >= ndim:
        raise ValueError(f"flow_axis '{axis}' is not compatible with a {ndim}D volume")
    return axis_index


def _mask_to_indices(mask: np.ndarray) -> list[list[int]] | None:
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        return None
    return [np.asarray(comp, dtype=int).tolist() for comp in coords]


def _superficial_velocity_profile(
    axial_velocity_lattice: np.ndarray,
    void_mask: np.ndarray,
) -> np.ndarray:
    """Return superficial axial velocity by plane along the aligned flow axis."""

    total_area_cells = int(np.prod(axial_velocity_lattice.shape[1:]))
    profile = np.zeros(axial_velocity_lattice.shape[0], dtype=float)
    for plane in range(axial_velocity_lattice.shape[0]):
        plane_void = np.asarray(void_mask[plane], dtype=bool)
        if np.any(plane_void):
            plane_flux = float(
                np.asarray(axial_velocity_lattice[plane], dtype=float)[plane_void].sum()
            )
            profile[plane] = plane_flux / float(total_area_cells)
    return profile


def _import_xlb():
    try:
        import jax
        import xlb
        from xlb.compute_backend import ComputeBackend
        from xlb.operator.boundary_condition import HalfwayBounceBackBC, RegularizedBC
        from xlb.operator.stepper import IncompressibleNavierStokesStepper
        from xlb.precision_policy import PrecisionPolicy
        from xlb.velocity_set import D2Q9, D3Q19
    except ImportError as exc:
        raise ImportError(
            "XLB benchmarks require the optional 'xlb' dependency. "
            "Install the Pixi 'lbm' environment or `pip install xlb`."
        ) from exc

    return {
        "jax": jax,
        "xlb": xlb,
        "ComputeBackend": ComputeBackend,
        "HalfwayBounceBackBC": HalfwayBounceBackBC,
        "RegularizedBC": RegularizedBC,
        "IncompressibleNavierStokesStepper": IncompressibleNavierStokesStepper,
        "PrecisionPolicy": PrecisionPolicy,
        "D2Q9": D2Q9,
        "D3Q19": D3Q19,
    }


@dataclass(slots=True)
class XLBOptions:
    """Numerical controls for the direct-image XLB benchmark.

    Notes
    -----
    The current `voids` adapter uses XLB's JAX backend only. This keeps the
    dependency path compatible with CPU-only macOS and Linux environments, which
    is the relevant portability target for the benchmark notebook.
    """

    backend: str = "jax"
    precision_policy: str = "FP32FP32"
    collision_model: str = "BGK"
    streaming_scheme: str = "pull"
    lattice_viscosity: float = 0.10
    rho_inlet: float = 1.001
    rho_outlet: float = 1.000
    inlet_outlet_buffer_cells: int = 6
    max_steps: int = 2000
    min_steps: int = 200
    check_interval: int = 100
    steady_rtol: float = 1.0e-3


@dataclass(slots=True)
class XLBDirectSimulationResult:
    """Store direct-image LBM outputs from an XLB run."""

    flow_axis: str
    voxel_size: float
    image_porosity: float
    sample_lengths: dict[str, float]
    sample_cross_sections: dict[str, float]
    lattice_viscosity: float
    lattice_density_inlet: float
    lattice_density_outlet: float
    lattice_pressure_drop: float
    inlet_outlet_buffer_cells: int
    omega: float
    superficial_velocity_lattice: float
    superficial_velocity_profile_lattice: np.ndarray
    axial_velocity_lattice: np.ndarray
    converged: bool
    n_steps: int
    convergence_metric: float
    permeability: float
    backend: str
    backend_version: str | None


@dataclass(slots=True)
class SegmentedVolumeXLBResult:
    """Store extraction, porosity, and direct-image XLB benchmark outputs."""

    extract: NetworkExtractionResult
    fluid: FluidSinglePhase
    bc: PressureBC
    options: SinglePhaseOptions
    xlb_options: XLBOptions
    image_porosity: float
    absolute_porosity: float
    effective_porosity: float
    voids_result: SinglePhaseResult
    xlb_result: XLBDirectSimulationResult
    permeability_abs_diff: float
    permeability_rel_diff: float

    def to_record(self) -> dict[str, Any]:
        """Return scalar diagnostics suitable for tabulation."""

        k_voids = float((self.voids_result.permeability or {}).get(self.extract.flow_axis, np.nan))
        return {
            "flow_axis": self.extract.flow_axis,
            "phi_image": float(self.image_porosity),
            "phi_abs": float(self.absolute_porosity),
            "phi_eff": float(self.effective_porosity),
            "Np": int(self.extract.net.Np),
            "Nt": int(self.extract.net.Nt),
            "k_voids": k_voids,
            "k_xlb": float(self.xlb_result.permeability),
            "k_abs_diff": float(self.permeability_abs_diff),
            "k_rel_diff": float(self.permeability_rel_diff),
            "voids_mass_balance_error": float(self.voids_result.mass_balance_error),
            "conductance_model": str(self.options.conductance_model),
            "solver_voids": str(self.options.solver),
            "extract_backend": str(self.extract.backend),
            "extract_backend_version": self.extract.backend_version,
            "xlb_backend": str(self.xlb_result.backend),
            "xlb_backend_version": self.xlb_result.backend_version,
            "xlb_steps": int(self.xlb_result.n_steps),
            "xlb_converged": bool(self.xlb_result.converged),
            "xlb_convergence_metric": float(self.xlb_result.convergence_metric),
            "xlb_lattice_viscosity": float(self.xlb_result.lattice_viscosity),
            "xlb_rho_inlet": float(self.xlb_result.lattice_density_inlet),
            "xlb_rho_outlet": float(self.xlb_result.lattice_density_outlet),
            "xlb_dp_lattice": float(self.xlb_result.lattice_pressure_drop),
            "xlb_buffer_cells": int(self.xlb_result.inlet_outlet_buffer_cells),
            "xlb_u_superficial_lattice": float(self.xlb_result.superficial_velocity_lattice),
        }


def solve_binary_volume_with_xlb(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    options: XLBOptions | None = None,
) -> XLBDirectSimulationResult:
    """Solve a binary segmented volume directly with XLB and estimate permeability.

    Notes
    -----
    The permeability conversion is based on lattice units:

    ``K_phys = nu_lu * U_lu * L_lu * dx_phys**2 / dp_lu``

    where ``U_lu`` is the superficial sample velocity, ``nu_lu`` is the lattice
    kinematic viscosity, ``L_lu`` is the voxel count along the flow axis, and
    ``dp_lu = c_s^2 (rho_in - rho_out)``.

    This keeps the benchmark focused on permeability, which is the transport
    quantity comparable to the PNM solve without requiring a full physical
    pressure-unit calibration of the lattice simulation.
    """

    arr = _as_binary_volume(phases)
    xlb_options = options or XLBOptions()

    if xlb_options.backend.lower() != "jax":
        raise ValueError("The current `voids` XLB adapter supports only backend='jax'")
    if xlb_options.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if xlb_options.min_steps < 0:
        raise ValueError("min_steps must be non-negative")
    if xlb_options.check_interval <= 0:
        raise ValueError("check_interval must be positive")
    if xlb_options.steady_rtol <= 0:
        raise ValueError("steady_rtol must be positive")
    if xlb_options.lattice_viscosity <= 0:
        raise ValueError("lattice_viscosity must be positive")
    if xlb_options.rho_inlet <= xlb_options.rho_outlet:
        raise ValueError(
            "rho_inlet must be greater than rho_outlet for positive pressure-driven flow"
        )
    if xlb_options.inlet_outlet_buffer_cells < 0:
        raise ValueError("inlet_outlet_buffer_cells must be non-negative")

    _, axis_lengths, axis_areas, inferred_axis = infer_sample_axes(arr.shape, voxel_size=voxel_size)
    axis = inferred_axis if flow_axis is None else flow_axis
    axis_index = _axis_to_index(axis, arr.ndim)

    aligned_void_sample = np.moveaxis(np.asarray(arr, dtype=bool), axis_index, 0)
    if not np.any(aligned_void_sample[0]):
        raise ValueError("The inlet plane contains no void voxels for the requested flow axis")
    if not np.any(aligned_void_sample[-1]):
        raise ValueError("The outlet plane contains no void voxels for the requested flow axis")

    buffer_cells = int(xlb_options.inlet_outlet_buffer_cells)
    if buffer_cells > 0:
        aligned_void = np.pad(
            aligned_void_sample,
            pad_width=((buffer_cells, buffer_cells),) + ((0, 0),) * (aligned_void_sample.ndim - 1),
            mode="constant",
            constant_values=True,
        )
    else:
        aligned_void = aligned_void_sample

    xlb_api = _import_xlb()
    jax = xlb_api["jax"]
    xlb = xlb_api["xlb"]
    ComputeBackend = xlb_api["ComputeBackend"]
    PrecisionPolicy = xlb_api["PrecisionPolicy"]
    IncompressibleNavierStokesStepper = xlb_api["IncompressibleNavierStokesStepper"]
    HalfwayBounceBackBC = xlb_api["HalfwayBounceBackBC"]
    RegularizedBC = xlb_api["RegularizedBC"]
    velocity_set_cls = xlb_api["D2Q9"] if arr.ndim == 2 else xlb_api["D3Q19"]

    precision_policy = getattr(PrecisionPolicy, xlb_options.precision_policy, None)
    if precision_policy is None:
        raise ValueError(f"Unknown XLB precision policy {xlb_options.precision_policy!r}")

    compute_backend = ComputeBackend.JAX
    velocity_set = velocity_set_cls(precision_policy, compute_backend)
    xlb.init(velocity_set, compute_backend, precision_policy)
    grid = xlb.grid.grid_factory(aligned_void.shape, compute_backend=compute_backend)

    sealed_side_mask = np.zeros_like(aligned_void, dtype=bool)
    for side_axis in range(1, aligned_void.ndim):
        lower = [slice(None)] * aligned_void.ndim
        upper = [slice(None)] * aligned_void.ndim
        lower[side_axis] = 0
        upper[side_axis] = -1
        sealed_side_mask[tuple(lower)] = True
        sealed_side_mask[tuple(upper)] = True

    inlet_mask = np.zeros_like(aligned_void, dtype=bool)
    outlet_mask = np.zeros_like(aligned_void, dtype=bool)
    # Pressure BCs are imposed on planar reservoir faces restricted to void voxels,
    # with side-wall edges excluded because those cells belong to the sealed sample
    # jacket.  Intersecting with ``aligned_void`` prevents solid voxels from being
    # assigned a pressure BC, which would otherwise "open" them and corrupt the
    # bounce-back mask assignment below.
    inlet_mask[0, ...] = aligned_void[0, ...] & ~sealed_side_mask[0, ...]
    outlet_mask[-1, ...] = aligned_void[-1, ...] & ~sealed_side_mask[-1, ...]
    if not np.any(inlet_mask):
        raise ValueError(
            "The trimmed inlet plane has no interior void voxels for the requested flow axis"
        )
    if not np.any(outlet_mask):
        raise ValueError(
            "The trimmed outlet plane has no interior void voxels for the requested flow axis"
        )

    bounceback_mask = ~aligned_void
    bounceback_mask |= sealed_side_mask
    bounceback_mask &= ~(inlet_mask | outlet_mask)

    inlet_indices = _mask_to_indices(inlet_mask)
    outlet_indices = _mask_to_indices(outlet_mask)
    bounceback_indices = _mask_to_indices(bounceback_mask)

    boundary_conditions = [
        RegularizedBC(
            "pressure", prescribed_value=float(xlb_options.rho_inlet), indices=inlet_indices
        ),
        RegularizedBC(
            "pressure", prescribed_value=float(xlb_options.rho_outlet), indices=outlet_indices
        ),
    ]
    if bounceback_indices is not None:
        boundary_conditions.append(HalfwayBounceBackBC(indices=bounceback_indices))

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=xlb_options.collision_model,
        streaming_scheme=xlb_options.streaming_scheme,
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    omega = 1.0 / (3.0 * float(xlb_options.lattice_viscosity) + 0.5)
    omega = np.asarray(omega, dtype=precision_policy.compute_precision.jax_dtype)

    axial_velocity_aligned = np.zeros_like(aligned_void, dtype=float)
    superficial_profile = np.zeros(aligned_void.shape[0], dtype=float)
    superficial_velocity = 0.0
    convergence_metric = np.inf
    previous_superficial_velocity: float | None = None
    converged = False
    n_steps = 0

    def _measure_current_state(f_current) -> tuple[np.ndarray, np.ndarray, float]:
        jax.block_until_ready(f_current)
        _, u = stepper.macroscopic(f_current)
        axial_full = np.asarray(u[0], dtype=float)
        sample_slice = slice(buffer_cells, buffer_cells + aligned_void_sample.shape[0])
        axial = axial_full[sample_slice, ...]
        profile = _superficial_velocity_profile(axial, aligned_void_sample)
        interior_profile = profile[1:-1] if profile.size > 2 else profile
        mean_superficial_velocity = float(np.mean(interior_profile))
        return axial, profile, mean_superficial_velocity

    for step in range(xlb_options.max_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        f_0, f_1 = f_1, f_0
        n_steps = step + 1

        if n_steps % xlb_options.check_interval != 0 and n_steps != xlb_options.max_steps:
            continue

        axial_velocity_aligned, superficial_profile, superficial_velocity = _measure_current_state(
            f_0
        )
        if previous_superficial_velocity is not None:
            convergence_metric = abs(superficial_velocity - previous_superficial_velocity) / max(
                abs(previous_superficial_velocity),
                1.0e-30,
            )
        previous_superficial_velocity = superficial_velocity

        if (
            n_steps >= xlb_options.min_steps
            and np.isfinite(convergence_metric)
            and convergence_metric < xlb_options.steady_rtol
        ):
            converged = True
            break

    if not np.any(axial_velocity_aligned):
        axial_velocity_aligned, superficial_profile, superficial_velocity = _measure_current_state(
            f_0
        )
        if previous_superficial_velocity is not None:
            convergence_metric = abs(superficial_velocity - previous_superficial_velocity) / max(
                abs(previous_superficial_velocity),
                1.0e-30,
            )

    lattice_pressure_drop = float(np.asarray(velocity_set.cs2)) * (
        float(xlb_options.rho_inlet) - float(xlb_options.rho_outlet)
    )
    permeability = (
        float(xlb_options.lattice_viscosity)
        * float(superficial_velocity)
        * float(axis_lengths[axis])
        * float(voxel_size)
        / float(lattice_pressure_drop)
    )
    if not converged:
        warnings.warn(
            "XLB direct-image benchmark did not satisfy the steady-state tolerance "
            f"after {n_steps} steps; the reported permeability may be biased. "
            f"Last relative velocity change: {convergence_metric:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.isfinite(permeability) or permeability <= 0.0:
        raise RuntimeError(
            "XLB produced a non-physical permeability estimate "
            f"({permeability:.6e} m^2-equivalent). "
            "This usually indicates incompatible boundary conditions, an "
            "insufficient inlet/outlet buffer, or a run that is too short to "
            "reach steady state."
        )

    axial_velocity_original = np.moveaxis(axial_velocity_aligned, 0, axis_index)

    return XLBDirectSimulationResult(
        flow_axis=axis,
        voxel_size=float(voxel_size),
        image_porosity=float(arr.mean()),
        sample_lengths=axis_lengths,
        sample_cross_sections=axis_areas,
        lattice_viscosity=float(xlb_options.lattice_viscosity),
        lattice_density_inlet=float(xlb_options.rho_inlet),
        lattice_density_outlet=float(xlb_options.rho_outlet),
        lattice_pressure_drop=float(lattice_pressure_drop),
        inlet_outlet_buffer_cells=buffer_cells,
        omega=float(np.asarray(omega)),
        superficial_velocity_lattice=float(superficial_velocity),
        superficial_velocity_profile_lattice=np.asarray(superficial_profile, dtype=float),
        axial_velocity_lattice=np.asarray(axial_velocity_original, dtype=float),
        converged=bool(converged),
        n_steps=int(n_steps),
        convergence_metric=float(convergence_metric),
        permeability=float(permeability),
        backend="jax",
        backend_version=getattr(xlb, "__version__", None),
    )


def benchmark_segmented_volume_with_xlb(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    fluid: FluidSinglePhase | None = None,
    pin: float = 2.0e5,
    pout: float = 1.0e5,
    options: SinglePhaseOptions | None = None,
    xlb_options: XLBOptions | None = None,
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    extraction_kwargs: dict[str, object] | None = None,
    provenance_notes: dict[str, object] | None = None,
    strict: bool = True,
) -> SegmentedVolumeXLBResult:
    """Benchmark a segmented volume against a direct-image XLB solve.

    Notes
    -----
    The `voids` side solves on the extracted pore network. The XLB side solves
    directly on the binary segmented image. This is the scientifically relevant
    comparison if the goal is to assess extraction loss and PNM model discrepancy
    against a higher-fidelity voxel-scale reference.
    """

    arr = _as_binary_volume(phases)
    image_phi = float(arr.mean())

    notes = dict(provenance_notes or {})
    notes.setdefault("benchmark_kind", "segmented_volume_xlb")

    extract = extract_spanning_pore_network(
        arr,
        voxel_size=voxel_size,
        flow_axis=flow_axis,
        length_unit=length_unit,
        pressure_unit=pressure_unit,
        extraction_kwargs=extraction_kwargs,
        provenance_notes=notes,
        strict=strict,
    )

    fluid_used = fluid or FluidSinglePhase(viscosity=1.0e-3)
    options_used = options or SinglePhaseOptions(
        conductance_model="valvatne_blunt_baseline",
        solver="direct",
    )
    xlb_options_used = xlb_options or XLBOptions()

    axis = extract.flow_axis
    inlet_count = int(
        np.asarray(extract.net.pore_labels.get(f"inlet_{axis}min", []), dtype=bool).sum()
    )
    outlet_count = int(
        np.asarray(extract.net.pore_labels.get(f"outlet_{axis}max", []), dtype=bool).sum()
    )
    if extract.net.Np == 0 or inlet_count == 0 or outlet_count == 0:
        raise ValueError(
            "The extracted spanning network is empty or lacks non-empty inlet/outlet pore labels "
            f"for axis '{axis}', so the XLB benchmark cannot be compared against `voids` on this case."
        )

    bc = PressureBC(f"inlet_{axis}min", f"outlet_{axis}max", pin=float(pin), pout=float(pout))
    voids_result = solve(
        extract.net,
        fluid=fluid_used,
        bc=bc,
        axis=axis,
        options=options_used,
    )
    xlb_result = solve_binary_volume_with_xlb(
        arr,
        voxel_size=voxel_size,
        flow_axis=axis,
        options=xlb_options_used,
    )

    k_voids = float((voids_result.permeability or {}).get(axis, np.nan))
    k_xlb = float(xlb_result.permeability)

    return SegmentedVolumeXLBResult(
        extract=extract,
        fluid=fluid_used,
        bc=bc,
        options=options_used,
        xlb_options=xlb_options_used,
        image_porosity=image_phi,
        absolute_porosity=float(absolute_porosity(extract.net)),
        effective_porosity=float(effective_porosity(extract.net, axis=axis)),
        voids_result=voids_result,
        xlb_result=xlb_result,
        permeability_abs_diff=abs(k_voids - k_xlb),
        permeability_rel_diff=_rel_diff(k_voids, k_xlb),
    )


__all__ = [
    "SegmentedVolumeXLBResult",
    "XLBDirectSimulationResult",
    "XLBOptions",
    "benchmark_segmented_volume_with_xlb",
    "solve_binary_volume_with_xlb",
]
