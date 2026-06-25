"""Direct-image single-phase XLB adapters for binary segmented volumes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import warnings

import numpy as np

from voids.image.network_extraction import infer_sample_axes
from voids.physics.singlephase import FluidSinglePhase

ISOTHERMAL_LATTICE_CS2 = 1.0 / 3.0
DEFAULT_REFERENCE_DENSITY_LATTICE = 1.0
DEFAULT_PRESSURE_DROP_LATTICE = ISOTHERMAL_LATTICE_CS2 * 1.0e-3
DEFAULT_STOKES_PRESSURE_DROP_LATTICE = ISOTHERMAL_LATTICE_CS2 * 2.0e-4
MAX_RECOMMENDED_DENSITY_DROP_LATTICE = 1.0e-2


class XLBConvergenceWarning(RuntimeWarning):
    """Warn that an XLB solve reached ``max_steps`` before steady convergence."""


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


def _reference_pressure_lattice(reference_density_lattice: float, *, cs2: float) -> float:
    if reference_density_lattice <= 0.0:
        raise ValueError("reference_density_lattice must be positive")
    return float(cs2) * float(reference_density_lattice)


def _resolve_lattice_pressure_bc(
    options: "XLBOptions",
    *,
    cs2: float,
) -> tuple[float, float]:
    """Resolve inlet and outlet lattice pressures from the configured options."""

    consistency_rtol = 1.0e-9
    consistency_atol = 1.0e-12

    p_in = options.pressure_inlet_lattice
    p_out = options.pressure_outlet_lattice
    dp = options.pressure_drop_lattice

    if p_in is not None and p_out is not None:
        pass
    elif p_in is not None and dp is not None:
        p_out = float(p_in) - float(dp)
    elif p_out is not None and dp is not None:
        p_in = float(p_out) + float(dp)
    elif dp is not None:
        p_out = _reference_pressure_lattice(
            options.reference_density_lattice,
            cs2=cs2,
        )
        p_in = float(p_out) + float(dp)
    elif options.rho_inlet is not None and options.rho_outlet is not None:
        p_in = float(cs2) * float(options.rho_inlet)
        p_out = float(cs2) * float(options.rho_outlet)
    else:
        raise ValueError(
            "XLBOptions must define a positive lattice pressure drop via "
            "`pressure_inlet_lattice` / `pressure_outlet_lattice`, "
            "`pressure_drop_lattice`, or legacy `rho_inlet` / `rho_outlet`."
        )

    assert p_in is not None
    assert p_out is not None
    p_in = float(p_in)
    p_out = float(p_out)

    if not np.isfinite(p_in) or not np.isfinite(p_out):
        raise ValueError("Resolved lattice pressure BCs must be finite")
    if p_in <= 0.0 or p_out <= 0.0:
        raise ValueError("Resolved lattice pressure BCs must be positive")
    if p_in <= p_out:
        raise ValueError(
            "The inlet lattice pressure must be greater than the outlet lattice pressure "
            "for positive pressure-driven flow"
        )

    if dp is not None:
        resolved_dp = p_in - p_out
        if not np.isclose(resolved_dp, float(dp), rtol=consistency_rtol, atol=consistency_atol):
            raise ValueError(
                "Inconsistent lattice pressure BCs: `pressure_drop_lattice` must match "
                "`pressure_inlet_lattice - pressure_outlet_lattice`."
            )

    if options.rho_inlet is not None:
        p_in_from_rho = float(cs2) * float(options.rho_inlet)
        if not np.isclose(p_in, p_in_from_rho, rtol=consistency_rtol, atol=consistency_atol):
            raise ValueError(
                "Inconsistent lattice BCs: `pressure_inlet_lattice` is not compatible with "
                "`rho_inlet` via `p = cs2 * rho`."
            )
    if options.rho_outlet is not None:
        p_out_from_rho = float(cs2) * float(options.rho_outlet)
        if not np.isclose(p_out, p_out_from_rho, rtol=consistency_rtol, atol=consistency_atol):
            raise ValueError(
                "Inconsistent lattice BCs: `pressure_outlet_lattice` is not compatible with "
                "`rho_outlet` via `p = cs2 * rho`."
            )
    return p_in, p_out


def _physical_pressure_drop_to_lattice(
    delta_p_physical: float,
    *,
    voxel_size: float,
    lattice_viscosity: float,
    fluid: FluidSinglePhase,
) -> float:
    """Map a physical pressure drop to lattice pressure units.

    Notes
    -----
    The mapping uses the standard LBM unit conversion

    ``nu_phys = mu_phys / rho_phys``

    and

    ``dt_phys = nu_lu * dx_phys**2 / nu_phys``.

    Pressure is then converted with

    ``delta_p_lu = delta_p_phys * dt_phys**2 / (rho_phys * dx_phys**2)``.
    """

    if delta_p_physical <= 0.0:
        raise ValueError("Physical pressure drop must be positive for the XLB solve")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if lattice_viscosity <= 0.0:
        raise ValueError("lattice_viscosity must be positive")
    if fluid.density is None or fluid.density <= 0.0:
        raise ValueError(
            "Fluid density must be positive to map a physical pressure drop into lattice units"
        )

    mu_phys = fluid.reference_viscosity()
    if mu_phys <= 0.0:
        raise ValueError("Fluid viscosity must be positive")
    nu_phys = float(mu_phys) / float(fluid.density)
    dt_phys = float(lattice_viscosity) * float(voxel_size) ** 2 / nu_phys
    return float(delta_p_physical) * dt_phys**2 / (float(fluid.density) * float(voxel_size) ** 2)


def _couple_xlb_options_to_physical_pressure_drop(
    options: "XLBOptions",
    *,
    delta_p_physical: float,
    voxel_size: float,
    fluid: FluidSinglePhase,
) -> "XLBOptions":
    """Return XLB options whose lattice pressure drop matches a physical ``delta_p``."""

    _, p_out_current = _resolve_lattice_pressure_bc(options, cs2=ISOTHERMAL_LATTICE_CS2)
    delta_p_lattice = _physical_pressure_drop_to_lattice(
        delta_p_physical,
        voxel_size=voxel_size,
        lattice_viscosity=options.lattice_viscosity,
        fluid=fluid,
    )
    density_drop_lattice = delta_p_lattice / ISOTHERMAL_LATTICE_CS2
    if density_drop_lattice > MAX_RECOMMENDED_DENSITY_DROP_LATTICE:
        raise ValueError(
            "The requested physical pressure drop maps to a lattice density jump of "
            f"{density_drop_lattice:.3e}, which is too large for the intended "
            "weakly compressible XLB solve. Reduce `pin - pout`, reduce "
            "`voxel_size`, or choose a fluid / lattice scaling that keeps the "
            "lattice pressure drop small."
        )

    return replace(
        options,
        pressure_inlet_lattice=float(p_out_current) + float(delta_p_lattice),
        pressure_outlet_lattice=float(p_out_current),
        pressure_drop_lattice=None,
        rho_inlet=None,
        rho_outlet=None,
    )


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
        import warp
        import warp.utils

        if not hasattr(warp.utils, "ScopedTimer") and hasattr(warp, "ScopedTimer"):
            warp.utils.ScopedTimer = warp.ScopedTimer

        import xlb
        from xlb.compute_backend import ComputeBackend
        from xlb.operator.boundary_condition import HalfwayBounceBackBC, RegularizedBC
        from xlb.operator.stepper import IncompressibleNavierStokesStepper
        from xlb.precision_policy import PrecisionPolicy
        from xlb.velocity_set import D2Q9, D3Q19
    except ImportError as exc:
        raise ImportError(
            "XLB direct-image solvers require the optional 'xlb' dependency. "
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
    """Numerical controls for the direct-image XLB solver.

    Attributes
    ----------
    formulation :
        Either ``"incompressible_navier_stokes"`` or
        ``"steady_stokes_limit"``.
    backend :
        XLB compute backend. The current `voids` adapter supports only
        ``"jax"``.
    precision_policy :
        XLB precision policy name, for example ``"FP32FP32"``.
    collision_model :
        XLB collision operator label passed to the stepper.
    streaming_scheme :
        XLB streaming scheme label passed to the stepper.
    lattice_viscosity :
        Kinematic viscosity in lattice units.
    pressure_inlet_lattice, pressure_outlet_lattice :
        Optional inlet and outlet lattice pressures. If both are provided they
        define the pressure BC directly.
    pressure_drop_lattice :
        Optional lattice pressure drop. When set without explicit inlet/outlet
        pressures, it is applied relative to ``reference_density_lattice``.
    reference_density_lattice :
        Reference lattice density used to construct a baseline outlet pressure
        when only ``pressure_drop_lattice`` is provided.
    rho_inlet, rho_outlet :
        Legacy density-based BC inputs retained for backward compatibility.
        They are converted internally to lattice pressure using
        ``p_lu = c_s^2 rho``.
    inlet_outlet_buffer_cells :
        Number of fluid reservoir layers inserted ahead of and behind the
        sample.
    max_steps, min_steps, check_interval, steady_rtol :
        Iteration and convergence controls for the steady-state solve.

    Notes
    -----
    The current `voids` adapter uses XLB's JAX backend only. This keeps the
    dependency path compatible with CPU-only macOS and Linux environments.

    The currently exposed XLB operator is the incompressible Navier-Stokes
    lattice-Boltzmann stepper. Setting ``formulation="steady_stokes_limit"``
    does not switch to a different PDE solver; it selects conservative forcing
    and convergence defaults so the converged solution can be interpreted in the
    steady creeping-flow limit.

    In this isothermal LBM setting, lattice pressure satisfies
    ``p_lu = c_s^2 rho``. The preferred public inputs are therefore the lattice
    pressure fields ``pressure_inlet_lattice`` / ``pressure_outlet_lattice`` or
    the pressure drop ``pressure_drop_lattice``. The legacy fields ``rho_inlet``
    and ``rho_outlet`` remain supported for backward compatibility and are
    converted internally to pressure.
    """

    formulation: str = "incompressible_navier_stokes"
    backend: str = "jax"
    precision_policy: str = "FP32FP32"
    collision_model: str = "BGK"
    streaming_scheme: str = "pull"
    lattice_viscosity: float = 0.10
    pressure_inlet_lattice: float | None = None
    pressure_outlet_lattice: float | None = None
    pressure_drop_lattice: float | None = DEFAULT_PRESSURE_DROP_LATTICE
    reference_density_lattice: float = DEFAULT_REFERENCE_DENSITY_LATTICE
    rho_inlet: float | None = None
    rho_outlet: float | None = None
    inlet_outlet_buffer_cells: int = 6
    max_steps: int = 2000
    min_steps: int = 200
    check_interval: int = 100
    steady_rtol: float = 1.0e-3

    @classmethod
    def steady_stokes_defaults(cls, **overrides: float | int | str) -> "XLBOptions":
        """Return a conservative preset for the steady creeping-flow limit.

        Notes
        -----
        XLB does not currently expose a separate Stokes-only stepper in the
        installed package used by `voids`. This preset therefore still uses the
        incompressible Navier-Stokes LBM operator, but with a smaller lattice
        pressure drop and tighter steady-state controls so the converged solution is
        interpreted in the low-Reynolds, low-Mach limit.

        The buffer and convergence controls are intentionally stricter than the
        generic :class:`XLBOptions` defaults. They were selected from same-ROI
        DRP-317 sensitivity runs as a conservative direct-image permeability
        preset, not as a fit to experimental permeability.
        """

        values: dict[str, Any] = {
            "formulation": "steady_stokes_limit",
            "lattice_viscosity": 0.10,
            "pressure_drop_lattice": DEFAULT_STOKES_PRESSURE_DROP_LATTICE,
            "inlet_outlet_buffer_cells": 12,
            "max_steps": 8000,
            "min_steps": 1200,
            "check_interval": 100,
            "steady_rtol": 1.0e-4,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(slots=True)
class XLBDirectSimulationResult:
    """Store direct-image LBM outputs from an XLB run.

    Attributes
    ----------
    lattice_pressure_inlet, lattice_pressure_outlet, lattice_pressure_drop :
        Resolved inlet, outlet, and differential pressure in lattice units.
    lattice_density_inlet, lattice_density_outlet :
        Equivalent lattice densities associated with the pressure BCs through
        ``p_lu = c_s^2 rho``.
    permeability :
        Apparent permeability mapped back to physical units.
    max_mach_lattice, reynolds_voxel_max :
        Low-inertia diagnostics useful when interpreting a run as a creeping-flow
        reference.
    """

    flow_axis: str
    voxel_size: float
    image_porosity: float
    sample_lengths: dict[str, float]
    sample_cross_sections: dict[str, float]
    lattice_viscosity: float
    lattice_pressure_inlet: float
    lattice_pressure_outlet: float
    lattice_density_inlet: float
    lattice_density_outlet: float
    lattice_pressure_drop: float
    inlet_outlet_buffer_cells: int
    omega: float
    superficial_velocity_lattice: float
    superficial_velocity_profile_lattice: np.ndarray
    velocity_lattice: np.ndarray
    axial_velocity_lattice: np.ndarray
    converged: bool
    n_steps: int
    convergence_metric: float
    permeability: float
    backend: str
    backend_version: str | None
    formulation: str
    velocity_set: str
    collision_model: str
    streaming_scheme: str
    max_speed_lattice: float
    max_mach_lattice: float
    reynolds_voxel_max: float


def solve_binary_volume_with_xlb(
    phases: np.ndarray,
    *,
    voxel_size: float,
    flow_axis: str | None = None,
    options: XLBOptions | None = None,
) -> XLBDirectSimulationResult:
    """Solve a binary segmented volume directly with XLB and estimate permeability.

    Parameters
    ----------
    phases :
        Binary segmented volume with ``void=1`` and ``solid=0``.
    voxel_size :
        Physical voxel size used when mapping lattice permeability back to
        physical units.
    flow_axis :
        Requested flow axis. When omitted, the canonical sample axis inferred by
        :func:`voids.image.network_extraction.infer_sample_axes` is used.
    options :
        XLB numerical controls. This low-level interface expects lattice-unit
        pressure inputs through ``pressure_inlet_lattice``,
        ``pressure_outlet_lattice``, or ``pressure_drop_lattice``.

    Returns
    -------
    XLBDirectSimulationResult
        Direct-image XLB solution summary and permeability diagnostics.

    Raises
    ------
    ValueError
        If the input volume, flow axis, or XLB numerical controls are invalid.
    RuntimeError
        If the run completes with a non-physical permeability estimate.

    Warns
    -----
    XLBConvergenceWarning
        If ``max_steps`` is reached before the steady-state velocity criterion
        is satisfied. The result is still returned with ``converged=False`` and
        the final ``n_steps`` / ``convergence_metric`` diagnostics.

    Notes
    -----
    The current adapter uses XLB's incompressible Navier-Stokes lattice
    Boltzmann stepper with BGK-style collision and pressure / bounce-back
    boundary conditions. If ``options.formulation == "steady_stokes_limit"``,
    the same stepper is still used; the solution is simply driven more gently
    and interpreted in the low-Reynolds, low-Mach steady limit.

    The inlet and outlet conditions are pressure boundary conditions. The public
    API accepts lattice pressure values or a lattice pressure drop, which are
    converted internally to the density values expected by the isothermal LBM
    boundary operator through ``p_lu = c_s^2 rho``.

    The permeability conversion is based on lattice units:

    ``K_phys = nu_lu * U_lu * L_lu * dx_phys**2 / dp_lu``

    where ``U_lu`` is the superficial sample velocity, ``nu_lu`` is the lattice
    kinematic viscosity, ``L_lu`` is the voxel count along the flow axis, and
    ``dp_lu = p_in,lu - p_out,lu``.

    This keeps the solve focused on permeability, which is the transport
    quantity comparable to the PNM solve without requiring a full physical
    pressure-unit calibration of the lattice simulation.
    """

    arr = _as_binary_volume(phases)
    xlb_options = options or XLBOptions()

    if xlb_options.backend.lower() != "jax":
        raise ValueError("The current `voids` XLB adapter supports only backend='jax'")
    if xlb_options.formulation not in {"incompressible_navier_stokes", "steady_stokes_limit"}:
        raise ValueError(
            "formulation must be 'incompressible_navier_stokes' or 'steady_stokes_limit'"
        )
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
    if xlb_options.inlet_outlet_buffer_cells < 0:
        raise ValueError("inlet_outlet_buffer_cells must be non-negative")
    pressure_inlet_lattice, pressure_outlet_lattice = _resolve_lattice_pressure_bc(
        xlb_options,
        cs2=ISOTHERMAL_LATTICE_CS2,
    )
    density_inlet_lattice = pressure_inlet_lattice / ISOTHERMAL_LATTICE_CS2
    density_outlet_lattice = pressure_outlet_lattice / ISOTHERMAL_LATTICE_CS2
    density_drop_lattice = density_inlet_lattice - density_outlet_lattice
    if xlb_options.formulation == "steady_stokes_limit" and density_drop_lattice > 5.0e-4:
        warnings.warn(
            "steady_stokes_limit is being used with a relatively large lattice "
            "pressure drop. The run still uses the same incompressible LBM stepper, "
            "so keep the imposed driving small if the goal is a steady Stokes-limit "
            "reference.",
            RuntimeWarning,
            stacklevel=2,
        )

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
        lower: list[slice | int] = [slice(None)] * aligned_void.ndim
        upper: list[slice | int] = [slice(None)] * aligned_void.ndim
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
            "pressure",
            prescribed_value=float(pressure_inlet_lattice / float(np.asarray(velocity_set.cs2))),
            indices=inlet_indices,
        ),
        RegularizedBC(
            "pressure",
            prescribed_value=float(pressure_outlet_lattice / float(np.asarray(velocity_set.cs2))),
            indices=outlet_indices,
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

    omega_float = 1.0 / (3.0 * float(xlb_options.lattice_viscosity) + 0.5)
    omega = np.asarray(omega_float, dtype=precision_policy.compute_precision.jax_dtype)

    velocity_aligned = np.zeros((arr.ndim, *aligned_void_sample.shape), dtype=float)
    axial_velocity_aligned = np.zeros_like(aligned_void, dtype=float)
    superficial_profile = np.zeros(aligned_void.shape[0], dtype=float)
    superficial_velocity = 0.0
    max_speed_lattice = 0.0
    convergence_metric = np.inf
    previous_superficial_velocity: float | None = None
    converged = False
    n_steps = 0

    # f_current is a JAX array; typed as `object` because `jaxlib` types are
    # not available in the default mypy environment (jax is an optional dep).
    def _measure_current_state(
        f_current: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        jax.block_until_ready(f_current)
        _, u = stepper.macroscopic(f_current)
        u_full = np.asarray(u, dtype=float)
        axial_full = np.asarray(u_full[0], dtype=float)
        speed_full = np.linalg.norm(u_full, axis=0)
        sample_slice = slice(buffer_cells, buffer_cells + aligned_void_sample.shape[0])
        velocity = np.asarray(u_full[:, sample_slice, ...], dtype=float)
        axial = axial_full[sample_slice, ...]
        speed = speed_full[sample_slice, ...]
        profile = _superficial_velocity_profile(axial, aligned_void_sample)
        interior_profile = profile[1:-1] if profile.size > 2 else profile
        mean_superficial_velocity = float(np.mean(interior_profile))
        max_speed = float(np.max(speed[aligned_void_sample]))
        return velocity, axial, profile, mean_superficial_velocity, max_speed

    for step in range(xlb_options.max_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        f_0, f_1 = f_1, f_0
        n_steps = step + 1

        if n_steps % xlb_options.check_interval != 0 and n_steps != xlb_options.max_steps:
            continue

        (
            velocity_aligned,
            axial_velocity_aligned,
            superficial_profile,
            superficial_velocity,
            max_speed_lattice,
        ) = _measure_current_state(f_0)
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
        (
            velocity_aligned,
            axial_velocity_aligned,
            superficial_profile,
            superficial_velocity,
            max_speed_lattice,
        ) = _measure_current_state(f_0)
        if previous_superficial_velocity is not None:
            convergence_metric = abs(superficial_velocity - previous_superficial_velocity) / max(
                abs(previous_superficial_velocity),
                1.0e-30,
            )

    lattice_pressure_drop = float(pressure_inlet_lattice) - float(pressure_outlet_lattice)
    permeability = (
        float(xlb_options.lattice_viscosity)
        * float(superficial_velocity)
        * float(axis_lengths[axis])
        * float(voxel_size)
        / float(lattice_pressure_drop)
    )
    cs_lattice = float(np.sqrt(np.asarray(velocity_set.cs2)))
    max_mach_lattice = float(max_speed_lattice) / cs_lattice
    reynolds_voxel_max = float(max_speed_lattice) / float(xlb_options.lattice_viscosity)
    if not converged:
        warnings.warn(
            "XLB direct-image solve did not satisfy the steady-state tolerance "
            f"after {n_steps} steps; the reported permeability may be biased. "
            f"Last relative velocity change: {convergence_metric:.3e}.",
            XLBConvergenceWarning,
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

    velocity_spatial_original = np.moveaxis(velocity_aligned, 1, axis_index + 1)
    aligned_axes = [axis_index, *[idx for idx in range(arr.ndim) if idx != axis_index]]
    velocity_original = np.empty_like(velocity_spatial_original)
    for aligned_component, original_component in enumerate(aligned_axes):
        velocity_original[original_component] = velocity_spatial_original[aligned_component]

    axial_velocity_original = np.asarray(velocity_original[axis_index], dtype=float)

    return XLBDirectSimulationResult(
        flow_axis=axis,
        voxel_size=float(voxel_size),
        image_porosity=float(arr.mean()),
        sample_lengths=axis_lengths,
        sample_cross_sections=axis_areas,
        lattice_viscosity=float(xlb_options.lattice_viscosity),
        lattice_pressure_inlet=float(pressure_inlet_lattice),
        lattice_pressure_outlet=float(pressure_outlet_lattice),
        lattice_density_inlet=float(pressure_inlet_lattice / float(np.asarray(velocity_set.cs2))),
        lattice_density_outlet=float(pressure_outlet_lattice / float(np.asarray(velocity_set.cs2))),
        lattice_pressure_drop=float(lattice_pressure_drop),
        inlet_outlet_buffer_cells=buffer_cells,
        omega=omega_float,
        superficial_velocity_lattice=float(superficial_velocity),
        superficial_velocity_profile_lattice=np.asarray(superficial_profile, dtype=float),
        velocity_lattice=np.asarray(velocity_original, dtype=float),
        axial_velocity_lattice=np.asarray(axial_velocity_original, dtype=float),
        converged=bool(converged),
        n_steps=int(n_steps),
        convergence_metric=float(convergence_metric),
        permeability=float(permeability),
        backend="jax",
        backend_version=getattr(xlb, "__version__", None),
        formulation=str(xlb_options.formulation),
        velocity_set=str(velocity_set_cls.__name__),
        collision_model=str(xlb_options.collision_model),
        streaming_scheme=str(xlb_options.streaming_scheme),
        max_speed_lattice=float(max_speed_lattice),
        max_mach_lattice=float(max_mach_lattice),
        reynolds_voxel_max=float(reynolds_voxel_max),
    )


__all__ = [
    "DEFAULT_PRESSURE_DROP_LATTICE",
    "DEFAULT_REFERENCE_DENSITY_LATTICE",
    "DEFAULT_STOKES_PRESSURE_DROP_LATTICE",
    "ISOTHERMAL_LATTICE_CS2",
    "MAX_RECOMMENDED_DENSITY_DROP_LATTICE",
    "XLBConvergenceWarning",
    "XLBDirectSimulationResult",
    "XLBOptions",
    "solve_binary_volume_with_xlb",
]
