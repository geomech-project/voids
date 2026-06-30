from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, cast

_FEM_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _apply_fem_thread_defaults() -> None:
    """Keep direct sparse factorizations away from unsafe threaded BLAS defaults."""

    for name, value in _FEM_THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


_apply_fem_thread_defaults()

import numpy as np  # noqa: E402

from voids.image.porosity import PermeabilityMap, PorosityMap  # noqa: E402
from voids.linalg.cudss import (  # noqa: E402
    nvmath_cudss_controls_from_arguments as _nvmath_cudss_controls_from_arguments,
    resolve_nvmath_cudss_controls as _resolve_nvmath_cudss_controls,
    solve_nvmath_cudss as _solve_nvmath_cudss,
)


_AXIS_NAMES = ("x", "y", "z")
_MIN_MARKER = {"x": 1, "y": 3, "z": 5}
_MAX_MARKER = {"x": 2, "y": 4, "z": 6}
LinearSolverBackend = Literal[
    "auto",
    "petsc",
    "scipy",
    "superlu",
    "umfpack",
    "pardiso",
    "nvmath_cudss",
]
LinearSystemDType = Literal["float32", "float64"]
BrinkmanVelocityScale = Literal["viscous", "unit_darcy"]
FEMSolverPreset = Literal[
    "custom",
    "direct_reference",
    "direct_parallel",
    "iterative_block_lgmres_experimental",
    "iterative_fieldsplit_experimental",
]
_SUPERLU_PERMC_SPECS = {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}
_SUPERLU_CONTROL_KEYS = {
    "permc_spec",
    "diag_pivot_thresh",
    "relax",
    "panel_size",
    "equil",
}
_FEM_LINEAR_SYSTEM_DTYPES: dict[LinearSystemDType, np.dtype[Any]] = {
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


def _resolve_fem_linear_system_dtype(value: object) -> LinearSystemDType:
    normalized = str(value).strip().lower()
    if normalized not in _FEM_LINEAR_SYSTEM_DTYPES:
        raise ValueError("linear_system_dtype must be either 'float32' or 'float64'")
    return normalized


def _copy_sparse_matrix_with_value_dtype(matrix: Any, dtype: np.dtype[Any]) -> Any:
    if hasattr(matrix, "astype"):
        return matrix.astype(dtype, copy=False)
    copied = matrix.copy() if hasattr(matrix, "copy") else matrix
    if hasattr(copied, "data"):
        copied.data = np.asarray(copied.data, dtype=dtype)
    return copied


def _superlu_controls_from_arguments(
    *,
    permc_spec: str | None = None,
    diag_pivot_thresh: float | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    equil: bool | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    superlu_controls: dict[str, Any] = {}
    for key, value in (controls or {}).items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key not in _SUPERLU_CONTROL_KEYS:
            supported = ", ".join(sorted(_SUPERLU_CONTROL_KEYS))
            raise ValueError(
                f"Unsupported SuperLU control {key!r}; supported controls: {supported}"
            )
        superlu_controls[normalized_key] = value
    if permc_spec is not None:
        superlu_controls["permc_spec"] = permc_spec
    if diag_pivot_thresh is not None:
        superlu_controls["diag_pivot_thresh"] = float(diag_pivot_thresh)
    if relax is not None:
        superlu_controls["relax"] = int(relax)
    if panel_size is not None:
        superlu_controls["panel_size"] = int(panel_size)
    if equil is not None:
        superlu_controls["equil"] = bool(equil)
    return superlu_controls


@dataclass(slots=True)
class FEniCSSolverOptions:
    """Linear solver controls for FEniCSx linear problems.

    The default ``linear_backend="auto"`` preserves the PETSc/MUMPS path on
    platforms with a full DOLFINx/PETSc stack. On native Windows, where that
    PETSc stack is not available in the conda-forge FEniCSx packages used by
    ``voids``, ``auto`` uses DOLFINx assembly plus SciPy/SuperLU.

    Use ``linear_backend="superlu"``, ``"scipy"``, ``"umfpack"``,
    ``"pardiso"``, or ``"nvmath_cudss"`` to request the serial
    DOLFINx-assembly/direct-sparse path explicitly. These paths use the same
    weak form and boundary conditions as PETSc; only the linear algebra backend
    changes. ``"scipy"`` is kept as a backward-compatible alias for the
    SciPy/SuperLU path. ``"umfpack"`` requires the optional
    ``scikits.umfpack`` package, ``"pardiso"`` requires the optional
    ``pypardiso`` package, and ``"nvmath_cudss"`` requires a CUDA-capable
    PyTorch/nvmath cuDSS runtime. ``linear_system_dtype`` selects the assembled
    value dtype for serial sparse backends. PETSc backends use the scalar type
    of the installed PETSc/DOLFINx stack and are therefore double precision in
    the supported Pixi FEM environment.
    """

    linear_backend: LinearSolverBackend = "auto"
    solver_preset: FEMSolverPreset = "direct_reference"
    linear_system_dtype: LinearSystemDType = "float64"
    superlu_controls: dict[str, Any] = field(default_factory=dict)
    umfpack_controls: dict[str, Any] = field(default_factory=dict)
    nvmath_cudss_controls: dict[str, Any] = field(default_factory=dict)
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

    @classmethod
    def direct_lu(
        cls,
        backend: str = "mumps",
        *,
        linear_backend: LinearSolverBackend = "petsc",
        petsc_options_prefix: str = "voids_fem_",
        shift_amount: float | None = 1.0e-12,
        mumps_memory_relaxation_percent: int | None = None,
        mumps_workspace_mb: int | None = None,
    ) -> FEniCSSolverOptions:
        """Create PETSc options for a direct sparse LU solve.

        Parameters
        ----------
        backend :
            PETSc factorization package, for example ``"mumps"`` or
            ``"superlu_dist"``.
        linear_backend :
            Linear algebra backend. This builder configures PETSc options, so
            the default is ``"petsc"``.
        petsc_options_prefix :
            Prefix used by DOLFINx for PETSc runtime options.
        shift_amount :
            Nonzero diagonal shift used during factorization. Pass ``None`` to
            omit the shift options.
        mumps_memory_relaxation_percent, mumps_workspace_mb :
            Optional MUMPS memory controls. They are added only when the backend
            is ``"mumps"``.
        """

        options: dict[str, Any] = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": backend,
            "ksp_error_if_not_converged": True,
        }
        if shift_amount is not None:
            options["pc_factor_shift_type"] = "nonzero"
            options["pc_factor_shift_amount"] = float(shift_amount)
        if backend == "mumps":
            if mumps_memory_relaxation_percent is not None:
                options["mat_mumps_icntl_14"] = int(mumps_memory_relaxation_percent)
            if mumps_workspace_mb is not None:
                options["mat_mumps_icntl_23"] = int(mumps_workspace_mb)
        return cls(
            linear_backend=linear_backend,
            solver_preset="direct_reference",
            petsc_options=options,
            petsc_options_prefix=petsc_options_prefix,
        )

    @classmethod
    def direct_reference(
        cls,
        backend: str = "mumps",
        *,
        petsc_options_prefix: str = "voids_fem_",
        shift_amount: float | None = 1.0e-12,
        mumps_memory_relaxation_percent: int | None = None,
        mumps_workspace_mb: int | None = None,
    ) -> FEniCSSolverOptions:
        """Create the conservative PETSc direct-solve preset.

        This preset is the stable baseline for permeability comparisons: one
        monolithic sparse LU factorization through PETSc, normally MUMPS. It is
        appropriate as the direct reference before testing faster iterative or
        distributed configurations.
        """

        return cls.direct_lu(
            backend,
            petsc_options_prefix=petsc_options_prefix,
            shift_amount=shift_amount,
            mumps_memory_relaxation_percent=mumps_memory_relaxation_percent,
            mumps_workspace_mb=mumps_workspace_mb,
        )

    @classmethod
    def direct_parallel(
        cls,
        backend: str = "mumps",
        *,
        petsc_options_prefix: str = "voids_fem_",
        shift_amount: float | None = 1.0e-12,
        mumps_memory_relaxation_percent: int | None = 500,
        mumps_workspace_mb: int | None = None,
    ) -> FEniCSSolverOptions:
        """Create an MPI-oriented PETSc direct-solve preset.

        This still uses a direct factorization, but records a distinct preset
        name and applies a larger default MUMPS memory-relaxation factor. It is
        intended for runs launched under MPI, where MUMPS or SuperLU_DIST can
        distribute the factorization.
        """

        options = cls.direct_lu(
            backend,
            petsc_options_prefix=petsc_options_prefix,
            shift_amount=shift_amount,
            mumps_memory_relaxation_percent=(
                mumps_memory_relaxation_percent if backend == "mumps" else None
            ),
            mumps_workspace_mb=mumps_workspace_mb if backend == "mumps" else None,
        )
        options.solver_preset = "direct_parallel"
        return options

    @classmethod
    def iterative_fieldsplit_experimental(
        cls,
        *,
        petsc_options_prefix: str = "voids_fem_",
        ksp_type: str = "fgmres",
        rtol: float = 1.0e-8,
        max_it: int = 500,
    ) -> FEniCSSolverOptions:
        """Create an experimental PETSc field-split preset for mixed systems.

        The preset is deliberately labelled experimental because scalable
        saddle-point preconditioning depends on the formulation, coefficient
        contrast, mesh, and PETSc build. Permeability results from this preset
        should be compared against a direct reference before being used for
        scientific conclusions.
        """

        return cls(
            linear_backend="petsc",
            solver_preset="iterative_fieldsplit_experimental",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": float(rtol),
                "ksp_max_it": int(max_it),
                "ksp_error_if_not_converged": True,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "upper",
                "pc_fieldsplit_detect_saddle_point": True,
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "hypre",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "jacobi",
            },
            petsc_options_prefix=petsc_options_prefix,
        )

    @classmethod
    def iterative_block_lgmres_experimental(
        cls,
        *,
        petsc_options_prefix: str = "voids_fem_",
        rtol: float = 1.0e-8,
        atol: float = 1.0e-10,
        max_it: int = 3000,
        block_lu_backend: str = "superlu_dist",
    ) -> FEniCSSolverOptions:
        """Create an experimental block-LGMRES preset for USFEM block solves.

        This preset is intended for ``solve_brinkman_usfem_block`` with
        ``matrix_kind="nest"`` and ``preconditioner="none"``. It uses a
        multiplicative velocity/pressure field split as the outer
        preconditioner, with direct LU subsolves on the two diagonal operator
        blocks. It is a correctness-oriented iterative baseline, not a scalable
        multigrid preconditioner.
        """

        return cls(
            linear_backend="petsc",
            solver_preset="iterative_block_lgmres_experimental",
            petsc_options={
                "ksp_type": "lgmres",
                "ksp_rtol": float(rtol),
                "ksp_atol": float(atol),
                "ksp_max_it": int(max_it),
                "ksp_norm_type": "unpreconditioned",
                "ksp_error_if_not_converged": True,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_u_0_ksp_type": "preonly",
                "fieldsplit_u_0_pc_type": "lu",
                "fieldsplit_u_0_pc_factor_mat_solver_type": block_lu_backend,
                "fieldsplit_p_1_ksp_type": "preonly",
                "fieldsplit_p_1_pc_type": "lu",
                "fieldsplit_p_1_pc_factor_mat_solver_type": block_lu_backend,
            },
            petsc_options_prefix=petsc_options_prefix,
        )

    @classmethod
    def scipy_direct(
        cls,
        *,
        linear_system_dtype: LinearSystemDType = "float64",
        permc_spec: str | None = None,
        diag_pivot_thresh: float | None = None,
        relax: int | None = None,
        panel_size: int | None = None,
        equil: bool | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> FEniCSSolverOptions:
        """Create options for the serial DOLFINx-assembly/SciPy SuperLU backend."""

        return cls(
            linear_backend="scipy",
            solver_preset="direct_reference",
            linear_system_dtype=linear_system_dtype,
            superlu_controls=_superlu_controls_from_arguments(
                permc_spec=permc_spec,
                diag_pivot_thresh=diag_pivot_thresh,
                relax=relax,
                panel_size=panel_size,
                equil=equil,
                controls=controls,
            ),
        )

    @classmethod
    def superlu_direct(
        cls,
        *,
        linear_system_dtype: LinearSystemDType = "float64",
        permc_spec: str | None = None,
        diag_pivot_thresh: float | None = None,
        relax: int | None = None,
        panel_size: int | None = None,
        equil: bool | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> FEniCSSolverOptions:
        """Create options for the serial DOLFINx-assembly/SuperLU backend."""

        return cls(
            linear_backend="superlu",
            solver_preset="direct_reference",
            linear_system_dtype=linear_system_dtype,
            superlu_controls=_superlu_controls_from_arguments(
                permc_spec=permc_spec,
                diag_pivot_thresh=diag_pivot_thresh,
                relax=relax,
                panel_size=panel_size,
                equil=equil,
                controls=controls,
            ),
        )

    @classmethod
    def umfpack_direct(
        cls,
        *,
        linear_system_dtype: LinearSystemDType = "float64",
        ordering: str | int | float | None = None,
        strategy: str | int | float | None = None,
        pivot_tolerance: float | None = None,
        sym_pivot_tolerance: float | None = None,
        scale: str | int | float | None = None,
        block_size: int | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> FEniCSSolverOptions:
        """Create options for the serial DOLFINx-assembly/UMFPACK backend."""

        umfpack_controls: dict[str, Any] = dict(controls or {})
        if ordering is not None:
            umfpack_controls["ordering"] = ordering
        if strategy is not None:
            umfpack_controls["strategy"] = strategy
        if pivot_tolerance is not None:
            umfpack_controls["pivot_tolerance"] = float(pivot_tolerance)
        if sym_pivot_tolerance is not None:
            umfpack_controls["sym_pivot_tolerance"] = float(sym_pivot_tolerance)
        if scale is not None:
            umfpack_controls["scale"] = scale
        if block_size is not None:
            umfpack_controls["block_size"] = int(block_size)
        return cls(
            linear_backend="umfpack",
            solver_preset="direct_reference",
            linear_system_dtype=linear_system_dtype,
            umfpack_controls=umfpack_controls,
        )

    @classmethod
    def nvmath_cudss_direct(
        cls,
        *,
        dtype: Literal["float32", "float64"] = "float64",
        device_ids: int | Sequence[int] | Literal["all"] | None = None,
        ir_steps: int = 5,
        use_matching: bool = True,
        pivot_type: Literal["col", "row", "none"] | None = None,
        check_residual: bool = True,
        residual_rtol: float | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> FEniCSSolverOptions:
        """Create options for the optional CUDA cuDSS direct-solver backend.

        The backend uses serial DOLFINx assembly, transfers the assembled sparse
        system to CUDA tensors, and solves it through ``nvmath.bindings.cudss``.
        It requires a CUDA-capable PyTorch/nvmath cuDSS runtime at solve time.
        Pass ``device_ids=(0, 1)`` or ``device_ids="all"`` to request a
        single-node multi-GPU cuDSS handle.
        ``dtype="float64"``, matching, and iterative refinement are the
        conservative defaults for mixed Brinkman systems.
        """

        resolved_controls = _nvmath_cudss_controls_from_arguments(
            dtype=dtype,
            device_ids=device_ids,
            ir_steps=ir_steps,
            use_matching=use_matching,
            pivot_type=pivot_type,
            check_residual=check_residual,
            residual_rtol=residual_rtol,
            controls=controls,
        )
        return cls(
            linear_backend="nvmath_cudss",
            solver_preset="direct_reference",
            linear_system_dtype=cast(LinearSystemDType, resolved_controls["dtype"]),
            nvmath_cudss_controls=resolved_controls,
        )

    @classmethod
    def pardiso_direct(
        cls,
        *,
        linear_system_dtype: LinearSystemDType = "float64",
    ) -> FEniCSSolverOptions:
        """Create options for the serial DOLFINx-assembly/PARDISO backend."""

        return cls(
            linear_backend="pardiso",
            solver_preset="direct_reference",
            linear_system_dtype=linear_system_dtype,
        )


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


@dataclass(frozen=True, slots=True)
class BrinkmanNondimensionalization:
    """Velocity scaling choice for Brinkman nondimensional forms.

    The internal unknowns are ``u* = u / U`` and ``p* = p / DeltaP``. Use
    ``velocity_scale="viscous"`` for ``U = DeltaP L / mu``. Use
    ``velocity_scale="unit_darcy"`` for ``U = DeltaP K / (mu L)`` on constant
    permeability maps. Results are converted back to physical velocity and
    pressure before permeability is reported.
    """

    velocity_scale: BrinkmanVelocityScale = "viscous"


@dataclass(frozen=True, slots=True)
class _BrinkmanNondimensionalScales:
    velocity_scale_type: BrinkmanVelocityScale
    length_scale: float
    pressure_scale: float
    velocity_scale: float
    coefficient_scale: float
    constant_permeability: float | None = None
    darcy_number: float | None = None


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
    la: Any
    mesh: Any
    petsc: Any | None
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


def _require_dolfinx_core() -> _DolfinxAPI:
    try:
        import basix.ufl as basix_ufl
        from dolfinx import fem, la, mesh
        from mpi4py import MPI
        import ufl
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        message = (
            "FEniCSx FEM backends require DOLFINx, Basix, UFL, and mpi4py. "
            "Use the Pixi 'fem' feature/environment or install a compatible "
            "fenics-dolfinx stack before calling voids.fem."
        )
        raise ImportError(message) from exc
    return _DolfinxAPI(
        MPI=MPI,
        basix_ufl=basix_ufl,
        fem=fem,
        la=la,
        mesh=mesh,
        petsc=None,
        ufl=ufl,
    )


def _require_dolfinx_petsc(api: _DolfinxAPI | None = None) -> _DolfinxAPI:
    api = api or _require_dolfinx_core()
    try:
        import dolfinx.fem.petsc as petsc
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        message = (
            "The PETSc FEM linear backend requires the full DOLFINx/PETSc "
            "Python stack, including dolfinx.fem.petsc and petsc4py. Use "
            "linear_backend='superlu' for a serial direct sparse solve, or "
            "install a compatible PETSc-enabled fenics-dolfinx stack."
        )
        if sys.platform.startswith("win"):
            message += (
                " Native Windows does not provide this PETSc-backed path in "
                "the conda-forge FEniCSx stack used by voids; "
                "linear_backend='auto' falls back to the SciPy/SuperLU direct backend "
                "on Windows."
            )
        raise ImportError(message) from exc
    return _DolfinxAPI(
        MPI=api.MPI,
        basix_ufl=api.basix_ufl,
        fem=api.fem,
        la=api.la,
        mesh=api.mesh,
        petsc=petsc,
        ufl=api.ufl,
    )


def _require_dolfinx() -> _DolfinxAPI:
    """Return a DOLFINx API object with the PETSc linear backend available."""

    return _require_dolfinx_petsc()


def _resolve_linear_backend(requested: LinearSolverBackend, api: _DolfinxAPI) -> str:
    if requested not in {
        "auto",
        "petsc",
        "scipy",
        "superlu",
        "umfpack",
        "pardiso",
        "nvmath_cudss",
    }:
        raise ValueError(
            "linear_backend must be one of 'auto', 'petsc', 'scipy', 'superlu', "
            "'umfpack', 'pardiso', or 'nvmath_cudss'"
        )
    if requested != "auto":
        return requested
    if sys.platform.startswith("win"):
        try:
            _require_dolfinx_petsc(api)
        except ImportError:
            return "scipy"
    return "petsc"


def _axis_index(axis: str, ndim: int) -> int:
    if axis not in _AXIS_NAMES[:ndim]:
        raise ValueError(f"flow_axis must be one of {_AXIS_NAMES[:ndim]}, got {axis!r}")
    return _AXIS_NAMES.index(axis)


def _cell_size_tuple(permeability_map: PermeabilityMap) -> tuple[float, ...]:
    return tuple(float(v) for v in cast(tuple[float, ...], permeability_map.cell_size))


def _origin_tuple(permeability_map: PermeabilityMap) -> tuple[float, ...]:
    return tuple(float(v) for v in cast(tuple[float, ...], permeability_map.origin))


def _close_coordinate(values: Any, coordinate: float, *, atol: float) -> np.ndarray:
    return np.asarray(
        np.isclose(np.asarray(values, dtype=float), float(coordinate), atol=float(atol)),
        dtype=bool,
    )


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


def _build_context(
    problem: FEMMapProblem,
    *,
    flow_axis: str,
    api: _DolfinxAPI | None = None,
) -> _FEMContext:
    api = api or _require_dolfinx_core()
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


def _boundary_geometry(context: _FEMContext) -> tuple[np.ndarray, np.ndarray, float]:
    local_coordinates = np.asarray(
        context.mesh.geometry.x[:, : context.mesh.geometry.dim],
        dtype=float,
    )
    if local_coordinates.size == 0:
        local_origin = np.full(context.mesh.geometry.dim, np.inf, dtype=float)
        local_upper = np.full(context.mesh.geometry.dim, -np.inf, dtype=float)
    else:
        local_origin = np.min(local_coordinates, axis=0)
        local_upper = np.max(local_coordinates, axis=0)
    problem_origin = np.empty_like(local_origin)
    problem_upper = np.empty_like(local_upper)
    context.mesh.comm.Allreduce(local_origin, problem_origin, op=context.api.MPI.MIN)
    context.mesh.comm.Allreduce(local_upper, problem_upper, op=context.api.MPI.MAX)
    extent = float(np.max(problem_upper - problem_origin))
    atol = max(extent * 1.0e-10, float(np.finfo(float).eps))
    return problem_origin, problem_upper, atol


def _side_wall_bcs(context: _FEMContext, mixed_space: Any, *, flow_axis: str) -> list[Any]:
    axis = _axis_index(flow_axis, context.mesh.geometry.dim)
    problem_origin, problem_upper, atol = _boundary_geometry(context)
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


def _velocity_side_wall_bcs(
    context: _FEMContext,
    velocity_space: Any,
    *,
    flow_axis: str,
) -> list[Any]:
    axis = _axis_index(flow_axis, context.mesh.geometry.dim)
    problem_origin, problem_upper, atol = _boundary_geometry(context)
    bcs: list[Any] = []
    for side_axis in range(context.mesh.geometry.dim):
        if side_axis == axis:
            continue
        component_space = velocity_space.sub(side_axis)
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

    problem_origin, _, atol = _boundary_geometry(context)
    dofs = context.api.fem.locate_dofs_geometrical(
        (pressure_space, collapsed),
        lambda x: _match_point(x, problem_origin, ndim=context.mesh.geometry.dim, atol=atol),
    )
    if dofs[0].size > 1:
        dofs = [dofs[0][:1], dofs[1][:1]]
    return context.api.fem.dirichletbc(zero, dofs, pressure_space)


def _standalone_pressure_gauge_bc(context: _FEMContext, pressure_space: Any) -> Any:
    if context.api.petsc is None:
        raise ImportError("standalone pressure gauge boundary conditions require PETSc support.")
    problem_origin, _, atol = _boundary_geometry(context)
    dofs = context.api.fem.locate_dofs_geometrical(
        pressure_space,
        lambda x: _match_point(x, problem_origin, ndim=context.mesh.geometry.dim, atol=atol),
    )
    if dofs.size == 0:
        index_map = pressure_space.dofmap.index_map
        block_size = int(pressure_space.dofmap.index_map_bs)
        local_size = int(index_map.size_local) * block_size
        local_start, _ = index_map.local_range
        local_candidate = int(local_start) * block_size if local_size > 0 else sys.maxsize
        owner_candidate = int(context.mesh.comm.allreduce(local_candidate, op=context.api.MPI.MIN))
        if local_candidate == owner_candidate:
            dofs = np.asarray([0], dtype=np.int32)
        else:
            dofs = np.asarray([], dtype=np.int32)
    else:
        dofs = dofs[:1]
    return context.api.fem.dirichletbc(
        context.api.petsc.PETSc.ScalarType(0),
        dofs,
        pressure_space,
    )


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


def _thread_environment_metadata() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in sorted(_FEM_THREAD_ENV_DEFAULTS)}


def _constant_permeability_value(values: np.ndarray) -> float:
    permeability_values = np.asarray(values, dtype=float)
    if permeability_values.size == 0:
        raise ValueError("permeability map must contain at least one value")
    if not np.all(np.isfinite(permeability_values)) or np.any(permeability_values <= 0.0):
        raise ValueError("permeability values must be positive and finite")
    value = float(permeability_values.flat[0])
    if not np.allclose(permeability_values, value, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "velocity_scale='unit_darcy' follows U = DeltaP K / (mu L) and "
            "requires a globally constant floored permeability map"
        )
    return value


def _brinkman_nondimensional_scales(
    context: _FEMContext,
    problem: FEMMapProblem,
    *,
    pressure_inlet: float,
    pressure_outlet: float,
    velocity_scale: BrinkmanVelocityScale,
) -> _BrinkmanNondimensionalScales:
    pressure_scale = float(pressure_inlet) - float(pressure_outlet)
    if pressure_scale <= 0.0 or not np.isfinite(pressure_scale):
        raise ValueError("pressure scale must be positive and finite")
    length_scale = float(context.domain_length)
    if length_scale <= 0.0 or not np.isfinite(length_scale):
        raise ValueError("length scale must be positive and finite")
    if velocity_scale == "viscous":
        velocity_scale_value = pressure_scale * length_scale / float(problem.viscosity)
        coefficient_scale = length_scale / float(problem.viscosity)
        constant_permeability = None
        darcy_number = None
    elif velocity_scale == "unit_darcy":
        constant_permeability = _constant_permeability_value(
            np.asarray(context.coefficients["permeability_values"], dtype=float)
        )
        velocity_scale_value = (
            pressure_scale * constant_permeability / (float(problem.viscosity) * length_scale)
        )
        coefficient_scale = constant_permeability / (float(problem.viscosity) * length_scale)
        darcy_number = constant_permeability / (length_scale * length_scale)
    else:
        raise ValueError("velocity_scale must be either 'viscous' or 'unit_darcy'")
    if velocity_scale_value <= 0.0 or not np.isfinite(velocity_scale_value):
        raise ValueError("nondimensional velocity scale must be positive and finite")
    return _BrinkmanNondimensionalScales(
        velocity_scale_type=velocity_scale,
        length_scale=length_scale,
        pressure_scale=pressure_scale,
        velocity_scale=velocity_scale_value,
        coefficient_scale=coefficient_scale,
        constant_permeability=constant_permeability,
        darcy_number=darcy_number,
    )


def _resolve_brinkman_nondimensionalization(
    nondimensional: bool | BrinkmanNondimensionalization,
) -> BrinkmanNondimensionalization | None:
    if isinstance(nondimensional, bool):
        return BrinkmanNondimensionalization() if nondimensional else None
    if isinstance(nondimensional, BrinkmanNondimensionalization):
        return nondimensional
    raise TypeError("nondimensional must be a bool or BrinkmanNondimensionalization")


def _brinkman_nondimensional_coefficients(
    context: _FEMContext,
    problem: FEMMapProblem,
    scales: _BrinkmanNondimensionalScales,
) -> tuple[Any, Any]:
    coefficient_scale = context.api.fem.Constant(
        context.mesh,
        float(scales.coefficient_scale),
    )
    gamma = coefficient_scale * context.coefficients["gamma"]
    nu_eff = coefficient_scale * context.coefficients["nu_eff"]
    return gamma, nu_eff


def _nondimensional_metadata(scales: _BrinkmanNondimensionalScales | None) -> dict[str, Any]:
    if scales is None:
        return {"nondimensional": False}
    return {
        "nondimensional": True,
        "nondimensional_velocity_scale_type": scales.velocity_scale_type,
        "nondimensional_length_scale": scales.length_scale,
        "nondimensional_pressure_scale": scales.pressure_scale,
        "nondimensional_velocity_scale": scales.velocity_scale,
        "nondimensional_coefficient_scale": scales.coefficient_scale,
        "nondimensional_constant_permeability": scales.constant_permeability,
        "nondimensional_darcy_number": scales.darcy_number,
    }


def _mpi_metadata(context: _FEMContext) -> dict[str, int]:
    return {
        "mpi_size": int(context.mesh.comm.size),
        "mpi_rank": int(context.mesh.comm.rank),
    }


def _petsc_solver_diagnostics(problem: Any) -> dict[str, int | float | str]:
    solver = getattr(problem, "solver", None)
    if solver is None:
        return {}

    diagnostics: dict[str, int | float | str] = {}
    for metadata_key, method_name in (
        ("petsc_ksp_type", "getType"),
        ("petsc_converged_reason", "getConvergedReason"),
        ("petsc_iteration_number", "getIterationNumber"),
        ("petsc_residual_norm", "getResidualNorm"),
    ):
        method = getattr(solver, method_name, None)
        if method is None:
            continue
        try:
            value = method()
        except Exception:  # pragma: no cover - defensive around petsc4py versions
            continue
        if isinstance(value, (str, int, float)):
            diagnostics[metadata_key] = value
    return diagnostics


def _apply_petsc_ksp_options_after_setup(
    petsc_module: Any,
    solver: Any,
    petsc_options: dict[str, Any],
    reapply_fieldsplit_is: Callable[[], bool] | None = None,
) -> bool:
    fieldsplit_is_reapplied = False
    opts = petsc_module.PETSc.Options()
    opts.prefixPush(solver.getOptionsPrefix())
    try:
        for key, value in petsc_options.items():
            opts[key] = value
        solver.setFromOptions()
        if reapply_fieldsplit_is is not None:
            fieldsplit_is_reapplied = reapply_fieldsplit_is()
            if fieldsplit_is_reapplied:
                solver.setFromOptions()
    finally:
        for key in petsc_options:
            del opts[key]
        opts.prefixPop()
    return fieldsplit_is_reapplied


def _set_nest_fieldsplit_is(problem: Any) -> bool:
    matrix = getattr(problem, "A", None)
    get_nest_iss = getattr(matrix, "getNestISs", None)
    solver = getattr(problem, "solver", None)
    get_pc = getattr(solver, "getPC", None)
    solution_fields = getattr(problem, "u", None)
    if get_nest_iss is None or get_pc is None or not isinstance(solution_fields, (list, tuple)):
        return False

    pc = get_pc()
    set_fieldsplit_is = getattr(pc, "setFieldSplitIS", None)
    if set_fieldsplit_is is None:
        return False

    nest_is = get_nest_iss()
    row_is = nest_is[0]
    fields = []
    for index, (solution_field, field_is) in enumerate(zip(solution_fields, row_is, strict=False)):
        name = getattr(solution_field, "name", "f")
        prefix = f"{name}_" if name != "f" else ""
        fields.append((f"{prefix}{index}", field_is))
    set_fieldsplit_is(*fields)
    return True


def _solve_mixed_problem(
    context: _FEMContext,
    *,
    form: Any,
    rhs: Any,
    bcs: list[Any],
    options: FEniCSSolverOptions | None,
    prefix_suffix: str,
) -> tuple[Any, float, dict[str, Any]]:
    solver_options = options or FEniCSSolverOptions()
    api = _require_dolfinx_petsc(context.api)
    start = perf_counter()
    petsc = cast(Any, api.petsc)
    petsc_options_prefix = f"{solver_options.petsc_options_prefix}{prefix_suffix}_"
    problem = petsc.LinearProblem(
        form,
        rhs,
        bcs=bcs,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=dict(solver_options.petsc_options),
    )
    solution = problem.solve()
    solve_seconds = perf_counter() - start
    solver_metadata = {
        "petsc_options_prefix_effective": petsc_options_prefix,
        **_petsc_solver_diagnostics(problem),
    }
    converged_reason = solver_metadata.get("petsc_converged_reason")
    if (
        bool(solver_options.petsc_options.get("ksp_error_if_not_converged", False))
        and isinstance(converged_reason, (int, float))
        and converged_reason < 0
    ):
        raise RuntimeError(
            "PETSc linear solve did not converge: "
            f"reason={converged_reason}, diagnostics={solver_metadata}"
        )
    return (
        solution,
        solve_seconds,
        solver_metadata,
    )


def _solve_block_problem_petsc(
    context: _FEMContext,
    *,
    forms: list[list[Any]],
    rhs: list[Any],
    bcs: list[Any],
    solution_functions: list[Any],
    options: FEniCSSolverOptions | None,
    prefix_suffix: str,
    matrix_kind: Literal["mpi", "nest"],
    preconditioner_forms: list[list[Any | None]] | None = None,
) -> tuple[list[Any], float, dict[str, Any]]:
    solver_options = options or FEniCSSolverOptions()
    api = _require_dolfinx_petsc(context.api)
    start = perf_counter()
    petsc = cast(Any, api.petsc)
    petsc_options_prefix = f"{solver_options.petsc_options_prefix}{prefix_suffix}_"
    deferred_petsc_options = dict(solver_options.petsc_options) if matrix_kind == "nest" else None
    problem = petsc.LinearProblem(
        forms,
        rhs,
        bcs=bcs,
        u=solution_functions,
        P=preconditioner_forms,
        kind=matrix_kind,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=None
        if deferred_petsc_options is not None
        else dict(solver_options.petsc_options),
    )
    fieldsplit_is_reapplied = False
    if deferred_petsc_options is not None:
        fieldsplit_is_reapplied = _apply_petsc_ksp_options_after_setup(
            petsc,
            problem.solver,
            deferred_petsc_options,
            reapply_fieldsplit_is=lambda: _set_nest_fieldsplit_is(problem),
        )
    solution = problem.solve()
    solve_seconds = perf_counter() - start
    solver_metadata = {
        "petsc_options_prefix_effective": petsc_options_prefix,
        "petsc_matrix_kind": matrix_kind,
        "petsc_has_preconditioner_forms": preconditioner_forms is not None,
        "petsc_options_applied_after_block_setup": deferred_petsc_options is not None,
        "petsc_nest_fieldsplit_is_reapplied": fieldsplit_is_reapplied,
        **_petsc_solver_diagnostics(problem),
    }
    converged_reason = solver_metadata.get("petsc_converged_reason")
    if (
        bool(solver_options.petsc_options.get("ksp_error_if_not_converged", False))
        and isinstance(converged_reason, (int, float))
        and converged_reason < 0
    ):
        raise RuntimeError(
            "PETSc linear solve did not converge: "
            f"reason={converged_reason}, diagnostics={solver_metadata}"
        )
    if isinstance(solution, tuple):
        solution = list(solution)
    elif not isinstance(solution, list):
        solution = [solution]
    return solution, solve_seconds, solver_metadata


def _set_dirichlet_bc_values(fem: Any, array: np.ndarray, bcs: list[Any]) -> None:
    """Apply Dirichlet values to an assembled vector without using deprecated DOLFINx APIs."""

    if all(hasattr(bc, "set") for bc in bcs):
        for bc in bcs:
            bc.set(array)
        return
    fem.set_bc(array, bcs)


def _copy_sparse_matrix_with_index_dtype(matrix: Any, dtype: np.dtype[Any]) -> Any:
    """Copy a SciPy sparse matrix and force its structural arrays to ``dtype``."""

    copied = matrix.copy()
    copied.indices = copied.indices.astype(dtype, copy=False)
    copied.indptr = copied.indptr.astype(dtype, copy=False)
    return copied


_UMFPACK_CONTROL_CONSTANTS = {
    "strategy": "UMFPACK_STRATEGY",
    "ordering": "UMFPACK_ORDERING",
    "pivot_tolerance": "UMFPACK_PIVOT_TOLERANCE",
    "sym_pivot_tolerance": "UMFPACK_SYM_PIVOT_TOLERANCE",
    "scale": "UMFPACK_SCALE",
    "block_size": "UMFPACK_BLOCK_SIZE",
    "alloc_init": "UMFPACK_ALLOC_INIT",
    "front_alloc_init": "UMFPACK_FRONT_ALLOC_INIT",
}
_UMFPACK_NAMED_CONTROL_VALUES = {
    "strategy": {
        "auto": "UMFPACK_STRATEGY_AUTO",
        "unsymmetric": "UMFPACK_STRATEGY_UNSYMMETRIC",
        "symmetric": "UMFPACK_STRATEGY_SYMMETRIC",
    },
    "ordering": {
        "cholmod": "UMFPACK_ORDERING_CHOLMOD",
        "amd": "UMFPACK_ORDERING_AMD",
        "metis": "UMFPACK_ORDERING_METIS",
        "best": "UMFPACK_ORDERING_BEST",
        "none": "UMFPACK_ORDERING_NONE",
        "metis_guard": "UMFPACK_ORDERING_METIS_GUARD",
    },
    "scale": {
        "none": "UMFPACK_SCALE_NONE",
        "sum": "UMFPACK_SCALE_SUM",
        "max": "UMFPACK_SCALE_MAX",
    },
}


def _normalize_umfpack_control_key(key: str) -> str:
    normalized = key.strip().lower().replace("-", "_")
    if normalized not in _UMFPACK_CONTROL_CONSTANTS:
        supported = ", ".join(sorted(_UMFPACK_CONTROL_CONSTANTS))
        raise ValueError(f"Unsupported UMFPACK control {key!r}; supported controls: {supported}")
    return normalized


def _resolve_umfpack_control_value(umfpack: Any, key: str, value: Any) -> float:
    if isinstance(value, str):
        normalized_value = value.strip().lower().replace("-", "_")
        constant_name = _UMFPACK_NAMED_CONTROL_VALUES.get(key, {}).get(normalized_value)
        if constant_name is None:
            raise ValueError(f"Unsupported UMFPACK {key!r} value {value!r}")
        if not hasattr(umfpack, constant_name):
            raise ValueError(
                f"UMFPACK control {key!r} value {value!r} requires unavailable "
                f"constant {constant_name}"
            )
        return float(getattr(umfpack, constant_name))
    return float(value)


def _apply_umfpack_controls(
    umfpack: Any,
    context: Any,
    controls: Mapping[str, Any],
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for key, value in controls.items():
        normalized_key = _normalize_umfpack_control_key(str(key))
        constant_name = _UMFPACK_CONTROL_CONSTANTS[normalized_key]
        if not hasattr(umfpack, constant_name):
            raise ValueError(
                f"UMFPACK control {normalized_key!r} requires unavailable constant {constant_name}"
            )
        resolved_value = _resolve_umfpack_control_value(umfpack, normalized_key, value)
        context.control[int(getattr(umfpack, constant_name))] = resolved_value
        resolved[normalized_key] = resolved_value
    return resolved


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def _resolve_superlu_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in controls.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key not in _SUPERLU_CONTROL_KEYS:
            supported = ", ".join(sorted(_SUPERLU_CONTROL_KEYS))
            raise ValueError(
                f"Unsupported SuperLU control {key!r}; supported controls: {supported}"
            )
        if normalized_key == "permc_spec":
            permc_spec = str(value).strip().upper()
            if permc_spec not in _SUPERLU_PERMC_SPECS:
                supported = ", ".join(sorted(_SUPERLU_PERMC_SPECS))
                raise ValueError(
                    f"Unsupported SuperLU permc_spec {value!r}; supported: {supported}"
                )
            resolved[normalized_key] = permc_spec
        elif normalized_key == "diag_pivot_thresh":
            pivot = float(value)
            if pivot < 0.0 or pivot > 1.0:
                raise ValueError("SuperLU diag_pivot_thresh must be between 0 and 1")
            resolved[normalized_key] = pivot
        elif normalized_key in {"relax", "panel_size"}:
            integer_value = int(value)
            if integer_value <= 0:
                raise ValueError(f"SuperLU {normalized_key} must be positive")
            resolved[normalized_key] = integer_value
        elif normalized_key == "equil":
            resolved[normalized_key] = bool(value)
    return resolved


def _solve_umfpack_int64(
    umfpack: Any,
    matrix: Any,
    rhs_array: Any,
    *,
    controls: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Solve with the 64-bit-index UMFPACK family exposed by scikit-umfpack."""

    matrix_int64 = _copy_sparse_matrix_with_index_dtype(matrix, np.dtype(np.int64))
    context = umfpack.UmfpackContext("dl")
    resolved_controls = _apply_umfpack_controls(umfpack, context, controls)
    context.numeric(matrix_int64)
    solution = context.solve(
        umfpack.UMFPACK_A,
        matrix_int64,
        rhs_array,
        autoTranspose=True,
    )
    return solution, {
        "serial_sparse_umfpack_family": "dl",
        "serial_sparse_matrix_indices_dtype": str(matrix_int64.indices.dtype),
        "serial_sparse_matrix_indptr_dtype": str(matrix_int64.indptr.dtype),
        "serial_sparse_umfpack_requested_controls": _json_safe_mapping(controls),
        "serial_sparse_umfpack_resolved_controls": resolved_controls,
    }


def _solve_mixed_problem_serial_direct(
    context: _FEMContext,
    *,
    mixed_space: Any,
    form: Any,
    rhs: Any,
    bcs: list[Any],
    linear_backend: Literal["scipy", "superlu", "umfpack", "pardiso", "nvmath_cudss"],
    linear_system_dtype: LinearSystemDType = "float64",
    superlu_controls: Mapping[str, Any] | None = None,
    umfpack_controls: Mapping[str, Any] | None = None,
    nvmath_cudss_controls: Mapping[str, Any] | None = None,
) -> tuple[Any, float, dict[str, Any]]:
    if context.mesh.comm.size != 1:
        raise NotImplementedError(
            "linear_backend='scipy', linear_backend='superlu', "
            "linear_backend='umfpack', linear_backend='pardiso', and "
            "linear_backend='nvmath_cudss' are "
            "serial-only; use linear_backend='petsc' for MPI-distributed FEM "
            "solves."
        )

    resolved_linear_system_dtype = _resolve_fem_linear_system_dtype(linear_system_dtype)
    if resolved_linear_system_dtype == "float32" and linear_backend in {"umfpack", "pardiso"}:
        backend_name = "scikit-umfpack" if linear_backend == "umfpack" else "pypardiso"
        raise ValueError(
            f"linear_backend={linear_backend!r} currently supports float64 only through "
            f"{backend_name}; use linear_backend='superlu' or linear_backend='scipy' for "
            "CPU single-precision FEM sparse solves."
        )
    value_dtype = _FEM_LINEAR_SYSTEM_DTYPES[resolved_linear_system_dtype]

    solve_linear_system: Callable[[Any, Any], Any]
    sparse_matrix_format = "csr"
    serial_solver_backend = "scipy.sparse.linalg.splu"
    serial_solver_metadata: dict[str, Any] = {}
    if linear_backend == "umfpack":
        try:
            umfpack = import_module("scikits.umfpack")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "linear_backend='umfpack' requires the optional scikits.umfpack package. "
                "Install scikit-umfpack or use linear_backend='scipy'."
            ) from exc

        def solve_umfpack(matrix: Any, rhs_array: Any) -> Any:
            solution, metadata = _solve_umfpack_int64(
                umfpack,
                matrix,
                rhs_array,
                controls=umfpack_controls or {},
            )
            serial_solver_metadata.update(metadata)
            return solution

        solve_linear_system = solve_umfpack
        sparse_matrix_format = "csc"
        serial_solver_backend = "scikits.umfpack.UmfpackContext(dl)"
    elif linear_backend == "nvmath_cudss":
        nvmath_cudss_controls_with_dtype = dict(nvmath_cudss_controls or {})
        has_explicit_cudss_dtype = any(
            str(key).strip().lower().replace("-", "_") in {"dtype", "value_dtype"}
            for key in nvmath_cudss_controls_with_dtype
        )
        if not has_explicit_cudss_dtype:
            nvmath_cudss_controls_with_dtype["dtype"] = resolved_linear_system_dtype
        resolved_nvmath_cudss_controls = _resolve_nvmath_cudss_controls(
            nvmath_cudss_controls_with_dtype
        )
        resolved_linear_system_dtype = cast(
            LinearSystemDType, resolved_nvmath_cudss_controls["dtype"]
        )
        value_dtype = _FEM_LINEAR_SYSTEM_DTYPES[resolved_linear_system_dtype]

        def solve_nvmath_cudss(matrix: Any, rhs_array: Any) -> Any:
            solution, metadata = _solve_nvmath_cudss(
                matrix,
                rhs_array,
                controls=resolved_nvmath_cudss_controls,
            )
            serial_solver_metadata.update(metadata)
            return solution

        solve_linear_system = solve_nvmath_cudss
        sparse_matrix_format = "csr"
        serial_solver_backend = "nvmath.bindings.cudss"
    elif linear_backend == "pardiso":
        try:
            pypardiso = import_module("pypardiso")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "linear_backend='pardiso' requires the optional pypardiso package. "
                "Install pypardiso or use linear_backend='scipy'."
            ) from exc
        solve_linear_system = cast(Callable[[Any, Any], Any], pypardiso.spsolve)
        serial_solver_backend = "pypardiso.spsolve"
    else:
        from scipy.sparse.linalg import splu

        resolved_superlu_controls = _resolve_superlu_controls(superlu_controls or {})
        sparse_matrix_format = "csc"

        def solve_superlu(matrix: Any, rhs_array: Any) -> Any:
            kwargs: dict[str, Any] = {}
            if "permc_spec" in resolved_superlu_controls:
                kwargs["permc_spec"] = resolved_superlu_controls["permc_spec"]
            if "diag_pivot_thresh" in resolved_superlu_controls:
                kwargs["diag_pivot_thresh"] = resolved_superlu_controls["diag_pivot_thresh"]
            if "relax" in resolved_superlu_controls:
                kwargs["relax"] = resolved_superlu_controls["relax"]
            if "panel_size" in resolved_superlu_controls:
                kwargs["panel_size"] = resolved_superlu_controls["panel_size"]
            if "equil" in resolved_superlu_controls:
                kwargs["options"] = {"Equil": resolved_superlu_controls["equil"]}
            lu = splu(matrix, **kwargs)
            serial_solver_metadata.update(
                {
                    "serial_sparse_superlu_requested_controls": _json_safe_mapping(
                        superlu_controls or {}
                    ),
                    "serial_sparse_superlu_resolved_controls": _json_safe_mapping(
                        resolved_superlu_controls
                    ),
                    "serial_sparse_superlu_l_nnz": int(lu.L.nnz),
                    "serial_sparse_superlu_u_nnz": int(lu.U.nnz),
                }
            )
            return lu.solve(rhs_array)

        solve_linear_system = solve_superlu

    fem = context.api.fem
    la = context.api.la
    a_form = fem.form(form)
    rhs_form = fem.form(rhs)

    start = perf_counter()
    matrix = fem.assemble_matrix(a_form, bcs=bcs)
    matrix.scatter_reverse()
    vector = fem.assemble_vector(rhs_form)
    fem.apply_lifting(vector.array, [a_form], [bcs])
    vector.scatter_reverse(la.InsertMode.add)
    _set_dirichlet_bc_values(fem, vector.array, bcs)
    sparse_matrix = getattr(matrix.to_scipy(), f"to{sparse_matrix_format}")().copy()
    sparse_matrix = _copy_sparse_matrix_with_value_dtype(sparse_matrix, value_dtype)
    rhs_array = np.ascontiguousarray(vector.array.copy(), dtype=value_dtype)
    solution_array = np.asarray(solve_linear_system(sparse_matrix, rhs_array))
    solve_seconds = perf_counter() - start

    solution = fem.Function(mixed_space)
    if solution_array.size != solution.x.array.size:
        raise RuntimeError(
            "Serial FEM solve returned a solution vector with incompatible size "
            f"{solution_array.size}; expected {solution.x.array.size}."
        )
    solution.x.array[:] = solution_array.real
    solution.x.scatter_forward()
    return (
        solution,
        solve_seconds,
        {
            "serial_sparse_linear_system_dtype": resolved_linear_system_dtype,
            "serial_sparse_matrix_value_dtype": str(
                getattr(sparse_matrix, "dtype", getattr(value_dtype, "name", value_dtype))
            ),
            "serial_sparse_rhs_dtype": str(rhs_array.dtype),
            "serial_sparse_matrix_nnz": int(sparse_matrix.nnz),
            "serial_sparse_matrix_format": sparse_matrix_format,
            "serial_sparse_solver_backend": serial_solver_backend,
            **serial_solver_metadata,
        },
    )


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
    velocity_scale: float = 1.0,
    pressure_scale: float = 1.0,
) -> FEMSinglePhaseResult:
    velocity, pressure = _collapse_solution(solution)
    if velocity_scale != 1.0:
        velocity.x.array[:] *= float(velocity_scale)
        velocity.x.scatter_forward()
    if pressure_scale != 1.0:
        pressure.x.array[:] *= float(pressure_scale)
        pressure.x.scatter_forward()
    return _result_from_velocity_pressure(
        context,
        velocity,
        pressure,
        method=method,
        formulation=formulation,
        flow_axis=flow_axis,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
        viscosity=viscosity,
        solve_seconds=solve_seconds,
        metadata=metadata,
    )


def _result_from_velocity_pressure(
    context: _FEMContext,
    velocity: Any,
    pressure: Any,
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
    velocity.name = "velocity"
    pressure.name = "pressure"
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
    boundary_pressure_inlet: float | None = None,
    boundary_pressure_outlet: float | None = None,
    velocity_scale: float = 1.0,
    pressure_scale: float = 1.0,
    extra_metadata: Mapping[str, Any] | None = None,
) -> FEMSinglePhaseResult:
    _validate_pressure_drop(pressure_inlet, pressure_outlet)
    solver_options = options or FEniCSSolverOptions()
    requested_linear_system_dtype = _resolve_fem_linear_system_dtype(
        solver_options.linear_system_dtype
    )
    api = _require_dolfinx_core()
    selected_linear_backend = _resolve_linear_backend(solver_options.linear_backend, api)
    if selected_linear_backend == "petsc" and requested_linear_system_dtype != "float64":
        raise ValueError(
            "linear_system_dtype='float32' is available for serial sparse FEM "
            "backends such as 'superlu'/'scipy' and for 'nvmath_cudss'. The PETSc "
            "backend uses the scalar type of the installed PETSc/DOLFINx build."
        )
    if selected_linear_backend == "petsc":
        api = _require_dolfinx_petsc(api)
    context = _build_context(problem, flow_axis=flow_axis, api=api)
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
        pressure_inlet=pressure_inlet
        if boundary_pressure_inlet is None
        else boundary_pressure_inlet,
        pressure_outlet=pressure_outlet
        if boundary_pressure_outlet is None
        else boundary_pressure_outlet,
    )
    bcs = _side_wall_bcs(context, W, flow_axis=flow_axis)
    bcs.append(_pressure_gauge_bc(context, W))
    if selected_linear_backend == "petsc":
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
            mixed_space=W,
            form=form,
            rhs=rhs,
            bcs=bcs,
            linear_backend=cast(
                Literal["scipy", "superlu", "umfpack", "pardiso", "nvmath_cudss"],
                selected_linear_backend,
            ),
            linear_system_dtype=requested_linear_system_dtype,
            superlu_controls=solver_options.superlu_controls,
            umfpack_controls=solver_options.umfpack_controls,
            nvmath_cudss_controls=solver_options.nvmath_cudss_controls,
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
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        metadata={
            "linear_backend": selected_linear_backend,
            "solver_preset": solver_options.solver_preset,
            "linear_system_dtype": requested_linear_system_dtype,
            "velocity_degree": velocity_degree,
            "pressure_family": pressure_family,
            "porosity_floor": problem.porosity_floor,
            "permeability_floor": problem.permeability_floor,
            "petsc_options": dict(solver_options.petsc_options),
            "petsc_options_prefix": solver_options.petsc_options_prefix,
            "superlu_controls": dict(solver_options.superlu_controls),
            "umfpack_controls": dict(solver_options.umfpack_controls),
            "nvmath_cudss_controls": dict(solver_options.nvmath_cudss_controls),
            "thread_environment": _thread_environment_metadata(),
            **dict(extra_metadata or {}),
            **_mpi_metadata(context),
            **solver_metadata,
        },
    )


__all__ = [
    "FEMSolverPreset",
    "BrinkmanNondimensionalization",
    "FEMMapProblem",
    "FEMSinglePhaseResult",
    "FEniCSSolverOptions",
    "LinearSolverBackend",
    "LinearSystemDType",
    "BrinkmanVelocityScale",
    "_brinkman_nondimensional_coefficients",
    "_brinkman_nondimensional_scales",
    "_constant_permeability_value",
    "_nondimensional_metadata",
    "_resolve_brinkman_nondimensionalization",
    "_build_context",
    "_solve_with_form_builder",
]
