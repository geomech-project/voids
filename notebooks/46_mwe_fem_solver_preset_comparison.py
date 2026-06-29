# %% [markdown]
# # MWE 46 - FEM solver preset comparison
#
# This notebook compares finite-element micro-continuum solver presets on small
# synthetic coefficient maps before trying them in expensive same-ROI DRP-317
# notebooks.
#
# The goal is deliberately narrow:
#
# - keep cases small enough for quick local checks;
# - run solver presets as single-process jobs, so each backend can use its own
#   thread-level resources;
# - compare permeability, solve time, wall time, and solver metadata;
# - separate stable direct-reference runs from experimental iterative runs.
#
# PETSc presets are kept as explicit opt-in stress tests. The default
# comparison below uses single-process direct backends that are either portable
# (SuperLU/UMFPACK) or intentionally Linux-specific for performance (PARDISO).

# %%
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from ctypes import CDLL, c_int
from ctypes.util import find_library
from typing import Callable


def available_thread_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(len(os.sched_getaffinity(0)), 1)
    return max(os.cpu_count() or 1, 1)


detected_thread_count = available_thread_count()
# Use up to 32 threads by default. This keeps the notebook portable on smaller
# machines while avoiding oversubscription-heavy defaults on large workstations.
linear_solver_thread_count = min(32, detected_thread_count)
thread_environment_policy = {
    "OMP_NUM_THREADS": str(linear_solver_thread_count),
    "OPENBLAS_NUM_THREADS": str(linear_solver_thread_count),
    "VECLIB_MAXIMUM_THREADS": str(linear_solver_thread_count),
    "NUMEXPR_NUM_THREADS": str(linear_solver_thread_count),
    "MKL_NUM_THREADS": str(linear_solver_thread_count),
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
for name, value in thread_environment_policy.items():
    os.environ[name] = value


def configure_mkl_threads(thread_count: int) -> dict[str, object]:
    """Best-effort runtime MKL thread configuration for PARDISO."""

    library_path = find_library("mkl_rt")
    if not library_path:
        return {"mkl_runtime": "not found", "mkl_runtime_threads": ""}
    try:
        mkl_rt = CDLL(library_path)
        set_num_threads = mkl_rt.MKL_Set_Num_Threads
        set_num_threads.argtypes = [c_int]
        set_num_threads.restype = None
        set_num_threads(c_int(thread_count))
        get_max_threads = mkl_rt.MKL_Get_Max_Threads
        get_max_threads.restype = c_int
        return {
            "mkl_runtime": library_path,
            "mkl_runtime_threads": int(get_max_threads()),
        }
    except Exception as exc:
        return {
            "mkl_runtime": library_path,
            "mkl_runtime_threads": "",
            "mkl_runtime_error": f"{type(exc).__name__}: {exc}",
        }


mkl_thread_configuration = configure_mkl_threads(linear_solver_thread_count)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook convenience fallback
    display = print

from voids.fem.singlephase import (
    FEMMapProblem,
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    solve_brinkman_taylor_hood,
    solve_brinkman_usfem,
    solve_darcy_taylor_hood,
)
from voids.fvm.singlephase import solve_tpfa
from voids.image.porosity import (
    PermeabilityMap,
    PorosityMap,
    permeability_map_from_porosity,
    porosity_map_from_binary,
)
from voids.paths import project_root

plt.ioff()

# %%
# User-editable inputs
output_dir = (
    project_root() / "notebooks" / "outputs" / "46_mwe_fem_solver_preset_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_prefix = "fem_solver_preset_comparison_block10"

mu_pa_s = 1.0
pressure_inlet = 1.0
pressure_outlet = 0.0
flow_axis = "x"

# Keep this small by default. Increase only after the preset behavior is clear.
map_shape = (3, 3)
cell_size = 1.0
reference_permeability = 2.0

run_solvers = True
run_small_2d_cases = False
include_single_process_superlu_direct = True
include_single_process_umfpack_direct = True
include_single_process_pardiso_direct = True
include_petsc_direct_rows = False
include_historical_petsc_comparison_rows = False
include_block_preconditioner_probe_summary = False
# The experimental field-split preset is opt-in because some PETSc/DOLFINx
# stacks can emit cleanup-time PETSc errors after failed Schur-complement
# preconditioner setup. Enable this only when actively developing that preset.
include_experimental_iterative = False
# The default run is USFEM-only. Enable Taylor-Hood rows explicitly when
# comparing formulations rather than searching for a stable USFEM setup.
include_taylor_hood_formulations = False
include_usfem_formulation = True

# Small 3-D constant case for portable direct-solver sanity checks.
run_optional_3d = True
optional_3d_shape = (2, 2, 2)

# Image-derived 3-D case coarsened with the same block_shape convention used by
# the synthetic porosity-map notebook. The default is 200^3 voxels with
# block_shape=(10, 10, 10), producing a 20^3 coefficient map. Use
# image_3d_shape=(300, 300, 300) with the same block_shape for a 30^3 map.
run_image_3d_case = True
image_3d_shape = (200, 200, 200)
image_3d_target_porosity = 0.35
image_3d_blobiness = 1
image_3d_seed = 2026
image_3d_voxel_size_m = 40.0e-6
image_3d_block_shape = (10, 10, 10)
include_umfpack_direct_on_image_3d = False
# UMFPACK is still exercised on the small 3-D sanity case. On the 20^3 image
# map, the default UMFPACK strategy exceeded a 30-minute guarded run. The
# unsymmetric strategy finished in about 7 minutes and matched PARDISO on this
# workstation, so keep the image row opt-in but use that safer tuned control.
umfpack_direct_controls: dict[str, object] = {"strategy": "unsymmetric"}
# SuperLU is kept as the last-resort fallback on the image case because it did
# not finish a 30^3 heterogeneous map within a 40-minute guarded run.
include_superlu_direct_on_image_3d = False

run_tpfa_baseline = True
tpfa_solver_method = "cg"
tpfa_solver_parameters: dict[str, object] = {
    "rtol": 1.0e-10,
    "atol": 0.0,
    "maxiter": 2000,
    "preconditioner": "pyamg",
    "pyamg_solver": "smoothed_aggregation",
}

# Optional PARDISO thread sweep. This runs each thread count in a fresh Python
# subprocess because MKL reads thread controls when it is loaded. Keep disabled
# during ordinary notebook runs; enable when calibrating a machine.
run_pardiso_thread_sweep = False
pardiso_thread_sweep_image_shape = (180, 180, 180)
pardiso_thread_sweep_block_shape = (9, 9, 9)
pardiso_thread_candidates = sorted({1, 2, 4, 8, 16, 32, detected_thread_count})

# Optional UMFPACK tuning sweep. This probes portable direct-solver controls on
# a map10 case and compares each row against a same-map PARDISO reference.
run_umfpack_tuning_sweep = False
umfpack_tuning_image_shape = (100, 100, 100)
umfpack_tuning_block_shape = (10, 10, 10)
umfpack_tuning_thread_candidates = (1, 8, 16, 32)
umfpack_tuning_control_sets = (
    {"name": "default", "controls": {}},
    {"name": "unsymmetric", "controls": {"strategy": "unsymmetric"}},
    {"name": "metis_guard", "controls": {"ordering": "metis_guard"}},
    {"name": "best", "controls": {"ordering": "best"}},
    {
        "name": "metis_guard_pivot_0p01",
        "controls": {"ordering": "metis_guard", "pivot_tolerance": 1.0e-2},
    },
    {
        "name": "amd_pivot_0p001",
        "controls": {"ordering": "amd", "pivot_tolerance": 1.0e-3},
    },
)
umfpack_tuning_row_timeout_s = 600

display(
    pd.DataFrame(
        [
            {
                "detected_threads": detected_thread_count,
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                "MKL_DYNAMIC": os.environ.get("MKL_DYNAMIC"),
                "OMP_DYNAMIC": os.environ.get("OMP_DYNAMIC"),
                **mkl_thread_configuration,
            }
        ]
    )
)
print(
    "Single-process direct solver rows are non-MPI runs. They may still use "
    "threaded BLAS/MKL kernels; the thread environment above is part of the "
    "benchmark provenance.",
    flush=True,
)

dolfinx_superlu_dist_available = (
    importlib.util.find_spec("dolfinx.la.superlu_dist") is not None
)
display(
    pd.DataFrame(
        [
            {
                "capability": "dolfinx.la.superlu_dist",
                "available": dolfinx_superlu_dist_available,
                "role": (
                    "parallel SuperLU_DIST MatrixCSR path"
                    if dolfinx_superlu_dist_available
                    else "unavailable in this DOLFINx build"
                ),
            }
        ]
    )
)


# %%
def constant_problem(shape: tuple[int, ...], permeability: float) -> FEMMapProblem:
    return FEMMapProblem(
        permeability_map=PermeabilityMap(
            np.full(shape, permeability, dtype=float),
            cell_size=cell_size,
        ),
        porosity_map=PorosityMap(np.ones(shape, dtype=float), cell_size=cell_size),
        viscosity=mu_pa_s,
    )


def mildly_heterogeneous_problem(
    shape: tuple[int, ...], permeability: float
) -> FEMMapProblem:
    grid = np.indices(shape, dtype=float)
    normalized = sum(
        (axis_values + 1.0) / float(size) for axis_values, size in zip(grid, shape)
    )
    normalized = normalized / float(len(shape))
    values = permeability * (0.75 + 0.5 * normalized)
    porosity = np.clip(0.2 + 0.1 * normalized, 1.0e-3, 1.0)
    return FEMMapProblem(
        permeability_map=PermeabilityMap(values, cell_size=cell_size),
        porosity_map=PorosityMap(porosity, cell_size=cell_size),
        viscosity=mu_pa_s,
    )


def synthetic_blobs_problem_3d() -> tuple[FEMMapProblem, dict[str, object]]:
    if any(
        image_size % block_size != 0
        for image_size, block_size in zip(
            image_3d_shape, image_3d_block_shape, strict=True
        )
    ):
        raise ValueError("image_3d_shape must be divisible by image_3d_block_shape")
    map_shape_3d = tuple(
        image_size // block_size
        for image_size, block_size in zip(
            image_3d_shape, image_3d_block_shape, strict=True
        )
    )
    case_name = f"synthetic_{image_3d_shape[0]}_block{image_3d_block_shape[0]}_map{map_shape_3d[0]}_3d"

    start = time.perf_counter()
    binary_void = ps.generators.blobs(
        shape=list(image_3d_shape),
        porosity=image_3d_target_porosity,
        blobiness=image_3d_blobiness,
        seed=image_3d_seed,
    ).astype(bool)
    binary_seconds = time.perf_counter() - start

    start = time.perf_counter()
    porosity_map = porosity_map_from_binary(
        binary_void,
        block_shape=image_3d_block_shape,
        voxel_size=image_3d_voxel_size_m,
        metadata={
            "case": case_name,
            "generator": "porespy.generators.blobs",
            "image_shape": image_3d_shape,
            "map_shape": map_shape_3d,
            "block_shape": image_3d_block_shape,
            "seed": image_3d_seed,
            "target_porosity": image_3d_target_porosity,
        },
    )
    permeability_map = permeability_map_from_porosity(
        porosity_map,
        characteristic_length=min(porosity_map.cell_size),
        kozeny_constant=180.0,
        solid_permeability=1.0e-20,
        max_permeability=1.0e-8,
    )
    map_seconds = time.perf_counter() - start

    problem = FEMMapProblem(
        permeability_map=permeability_map,
        porosity_map=porosity_map,
        viscosity=mu_pa_s,
        porosity_floor=1.0e-3,
        permeability_floor=1.0e-20,
    )
    metadata = {
        "case": case_name,
        "image_shape": image_3d_shape,
        "map_shape": porosity_map.shape,
        "block_shape": image_3d_block_shape,
        "binary_seconds": binary_seconds,
        "map_seconds": map_seconds,
        "binary_porosity": float(binary_void.mean()),
        "mean_porosity": porosity_map.mean_porosity,
        "cell_size_m": porosity_map.cell_size,
        "k_min_m2": float(np.min(permeability_map.values)),
        "k_mean_m2": float(np.mean(permeability_map.values)),
        "k_max_m2": float(np.max(permeability_map.values)),
    }
    return problem, metadata


FormulationSolver = Callable[..., FEMSinglePhaseResult]
DARCY_TAYLOR_HOOD = "darcy_taylor_hood_p2p1"
BRINKMAN_TAYLOR_HOOD = "brinkman_taylor_hood_p2p1"
BRINKMAN_USFEM = "brinkman_usfem_p1dg1"


all_formulations: list[dict[str, object]] = [
    {
        "formulation": DARCY_TAYLOR_HOOD,
        "label": "Darcy TH",
        "solver": solve_darcy_taylor_hood,
        "enabled": include_taylor_hood_formulations,
    },
    {
        "formulation": BRINKMAN_TAYLOR_HOOD,
        "label": "Brinkman TH",
        "solver": solve_brinkman_taylor_hood,
        "enabled": include_taylor_hood_formulations,
    },
    {
        "formulation": BRINKMAN_USFEM,
        "label": "USFEM",
        "solver": solve_brinkman_usfem,
        "enabled": include_usfem_formulation,
    },
]
formulations = [
    formulation for formulation in all_formulations if formulation["enabled"]
]


single_process_formulations = tuple(
    formulation["formulation"] for formulation in all_formulations
)

solver_specs: list[dict[str, object]] = []
if include_petsc_direct_rows:
    solver_specs.extend(
        [
            {
                "preset": "direct_reference_mumps",
                "label": "direct ref MUMPS",
                "options": FEniCSSolverOptions.direct_reference(
                    "mumps",
                    petsc_options_prefix="voids_fem_direct_reference_mumps_",
                ),
                "expected": "petsc_stress_test",
                "formulations": single_process_formulations,
            },
            {
                "preset": "direct_parallel_mumps",
                "label": "direct parallel MUMPS",
                "options": FEniCSSolverOptions.direct_parallel(
                    "mumps",
                    petsc_options_prefix="voids_fem_direct_parallel_mumps_",
                ),
                "expected": "petsc_stress_test",
                "formulations": single_process_formulations,
            },
            {
                "preset": "direct_parallel_superlu_dist",
                "label": "direct parallel SuperLU_DIST",
                "options": FEniCSSolverOptions.direct_parallel(
                    "superlu_dist",
                    petsc_options_prefix="voids_fem_direct_parallel_superlu_dist_",
                ),
                "expected": "petsc_stress_test",
                "formulations": single_process_formulations,
            },
        ]
    )
if include_single_process_pardiso_direct:
    solver_specs.append(
        {
            "preset": "single_process_pardiso_direct",
            "label": "single-process PARDISO",
            "options": FEniCSSolverOptions.pardiso_direct(),
            "expected": "linux_performance_direct",
            "formulations": single_process_formulations,
        }
    )
if include_single_process_umfpack_direct:
    solver_specs.append(
        {
            "preset": "single_process_umfpack_direct",
            "label": "single-process UMFPACK tuned",
            "options": FEniCSSolverOptions.umfpack_direct(
                controls=umfpack_direct_controls
            ),
            "expected": "portable_single_process_direct",
            "formulations": single_process_formulations,
        }
    )
if include_single_process_superlu_direct:
    solver_specs.append(
        {
            "preset": "single_process_superlu_direct",
            "label": "single-process SuperLU",
            "options": FEniCSSolverOptions.superlu_direct(),
            "expected": "portable_single_process_fallback",
            "formulations": single_process_formulations,
        }
    )
if include_experimental_iterative:
    solver_specs.append(
        {
            "preset": "iterative_fieldsplit_experimental",
            "label": "fieldsplit experimental",
            "options": FEniCSSolverOptions.iterative_fieldsplit_experimental(
                petsc_options_prefix="voids_fem_iterative_fieldsplit_experimental_",
                rtol=1.0e-8,
                max_it=300,
            ),
            "expected": "experimental",
            "formulations": tuple(
                formulation["formulation"] for formulation in all_formulations
            ),
        }
    )


def solver_spec_applies_to_formulation(
    solver_spec: dict[str, object],
    formulation: dict[str, object],
) -> bool:
    allowed = solver_spec.get("formulations")
    if allowed is None:
        return True
    return formulation["formulation"] in allowed


display(
    pd.DataFrame(
        [
            {
                "preset": spec["preset"],
                "label": spec["label"],
                "expected": spec["expected"],
                "formulations": ", ".join(
                    str(item) for item in spec.get("formulations", ())
                ),
                "linear_backend": spec["options"].linear_backend,
                "solver_preset": spec["options"].solver_preset,
                "petsc_options": json.dumps(
                    spec["options"].petsc_options, sort_keys=True
                ),
            }
            for spec in solver_specs
        ]
    )
)

if not include_petsc_direct_rows:
    print(
        "PETSc direct rows are disabled by default. Notebook 46 compares "
        "single-process PARDISO, single-process UMFPACK, and fallback "
        "single-process SuperLU unless "
        "include_petsc_direct_rows=True is set for a deliberate stress test.",
        flush=True,
    )
if run_image_3d_case and not include_umfpack_direct_on_image_3d:
    print(
        "The heterogeneous image case runs PARDISO by default. UMFPACK is "
        "available as a portable direct backend and is exercised on the small "
        "3-D sanity case, but its image-case row is opt-in because the 20^3 "
        "stress row remains much slower than PARDISO.",
        flush=True,
    )
if run_image_3d_case and not include_superlu_direct_on_image_3d:
    print(
        "SuperLU remains the last fallback and is disabled on that case unless "
        "include_superlu_direct_on_image_3d=True.",
        flush=True,
    )


# %% [markdown]
# ## Run Quick Synthetic Cases
#
# The constant-permeability case has a known reference permeability. The mildly
# heterogeneous case is included for solver robustness and timing diagnostics,
# but it does not have an analytic effective permeability target in this
# notebook.

# %%
def run_one_case(
    *,
    case_name: str,
    problem: FEMMapProblem,
    expected_k: float | None,
    formulation: dict[str, object],
    solver_spec: dict[str, object],
) -> dict[str, object]:
    solver = formulation["solver"]
    if not callable(solver):
        raise TypeError("formulation solver must be callable")
    options = solver_spec["options"]
    if not isinstance(options, FEniCSSolverOptions):
        raise TypeError("solver options must be FEniCSSolverOptions")

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = solver(
                problem,
                flow_axis=flow_axis,
                pressure_inlet=pressure_inlet,
                pressure_outlet=pressure_outlet,
                options=options,
            )
        except Exception as exc:
            return {
                "case": case_name,
                "formulation": formulation["formulation"],
                "formulation_label": formulation["label"],
                "preset": solver_spec["preset"],
                "preset_label": solver_spec["label"],
                "expected": solver_spec["expected"],
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.perf_counter() - start,
                "warning_count": len(caught),
                "warnings": "; ".join(str(item.message) for item in caught),
            }

    wall_seconds = time.perf_counter() - start
    expected_value = np.nan if expected_k is None else float(expected_k)
    relative_error = (
        np.nan
        if expected_k is None
        else abs(float(result.permeability) - float(expected_k))
        / max(abs(float(expected_k)), 1.0e-300)
    )
    metadata = dict(result.metadata)
    return {
        "case": case_name,
        "formulation": formulation["formulation"],
        "formulation_label": formulation["label"],
        "preset": solver_spec["preset"],
        "preset_label": solver_spec["label"],
        "expected": solver_spec["expected"],
        "status": "ok",
        "failure": "",
        "K_m2": float(result.permeability),
        "K_expected_m2": expected_value,
        "K_relative_error": relative_error,
        "flow_rate": float(result.flow_rate),
        "solve_seconds": float(result.solve_seconds),
        "wall_seconds": wall_seconds,
        "linear_backend": metadata.get("linear_backend", ""),
        "solver_preset": metadata.get("solver_preset", ""),
        "mpi_size": metadata.get("mpi_size", ""),
        "mpi_rank": metadata.get("mpi_rank", ""),
        "petsc_ksp_type": metadata.get("petsc_ksp_type", ""),
        "petsc_converged_reason": metadata.get("petsc_converged_reason", ""),
        "petsc_iteration_number": metadata.get("petsc_iteration_number", ""),
        "petsc_residual_norm": metadata.get("petsc_residual_norm", ""),
        "serial_sparse_matrix_nnz": metadata.get("serial_sparse_matrix_nnz", ""),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
        "warning_count": len(caught),
        "warnings": "; ".join(str(item.message) for item in caught),
    }


case_specs: list[dict[str, object]] = []
if run_small_2d_cases:
    case_specs.extend(
        [
            {
                "case": f"constant_{len(map_shape)}d",
                "problem": constant_problem(map_shape, reference_permeability),
                "expected_k": reference_permeability,
            },
            {
                "case": f"mildly_heterogeneous_{len(map_shape)}d",
                "problem": mildly_heterogeneous_problem(
                    map_shape, reference_permeability
                ),
                "expected_k": None,
            },
        ]
    )
if run_optional_3d:
    case_specs.append(
        {
            "case": "constant_3d",
            "problem": constant_problem(optional_3d_shape, reference_permeability),
            "expected_k": reference_permeability,
        }
    )
map_generation_rows: list[dict[str, object]] = []
if run_image_3d_case:
    image_3d_problem, image_3d_metadata = synthetic_blobs_problem_3d()
    map_generation_rows.append(image_3d_metadata)
    image_3d_solver_presets = {"single_process_pardiso_direct"}
    if include_umfpack_direct_on_image_3d:
        image_3d_solver_presets.add("single_process_umfpack_direct")
    if include_superlu_direct_on_image_3d:
        image_3d_solver_presets.add("single_process_superlu_direct")
    case_specs.append(
        {
            "case": str(image_3d_metadata["case"]),
            "problem": image_3d_problem,
            "expected_k": None,
            "solver_presets": image_3d_solver_presets,
        }
    )

map_generation = pd.DataFrame(map_generation_rows)
map_generation_path = output_dir / f"{output_prefix}_map_generation.csv"
map_generation.to_csv(map_generation_path, index=False)
display(map_generation)
print(f"Saved map-generation metadata: {map_generation_path}")


def run_tpfa_baselines(case_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in case_specs:
        problem = case["problem"]
        if (
            not isinstance(problem, FEMMapProblem)
            or len(problem.permeability_map.shape) != 3
        ):
            continue
        for axis in ("x", "y", "z"):
            start = time.perf_counter()
            try:
                result = solve_tpfa(
                    problem.permeability_map,
                    viscosity=problem.viscosity,
                    flow_axis=axis,
                    pressure_inlet=pressure_inlet,
                    pressure_outlet=pressure_outlet,
                    solver_method=tpfa_solver_method,
                    solver_parameters=tpfa_solver_parameters,
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case["case"],
                        "axis": axis,
                        "status": "failed",
                        "failure": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": time.perf_counter() - start,
                    }
                )
            else:
                rows.append(
                    {
                        "case": case["case"],
                        "axis": axis,
                        "status": "ok",
                        "failure": "",
                        "K_m2": float(result.permeability),
                        "flow_rate": float(result.flow_rate),
                        "solve_seconds": float(result.solve_seconds),
                        "wall_seconds": time.perf_counter() - start,
                        "mass_balance_error": float(result.mass_balance_error),
                        "residual_relative": float(result.residual_relative),
                        "matrix_nnz": result.matrix_nnz,
                        "solver": result.solver_method,
                        "solver_info": json.dumps(result.solver_info, sort_keys=True),
                        "solver_parameters": json.dumps(
                            tpfa_solver_parameters, sort_keys=True
                        ),
                    }
                )
    return pd.DataFrame(rows)


tpfa_results = run_tpfa_baselines(case_specs) if run_tpfa_baseline else pd.DataFrame()
tpfa_results_path = output_dir / f"{output_prefix}_tpfa_baseline.csv"
tpfa_results.to_csv(tpfa_results_path, index=False)
display(tpfa_results)
print(f"Saved TPFA baseline: {tpfa_results_path}")

rows: list[dict[str, object]] = []
results_path = output_dir / f"{output_prefix}_results.csv"
if run_solvers:
    for case in case_specs:
        for formulation in formulations:
            for solver_spec in solver_specs:
                if not solver_spec_applies_to_formulation(solver_spec, formulation):
                    continue
                allowed_solver_presets = case.get("solver_presets")
                if (
                    allowed_solver_presets is not None
                    and solver_spec["preset"] not in allowed_solver_presets
                ):
                    continue
                row = run_one_case(
                    case_name=str(case["case"]),
                    problem=case["problem"],
                    expected_k=case["expected_k"],
                    formulation=formulation,
                    solver_spec=solver_spec,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                print(
                    "Saved FEM row: "
                    f"{row['case']} | {row['formulation_label']} | "
                    f"{row['preset_label']} | {row['status']}",
                    flush=True,
                )

results = pd.DataFrame(rows)
results.to_csv(results_path, index=False)
display(results)
print(f"Saved results: {results_path}")

# %%
ok = results[results["status"] == "ok"].copy() if not results.empty else pd.DataFrame()
failures = (
    results[results["status"] != "ok"].copy()
    if not results.empty
    else pd.DataFrame(columns=["case", "formulation", "preset", "failure"])
)
display(failures)

summary = (
    ok.groupby(["case", "formulation_label", "preset_label"], dropna=False)
    .agg(
        K_m2=("K_m2", "first"),
        K_relative_error=("K_relative_error", "first"),
        solve_seconds=("solve_seconds", "first"),
        wall_seconds=("wall_seconds", "first"),
        mpi_size=("mpi_size", "first"),
        petsc_iteration_number=("petsc_iteration_number", "first"),
        petsc_residual_norm=("petsc_residual_norm", "first"),
    )
    .reset_index()
    if not ok.empty
    else pd.DataFrame()
)
summary_path = output_dir / f"{output_prefix}_summary.csv"
summary.to_csv(summary_path, index=False)
display(summary)
print(f"Saved summary: {summary_path}")

# %% [markdown]
# ## Plots
#
# Plotting focuses on successful rows. Failed rows remain in the CSV and table
# above so solver/package availability problems are visible instead of silently
# hidden.

# %%
if ok.empty:
    print("No successful FEM rows to plot.")
else:
    preferred_plot_case = (
        f"synthetic_{image_3d_shape[0]}_block{image_3d_block_shape[0]}_map"
        f"{image_3d_shape[0] // image_3d_block_shape[0]}_3d"
    )
    plot_case = (
        preferred_plot_case
        if preferred_plot_case in set(ok["case"])
        else str(ok["case"].iloc[0])
    )
    plot_df = ok[ok["case"] == plot_case].copy()
    plot_df["case_formulation"] = (
        plot_df["formulation_label"].astype(str)
        + "\n"
        + plot_df["preset_label"].astype(str)
    )
    x = np.arange(len(plot_df), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), constrained_layout=True)
    axes[0].bar(x, plot_df["wall_seconds"], color="tab:blue", alpha=0.75, label="wall")
    axes[0].bar(
        x,
        plot_df["solve_seconds"],
        color="tab:orange",
        alpha=0.75,
        label="reported solve",
    )
    axes[0].set_ylabel("seconds")
    axes[0].set_title(f"FEM solver preset timing: {plot_case}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["case_formulation"], rotation=70, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    if plot_df["K_relative_error"].notna().any():
        axes[1].bar(x, plot_df["K_relative_error"], color="tab:green", alpha=0.75)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("relative K error")
    else:
        axes[1].bar(x, plot_df["K_m2"], color="tab:green", alpha=0.75)
        axes[1].set_ylabel("permeability [m2]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["case_formulation"], rotation=70, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    timing_plot_path = output_dir / f"{output_prefix}_{plot_case}_timing_error.png"
    fig.savefig(timing_plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved timing/error plot: {timing_plot_path}")

# %%
if ok.empty or not ok["K_relative_error"].notna().any():
    print("No analytic-reference rows available for error-versus-time plot.")
else:
    reference_rows = ok[ok["K_relative_error"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for formulation_label, subset in reference_rows.groupby("formulation_label"):
        ax.scatter(
            subset["wall_seconds"],
            subset["K_relative_error"],
            s=70,
            label=str(formulation_label),
        )
        for row in subset.itertuples(index=False):
            ax.annotate(
                str(row.preset_label),
                (float(row.wall_seconds), float(row.K_relative_error)),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel("relative K error")
    ax.set_title("FEM preset accuracy versus runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    scatter_plot_path = output_dir / f"{output_prefix}_accuracy_vs_runtime.png"
    fig.savefig(scatter_plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved accuracy/runtime plot: {scatter_plot_path}")

# %%
if ok.empty:
    print("No successful rows available for PETSc diagnostics plot.")
else:
    diag_df = ok.copy()
    diag_df["petsc_iteration_number_numeric"] = pd.to_numeric(
        diag_df["petsc_iteration_number"],
        errors="coerce",
    )
    diag_df = diag_df[diag_df["petsc_iteration_number_numeric"].notna()]
    if diag_df.empty:
        print(
            "No PETSc iteration diagnostics were exposed by this DOLFINx/PETSc stack."
        )
    else:
        diag_df["label"] = (
            diag_df["formulation_label"].astype(str)
            + "\n"
            + diag_df["preset_label"].astype(str)
        )
        fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
        ax.bar(
            np.arange(len(diag_df), dtype=float),
            diag_df["petsc_iteration_number_numeric"],
            color="tab:purple",
            alpha=0.75,
        )
        ax.set_xticks(np.arange(len(diag_df), dtype=float))
        ax.set_xticklabels(diag_df["label"], rotation=70, ha="right")
        ax.set_ylabel("PETSc KSP iterations")
        ax.set_title("PETSc iteration diagnostics")
        ax.grid(axis="y", alpha=0.25)
        diagnostics_plot_path = output_dir / f"{output_prefix}_petsc_iterations.png"
        fig.savefig(diagnostics_plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved PETSc diagnostics plot: {diagnostics_plot_path}")

# %% [markdown]
# ## PARDISO Thread Sweep
#
# PARDISO thread controls must be set before MKL is loaded. The optional sweep
# below therefore launches one fresh Python subprocess per thread count. The
# default notebook run only reads and plots existing sweep CSVs; set
# `run_pardiso_thread_sweep = True` above to recalibrate this machine.

# %%
thread_sweep_map_shape = tuple(
    image_size // block_size
    for image_size, block_size in zip(
        pardiso_thread_sweep_image_shape,
        pardiso_thread_sweep_block_shape,
        strict=True,
    )
)
thread_sweep_output_prefix = (
    f"usfem_pardiso_thread_sweep_map{thread_sweep_map_shape[0]}"
)
thread_sweep_path = output_dir / f"{thread_sweep_output_prefix}.csv"
thread_sweep_row_dir = output_dir / f"{thread_sweep_output_prefix}_rows"
thread_sweep_row_dir.mkdir(parents=True, exist_ok=True)


def run_pardiso_thread_sweep_row(thread_count: int) -> dict[str, object]:
    row_path = thread_sweep_row_dir / f"threads_{thread_count}.csv"
    config = {
        "thread_count": int(thread_count),
        "row_path": str(row_path),
        "image_shape": pardiso_thread_sweep_image_shape,
        "block_shape": pardiso_thread_sweep_block_shape,
        "target_porosity": image_3d_target_porosity,
        "blobiness": image_3d_blobiness,
        "seed": image_3d_seed,
        "voxel_size_m": image_3d_voxel_size_m,
    }
    child_code = r"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import porespy as ps

from voids.fem.singlephase import FEMMapProblem, FEniCSSolverOptions, solve_brinkman_usfem
from voids.image.porosity import porosity_map_from_binary, permeability_map_from_porosity

cfg = json.loads(os.environ["VOIDS_FEM_PARDISO_THREAD_SWEEP_CONFIG"])
thread_count = int(cfg["thread_count"])
row = {
    "threads": thread_count,
    "image_shape": tuple(cfg["image_shape"]),
    "block_shape": tuple(cfg["block_shape"]),
}
try:
    start = time.perf_counter()
    binary_void = ps.generators.blobs(
        shape=list(cfg["image_shape"]),
        porosity=float(cfg["target_porosity"]),
        blobiness=float(cfg["blobiness"]),
        seed=int(cfg["seed"]),
    ).astype(bool)
    row["binary_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    porosity_map = porosity_map_from_binary(
        binary_void,
        block_shape=tuple(cfg["block_shape"]),
        voxel_size=float(cfg["voxel_size_m"]),
    )
    permeability_map = permeability_map_from_porosity(
        porosity_map,
        characteristic_length=min(porosity_map.cell_size),
        kozeny_constant=180.0,
        solid_permeability=1.0e-20,
        max_permeability=1.0e-8,
    )
    row["map_seconds"] = time.perf_counter() - start

    problem = FEMMapProblem(
        permeability_map,
        porosity_map,
        viscosity=1.0,
        porosity_floor=1.0e-3,
        permeability_floor=1.0e-20,
    )
    start = time.perf_counter()
    result = solve_brinkman_usfem(
        problem,
        flow_axis="x",
        pressure_inlet=1.0,
        pressure_outlet=0.0,
        options=FEniCSSolverOptions.pardiso_direct(),
    )
    row.update(
        {
            "status": "ok",
            "K_m2": result.permeability,
            "flow_rate": result.flow_rate,
            "solve_seconds": result.solve_seconds,
            "wall_seconds": time.perf_counter() - start,
            "serial_sparse_matrix_nnz": result.metadata.get("serial_sparse_matrix_nnz"),
            "serial_sparse_solver_backend": result.metadata.get("serial_sparse_solver_backend"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        }
    )
except Exception as exc:
    row.update(
        {
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
        }
    )

pd.DataFrame([row]).to_csv(cfg["row_path"], index=False)
print(pd.DataFrame([row]).to_string(index=False), flush=True)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(thread_count),
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MKL_NUM_THREADS": str(thread_count),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "VOIDS_FEM_PARDISO_THREAD_SWEEP_CONFIG": json.dumps(config),
        }
    )
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=project_root(),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    subprocess_seconds = time.perf_counter() - start
    if row_path.exists():
        row = pd.read_csv(row_path).iloc[0].to_dict()
    else:
        row = {"threads": thread_count, "status": "failed"}
    row["returncode"] = int(completed.returncode)
    row["subprocess_seconds"] = subprocess_seconds
    row["stdout_tail"] = completed.stdout[-2000:]
    row["stderr_tail"] = completed.stderr[-2000:]
    pd.DataFrame([row]).to_csv(row_path, index=False)
    return row


thread_sweep_rows: list[dict[str, object]] = []
if run_pardiso_thread_sweep:
    for thread_count in pardiso_thread_candidates:
        row = run_pardiso_thread_sweep_row(int(thread_count))
        thread_sweep_rows.append(row)
        pd.DataFrame(thread_sweep_rows).to_csv(thread_sweep_path, index=False)
        print(
            f"Saved PARDISO thread-sweep row: threads={thread_count}, status={row['status']}",
            flush=True,
        )
elif thread_sweep_path.exists():
    thread_sweep_rows = pd.read_csv(thread_sweep_path).to_dict("records")
else:
    row_paths = sorted(thread_sweep_row_dir.glob("threads_*.csv"))
    for path in row_paths:
        thread_sweep_rows.extend(pd.read_csv(path).to_dict("records"))
    if thread_sweep_rows:
        pd.DataFrame(thread_sweep_rows).to_csv(thread_sweep_path, index=False)

if not thread_sweep_rows:
    print("No PARDISO thread-sweep rows found.")
else:
    thread_sweep = pd.DataFrame(thread_sweep_rows).sort_values("threads")
    thread_sweep.to_csv(thread_sweep_path, index=False)
    display(thread_sweep)
    ok_thread_sweep = thread_sweep[thread_sweep["status"] == "ok"].copy()
    if ok_thread_sweep.empty:
        print("No successful PARDISO thread-sweep rows to plot.")
    else:
        best_row = ok_thread_sweep.loc[ok_thread_sweep["wall_seconds"].idxmin()]
        print(
            "Fastest PARDISO thread-sweep row: "
            f"threads={int(best_row['threads'])}, "
            f"wall={float(best_row['wall_seconds']):.3f} s, "
            f"K={float(best_row['K_m2']):.6e} m2",
            flush=True,
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(
            ok_thread_sweep["threads"],
            ok_thread_sweep["wall_seconds"],
            marker="o",
            label="wall",
        )
        ax.plot(
            ok_thread_sweep["threads"],
            ok_thread_sweep["solve_seconds"],
            marker="s",
            label="reported solve",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MKL_NUM_THREADS")
        ax.set_ylabel("seconds")
        ax.set_title(f"USFEM PARDISO thread sweep: map{thread_sweep_map_shape[0]}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        thread_sweep_plot_path = output_dir / f"{thread_sweep_output_prefix}.png"
        fig.savefig(thread_sweep_plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved PARDISO thread-sweep plot: {thread_sweep_plot_path}")

# %% [markdown]
# ## UMFPACK Tuning Sweep
#
# UMFPACK is the portable direct-solver fallback. Its important controls are
# ordering, strategy, pivot tolerance, and the BLAS/OpenMP thread environment.
# The optional sweep below runs each candidate in a fresh subprocess, compares
# permeability against a same-map PARDISO reference, and treats large relative
# deviations as rejected configurations.

# %%
umfpack_sweep_map_shape = tuple(
    image_size // block_size
    for image_size, block_size in zip(
        umfpack_tuning_image_shape,
        umfpack_tuning_block_shape,
        strict=True,
    )
)
umfpack_sweep_output_prefix = f"usfem_umfpack_tuning_map{umfpack_sweep_map_shape[0]}"
umfpack_sweep_path = output_dir / f"{umfpack_sweep_output_prefix}.csv"
umfpack_sweep_row_dir = output_dir / f"{umfpack_sweep_output_prefix}_rows"
umfpack_sweep_row_dir.mkdir(parents=True, exist_ok=True)


def run_direct_solver_tuning_row(
    *,
    row_name: str,
    backend: str,
    thread_count: int,
    controls: dict[str, object],
) -> dict[str, object]:
    safe_row_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in row_name
    )
    row_path = umfpack_sweep_row_dir / f"{safe_row_name}.csv"
    config = {
        "row_name": row_name,
        "row_path": str(row_path),
        "backend": backend,
        "thread_count": int(thread_count),
        "controls": controls,
        "image_shape": umfpack_tuning_image_shape,
        "block_shape": umfpack_tuning_block_shape,
        "target_porosity": image_3d_target_porosity,
        "blobiness": image_3d_blobiness,
        "seed": image_3d_seed,
        "voxel_size_m": image_3d_voxel_size_m,
    }
    child_code = r"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import porespy as ps

from voids.fem.singlephase import FEMMapProblem, FEniCSSolverOptions, solve_brinkman_usfem
from voids.image.porosity import porosity_map_from_binary, permeability_map_from_porosity

cfg = json.loads(os.environ["VOIDS_FEM_UMFPACK_TUNING_CONFIG"])
backend = str(cfg["backend"])
thread_count = int(cfg["thread_count"])
controls = dict(cfg.get("controls", {}))
row = {
    "row_name": cfg["row_name"],
    "backend": backend,
    "threads": thread_count,
    "controls_json": json.dumps(controls, sort_keys=True),
    "image_shape": tuple(cfg["image_shape"]),
    "block_shape": tuple(cfg["block_shape"]),
}
try:
    start = time.perf_counter()
    binary_void = ps.generators.blobs(
        shape=list(cfg["image_shape"]),
        porosity=float(cfg["target_porosity"]),
        blobiness=float(cfg["blobiness"]),
        seed=int(cfg["seed"]),
    ).astype(bool)
    row["binary_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    porosity_map = porosity_map_from_binary(
        binary_void,
        block_shape=tuple(cfg["block_shape"]),
        voxel_size=float(cfg["voxel_size_m"]),
    )
    permeability_map = permeability_map_from_porosity(
        porosity_map,
        characteristic_length=min(porosity_map.cell_size),
        kozeny_constant=180.0,
        solid_permeability=1.0e-20,
        max_permeability=1.0e-8,
    )
    row["map_seconds"] = time.perf_counter() - start

    problem = FEMMapProblem(
        permeability_map,
        porosity_map,
        viscosity=1.0,
        porosity_floor=1.0e-3,
        permeability_floor=1.0e-20,
    )
    if backend == "pardiso":
        options = FEniCSSolverOptions.pardiso_direct()
    elif backend == "umfpack":
        options = FEniCSSolverOptions.umfpack_direct(controls=controls)
    else:
        raise ValueError(f"unsupported backend: {backend}")

    start = time.perf_counter()
    result = solve_brinkman_usfem(
        problem,
        flow_axis="x",
        pressure_inlet=1.0,
        pressure_outlet=0.0,
        options=options,
    )
    row.update(
        {
            "status": "ok",
            "K_m2": result.permeability,
            "flow_rate": result.flow_rate,
            "solve_seconds": result.solve_seconds,
            "wall_seconds": time.perf_counter() - start,
            "serial_sparse_matrix_nnz": result.metadata.get("serial_sparse_matrix_nnz"),
            "serial_sparse_solver_backend": result.metadata.get("serial_sparse_solver_backend"),
            "serial_sparse_umfpack_family": result.metadata.get("serial_sparse_umfpack_family"),
            "serial_sparse_umfpack_resolved_controls": json.dumps(
                result.metadata.get("serial_sparse_umfpack_resolved_controls", {}),
                sort_keys=True,
            ),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        }
    )
except Exception as exc:
    row.update(
        {
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
        }
    )

pd.DataFrame([row]).to_csv(cfg["row_path"], index=False)
print(pd.DataFrame([row]).to_string(index=False), flush=True)
"""
    environment = os.environ.copy()
    thread_text = str(thread_count)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_text,
            "OPENBLAS_NUM_THREADS": thread_text,
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1" if backend == "umfpack" else thread_text,
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "VOIDS_FEM_UMFPACK_TUNING_CONFIG": json.dumps(config),
        }
    )
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=project_root(),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=umfpack_tuning_row_timeout_s,
    )
    subprocess_seconds = time.perf_counter() - start
    if row_path.exists():
        row = pd.read_csv(row_path).iloc[0].to_dict()
    else:
        row = {
            "row_name": row_name,
            "backend": backend,
            "threads": thread_count,
            "status": "failed",
        }
    row["returncode"] = int(completed.returncode)
    row["subprocess_seconds"] = subprocess_seconds
    row["stdout_tail"] = completed.stdout[-2000:]
    row["stderr_tail"] = completed.stderr[-2000:]
    pd.DataFrame([row]).to_csv(row_path, index=False)
    return row


umfpack_sweep_rows: list[dict[str, object]] = []
if run_umfpack_tuning_sweep:
    reference_row = run_direct_solver_tuning_row(
        row_name="pardiso_reference",
        backend="pardiso",
        thread_count=min(32, detected_thread_count),
        controls={},
    )
    umfpack_sweep_rows.append(reference_row)
    pd.DataFrame(umfpack_sweep_rows).to_csv(umfpack_sweep_path, index=False)
    for thread_count in umfpack_tuning_thread_candidates:
        for control_set in umfpack_tuning_control_sets:
            row = run_direct_solver_tuning_row(
                row_name=f"umfpack_{control_set['name']}_threads{thread_count}",
                backend="umfpack",
                thread_count=int(thread_count),
                controls=dict(control_set["controls"]),
            )
            umfpack_sweep_rows.append(row)
            pd.DataFrame(umfpack_sweep_rows).to_csv(umfpack_sweep_path, index=False)
            print(
                f"Saved UMFPACK tuning row: {row['row_name']}, status={row['status']}",
                flush=True,
            )
elif umfpack_sweep_path.exists():
    umfpack_sweep_rows = pd.read_csv(umfpack_sweep_path).to_dict("records")
else:
    row_paths = sorted(umfpack_sweep_row_dir.glob("*.csv"))
    for path in row_paths:
        umfpack_sweep_rows.extend(pd.read_csv(path).to_dict("records"))
    if umfpack_sweep_rows:
        pd.DataFrame(umfpack_sweep_rows).to_csv(umfpack_sweep_path, index=False)

if not umfpack_sweep_rows:
    print("No UMFPACK tuning rows found.")
else:
    umfpack_sweep = pd.DataFrame(umfpack_sweep_rows)
    reference_ok = umfpack_sweep[
        (umfpack_sweep["backend"] == "pardiso") & (umfpack_sweep["status"] == "ok")
    ]
    if reference_ok.empty:
        print("No successful PARDISO reference row for UMFPACK tuning.")
    else:
        reference_k = float(reference_ok.iloc[0]["K_m2"])
        umfpack_sweep["relative_error_vs_pardiso"] = np.nan
        ok_mask = umfpack_sweep["status"] == "ok"
        umfpack_sweep.loc[ok_mask, "relative_error_vs_pardiso"] = (
            pd.to_numeric(umfpack_sweep.loc[ok_mask, "K_m2"], errors="coerce")
            - reference_k
        ).abs() / abs(reference_k)
        umfpack_sweep.to_csv(umfpack_sweep_path, index=False)
        display(umfpack_sweep)

        accepted = umfpack_sweep[
            (umfpack_sweep["backend"] == "umfpack")
            & (umfpack_sweep["status"] == "ok")
            & (umfpack_sweep["relative_error_vs_pardiso"] <= 1.0e-8)
        ].copy()
        if accepted.empty:
            print(
                "No accepted UMFPACK tuning rows matched the PARDISO reference tolerance."
            )
        else:
            best_umfpack = accepted.loc[
                pd.to_numeric(accepted["wall_seconds"]).idxmin()
            ]
            print(
                "Fastest accepted UMFPACK tuning row: "
                f"{best_umfpack['row_name']}, "
                f"wall={float(best_umfpack['wall_seconds']):.3f} s, "
                f"relative_error={float(best_umfpack['relative_error_vs_pardiso']):.3e}",
                flush=True,
            )
            plot_df = accepted.sort_values("wall_seconds").head(12).copy()
            plot_df["label"] = plot_df["row_name"].astype(str)
            fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
            ax.barh(
                np.arange(len(plot_df), dtype=float),
                plot_df["wall_seconds"],
                color="tab:green",
            )
            ax.set_yticks(np.arange(len(plot_df), dtype=float))
            ax.set_yticklabels(plot_df["label"])
            ax.invert_yaxis()
            ax.set_xlabel("wall time [s]")
            ax.set_title(
                f"Accepted UMFPACK tuning rows: map{umfpack_sweep_map_shape[0]}"
            )
            ax.grid(axis="x", alpha=0.25)
            umfpack_sweep_plot_path = output_dir / f"{umfpack_sweep_output_prefix}.png"
            fig.savefig(umfpack_sweep_plot_path, dpi=180)
            plt.close(fig)
            print(f"Saved UMFPACK tuning plot: {umfpack_sweep_plot_path}")

# %% [markdown]
# ## Model And Solver Timing Comparison
#
# This table summarizes the current run. Previous PETSc direct or stress-test
# rows can be folded in for provenance by setting
# `include_historical_petsc_comparison_rows=True`.

# %%
model_solver_frames: list[pd.DataFrame] = []
current_results_path = results_path
if current_results_path.exists():
    frame = pd.read_csv(current_results_path)
    frame["comparison_source"] = current_results_path.name
    model_solver_frames.append(frame)

if include_historical_petsc_comparison_rows:
    previous_direct_path = output_dir / f"{output_prefix}_partial_summary.csv"
    if previous_direct_path.exists():
        frame = pd.read_csv(previous_direct_path)
        frame["comparison_source"] = previous_direct_path.name
        model_solver_frames.append(frame)

    stress_path = output_dir / f"{output_prefix}_stress_outcomes.csv"
    if stress_path.exists():
        frame = pd.read_csv(stress_path)
        frame["comparison_source"] = stress_path.name
        model_solver_frames.append(frame)

if not model_solver_frames:
    print("No model/solver timing rows found.")
else:
    model_solver_timing = pd.concat(model_solver_frames, ignore_index=True, sort=False)
    model_solver_timing = model_solver_timing.drop_duplicates(
        subset=["case", "formulation_label", "preset_label", "status"],
        keep="first",
    )
    preferred_columns = [
        "case",
        "formulation_label",
        "preset_label",
        "status",
        "K_m2",
        "solve_seconds",
        "wall_seconds",
        "comparison_source",
        "failure",
    ]
    present_columns = [
        column for column in preferred_columns if column in model_solver_timing
    ]
    model_solver_timing = model_solver_timing[present_columns].copy()
    model_solver_timing_path = output_dir / f"{output_prefix}_model_solver_timing.csv"
    model_solver_timing.to_csv(model_solver_timing_path, index=False)
    display(model_solver_timing)
    print(f"Saved model/solver timing comparison: {model_solver_timing_path}")

    timed_rows = model_solver_timing[model_solver_timing["wall_seconds"].notna()].copy()
    if timed_rows.empty:
        print("No timed model/solver rows to plot.")
    else:
        timed_rows["label"] = (
            timed_rows["case"].astype(str)
            + " | "
            + timed_rows["formulation_label"].astype(str)
            + " | "
            + timed_rows["preset_label"].astype(str)
        )
        timed_rows = timed_rows.sort_values("wall_seconds")
        colors = np.where(
            timed_rows["status"].astype(str) == "ok", "tab:blue", "tab:red"
        )
        fig_height = max(5.0, 0.42 * len(timed_rows))
        fig, ax = plt.subplots(figsize=(11.0, fig_height), constrained_layout=True)
        ax.barh(
            np.arange(len(timed_rows)),
            timed_rows["wall_seconds"],
            color=colors,
            alpha=0.75,
        )
        ax.set_xscale("log")
        ax.set_yticks(np.arange(len(timed_rows)))
        ax.set_yticklabels(timed_rows["label"])
        ax.set_xlabel("wall time [s]")
        ax.set_title("Model and solver timing comparison")
        ax.grid(axis="x", which="both", alpha=0.25)
        timing_plot_path = output_dir / f"{output_prefix}_model_solver_timing.png"
        fig.savefig(timing_plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved model/solver timing plot: {timing_plot_path}")

# %% [markdown]
# ## USFEM Block Preconditioner Probes
#
# The rows below summarize optional probe CSVs produced while developing USFEM
# block preconditioners. They are intentionally separate from the main preset
# table because they exercise low-level experimental assemblies and are not
# shipped as recommended solver presets.

# %%
probe_file_specs = (
    [
        ("monolithic PyAMG root-node", "map5", "usfem_pyamg_rootnode_probe_map5.csv"),
        ("monolithic PyAMG root-node", "map10", "usfem_pyamg_rootnode_probe_map10.csv"),
        (
            "block lower triangular AMG",
            "map5",
            "usfem_block_aware_pyamg_probe_map5.csv",
        ),
        (
            "block lower triangular AMG",
            "map10",
            "usfem_block_aware_pyamg_probe_map10.csv",
        ),
        (
            "block lower triangular AMG with exact pressure",
            "map10",
            "usfem_block_aware_pyamg_pressure_exact_probe_map10.csv",
        ),
        (
            "Schur diagonal with SuperLU Schur solve",
            "map5",
            "usfem_block_schur_diag_probe_map5.csv",
        ),
        (
            "Schur diagonal with SuperLU Schur solve",
            "map10",
            "usfem_block_schur_diag_probe_map10.csv",
        ),
        (
            "Schur diagonal with PARDISO Schur solve",
            "map10",
            "usfem_block_schur_pardiso_probe_map10.csv",
        ),
        (
            "Schur diagonal with PARDISO Schur solve",
            "map20",
            "usfem_block_schur_pardiso_probe_map20.csv",
        ),
        (
            "Schur diagonal with incomplete LU",
            "map10",
            "usfem_block_schur_spilu_probe_map10.csv",
        ),
    ]
    if include_block_preconditioner_probe_summary
    else []
)

if not include_block_preconditioner_probe_summary:
    print(
        "USFEM block-preconditioner probe summaries are disabled by default; "
        "set include_block_preconditioner_probe_summary=True to load historical "
        "probe CSVs.",
        flush=True,
    )

probe_frames: list[pd.DataFrame] = []
for probe_label, problem_size, filename in probe_file_specs:
    path = output_dir / filename
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    frame["probe"] = probe_label
    frame["problem_size"] = problem_size
    frame["source_file"] = filename
    probe_frames.append(frame)

if not probe_frames:
    print("No USFEM block preconditioner probe CSVs found.")
else:
    probe_results = pd.concat(probe_frames, ignore_index=True, sort=False)
    preferred_columns = [
        "problem_size",
        "probe",
        "config",
        "status",
        "K_m2",
        "K_relative_to_reference",
        "relative_residual",
        "iterations",
        "setup_seconds",
        "solve_seconds",
        "wall_seconds",
        "assembly_seconds",
        "ndof",
        "nnz",
        "Sdiag_nnz",
        "source_file",
        "error",
    ]
    present_columns = [
        column for column in preferred_columns if column in probe_results
    ]
    probe_summary = probe_results[present_columns].copy()
    probe_summary_path = output_dir / "usfem_block_preconditioner_probe_summary.csv"
    probe_summary.to_csv(probe_summary_path, index=False)
    display(probe_summary)
    print(f"Saved USFEM block-preconditioner probe summary: {probe_summary_path}")

# %%
if not probe_frames:
    print("No USFEM block preconditioner probe rows to plot.")
else:
    plot_probe = probe_results.copy()
    plot_probe = plot_probe[plot_probe["wall_seconds"].notna()].copy()
    if plot_probe.empty:
        print("No timed USFEM block preconditioner probe rows to plot.")
    else:
        plot_probe["short_label"] = (
            plot_probe["problem_size"].astype(str)
            + " | "
            + plot_probe["config"].astype(str)
        )
        y = np.arange(len(plot_probe), dtype=float)
        figure_height = max(8.0, 0.36 * len(plot_probe))
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16.0, figure_height),
            constrained_layout=True,
            sharey=True,
        )

        colors = np.where(
            plot_probe["status"].astype(str) == "ok", "tab:blue", "tab:red"
        )
        axes[0].barh(y, plot_probe["wall_seconds"], color=colors, alpha=0.75)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("wall time [s]")
        axes[0].set_title("USFEM block preconditioner probe timing")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(plot_probe["short_label"])
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", which="both", alpha=0.25)

        error_values = pd.to_numeric(
            plot_probe.get(
                "K_relative_to_reference", pd.Series(np.nan, index=plot_probe.index)
            ),
            errors="coerce",
        ).replace(0.0, np.nan)
        axes[1].barh(y, error_values, color=colors, alpha=0.75)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("relative K difference")
        axes[1].set_title("Permeability difference from direct reference")
        axes[1].grid(axis="x", which="both", alpha=0.25)

        probe_plot_path = (
            output_dir / "usfem_block_preconditioner_probe_timing_accuracy.png"
        )
        fig.savefig(probe_plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved USFEM block-preconditioner probe plot: {probe_plot_path}")

# %% [markdown]
# ## Interpretation Checklist
#
# Before using a faster preset in a same-ROI rock comparison:
#
# - compare permeability against `direct_reference_mumps` on the constant case;
# - compare permeability against direct rows on a small heterogeneous case;
# - inspect `petsc_converged_reason`, iteration count, residual norm, and any
#   warnings or failures;
# - record whether `mpi_size` is actually greater than one for parallel-direct
#   runs;
# - only relax tolerances after permeability, flow rate, and solver residuals
#   are insensitive at the required scientific precision.
