from __future__ import annotations

from typing import cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, gmres, splu, spsolve

from voids.linalg.cudss import solve_nvmath_cudss
from voids.linalg._typing import (
    LinearSystemDType,
    PyAMGModule,
    PyPardisoSolver,
    SolverInfo,
    SolverParameters,
    SolverParameterValue,
    UmfpackSolver,
)

_SUPPORTED_LINEAR_SYSTEM_DTYPES: dict[np.dtype[np.generic], LinearSystemDType] = {
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
}


def _resolve_linear_system_dtype(value: SolverParameterValue | None) -> np.dtype[np.generic]:
    """Normalize the requested floating-point dtype for sparse solves."""

    if value is None:
        return np.dtype("float64")
    if not isinstance(value, str):
        raise ValueError("linear solver dtype must be 'float32' or 'float64'")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("linear solver dtype must be 'float32' or 'float64'") from exc
    if dtype not in _SUPPORTED_LINEAR_SYSTEM_DTYPES:
        raise ValueError("linear solver dtype must be 'float32' or 'float64'")
    return dtype


def _linear_system_dtype_metadata(dtype: np.dtype[np.generic]) -> dict[str, str]:
    return {"linear_system_dtype": _SUPPORTED_LINEAR_SYSTEM_DTYPES[dtype]}


def _cast_linear_system(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    dtype: np.dtype[np.generic],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    return matrix.tocsr().astype(dtype, copy=False), np.ascontiguousarray(
        np.asarray(rhs, dtype=dtype)
    )


def _superlu_kwargs(parameters: SolverParameters) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if "permc_spec" in parameters:
        kwargs["permc_spec"] = parameters["permc_spec"]
    if "diag_pivot_thresh" in parameters:
        kwargs["diag_pivot_thresh"] = parameters["diag_pivot_thresh"]
    if "relax" in parameters:
        kwargs["relax"] = parameters["relax"]
    if "panel_size" in parameters:
        kwargs["panel_size"] = parameters["panel_size"]
    if "equil" in parameters:
        kwargs["options"] = {"Equil": bool(parameters["equil"])}
    return kwargs


def _import_pyamg() -> PyAMGModule:
    """Import PyAMG lazily so the dependency remains easy to diagnose."""

    try:
        import pyamg  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyAMG preconditioning requires the 'pyamg' package to be installed."
        ) from exc
    return cast(PyAMGModule, pyamg)


def _import_pypardiso() -> PyPardisoSolver:
    """Import pypardiso lazily so the dependency remains easy to diagnose."""

    try:
        import pypardiso  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PARDISO solver requires the 'pypardiso' package to be installed. "
            "This is currently only supported on Linux systems."
        ) from exc
    return cast(PyPardisoSolver, pypardiso.spsolve)


def _import_umfpack() -> UmfpackSolver:
    """Import scikit-umfpack lazily so missing SuiteSparse support is clear."""

    try:
        from scikits.umfpack import spsolve as umfpack_spsolve  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "UMFPACK solver requires the optional 'scikit-umfpack' package and "
            "SuiteSparse/UMFPACK libraries to be installed."
        ) from exc
    return cast(UmfpackSolver, umfpack_spsolve)


def _build_preconditioner(
    A: sparse.csr_matrix,
    *,
    solver_parameters: SolverParameters | None,
) -> tuple[LinearOperator[np.float64] | None, dict[str, str | float | int]]:
    """Build an optional Krylov preconditioner from solver parameters."""

    parameters = dict(solver_parameters or {})
    name = parameters.get("preconditioner")
    if name is None:
        return None, {}
    if name != "pyamg":
        raise ValueError(f"Unknown preconditioner '{name}'")

    pyamg = _import_pyamg()
    amg_kind = str(parameters.get("pyamg_solver", "smoothed_aggregation"))
    amg_kwargs = parameters.get("pyamg_kwargs", {})
    if not isinstance(amg_kwargs, dict):
        raise ValueError("pyamg_kwargs must be a dictionary")

    matrix = sparse.csr_matrix(A)
    if amg_kind == "smoothed_aggregation":
        hierarchy = pyamg.smoothed_aggregation_solver(matrix, **amg_kwargs)
    elif amg_kind == "rootnode":
        hierarchy = pyamg.rootnode_solver(matrix, **amg_kwargs)
    elif amg_kind == "ruge_stuben":
        hierarchy = pyamg.ruge_stuben_solver(matrix, **amg_kwargs)
    else:
        raise ValueError(
            f"Unknown pyamg_solver '{amg_kind}'. Expected 'smoothed_aggregation', "
            "'rootnode', or 'ruge_stuben'."
        )
    return (
        hierarchy.aspreconditioner(),
        {
            "preconditioner": "pyamg",
            "pyamg_solver": amg_kind,
            "pyamg_levels": int(len(hierarchy.levels)),
            "pyamg_operator_complexity": float(hierarchy.operator_complexity()),
        },
    )


def solve_linear_system(
    A: sparse.csr_matrix,
    b: np.ndarray,
    *,
    method: str = "direct",
    solver_parameters: SolverParameters | None = None,
) -> tuple[np.ndarray, SolverInfo]:
    """Solve a sparse linear system with one of the supported backends.

    Parameters
    ----------
    A :
        Sparse system matrix.
    b :
        Right-hand-side vector.
    method :
        Solver backend. Supported values are ``"direct"``, ``"superlu"``,
        ``"umfpack"``, ``"pardiso"``, ``"nvmath_cudss"``, ``"cg"``, and
        ``"gmres"``.
    solver_parameters :
        Optional backend-specific solver options. For SciPy Krylov methods this
        maps directly to supported keyword arguments such as ``rtol``,
        ``atol``, ``restart``, and ``maxiter``. Setting
        ``{"preconditioner": "pyamg"}`` attaches a PyAMG preconditioner to
        ``cg`` or ``gmres``. Setting ``{"dtype": "float32"}`` or
        ``{"dtype": "float64"}`` controls the value dtype used by backends that
        support runtime precision selection. ``scikit-umfpack`` and
        ``pypardiso`` currently expose double-precision solves only.

    Returns
    -------
    numpy.ndarray
        Solution vector.
    SolverInfo
        Solver metadata containing the method name and the iterative solver
        status code ``info``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.

    Notes
    -----
    The ``"direct"`` method uses :func:`scipy.sparse.linalg.spsolve`. The
    ``"superlu"`` method uses :func:`scipy.sparse.linalg.splu` explicitly and is
    the portable CPU direct backend with runtime ``float32``/``float64`` value
    dtype selection. The ``"umfpack"`` method requests SuiteSparse/UMFPACK
    explicitly through ``scikit-umfpack``. The ``"pardiso"`` method uses Intel
    MKL PARDISO through ``pypardiso``; this is typically only available on Linux
    systems. The ``"nvmath_cudss"`` method uses the optional nvmath/cuDSS CUDA
    direct solver and accepts controls such as
    ``{"device_ids": 0, "dtype": "float64"}`` or
    ``{"device_ids": (0, 1), "dtype": "float64"}``.
    """

    parameters = dict(solver_parameters or {})
    dtype = _resolve_linear_system_dtype(parameters.get("dtype"))
    A_work, b_work = _cast_linear_system(A, b, dtype)
    dtype_info = _linear_system_dtype_metadata(dtype)

    if method == "direct":
        kwargs: dict[str, object] = {}
        if dtype == np.dtype("float32"):
            # SciPy's default spsolve path may dispatch to UMFPACK when
            # scikit-umfpack is installed; that wrapper is double-only.
            kwargs["use_umfpack"] = False
        x = spsolve(A_work, b_work, **kwargs)
        return np.asarray(x), {
            "method": method,
            "backend": "scipy.sparse.linalg.spsolve",
            "info": 0,
            **dtype_info,
        }
    if method == "superlu":
        lu = splu(A_work.tocsc(), **_superlu_kwargs(parameters))
        x = lu.solve(b_work)
        return np.asarray(x), {
            "method": method,
            "backend": "scipy.sparse.linalg.splu",
            "info": 0,
            "superlu_l_nnz": int(lu.L.nnz),
            "superlu_u_nnz": int(lu.U.nnz),
            **dtype_info,
        }
    if method == "umfpack":
        if dtype == np.dtype("float32"):
            raise ValueError(
                "solver method 'umfpack' currently supports float64 only through "
                "scikit-umfpack; use method='direct' or method='superlu' for CPU "
                "single-precision sparse solves."
            )
        umfpack_spsolve = _import_umfpack()
        x = umfpack_spsolve(
            sparse.csc_matrix(A_work, dtype=dtype),
            b_work,
        )
        return np.asarray(x), {
            "method": method,
            "backend": "scikits.umfpack.spsolve",
            "info": 0,
            **dtype_info,
        }
    if method == "pardiso":
        if dtype == np.dtype("float32"):
            raise ValueError(
                "solver method 'pardiso' currently supports float64 only through "
                "pypardiso; use method='direct' or method='superlu' for CPU "
                "single-precision sparse solves."
            )
        pardiso_spsolve = _import_pypardiso()
        x = pardiso_spsolve(A_work, b_work)
        return np.asarray(x), {
            "method": method,
            "backend": "pypardiso",
            "info": 0,
            **dtype_info,
        }
    if method == "nvmath_cudss":
        x, metadata = solve_nvmath_cudss(
            A,
            np.ascontiguousarray(np.asarray(b, dtype=float)),
            controls=parameters,
        )
        return np.asarray(x, dtype=float), {
            "method": method,
            "backend": "nvmath.bindings.cudss",
            "info": 0,
            "linear_system_dtype": str(metadata.get("serial_sparse_nvmath_cudss_dtype", "")),
            **metadata,
        }
    if method == "cg":
        preconditioner, preconditioner_info = _build_preconditioner(
            A_work, solver_parameters=parameters
        )
        cg_kwargs = {
            key: parameters[key] for key in ("rtol", "atol", "maxiter", "M") if key in parameters
        }
        if preconditioner is not None and "M" not in cg_kwargs:
            cg_kwargs["M"] = preconditioner
        x, info = cg(A_work, b_work, **cg_kwargs)
        return np.asarray(x), {
            "method": method,
            "info": int(info),
            **dtype_info,
            **preconditioner_info,
        }
    if method == "gmres":
        preconditioner, preconditioner_info = _build_preconditioner(
            A_work, solver_parameters=parameters
        )
        gmres_kwargs = {
            key: parameters[key]
            for key in ("rtol", "atol", "restart", "maxiter", "M")
            if key in parameters
        }
        if preconditioner is not None and "M" not in gmres_kwargs:
            gmres_kwargs["M"] = preconditioner
        x, info = gmres(A_work, b_work, **gmres_kwargs)
        return np.asarray(x), {
            "method": method,
            "info": int(info),
            **dtype_info,
            **preconditioner_info,
        }
    raise ValueError(f"Unknown solver method '{method}'")
