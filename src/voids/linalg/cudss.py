from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

NVMATH_CUDSS_CONTROL_KEYS = {
    "check_residual",
    "device_ids",
    "dtype",
    "ir_steps",
    "pivot_type",
    "residual_rtol",
    "use_matching",
    "value_dtype",
}
NVMATH_CUDSS_DTYPES = {"float32", "float64"}
NVMATH_CUDSS_PIVOT_TYPES = {"col", "row", "none"}


def json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def normalize_nvmath_cudss_device_ids(value: Any) -> tuple[int, ...] | Literal["all"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized != "all":
            raise ValueError("nvmath_cudss device_ids must be 'all', an int, or a sequence of ints")
        return "all"
    ids: tuple[int, ...]
    if isinstance(value, int):
        ids = (int(value),)
    elif isinstance(value, Sequence):
        ids = tuple(int(item) for item in value)
    else:
        raise ValueError("nvmath_cudss device_ids must be 'all', an int, or a sequence of ints")
    if not ids:
        raise ValueError("nvmath_cudss device_ids cannot be empty")
    if any(device_id < 0 for device_id in ids):
        raise ValueError("nvmath_cudss device_ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError("nvmath_cudss device_ids must not contain duplicates")
    return ids


def resolve_nvmath_cudss_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize controls for the optional nvmath/cuDSS backend."""

    resolved: dict[str, Any] = {
        "dtype": "float64",
        "ir_steps": 5,
        "use_matching": True,
        "check_residual": True,
    }
    for key, value in controls.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key not in NVMATH_CUDSS_CONTROL_KEYS:
            supported = ", ".join(sorted(NVMATH_CUDSS_CONTROL_KEYS))
            raise ValueError(
                f"Unsupported nvmath_cudss control {key!r}; supported controls: {supported}"
            )
        if normalized_key in {"dtype", "value_dtype"}:
            dtype = str(value).strip().lower()
            if dtype not in NVMATH_CUDSS_DTYPES:
                supported = ", ".join(sorted(NVMATH_CUDSS_DTYPES))
                raise ValueError(f"nvmath_cudss dtype must be one of: {supported}")
            resolved["dtype"] = dtype
        elif normalized_key == "device_ids":
            resolved["device_ids"] = normalize_nvmath_cudss_device_ids(value)
        elif normalized_key == "ir_steps":
            ir_steps = int(value)
            if ir_steps < 0:
                raise ValueError("nvmath_cudss ir_steps must be non-negative")
            resolved["ir_steps"] = ir_steps
        elif normalized_key == "use_matching":
            resolved["use_matching"] = bool(value)
        elif normalized_key == "pivot_type":
            if value is None:
                resolved.pop("pivot_type", None)
            else:
                pivot_type = str(value).strip().lower()
                if pivot_type not in NVMATH_CUDSS_PIVOT_TYPES:
                    supported = ", ".join(sorted(NVMATH_CUDSS_PIVOT_TYPES))
                    raise ValueError(f"nvmath_cudss pivot_type must be one of: {supported}")
                resolved["pivot_type"] = pivot_type
        elif normalized_key == "check_residual":
            resolved["check_residual"] = bool(value)
        elif normalized_key == "residual_rtol":
            residual_rtol = float(value)
            if residual_rtol <= 0.0 or not np.isfinite(residual_rtol):
                raise ValueError("nvmath_cudss residual_rtol must be positive and finite")
            resolved["residual_rtol"] = residual_rtol
    if "residual_rtol" not in resolved:
        resolved["residual_rtol"] = 1.0e-8 if resolved["dtype"] == "float64" else 1.0e-4
    return resolved


def nvmath_cudss_controls_from_arguments(
    *,
    dtype: Literal["float32", "float64"] = "float64",
    device_ids: int | Sequence[int] | Literal["all"] | None = None,
    ir_steps: int = 5,
    use_matching: bool = True,
    pivot_type: Literal["col", "row", "none"] | None = None,
    check_residual: bool = True,
    residual_rtol: float | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build validated cuDSS controls from keyword-style arguments."""

    nvmath_cudss_controls: dict[str, Any] = dict(controls or {})
    nvmath_cudss_controls["dtype"] = dtype
    nvmath_cudss_controls["ir_steps"] = int(ir_steps)
    nvmath_cudss_controls["use_matching"] = bool(use_matching)
    nvmath_cudss_controls["check_residual"] = bool(check_residual)
    if device_ids is not None:
        nvmath_cudss_controls["device_ids"] = device_ids
    if pivot_type is not None:
        nvmath_cudss_controls["pivot_type"] = pivot_type
    if residual_rtol is not None:
        nvmath_cudss_controls["residual_rtol"] = float(residual_rtol)
    return resolve_nvmath_cudss_controls(nvmath_cudss_controls)


def require_nvmath_cudss() -> tuple[Any, Any]:
    """Return imported PyTorch and cuDSS bindings or raise a backend-specific error."""

    try:
        torch = import_module("torch")
        cudss = import_module("nvmath.bindings.cudss")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "linear_backend='nvmath_cudss' requires optional CUDA sparse-solver "
            "dependencies: PyTorch with CUDA support and nvmath-python with cuDSS. "
            "Install a compatible nvmath/cuDSS stack or choose a portable backend "
            "such as solver='direct' or linear_backend='superlu'."
        ) from exc
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise RuntimeError(
            "linear_backend='nvmath_cudss' requires a CUDA-capable GPU visible to "
            "PyTorch. Choose a CPU backend such as solver='direct' or "
            "linear_backend='superlu' on this platform."
        )
    return torch, cudss


def nvmath_cudss_device_ids(
    torch: Any,
    device_ids: tuple[int, ...] | Literal["all"] | None,
) -> tuple[int, ...]:
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError("linear_backend='nvmath_cudss' requires at least one CUDA device")
    if device_ids == "all":
        resolved_device_ids = tuple(range(device_count))
    elif device_ids is None:
        resolved_device_ids = (int(torch.cuda.current_device()),)
    else:
        resolved_device_ids = device_ids
    if any(device_id >= device_count for device_id in resolved_device_ids):
        raise ValueError(
            "nvmath_cudss device_ids contains an unavailable CUDA device; "
            f"available device ids are 0..{device_count - 1}"
        )
    return resolved_device_ids


def _set_nvmath_cudss_config_scalar(cudss: Any, config: Any, param: Any, value: Any) -> None:
    dtype = cudss.get_config_param_dtype(param)
    scalar = np.zeros((1,), dtype=dtype)
    scalar[0] = value
    cudss.config_set(config, param, scalar.ctypes.data, scalar.dtype.itemsize)


def _set_nvmath_cudss_config_array(cudss: Any, config: Any, param: Any, values: Any) -> None:
    dtype = cudss.get_config_param_dtype(param)
    array = np.ascontiguousarray(values, dtype=dtype)
    cudss.config_set(config, param, array.ctypes.data, array.dtype.itemsize * array.size)


_NVMATH_CUDSS_MEMORY_ESTIMATES_DTYPE = np.dtype(
    [
        ("permanent_device_memory", "<u8"),
        ("peak_device_memory", "<u8"),
        ("permanent_host_memory", "<u8"),
        ("peak_host_memory", "<u8"),
        ("hybrid_min_device_memory", "<u8"),
        ("hybrid_max_device_memory", "<u8"),
        ("reserved", "<u8", (10,)),
    ]
)


def _nvmath_cudss_memory_estimates(cudss: Any, handle: Any, data: Any) -> dict[str, int]:
    estimates = np.zeros((), dtype=_NVMATH_CUDSS_MEMORY_ESTIMATES_DTYPE)
    size_written = np.zeros((1,), dtype=np.uint64)
    cudss.data_get(
        handle,
        data,
        cudss.DataParam.MEMORY_ESTIMATES,
        estimates.ctypes.data,
        estimates.dtype.itemsize,
        size_written.ctypes.data,
    )
    record = estimates.view(np.recarray)
    return {
        name: int(getattr(record, name))
        for name in _NVMATH_CUDSS_MEMORY_ESTIMATES_DTYPE.names
        if name != "reserved"
    }


def _nvmath_cudss_relative_residual(
    matrix: sparse.spmatrix,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    residual = matrix @ solution - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(residual))
    if rhs_norm == 0.0:
        return residual_norm
    return residual_norm / rhs_norm


def solve_nvmath_cudss(
    matrix: sparse.spmatrix,
    rhs_array: np.ndarray,
    *,
    controls: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a CSR-compatible sparse system with nvmath/cuDSS on CUDA devices."""

    torch, cudss = require_nvmath_cudss()
    requested_controls = dict(controls or {})
    resolved_controls = resolve_nvmath_cudss_controls(requested_controls)
    device_ids = nvmath_cudss_device_ids(
        torch,
        cast(tuple[int, ...] | Literal["all"] | None, resolved_controls.get("device_ids")),
    )
    primary_device = device_ids[0]
    torch.cuda.set_device(primary_device)
    torch_device = torch.device(f"cuda:{primary_device}")
    torch_dtype = torch.float64 if resolved_controls["dtype"] == "float64" else torch.float32
    numpy_dtype = np.float64 if resolved_controls["dtype"] == "float64" else np.float32

    csr_matrix = matrix.tocsr()
    rows, cols = csr_matrix.shape
    if rows != cols:
        raise ValueError("linear_backend='nvmath_cudss' requires a square sparse matrix")
    int32_max = np.iinfo(np.int32).max
    if rows > int32_max or cols > int32_max or int(csr_matrix.nnz) > int32_max:
        raise ValueError("linear_backend='nvmath_cudss' currently requires 32-bit CSR indices")

    row_offsets = torch.as_tensor(
        np.ascontiguousarray(csr_matrix.indptr, dtype=np.int32),
        device=torch_device,
    )
    col_indices = torch.as_tensor(
        np.ascontiguousarray(csr_matrix.indices, dtype=np.int32),
        device=torch_device,
    )
    values = torch.as_tensor(
        np.ascontiguousarray(csr_matrix.data, dtype=numpy_dtype),
        dtype=torch_dtype,
        device=torch_device,
    )
    rhs = torch.as_tensor(
        np.ascontiguousarray(rhs_array, dtype=numpy_dtype),
        dtype=torch_dtype,
        device=torch_device,
    )
    rhs_is_vector = bool(rhs.dim() == 1)
    rhs_matrix = rhs.unsqueeze(1) if rhs_is_vector else rhs
    nrhs = int(rhs_matrix.size(1))
    rhs_col_major = rhs_matrix.t().contiguous()
    solution_col_major = torch.zeros_like(rhs_col_major)

    cuda_r_32i = 10
    value_type = 1 if resolved_controls["dtype"] == "float64" else 0
    pivot_type_map = {
        "col": cudss.PivotType.PIVOT_COL,
        "row": cudss.PivotType.PIVOT_ROW,
        "none": cudss.PivotType.PIVOT_NONE,
    }
    handle = (
        cudss.create()
        if len(device_ids) == 1
        else cudss.create_mg(len(device_ids), list(device_ids))
    )
    matrix_desc = rhs_desc = solution_desc = config = data = None
    for device_id in device_ids:
        torch.cuda.reset_peak_memory_stats(device_id)
    memory_estimates: dict[str, int] = {}
    memory_estimates_error = ""
    start = perf_counter()
    try:
        cudss.set_stream(handle, torch.cuda.current_stream(primary_device).cuda_stream)
        matrix_desc = cudss.matrix_create_csr(
            rows,
            cols,
            int(values.numel()),
            row_offsets.data_ptr(),
            0,
            col_indices.data_ptr(),
            values.data_ptr(),
            cuda_r_32i,
            value_type,
            cudss.MatrixType.GENERAL.value,
            cudss.MatrixViewType.FULL.value,
            cudss.IndexBase.ZERO.value,
        )
        rhs_desc = cudss.matrix_create_dn(
            rows,
            nrhs,
            rows,
            rhs_col_major.data_ptr(),
            value_type,
            cudss.Layout.COL_MAJOR.value,
        )
        solution_desc = cudss.matrix_create_dn(
            rows,
            nrhs,
            rows,
            solution_col_major.data_ptr(),
            value_type,
            cudss.Layout.COL_MAJOR.value,
        )
        config = cudss.config_create()
        data = cudss.data_create(handle)
        if len(device_ids) > 1:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.DEVICE_COUNT,
                len(device_ids),
            )
            _set_nvmath_cudss_config_array(
                cudss,
                config,
                cudss.ConfigParam.DEVICE_INDICES,
                device_ids,
            )
        _set_nvmath_cudss_config_scalar(
            cudss,
            config,
            cudss.ConfigParam.IR_N_STEPS,
            int(resolved_controls["ir_steps"]),
        )
        _set_nvmath_cudss_config_scalar(
            cudss,
            config,
            cudss.ConfigParam.USE_MATCHING,
            int(bool(resolved_controls["use_matching"])),
        )
        if "pivot_type" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.PIVOT_TYPE,
                pivot_type_map[str(resolved_controls["pivot_type"])].value,
            )
        for phase_name, phase in (
            ("analysis", cudss.Phase.ANALYSIS.value),
            ("factorization", cudss.Phase.FACTORIZATION.value),
            ("solve", cudss.Phase.SOLVE.value),
        ):
            try:
                cudss.execute(handle, phase, config, data, matrix_desc, solution_desc, rhs_desc)
            except Exception as exc:
                failure_memory_estimates: dict[str, int] = {}
                try:
                    failure_memory_estimates = _nvmath_cudss_memory_estimates(cudss, handle, data)
                except Exception:
                    pass
                memory_message = (
                    f"; cuDSS memory estimates (bytes): {json_safe_mapping(failure_memory_estimates)}"
                    if failure_memory_estimates
                    else ""
                )
                raise RuntimeError(
                    f"nvmath_cudss failed during {phase_name} phase on CUDA devices "
                    f"{tuple(device_ids)} with dtype={resolved_controls['dtype']}: "
                    f"{type(exc).__name__}: {exc}{memory_message}"
                ) from exc
        torch.cuda.synchronize(primary_device)
        for device_id in device_ids[1:]:
            torch.cuda.synchronize(device_id)
        try:
            memory_estimates = _nvmath_cudss_memory_estimates(cudss, handle, data)
        except Exception as exc:  # pragma: no cover - defensive around cuDSS versions
            memory_estimates_error = f"{type(exc).__name__}: {exc}"
    finally:
        if data is not None:
            cudss.data_destroy(handle, data)
        if config is not None:
            cudss.config_destroy(config)
        if solution_desc is not None:
            cudss.matrix_destroy(solution_desc)
        if rhs_desc is not None:
            cudss.matrix_destroy(rhs_desc)
        if matrix_desc is not None:
            cudss.matrix_destroy(matrix_desc)
        cudss.destroy(handle)
    backend_solve_seconds = perf_counter() - start

    solution = solution_col_major.t()
    if rhs_is_vector:
        solution = solution.squeeze(1)
    solution_array = np.asarray(solution.detach().cpu().numpy(), dtype=float)
    relative_residual = _nvmath_cudss_relative_residual(
        csr_matrix,
        solution_array,
        np.asarray(rhs_array, dtype=float),
    )
    if bool(resolved_controls["check_residual"]) and relative_residual > float(
        resolved_controls["residual_rtol"]
    ):
        raise RuntimeError(
            "nvmath_cudss residual check failed: "
            f"relative_residual={relative_residual:.3e}, "
            f"residual_rtol={float(resolved_controls['residual_rtol']):.3e}. "
            "Use float64 and matching/iterative-refinement controls, or choose a "
            "CPU/PETSc direct reference backend."
        )
    return solution_array, {
        "serial_sparse_nvmath_cudss_requested_controls": json_safe_mapping(requested_controls),
        "serial_sparse_nvmath_cudss_resolved_controls": json_safe_mapping(resolved_controls),
        "serial_sparse_nvmath_cudss_dtype": str(resolved_controls["dtype"]),
        "serial_sparse_nvmath_cudss_device_ids": tuple(device_ids),
        "serial_sparse_nvmath_cudss_device_names": tuple(
            str(torch.cuda.get_device_name(device_id)) for device_id in device_ids
        ),
        "serial_sparse_nvmath_cudss_backend_seconds": backend_solve_seconds,
        "serial_sparse_nvmath_cudss_relative_residual": relative_residual,
        "serial_sparse_nvmath_cudss_max_memory_allocated_bytes": tuple(
            int(torch.cuda.max_memory_allocated(device_id)) for device_id in device_ids
        ),
        "serial_sparse_nvmath_cudss_primary_max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(primary_device)
        ),
        "serial_sparse_nvmath_cudss_memory_estimates": json_safe_mapping(memory_estimates),
        "serial_sparse_nvmath_cudss_memory_estimates_error": memory_estimates_error,
        "serial_sparse_nvmath_cudss_torch_version": str(getattr(torch, "__version__", "")),
        "serial_sparse_nvmath_cudss_torch_cuda_version": str(
            getattr(getattr(torch, "version", None), "cuda", "")
        ),
    }


__all__ = [
    "NVMATH_CUDSS_CONTROL_KEYS",
    "NVMATH_CUDSS_DTYPES",
    "NVMATH_CUDSS_PIVOT_TYPES",
    "json_safe_mapping",
    "normalize_nvmath_cudss_device_ids",
    "nvmath_cudss_controls_from_arguments",
    "nvmath_cudss_device_ids",
    "require_nvmath_cudss",
    "resolve_nvmath_cudss_controls",
    "solve_nvmath_cudss",
]
