from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

NVMATH_CUDSS_CONTROL_KEYS = {
    "check_residual",
    "deterministic_mode",
    "device_ids",
    "dtype",
    "factorization_alg",
    "host_nthreads",
    "hybrid_device_memory_limit",
    "hybrid_execute_mode",
    "hybrid_mode",
    "ir_steps",
    "matching_alg",
    "nd_nlevels",
    "pivot_epsilon",
    "pivot_epsilon_alg",
    "pivot_threshold",
    "pivot_type",
    "reordering_alg",
    "residual_rtol",
    "solve_alg",
    "threading_lib",
    "use_cuda_register_memory",
    "use_superpanels",
    "use_matching",
    "value_dtype",
}
NVMATH_CUDSS_DTYPES = {"float32", "float64"}
NVMATH_CUDSS_PIVOT_TYPES = {"col", "row", "none"}
NVMATH_CUDSS_ALGORITHMS = {
    "default": 0,
    "alg_default": 0,
    "alg0": 0,
    "alg_0": 0,
    "0": 0,
    "alg1": 1,
    "alg_1": 1,
    "1": 1,
    "alg2": 2,
    "alg_2": 2,
    "2": 2,
    "alg3": 3,
    "alg_3": 3,
    "3": 3,
    "alg4": 4,
    "alg_4": 4,
    "4": 4,
    "alg5": 5,
    "alg_5": 5,
    "5": 5,
}


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


def _resolve_nvmath_cudss_algorithm(value: Any, *, control_name: str) -> int:
    if isinstance(value, int):
        algorithm = int(value)
    else:
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized not in NVMATH_CUDSS_ALGORITHMS:
            supported = ", ".join(sorted(NVMATH_CUDSS_ALGORITHMS))
            raise ValueError(f"nvmath_cudss {control_name} must be one of: {supported}")
        algorithm = NVMATH_CUDSS_ALGORITHMS[normalized]
    if algorithm < 0 or algorithm > 5:
        raise ValueError(f"nvmath_cudss {control_name} must be in the range 0..5")
    return algorithm


def _resolve_finite_float(value: Any, *, control_name: str, positive: bool = False) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0 or (positive and resolved == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"nvmath_cudss {control_name} must be {qualifier} and finite")
    return resolved


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
        elif normalized_key in {
            "reordering_alg",
            "matching_alg",
            "factorization_alg",
            "solve_alg",
            "pivot_epsilon_alg",
        }:
            resolved[normalized_key] = _resolve_nvmath_cudss_algorithm(
                value,
                control_name=normalized_key,
            )
        elif normalized_key == "pivot_type":
            if value is None:
                resolved.pop("pivot_type", None)
            else:
                pivot_type = str(value).strip().lower()
                if pivot_type not in NVMATH_CUDSS_PIVOT_TYPES:
                    supported = ", ".join(sorted(NVMATH_CUDSS_PIVOT_TYPES))
                    raise ValueError(f"nvmath_cudss pivot_type must be one of: {supported}")
                resolved["pivot_type"] = pivot_type
        elif normalized_key == "pivot_threshold":
            resolved["pivot_threshold"] = _resolve_finite_float(
                value,
                control_name="pivot_threshold",
            )
        elif normalized_key == "pivot_epsilon":
            resolved["pivot_epsilon"] = _resolve_finite_float(
                value,
                control_name="pivot_epsilon",
            )
        elif normalized_key == "nd_nlevels":
            nd_nlevels = int(value)
            if nd_nlevels < 0:
                raise ValueError("nvmath_cudss nd_nlevels must be non-negative")
            resolved["nd_nlevels"] = nd_nlevels
        elif normalized_key == "host_nthreads":
            host_nthreads = int(value)
            if host_nthreads <= 0:
                raise ValueError("nvmath_cudss host_nthreads must be positive")
            resolved["host_nthreads"] = host_nthreads
        elif normalized_key == "threading_lib":
            if value is None:
                resolved.pop("threading_lib", None)
            else:
                threading_lib = str(value).strip()
                if not threading_lib:
                    raise ValueError("nvmath_cudss threading_lib must be a path or 'auto'")
                resolved["threading_lib"] = threading_lib
        elif normalized_key == "hybrid_device_memory_limit":
            memory_limit = int(value)
            if memory_limit <= 0:
                raise ValueError("nvmath_cudss hybrid_device_memory_limit must be positive")
            resolved["hybrid_device_memory_limit"] = memory_limit
        elif normalized_key in {
            "hybrid_mode",
            "hybrid_execute_mode",
            "use_cuda_register_memory",
        }:
            resolved[normalized_key] = bool(value)
        elif normalized_key == "use_superpanels":
            resolved["use_superpanels"] = bool(value)
        elif normalized_key == "deterministic_mode":
            resolved["deterministic_mode"] = bool(value)
        elif normalized_key == "check_residual":
            resolved["check_residual"] = bool(value)
        elif normalized_key == "residual_rtol":
            residual_rtol = float(value)
            if residual_rtol <= 0.0 or not np.isfinite(residual_rtol):
                raise ValueError("nvmath_cudss residual_rtol must be positive and finite")
            resolved["residual_rtol"] = residual_rtol
    if "residual_rtol" not in resolved:
        resolved["residual_rtol"] = 1.0e-8 if resolved["dtype"] == "float64" else 1.0e-4
    if bool(resolved.get("hybrid_mode", False)) and bool(
        resolved.get("hybrid_execute_mode", False)
    ):
        raise ValueError(
            "nvmath_cudss hybrid_mode and hybrid_execute_mode cannot both be enabled "
            "in the tested cuDSS runtime"
        )
    return resolved


def nvmath_cudss_controls_from_arguments(
    *,
    dtype: Literal["float32", "float64"] = "float64",
    device_ids: int | Sequence[int] | Literal["all"] | None = None,
    ir_steps: int = 5,
    use_matching: bool = True,
    reordering_alg: str | int | None = None,
    matching_alg: str | int | None = None,
    factorization_alg: str | int | None = None,
    solve_alg: str | int | None = None,
    pivot_type: Literal["col", "row", "none"] | None = None,
    pivot_threshold: float | None = None,
    pivot_epsilon: float | None = None,
    pivot_epsilon_alg: str | int | None = None,
    nd_nlevels: int | None = None,
    host_nthreads: int | None = None,
    threading_lib: str | None = None,
    hybrid_mode: bool | None = None,
    hybrid_device_memory_limit: int | None = None,
    hybrid_execute_mode: bool | None = None,
    use_cuda_register_memory: bool | None = None,
    use_superpanels: bool | None = None,
    deterministic_mode: bool | None = None,
    check_residual: bool = True,
    residual_rtol: float | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build validated low-level controls for the optional nvmath/cuDSS backend.

    This helper merges explicit keyword arguments with an optional existing
    control mapping and then calls :func:`resolve_nvmath_cudss_controls`.
    The returned dictionary is normalized to the names and value types consumed
    by :func:`solve_nvmath_cudss`; it is safe to store in solver metadata or
    pass to the shared sparse solver. These controls affect only the numerical
    linear solve. They do not change the physical model, boundary conditions,
    permeability, porosity, viscosity, pressure drop, or nondimensionalization.

    Parameters
    ----------
    dtype :
        Floating-point value precision for cuDSS matrix values, right-hand side,
        and solution. Supported values are ``"float64"`` and ``"float32"``.
        The default is ``"float64"``. Single precision should be accepted only
        with the residual check enabled and a same-problem reference comparison.
    device_ids :
        CUDA device selection. ``None`` leaves device choice to PyTorch's
        current CUDA device at solve time. An integer selects one GPU, a
        sequence such as ``(0, 1)`` requests a single-node multi-GPU cuDSS
        handle, and ``"all"`` requests all CUDA devices visible to PyTorch.
    ir_steps :
        cuDSS iterative-refinement step count
        (``ConfigParam.IR_N_STEPS``). ``voids`` sets this to ``5`` by default;
        a fresh cuDSS config in the tested nvmath/cuDSS stack reports
        ``IR_N_STEPS = 0``. Larger values can improve lower-precision residuals
        on some systems, but the effect is not guaranteed to be monotonic.
    use_matching :
        Whether to enable cuDSS matching/scaling
        (``ConfigParam.USE_MATCHING``). Matching is enabled by default because
        it can reduce pivot perturbations for general sparse matrices.
    reordering_alg, matching_alg, factorization_alg, solve_alg, pivot_epsilon_alg :
        Optional cuDSS algorithm selectors for
        ``REORDERING_ALG``, ``MATCHING_ALG``, ``FACTORIZATION_ALG``,
        ``SOLVE_ALG``, and ``PIVOT_EPSILON_ALG``. Values may be integers
        ``0`` through ``5`` or strings such as ``"default"``, ``"alg_1"``, or
        ``"3"``. Support and exact meaning are defined by the installed cuDSS
        version.
    pivot_type :
        Optional pivoting mode (``ConfigParam.PIVOT_TYPE``): ``"col"``,
        ``"row"``, or ``"none"``. ``None`` leaves the cuDSS default unchanged.
    pivot_threshold :
        Optional non-negative pivoting threshold
        (``ConfigParam.PIVOT_THRESHOLD``).
    pivot_epsilon :
        Optional non-negative pivot perturbation/floor
        (``ConfigParam.PIVOT_EPSILON``). This can help stabilize very small
        pivots in ill-conditioned single-precision systems, but should be
        treated as a solver-stabilization experiment rather than a physics
        change.
    nd_nlevels :
        Optional non-negative nested-dissection level control
        (``ConfigParam.ND_NLEVELS``) for cuDSS reordering algorithms that support
        it.
    host_nthreads :
        Optional positive host-thread count for cuDSS host-side work
        (``ConfigParam.HOST_NTHREADS``). This affects execution only when a
        cuDSS threading-layer library is loaded.
    threading_lib :
        Optional path to a cuDSS threading-layer library. Use ``"auto"`` to
        request the packaged ``libcudss_mtlayer_gomp`` library when available.
        If host threading is requested and this is omitted, ``voids`` uses
        ``CUDSS_THREADING_LIB`` when set and otherwise auto-loads the packaged
        threading layer when ``host_nthreads`` or ``hybrid_execute_mode=True`` is
        requested.
    hybrid_mode :
        Optional request to enable cuDSS hybrid memory mode
        (``ConfigParam.HYBRID_MODE``). This must be applied before the analysis
        phase and can reduce required device memory by using host memory.
    hybrid_device_memory_limit :
        Optional positive device-memory limit in bytes
        (``ConfigParam.HYBRID_DEVICE_MEMORY_LIMIT``). ``voids`` applies this
        after analysis and before factorization, following cuDSS' phase
        ordering for manual hybrid-memory control. Multi-GPU hybrid-memory
        limit handling is runtime dependent in the low-level nvmath binding;
        compare against a small same-configuration probe before relying on it.
    hybrid_execute_mode :
        Optional request to enable cuDSS hybrid execute mode
        (``ConfigParam.HYBRID_EXECUTE_MODE``). This must be applied before the
        analysis phase and is runtime/backend dependent.
    use_cuda_register_memory :
        Optional request to register host memory with CUDA
        (``ConfigParam.USE_CUDA_REGISTER_MEMORY``) for hybrid-memory execution.
    use_superpanels :
        Optional flag for cuDSS superpanel optimization
        (``ConfigParam.USE_SUPERPANELS``). ``None`` leaves the cuDSS default
        unchanged.
    deterministic_mode :
        Optional request for deterministic cuDSS execution
        (``ConfigParam.DETERMINISTIC_MODE``). Support is runtime/backend
        dependent.
    check_residual :
        Whether :func:`solve_nvmath_cudss` should verify the assembled-system
        relative residual after cuDSS returns. Keep this enabled for lower
        precision and high-contrast systems.
    residual_rtol :
        Relative residual tolerance used when ``check_residual`` is enabled.
        If omitted, ``voids`` uses ``1.0e-8`` for ``float64`` and ``1.0e-4`` for
        ``float32``.
    controls :
        Optional base mapping of cuDSS controls. Control keys are normalized by
        stripping whitespace, lowercasing, and replacing hyphens with
        underscores. The always-present keyword arguments ``dtype``,
        ``ir_steps``, ``use_matching``, and ``check_residual`` override any same
        keys in this mapping. Optional keyword arguments override matching keys
        only when they are not ``None``.

    Returns
    -------
    dict[str, Any]
        Validated controls with normalized keys. Algorithm selectors are stored
        as cuDSS integer algorithm ids, device ids are normalized to a tuple or
        ``"all"``, and a default ``residual_rtol`` is inserted when omitted.

    Raises
    ------
    ValueError
        If a control name is unsupported or a control value is outside the
        accepted range.
    """

    nvmath_cudss_controls: dict[str, Any] = dict(controls or {})
    nvmath_cudss_controls["dtype"] = dtype
    nvmath_cudss_controls["ir_steps"] = int(ir_steps)
    nvmath_cudss_controls["use_matching"] = bool(use_matching)
    nvmath_cudss_controls["check_residual"] = bool(check_residual)
    if device_ids is not None:
        nvmath_cudss_controls["device_ids"] = device_ids
    if reordering_alg is not None:
        nvmath_cudss_controls["reordering_alg"] = reordering_alg
    if matching_alg is not None:
        nvmath_cudss_controls["matching_alg"] = matching_alg
    if factorization_alg is not None:
        nvmath_cudss_controls["factorization_alg"] = factorization_alg
    if solve_alg is not None:
        nvmath_cudss_controls["solve_alg"] = solve_alg
    if pivot_type is not None:
        nvmath_cudss_controls["pivot_type"] = pivot_type
    if pivot_threshold is not None:
        nvmath_cudss_controls["pivot_threshold"] = pivot_threshold
    if pivot_epsilon is not None:
        nvmath_cudss_controls["pivot_epsilon"] = pivot_epsilon
    if pivot_epsilon_alg is not None:
        nvmath_cudss_controls["pivot_epsilon_alg"] = pivot_epsilon_alg
    if nd_nlevels is not None:
        nvmath_cudss_controls["nd_nlevels"] = int(nd_nlevels)
    if host_nthreads is not None:
        nvmath_cudss_controls["host_nthreads"] = int(host_nthreads)
    if threading_lib is not None:
        nvmath_cudss_controls["threading_lib"] = threading_lib
    if hybrid_mode is not None:
        nvmath_cudss_controls["hybrid_mode"] = bool(hybrid_mode)
    if hybrid_device_memory_limit is not None:
        nvmath_cudss_controls["hybrid_device_memory_limit"] = int(hybrid_device_memory_limit)
    if hybrid_execute_mode is not None:
        nvmath_cudss_controls["hybrid_execute_mode"] = bool(hybrid_execute_mode)
    if use_cuda_register_memory is not None:
        nvmath_cudss_controls["use_cuda_register_memory"] = bool(use_cuda_register_memory)
    if use_superpanels is not None:
        nvmath_cudss_controls["use_superpanels"] = bool(use_superpanels)
    if deterministic_mode is not None:
        nvmath_cudss_controls["deterministic_mode"] = bool(deterministic_mode)
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


def _find_nvmath_cudss_threading_lib(cudss: Any) -> str | None:
    module_file = getattr(cudss, "__file__", None)
    candidates: list[Path] = []
    if module_file is not None:
        for parent in Path(module_file).resolve().parents:
            candidates.extend(sorted(parent.glob("nvidia/cu*/lib/libcudss_mtlayer_gomp.so*")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_nvmath_cudss_threading_lib(
    cudss: Any,
    controls: Mapping[str, Any],
) -> str | None:
    requested = controls.get("threading_lib")
    needs_threading = (
        requested is not None
        or "host_nthreads" in controls
        or bool(controls.get("hybrid_execute_mode", False))
    )
    if not needs_threading:
        return None
    if requested is not None:
        requested_text = str(requested).strip()
        if requested_text.lower() not in {"auto", "default"}:
            requested_path = Path(requested_text).expanduser()
            if not requested_path.is_file():
                raise RuntimeError(
                    "nvmath_cudss threading_lib points to a missing cuDSS threading "
                    f"layer library: {requested_text!r}"
                )
            return str(requested_path)
    else:
        env_path = os.getenv("CUDSS_THREADING_LIB")
        if env_path:
            requested_path = Path(env_path).expanduser()
            if requested_path.is_file():
                return str(requested_path)

    discovered = _find_nvmath_cudss_threading_lib(cudss)
    if discovered is None:
        raise RuntimeError(
            "nvmath_cudss host threading was requested, but no cuDSS threading "
            "layer library was found. Pass threading_lib=<path> or set "
            "CUDSS_THREADING_LIB."
        )
    return discovered


def _set_nvmath_cudss_threading_layer(
    cudss: Any,
    handle: Any,
    controls: Mapping[str, Any],
) -> str | None:
    threading_lib = _resolve_nvmath_cudss_threading_lib(cudss, controls)
    if threading_lib is not None:
        cudss.set_threading_layer(handle, threading_lib)
    return threading_lib


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
    names = _NVMATH_CUDSS_MEMORY_ESTIMATES_DTYPE.names or ()
    return {name: int(getattr(record, name)) for name in names if name != "reserved"}


def _validate_nvmath_cudss_runtime_controls(
    controls: Mapping[str, Any],
    device_ids: Sequence[int],
) -> None:
    if len(device_ids) <= 1:
        return
    if (
        "host_nthreads" in controls
        or "threading_lib" in controls
        or bool(controls.get("hybrid_execute_mode", False))
    ):
        raise RuntimeError(
            "nvmath_cudss host threading controls are not supported with the "
            "multi-GPU cuDSS handle in the tested nvmath/cuDSS runtime. Omit "
            "host_nthreads/threading_lib for multi-GPU runs, or select a single "
            "CUDA device."
        )


class NvmathCudssFactor:
    """Reusable cuDSS sparse factorization for repeated single-RHS solves.

    The direct :func:`solve_nvmath_cudss` helper creates and destroys a cuDSS
    handle for one linear solve. This class keeps the cuDSS analysis and
    factorization data alive so iterative methods can reuse the same sparse
    factor for many right-hand sides. It is intended for internal solver
    preconditioners and advanced benchmarks; the matrix structure and values are
    fixed for the lifetime of the object.
    """

    def __init__(
        self,
        matrix: sparse.spmatrix,
        *,
        controls: Mapping[str, Any] | None = None,
    ) -> None:
        self.torch, self.cudss = require_nvmath_cudss()
        requested_controls = dict(controls or {})
        self.resolved_controls = resolve_nvmath_cudss_controls(requested_controls)
        self.device_ids = nvmath_cudss_device_ids(
            self.torch,
            cast(tuple[int, ...] | Literal["all"] | None, self.resolved_controls.get("device_ids")),
        )
        _validate_nvmath_cudss_runtime_controls(self.resolved_controls, self.device_ids)
        self.primary_device = int(self.device_ids[0])
        self.torch.cuda.set_device(self.primary_device)
        self.torch_device = self.torch.device(f"cuda:{self.primary_device}")
        self.torch_dtype = (
            self.torch.float64
            if self.resolved_controls["dtype"] == "float64"
            else self.torch.float32
        )
        self.numpy_dtype = (
            np.float64 if self.resolved_controls["dtype"] == "float64" else np.float32
        )
        self.value_type = 1 if self.resolved_controls["dtype"] == "float64" else 0
        self.rows = 0
        self.nnz = 0
        self.analysis_seconds = 0.0
        self.factorization_seconds = 0.0
        self.solve_calls = 0
        self.solve_seconds = 0.0
        self.memory_estimates: dict[str, int] = {}
        self.memory_estimates_error = ""
        self.threading_lib: str | None = None
        self._closed = False
        self._device_indices_array: np.ndarray | None = None
        self.handle = None
        self.matrix_desc = None
        self.rhs_desc = None
        self.solution_desc = None
        self.config = None
        self.data = None

        csr_matrix = matrix.tocsr()
        rows, cols = csr_matrix.shape
        if rows != cols:
            raise ValueError("NvmathCudssFactor requires a square sparse matrix")
        int32_max = np.iinfo(np.int32).max
        if rows > int32_max or cols > int32_max or int(csr_matrix.nnz) > int32_max:
            raise ValueError("NvmathCudssFactor currently requires 32-bit CSR indices")
        self.rows = int(rows)
        self.nnz = int(csr_matrix.nnz)

        self.row_offsets = self.torch.as_tensor(
            np.ascontiguousarray(csr_matrix.indptr, dtype=np.int32),
            device=self.torch_device,
        )
        self.col_indices = self.torch.as_tensor(
            np.ascontiguousarray(csr_matrix.indices, dtype=np.int32),
            device=self.torch_device,
        )
        self.values = self.torch.as_tensor(
            np.ascontiguousarray(csr_matrix.data, dtype=self.numpy_dtype),
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        self.rhs_col_major = self.torch.zeros(
            (1, self.rows),
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        self.solution_col_major = self.torch.zeros_like(self.rhs_col_major)

        for device_id in self.device_ids:
            self.torch.cuda.reset_peak_memory_stats(device_id)

        try:
            self._create_descriptors()
            self._apply_controls()
            self._factorize()
        except Exception:
            self.close()
            raise

    def _create_descriptors(self) -> None:
        cudss = self.cudss
        if len(self.device_ids) == 1:
            self.handle = cudss.create()
        else:
            self._device_indices_array = np.asarray(self.device_ids, dtype=np.int32)
            self.handle = cudss.create_mg(
                len(self.device_ids),
                self._device_indices_array.ctypes.data,
            )
        self.threading_lib = _set_nvmath_cudss_threading_layer(
            cudss,
            self.handle,
            self.resolved_controls,
        )
        cudss.set_stream(
            self.handle,
            self.torch.cuda.current_stream(self.primary_device).cuda_stream,
        )
        self.matrix_desc = cudss.matrix_create_csr(
            self.rows,
            self.rows,
            self.nnz,
            self.row_offsets.data_ptr(),
            0,
            self.col_indices.data_ptr(),
            self.values.data_ptr(),
            10,  # CUDA_R_32I
            self.value_type,
            cudss.MatrixType.GENERAL.value,
            cudss.MatrixViewType.FULL.value,
            cudss.IndexBase.ZERO.value,
        )
        self.rhs_desc = cudss.matrix_create_dn(
            self.rows,
            1,
            self.rows,
            self.rhs_col_major.data_ptr(),
            self.value_type,
            cudss.Layout.COL_MAJOR.value,
        )
        self.solution_desc = cudss.matrix_create_dn(
            self.rows,
            1,
            self.rows,
            self.solution_col_major.data_ptr(),
            self.value_type,
            cudss.Layout.COL_MAJOR.value,
        )
        self.config = cudss.config_create()
        self.data = cudss.data_create(self.handle)

    def _apply_controls(self) -> None:
        cudss = self.cudss
        controls = self.resolved_controls
        pivot_type_map = {
            "col": cudss.PivotType.PIVOT_COL,
            "row": cudss.PivotType.PIVOT_ROW,
            "none": cudss.PivotType.PIVOT_NONE,
        }
        if len(self.device_ids) > 1:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.DEVICE_COUNT,
                len(self.device_ids),
            )
            _set_nvmath_cudss_config_array(
                cudss,
                self.config,
                cudss.ConfigParam.DEVICE_INDICES,
                self.device_ids,
            )
        if "reordering_alg" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.REORDERING_ALG,
                int(controls["reordering_alg"]),
            )
        if "matching_alg" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.MATCHING_ALG,
                int(controls["matching_alg"]),
            )
        if "factorization_alg" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.FACTORIZATION_ALG,
                int(controls["factorization_alg"]),
            )
        if "solve_alg" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.SOLVE_ALG,
                int(controls["solve_alg"]),
            )
        _set_nvmath_cudss_config_scalar(
            cudss,
            self.config,
            cudss.ConfigParam.IR_N_STEPS,
            int(controls["ir_steps"]),
        )
        _set_nvmath_cudss_config_scalar(
            cudss,
            self.config,
            cudss.ConfigParam.USE_MATCHING,
            int(bool(controls["use_matching"])),
        )
        if "pivot_type" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.PIVOT_TYPE,
                pivot_type_map[str(controls["pivot_type"])].value,
            )
        if "pivot_threshold" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.PIVOT_THRESHOLD,
                float(controls["pivot_threshold"]),
            )
        if "pivot_epsilon" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.PIVOT_EPSILON,
                float(controls["pivot_epsilon"]),
            )
        if "pivot_epsilon_alg" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.PIVOT_EPSILON_ALG,
                int(controls["pivot_epsilon_alg"]),
            )
        if "nd_nlevels" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.ND_NLEVELS,
                int(controls["nd_nlevels"]),
            )
        if "host_nthreads" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.HOST_NTHREADS,
                int(controls["host_nthreads"]),
            )
        if "hybrid_mode" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.HYBRID_MODE,
                int(bool(controls["hybrid_mode"])),
            )
        if "hybrid_execute_mode" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.HYBRID_EXECUTE_MODE,
                int(bool(controls["hybrid_execute_mode"])),
            )
        if "use_cuda_register_memory" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.USE_CUDA_REGISTER_MEMORY,
                int(bool(controls["use_cuda_register_memory"])),
            )
        if "use_superpanels" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.USE_SUPERPANELS,
                int(bool(controls["use_superpanels"])),
            )
        if "deterministic_mode" in controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                self.config,
                cudss.ConfigParam.DETERMINISTIC_MODE,
                int(bool(controls["deterministic_mode"])),
            )

    def _sync_devices(self) -> None:
        self.torch.cuda.synchronize(self.primary_device)
        for device_id in self.device_ids[1:]:
            self.torch.cuda.synchronize(device_id)

    def _execute_phase(self, phase_name: str, phase: int) -> None:
        try:
            self.cudss.execute(
                self.handle,
                phase,
                self.config,
                self.data,
                self.matrix_desc,
                self.solution_desc,
                self.rhs_desc,
            )
            self._sync_devices()
        except Exception as exc:
            failure_memory_estimates: dict[str, int] = {}
            try:
                failure_memory_estimates = _nvmath_cudss_memory_estimates(
                    self.cudss,
                    self.handle,
                    self.data,
                )
            except Exception:
                pass
            memory_message = (
                f"; cuDSS memory estimates (bytes): {json_safe_mapping(failure_memory_estimates)}"
                if failure_memory_estimates
                else ""
            )
            raise RuntimeError(
                f"NvmathCudssFactor failed during {phase_name} phase on CUDA devices "
                f"{tuple(self.device_ids)} with dtype={self.resolved_controls['dtype']}: "
                f"{type(exc).__name__}: {exc}{memory_message}"
            ) from exc

    def _factorize(self) -> None:
        start = perf_counter()
        self._execute_phase("analysis", self.cudss.Phase.ANALYSIS.value)
        self.analysis_seconds = perf_counter() - start
        if "hybrid_device_memory_limit" in self.resolved_controls:
            _set_nvmath_cudss_config_scalar(
                self.cudss,
                self.config,
                self.cudss.ConfigParam.HYBRID_DEVICE_MEMORY_LIMIT,
                int(self.resolved_controls["hybrid_device_memory_limit"]),
            )
        start = perf_counter()
        self._execute_phase("factorization", self.cudss.Phase.FACTORIZATION.value)
        self.factorization_seconds = perf_counter() - start
        try:
            self.memory_estimates = _nvmath_cudss_memory_estimates(
                self.cudss,
                self.handle,
                self.data,
            )
        except Exception as exc:  # pragma: no cover - defensive around cuDSS versions
            self.memory_estimates_error = f"{type(exc).__name__}: {exc}"

    def solve(self, rhs_array: np.ndarray) -> np.ndarray:
        """Solve the factored sparse system for one right-hand-side vector."""

        if self._closed:
            raise RuntimeError("NvmathCudssFactor is closed")
        rhs = np.asarray(rhs_array, dtype=self.numpy_dtype)
        if rhs.shape != (self.rows,):
            raise ValueError(f"expected vector with shape {(self.rows,)}, got {rhs.shape}")
        start = perf_counter()
        rhs_tensor = self.torch.as_tensor(
            np.ascontiguousarray(rhs),
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        self.rhs_col_major.view(-1).copy_(rhs_tensor)
        self.solution_col_major.zero_()
        self._execute_phase("solve", self.cudss.Phase.SOLVE.value)
        self.solve_seconds += perf_counter() - start
        self.solve_calls += 1
        return np.asarray(self.solution_col_major.view(-1).detach().cpu().numpy(), dtype=float)

    def metadata(self) -> dict[str, Any]:
        """Return diagnostic metadata for the reusable factorization."""

        return {
            "nvmath_cudss_factor_dtype": str(self.resolved_controls["dtype"]),
            "nvmath_cudss_factor_device_ids": tuple(self.device_ids),
            "nvmath_cudss_factor_device_names": tuple(
                str(self.torch.cuda.get_device_name(device_id)) for device_id in self.device_ids
            ),
            "nvmath_cudss_factor_resolved_controls": json_safe_mapping(self.resolved_controls),
            "nvmath_cudss_factor_threading_lib": self.threading_lib or "",
            "nvmath_cudss_factor_analysis_seconds": self.analysis_seconds,
            "nvmath_cudss_factor_factorization_seconds": self.factorization_seconds,
            "nvmath_cudss_factor_solve_calls": self.solve_calls,
            "nvmath_cudss_factor_solve_seconds": self.solve_seconds,
            "nvmath_cudss_factor_solve_seconds_per_call": (
                self.solve_seconds / self.solve_calls if self.solve_calls else np.nan
            ),
            "nvmath_cudss_factor_max_memory_allocated_bytes": tuple(
                int(self.torch.cuda.max_memory_allocated(device_id))
                for device_id in self.device_ids
            ),
            "nvmath_cudss_factor_primary_max_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.primary_device)
            ),
            "nvmath_cudss_factor_memory_estimates": json_safe_mapping(self.memory_estimates),
            "nvmath_cudss_factor_memory_estimates_error": self.memory_estimates_error,
            "nvmath_cudss_factor_torch_version": str(getattr(self.torch, "__version__", "")),
            "nvmath_cudss_factor_torch_cuda_version": str(
                getattr(getattr(self.torch, "version", None), "cuda", "")
            ),
        }

    def close(self) -> None:
        """Release cuDSS descriptors, config, data, and handle."""

        if self._closed:
            return
        if self.data is not None:
            self.cudss.data_destroy(self.handle, self.data)
            self.data = None
        if self.config is not None:
            self.cudss.config_destroy(self.config)
            self.config = None
        if self.solution_desc is not None:
            self.cudss.matrix_destroy(self.solution_desc)
            self.solution_desc = None
        if self.rhs_desc is not None:
            self.cudss.matrix_destroy(self.rhs_desc)
            self.rhs_desc = None
        if self.matrix_desc is not None:
            self.cudss.matrix_destroy(self.matrix_desc)
            self.matrix_desc = None
        if self.handle is not None:
            self.cudss.destroy(self.handle)
            self.handle = None
        self._closed = True

    def __enter__(self) -> NvmathCudssFactor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


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
    _validate_nvmath_cudss_runtime_controls(resolved_controls, device_ids)
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
    threading_lib: str | None = None
    matrix_desc = rhs_desc = solution_desc = config = data = None
    for device_id in device_ids:
        torch.cuda.reset_peak_memory_stats(device_id)
    memory_estimates: dict[str, int] = {}
    memory_estimates_error = ""
    start = perf_counter()
    try:
        threading_lib = _set_nvmath_cudss_threading_layer(cudss, handle, resolved_controls)
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
        if "reordering_alg" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.REORDERING_ALG,
                int(resolved_controls["reordering_alg"]),
            )
        if "matching_alg" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.MATCHING_ALG,
                int(resolved_controls["matching_alg"]),
            )
        if "factorization_alg" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.FACTORIZATION_ALG,
                int(resolved_controls["factorization_alg"]),
            )
        if "solve_alg" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.SOLVE_ALG,
                int(resolved_controls["solve_alg"]),
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
        if "pivot_threshold" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.PIVOT_THRESHOLD,
                float(resolved_controls["pivot_threshold"]),
            )
        if "pivot_epsilon" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.PIVOT_EPSILON,
                float(resolved_controls["pivot_epsilon"]),
            )
        if "pivot_epsilon_alg" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.PIVOT_EPSILON_ALG,
                int(resolved_controls["pivot_epsilon_alg"]),
            )
        if "nd_nlevels" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.ND_NLEVELS,
                int(resolved_controls["nd_nlevels"]),
            )
        if "host_nthreads" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.HOST_NTHREADS,
                int(resolved_controls["host_nthreads"]),
            )
        if "hybrid_mode" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.HYBRID_MODE,
                int(bool(resolved_controls["hybrid_mode"])),
            )
        if "hybrid_execute_mode" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.HYBRID_EXECUTE_MODE,
                int(bool(resolved_controls["hybrid_execute_mode"])),
            )
        if "use_cuda_register_memory" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.USE_CUDA_REGISTER_MEMORY,
                int(bool(resolved_controls["use_cuda_register_memory"])),
            )
        if "use_superpanels" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.USE_SUPERPANELS,
                int(bool(resolved_controls["use_superpanels"])),
            )
        if "deterministic_mode" in resolved_controls:
            _set_nvmath_cudss_config_scalar(
                cudss,
                config,
                cudss.ConfigParam.DETERMINISTIC_MODE,
                int(bool(resolved_controls["deterministic_mode"])),
            )
        for phase_name, phase in (
            ("analysis", cudss.Phase.ANALYSIS.value),
            ("factorization", cudss.Phase.FACTORIZATION.value),
            ("solve", cudss.Phase.SOLVE.value),
        ):
            try:
                cudss.execute(handle, phase, config, data, matrix_desc, solution_desc, rhs_desc)
                if phase_name == "analysis" and "hybrid_device_memory_limit" in resolved_controls:
                    _set_nvmath_cudss_config_scalar(
                        cudss,
                        config,
                        cudss.ConfigParam.HYBRID_DEVICE_MEMORY_LIMIT,
                        int(resolved_controls["hybrid_device_memory_limit"]),
                    )
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
    solution_array = np.asarray(solution.detach().cpu().numpy(), dtype=numpy_dtype)
    relative_residual = _nvmath_cudss_relative_residual(
        csr_matrix,
        solution_array,
        np.asarray(rhs_array, dtype=numpy_dtype),
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
        "serial_sparse_nvmath_cudss_threading_lib": threading_lib or "",
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
    "NVMATH_CUDSS_ALGORITHMS",
    "NVMATH_CUDSS_DTYPES",
    "NVMATH_CUDSS_PIVOT_TYPES",
    "NvmathCudssFactor",
    "json_safe_mapping",
    "normalize_nvmath_cudss_device_ids",
    "nvmath_cudss_controls_from_arguments",
    "nvmath_cudss_device_ids",
    "require_nvmath_cudss",
    "resolve_nvmath_cudss_controls",
    "solve_nvmath_cudss",
]
