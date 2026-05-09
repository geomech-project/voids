from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from typing import cast

import numpy as np
import porespy as ps

from voids.image.connectivity import has_spanning_cluster
from voids.image._utils import normalize_shape, validate_axis_index


@dataclass(slots=True)
class MacroMicroPorousImage:
    """Synthetic image with resolved macropores and matrix-hosted micropores.

    Attributes
    ----------
    void :
        Combined binary void image. ``True`` denotes void.
    macro_void :
        Resolved/macropore image generated at the larger feature scale.
    micropore_void :
        Small-pore image clipped to the matrix phase of ``macro_void``.
    metadata :
        JSON-serializable generation metadata.

    Notes
    -----
    ``micropore_void`` is always a subset of ``~macro_void``. This keeps the
    two porosity scales interpretable: macropores define fully resolved voids,
    while micropores add small voids inside the matrix region only.
    """

    void: np.ndarray
    macro_void: np.ndarray
    micropore_void: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate mask relationships."""

        void = np.asarray(self.void, dtype=bool)
        macro = np.asarray(self.macro_void, dtype=bool)
        micro = np.asarray(self.micropore_void, dtype=bool)
        normalize_shape(void.shape, allowed_ndim=(2, 3))
        if macro.shape != void.shape or micro.shape != void.shape:
            raise ValueError("void, macro_void, and micropore_void must have the same shape")
        if np.any(micro & macro):
            raise ValueError("micropore_void must be confined to the macro matrix phase")
        if not np.array_equal(void, macro | micro):
            raise ValueError("void must equal macro_void | micropore_void")
        self.void = void
        self.macro_void = macro
        self.micropore_void = micro
        self.metadata = dict(self.metadata)

    @property
    def ndim(self) -> int:
        """Return image dimensionality."""

        return int(self.void.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return image shape."""

        return tuple(int(v) for v in self.void.shape)

    @property
    def porosity(self) -> float:
        """Return total void fraction."""

        return float(np.mean(self.void))

    @property
    def macro_porosity(self) -> float:
        """Return resolved/macropore void fraction."""

        return float(np.mean(self.macro_void))

    @property
    def matrix_microporosity(self) -> float:
        """Return micropore fraction measured only inside the macro matrix."""

        matrix = ~self.macro_void
        if not np.any(matrix):
            return float("nan")
        return float(np.mean(self.micropore_void[matrix]))

    @property
    def total_microporosity(self) -> float:
        """Return micropore fraction measured over the full image support."""

        return float(np.mean(self.micropore_void))


def _coerce_blobiness(
    blobiness: float | Sequence[float],
    *,
    ndim: int,
    name: str,
) -> float | tuple[float, ...]:
    """Normalize blobiness controls to the format expected by PoreSpy."""

    if np.isscalar(blobiness):
        value = float(cast(float, blobiness))
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    values = tuple(float(v) for v in cast(Sequence[float], blobiness))
    if len(values) != ndim:
        raise ValueError(f"{name} must have length {ndim}")
    if min(values) <= 0:
        raise ValueError(f"All entries in {name} must be positive")
    return values


def _matrix_quantile_mask(
    score: np.ndarray,
    matrix_mask: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    """Threshold a score field so a target fraction of the matrix is selected."""

    matrix_values = np.asarray(score, dtype=float)[matrix_mask]
    if matrix_values.size == 0:
        raise ValueError("macro image leaves no matrix voxels for micropores")
    threshold = float(np.quantile(matrix_values, float(fraction)))
    selected = (np.asarray(score, dtype=float) <= threshold) & matrix_mask

    # Quantiles can include ties. Trim deterministically so the matrix fraction
    # matches the requested microporosity as closely as the voxel count allows.
    target_count = int(round(float(fraction) * matrix_values.size))
    current_count = int(np.count_nonzero(selected))
    if current_count > target_count:
        selected_indices = np.flatnonzero(selected)
        selected.flat[selected_indices[target_count:]] = False
    elif current_count < target_count:
        candidates = np.flatnonzero(matrix_mask & ~selected)
        order = np.argsort(np.asarray(score, dtype=float).flat[candidates], kind="mergesort")
        selected.flat[candidates[order[: target_count - current_count]]] = True
    return np.asarray(selected, dtype=bool)


def generate_macro_micro_blobs_matrix(
    *,
    shape: Sequence[int],
    macro_porosity: float,
    matrix_microporosity: float,
    macro_blobiness: float | Sequence[float],
    micropore_blobiness: float | Sequence[float],
    seed_start: int,
    max_tries: int,
    axis_index: int | None = None,
    periodic: bool = True,
) -> MacroMicroPorousImage:
    """Generate PoreSpy blobs with small micropores inside the matrix phase.

    Parameters
    ----------
    shape :
        Image shape in voxels. Supports 2D and 3D.
    macro_porosity :
        Target porosity of the resolved/macropore PoreSpy ``blobs`` image.
    matrix_microporosity :
        Fraction of the macro-matrix phase converted to small micropores.
        The expected total porosity is approximately
        ``macro_porosity + (1 - macro_porosity) * matrix_microporosity``.
    macro_blobiness :
        PoreSpy blobiness control for the resolved/macropore field.
    micropore_blobiness :
        PoreSpy blobiness control for the matrix micropore field. Larger values
        create smaller features in PoreSpy's ``blobs`` generator.
    seed_start, max_tries :
        Seed-stream controls. Trial ``i`` uses seeds ``seed_start + 2*i`` and
        ``seed_start + 2*i + 1`` for macro and micropore fields.
    axis_index :
        Optional axis used to require the combined void image to span. If
        ``None``, no connectivity acceptance is applied.
    periodic :
        Forwarded to ``porespy.generators.blobs`` for both fields.

    Returns
    -------
    MacroMicroPorousImage
        Combined void image plus separate macro and micropore masks.

    Scientific interpretation
    -------------------------
    This is a synthetic two-porosity construction. The micropore image is not an
    experimentally calibrated unresolved-porosity model; it is a controlled way
    to place small pores in the matrix while preserving explicit porosity knobs.
    """

    dims = normalize_shape(shape, allowed_ndim=(2, 3))
    ndim = len(dims)
    if not (0.0 < macro_porosity < 1.0):
        raise ValueError("macro_porosity must be in (0, 1)")
    if not (0.0 <= matrix_microporosity <= 1.0):
        raise ValueError("matrix_microporosity must be in [0, 1]")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    axis = None if axis_index is None else validate_axis_index(axis_index=axis_index, ndim=ndim)
    macro_blob = _coerce_blobiness(macro_blobiness, ndim=ndim, name="macro_blobiness")
    micro_blob = _coerce_blobiness(micropore_blobiness, ndim=ndim, name="micropore_blobiness")

    for i in range(int(max_tries)):
        macro_seed = int(seed_start + 2 * i)
        micro_seed = int(macro_seed + 1)
        macro_void = np.asarray(
            ps.generators.blobs(
                shape=dims,
                porosity=float(macro_porosity),
                blobiness=macro_blob,
                seed=macro_seed,
                periodic=bool(periodic),
            ),
            dtype=bool,
        )
        matrix_mask = ~macro_void
        if matrix_microporosity == 0.0:
            micropore_void = np.zeros(dims, dtype=bool)
        else:
            micropore_score = np.asarray(
                ps.generators.blobs(
                    shape=dims,
                    porosity=None,
                    blobiness=micro_blob,
                    seed=micro_seed,
                    periodic=bool(periodic),
                ),
                dtype=float,
            )
            micropore_void = _matrix_quantile_mask(
                micropore_score,
                matrix_mask,
                fraction=float(matrix_microporosity),
            )

        combined = macro_void | micropore_void
        if axis is not None and not has_spanning_cluster(combined, axis_index=axis):
            continue

        return MacroMicroPorousImage(
            void=combined,
            macro_void=macro_void,
            micropore_void=micropore_void,
            metadata={
                "source_kind": "macro_micro_porespy_blobs",
                "shape": dims,
                "macro_seed": macro_seed,
                "micropore_seed": micro_seed,
                "macro_porosity_target": float(macro_porosity),
                "matrix_microporosity_target": float(matrix_microporosity),
                "macro_blobiness": macro_blob,
                "micropore_blobiness": micro_blob,
                "periodic": bool(periodic),
                "axis_index": axis,
            },
        )

    raise RuntimeError(
        "Could not generate accepted macro/micro blobs matrix for "
        f"seed_start={int(seed_start)} after {int(max_tries)} trials"
    )


def generate_spanning_multiscale_blobs_matrix(
    *,
    shape: Sequence[int],
    porosity: float,
    blobiness_primary: float | Sequence[float],
    blobiness_secondary: float | Sequence[float],
    axis_index: int,
    seed_start: int,
    max_tries: int,
    primary_weight: float = 0.75,
    periodic: bool = True,
) -> tuple[np.ndarray, int]:
    """Generate a spanning binary image from a two-scale PoreSpy blobs field.

    Parameters
    ----------
    shape :
        Image shape in voxels. Supports 2D and 3D.
    porosity :
        Target void fraction in ``(0, 1)``. This is enforced by thresholding the
        combined multiscale field at its porosity quantile, so the achieved
        porosity closely matches the requested value up to voxel discretization.
    blobiness_primary, blobiness_secondary :
        Correlation-length controls for the two component blob fields. Each can
        be either a positive scalar or a length-``ndim`` sequence to create
        anisotropic correlation by axis.
    axis_index :
        Axis used for percolation (spanning) acceptance.
    seed_start :
        Initial random seed. Each trial uses ``(seed_start + 2*i, seed_start + 2*i + 1)``
        for the primary and secondary fields.
    max_tries :
        Maximum number of independent multiscale fields to test.
    primary_weight :
        Convex combination weight for the primary field in ``[0, 1]``.
    periodic :
        Forwarded to ``porespy.generators.blobs``.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(matrix, seed_used)`` where ``matrix`` is boolean with ``True`` as
        void and ``seed_used`` is the primary seed of the accepted realization.

    Notes
    -----
    This helper is inspired by the official PoreSpy multiscale-image workflow:
    two correlated noise fields are generated with ``porosity=None``, blended,
    then thresholded to the target void fraction.

    Scientific interpretation
    -------------------------
    This generator is still synthetic, but it can represent broader
    multi-resolution structure than single-scale blobs while retaining explicit
    porosity control.
    """

    dims = normalize_shape(shape, allowed_ndim=(2, 3))
    axis = validate_axis_index(axis_index=axis_index, ndim=len(dims))
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must be in (0, 1)")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    if not (0.0 <= primary_weight <= 1.0):
        raise ValueError("primary_weight must be in [0, 1]")

    primary = _coerce_blobiness(
        blobiness_primary,
        ndim=len(dims),
        name="blobiness_primary",
    )
    secondary = _coerce_blobiness(
        blobiness_secondary,
        ndim=len(dims),
        name="blobiness_secondary",
    )

    for i in range(max_tries):
        seed_primary = int(seed_start + 2 * i)
        seed_secondary = int(seed_primary + 1)
        field_primary = np.asarray(
            ps.generators.blobs(
                shape=dims,
                porosity=None,
                blobiness=primary,
                seed=seed_primary,
                periodic=bool(periodic),
            ),
            dtype=float,
        )
        field_secondary = np.asarray(
            ps.generators.blobs(
                shape=dims,
                porosity=None,
                blobiness=secondary,
                seed=seed_secondary,
                periodic=bool(periodic),
            ),
            dtype=float,
        )
        score = (
            float(primary_weight) * field_primary + (1.0 - float(primary_weight)) * field_secondary
        )
        threshold = float(np.quantile(score, float(porosity)))
        matrix = np.asarray(score <= threshold, dtype=bool)
        if has_spanning_cluster(matrix, axis_index=axis):
            return matrix, seed_primary

    raise RuntimeError(
        "Could not generate spanning multiscale blobs matrix for "
        f"seed_start={int(seed_start)} after {int(max_tries)} trials"
    )


def generate_spanning_blobs_matrix(
    *,
    shape: Sequence[int],
    porosity: float,
    blobiness: float,
    axis_index: int,
    seed_start: int,
    max_tries: int,
) -> tuple[np.ndarray, int]:
    """Generate a percolating porous matrix using PoreSpy's `blobs` model.

    Parameters
    ----------
    shape :
        Image shape in voxels. Supports 2D and 3D.
    porosity :
        Target void fraction passed to ``porespy.generators.blobs``.
    blobiness :
        Correlation-length control used by PoreSpy; larger values generally
        produce smoother/larger features.
    axis_index :
        Axis used for percolation (spanning) acceptance.
    seed_start :
        Initial random seed. Subsequent attempts use ``seed_start + i``.
    max_tries :
        Maximum number of seeds to test.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(matrix, seed_used)`` where ``matrix`` is boolean with ``True`` as
        void and ``seed_used`` is the accepted seed.

    Raises
    ------
    ValueError
        If input controls are outside valid ranges.
    RuntimeError
        If no percolating realization is found in ``max_tries`` attempts.

    Scientific interpretation
    -------------------------
    The acceptance criterion is topological percolation only. It ensures a
    connected pathway exists, but does not guarantee a target hydraulic
    conductance or morphological realism for specific rock classes.
    """

    dims = normalize_shape(shape, allowed_ndim=(2, 3))
    axis = validate_axis_index(axis_index=axis_index, ndim=len(dims))
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must be in (0, 1)")
    if blobiness <= 0:
        raise ValueError("blobiness must be positive")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")

    for i in range(max_tries):
        seed = int(seed_start + i)
        matrix = np.asarray(
            ps.generators.blobs(
                shape=dims,
                porosity=float(porosity),
                blobiness=float(blobiness),
                seed=seed,
            ),
            dtype=bool,
        )
        if has_spanning_cluster(matrix, axis_index=axis):
            return matrix, seed
    raise RuntimeError(
        f"Could not generate spanning blobs matrix for seed_start={int(seed_start)} "
        f"after {int(max_tries)} trials"
    )


def generate_connected_matrix(
    *,
    shape: tuple[int, int, int],
    porosity: float,
    blobiness: float,
    axis_index: int,
    seed_start: int,
    max_tries: int,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, int]:
    """Backward-compatible wrapper for notebook `08` API signatures.

    Parameters
    ----------
    shape, porosity, blobiness, axis_index, seed_start, max_tries :
        Same physical meaning as :func:`generate_spanning_blobs_matrix`.
    show_progress, progress_desc :
        Retained for compatibility with notebook code. Currently ignored by the
        packaged implementation.

    Returns
    -------
    tuple[numpy.ndarray, int]
        Same as :func:`generate_spanning_blobs_matrix`.

    Notes
    -----
    This wrapper exists to reduce migration cost for notebook scripts while
    preserving deterministic behavior in the core implementation.
    """

    del show_progress, progress_desc
    return generate_spanning_blobs_matrix(
        shape=shape,
        porosity=porosity,
        blobiness=blobiness,
        axis_index=axis_index,
        seed_start=seed_start,
        max_tries=max_tries,
    )


def estimate_voronoi_ncells_for_porosity_2d(
    shape: tuple[int, int],
    porosity: float,
    *,
    intercept: float = 0.080,
    slope: float = 3.22e-4,
    reference_shape: tuple[int, int] = (180, 180),
    min_ncells: int = 40,
) -> int:
    """Estimate Voronoi-cell count needed for a target 2D void fraction.

    Parameters
    ----------
    shape :
        Output image shape ``(nx, ny)``.
    porosity :
        Target void porosity in ``(0, 1)``.
    intercept, slope :
        Empirical linear model coefficients for
        ``phi_void ~= intercept + slope * ncells`` at ``reference_shape``.
    reference_shape :
        Calibration image shape for the linear model.
    min_ncells :
        Lower bound applied to the returned estimate.

    Returns
    -------
    int
        Estimated ``ncells`` for ``porespy.generators.voronoi_edges``.

    Notes
    -----
    The default calibration was measured in notebook `09` for
    ``reference_shape=(180, 180)`` and ``r=0`` with ``void = ~voronoi_edges``.

    Assumptions and caveats
    -----------------------
    The relation is empirical, not universal. Different resolutions, filtering,
    or post-processing can shift the mapping significantly.
    """

    dims = normalize_shape(shape, allowed_ndim=(2,))
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must be in (0, 1)")
    if slope <= 0:
        raise ValueError("slope must be positive")
    if min_ncells < 1:
        raise ValueError("min_ncells must be >= 1")

    ref = normalize_shape(reference_shape, allowed_ndim=(2,))
    area = float(dims[0] * dims[1])
    area_ref = float(ref[0] * ref[1])
    ncells_ref = (float(porosity) - float(intercept)) / float(slope)
    ncells_scaled = ncells_ref * (area / area_ref)
    return int(max(int(min_ncells), round(ncells_scaled)))


def generate_spanning_voronoi_matrix_2d(
    *,
    shape: tuple[int, int],
    porosity: float,
    axis_index: int,
    seed_start: int,
    max_tries: int,
    edge_radius_vox: int = 0,
    target_tol: float = 0.003,
    ncells_step: int = 10,
    search_half_window: int = 70,
    min_ncells: int = 40,
) -> tuple[np.ndarray, int]:
    """Generate a percolating 2D matrix from Voronoi-edge microstructures.

    Parameters
    ----------
    shape :
        Output image shape ``(nx, ny)``.
    porosity :
        Target void porosity.
    axis_index :
        Axis used for percolation acceptance.
    seed_start, max_tries :
        Seed-search controls.
    edge_radius_vox :
        Edge thickening radius passed to ``voronoi_edges(..., r=...)``.
    target_tol :
        Relative acceptance tolerance on porosity mismatch.
    ncells_step, search_half_window :
        Search controls around the estimated ``ncells`` value.
    min_ncells :
        Lower bound for candidate ``ncells``.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(matrix, seed_used)`` with boolean matrix encoded as ``True=void``.

    Raises
    ------
    ValueError
        If controls are invalid.
    RuntimeError
        If no spanning realization is found.

    Scientific interpretation
    -------------------------
    The generator targets low-porosity, edge-dominated connectivity. This is
    useful for sensitivity studies but remains a synthetic morphology model, not
    a physically faithful reconstruction of any specific rock sample.
    """

    dims = normalize_shape(shape, allowed_ndim=(2,))
    dims2 = (int(dims[0]), int(dims[1]))
    axis = validate_axis_index(axis_index=axis_index, ndim=2)
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must be in (0, 1)")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    if edge_radius_vox < 0:
        raise ValueError("edge_radius_vox must be >= 0")
    if target_tol < 0:
        raise ValueError("target_tol must be >= 0")
    if ncells_step < 1:
        raise ValueError("ncells_step must be >= 1")
    if search_half_window < 0:
        raise ValueError("search_half_window must be >= 0")
    if min_ncells < 1:
        raise ValueError("min_ncells must be >= 1")

    ncells_guess = estimate_voronoi_ncells_for_porosity_2d(
        dims2,
        float(porosity),
        min_ncells=min_ncells,
    )
    ncells_candidates = sorted(
        {
            max(int(min_ncells), ncells_guess + delta)
            for delta in range(-search_half_window, search_half_window + 1, ncells_step)
        }
    )

    best_matrix: np.ndarray | None = None
    best_seed: int | None = None
    best_error = np.inf
    for i in range(max_tries):
        seed = int(seed_start + i)
        for ncells in ncells_candidates:
            matrix = np.asarray(
                ps.generators.voronoi_edges(
                    shape=dims2,
                    ncells=int(ncells),
                    r=int(edge_radius_vox),
                    seed=seed,
                ),
                dtype=bool,
            )
            void = ~matrix
            if not has_spanning_cluster(void, axis_index=axis):
                continue

            err = abs(float(void.mean()) - float(porosity))
            if err < best_error:
                best_matrix = void
                best_seed = seed
                best_error = err
            if err <= target_tol:
                return void, seed

    if best_matrix is not None and best_seed is not None:
        return best_matrix, best_seed

    raise RuntimeError(
        "Could not generate spanning Voronoi matrix "
        f"for seed_start={int(seed_start)} after {int(max_tries)} trials"
    )


def generate_spanning_matrix_2d(
    *,
    shape: tuple[int, int],
    porosity: float,
    axis_index: int,
    generator_name: str,
    seed_start: int,
    max_tries: int,
    blobs_blobiness: float = 1.8,
    blobs_fallback_porosity_levels: Sequence[float] = (),
    voronoi_edge_radius_vox: int = 0,
    voronoi_target_tol: float = 0.003,
    voronoi_ncells_step: int = 10,
    voronoi_search_half_window: int = 70,
    voronoi_min_ncells: int = 40,
) -> tuple[np.ndarray, int, float]:
    """Dispatch 2D matrix generation across supported synthetic families.

    Parameters
    ----------
    shape, porosity, axis_index, seed_start, max_tries :
        Base generation/percolation controls.
    generator_name :
        Generator family selector: ``"voronoi_edges"`` or ``"blobs"``.
    blobs_blobiness, blobs_fallback_porosity_levels :
        Controls for ``blobs`` mode, including optional fallback porosity trials
        if the requested porosity fails to percolate.
    voronoi_edge_radius_vox :
        Edge-thickening radius for Voronoi mode.
    voronoi_target_tol :
        Porosity mismatch tolerance for Voronoi mode.
    voronoi_ncells_step, voronoi_search_half_window, voronoi_min_ncells :
        Candidate-search controls for Voronoi mode.

    Returns
    -------
    tuple[numpy.ndarray, int, float]
        ``(matrix, seed_used, porosity_used)`` where ``porosity_used`` is:
        - requested/fallback porosity parameter for ``blobs``;
        - achieved void porosity for ``voronoi_edges``.

    Raises
    ------
    ValueError
        If ``generator_name`` is unsupported or controls are invalid.
    RuntimeError
        If a spanning realization cannot be found.
    """

    generator = str(generator_name).strip().lower()
    if generator == "voronoi_edges":
        matrix, seed = generate_spanning_voronoi_matrix_2d(
            shape=shape,
            porosity=float(porosity),
            axis_index=axis_index,
            seed_start=seed_start,
            max_tries=max_tries,
            edge_radius_vox=voronoi_edge_radius_vox,
            target_tol=voronoi_target_tol,
            ncells_step=voronoi_ncells_step,
            search_half_window=voronoi_search_half_window,
            min_ncells=voronoi_min_ncells,
        )
        return matrix, seed, float(matrix.mean())

    if generator == "blobs":
        if not (0.0 < porosity < 1.0):
            raise ValueError("porosity must be in (0, 1)")
        porosity_trials = [float(porosity)] + [
            float(v) for v in blobs_fallback_porosity_levels if float(v) > float(porosity)
        ]
        porosity_trials = list(dict.fromkeys(porosity_trials))
        last_error: Exception | None = None
        for porosity_try in porosity_trials:
            try:
                matrix, seed = generate_spanning_blobs_matrix(
                    shape=shape,
                    porosity=porosity_try,
                    blobiness=blobs_blobiness,
                    axis_index=axis_index,
                    seed_start=seed_start,
                    max_tries=max_tries,
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            return matrix, seed, float(porosity_try)

        assert last_error is not None
        raise RuntimeError(
            "Could not generate spanning blobs matrix for requested or fallback porosities"
        ) from last_error

    raise ValueError(f"Unsupported generator_name: {generator_name}")


def generate_connected_matrix_2d(
    *,
    shape: tuple[int, int],
    porosity: float,
    axis_index: int,
    generator_name: str,
    seed_start: int,
    max_tries: int,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, int, float]:
    """Backward-compatible wrapper for notebook `09` matrix API.

    Parameters
    ----------
    shape, porosity, axis_index, generator_name, seed_start, max_tries :
        Same as :func:`generate_spanning_matrix_2d`.
    show_progress, progress_desc :
        Retained for compatibility. Currently ignored.

    Returns
    -------
    tuple[numpy.ndarray, int, float]
        Same as :func:`generate_spanning_matrix_2d`.
    """

    del show_progress, progress_desc
    return generate_spanning_matrix_2d(
        shape=shape,
        porosity=porosity,
        axis_index=axis_index,
        generator_name=generator_name,
        seed_start=seed_start,
        max_tries=max_tries,
    )


def insert_ellipsoidal_vug(
    matrix_void: np.ndarray,
    *,
    radii_vox: tuple[int, int, int],
    center: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert an axis-aligned ellipsoidal vug into a 3D binary void image.

    Parameters
    ----------
    matrix_void :
        Input 3D boolean array where ``True`` denotes void.
    radii_vox :
        Ellipsoid semi-axes ``(rx, ry, rz)`` in voxels.
    center :
        Ellipsoid center index. Defaults to the image center.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(updated_void, inserted_mask)`` where ``inserted_mask`` identifies the
        ellipsoidal support.

    Raises
    ------
    ValueError
        If dimensionality or radii are invalid.

    Notes
    -----
    The operation is a boolean union: pre-existing void voxels remain void.
    """

    arr = np.asarray(matrix_void, dtype=bool)
    if arr.ndim != 3:
        raise ValueError("matrix_void must be a 3D array")

    out = arr.copy()
    nx, ny, nz = out.shape
    if center is None:
        cx, cy, cz = nx // 2, ny // 2, nz // 2
    else:
        if len(center) != 3:
            raise ValueError("center must have length 3")
        cx, cy, cz = center
    rx, ry, rz = (float(radii_vox[0]), float(radii_vox[1]), float(radii_vox[2]))
    if min(rx, ry, rz) <= 0:
        raise ValueError("All ellipsoid radii must be positive")

    x = np.arange(nx, dtype=float) - float(cx)
    y = np.arange(ny, dtype=float) - float(cy)
    z = np.arange(nz, dtype=float) - float(cz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    ellipsoid_mask = (xx / rx) ** 2 + (yy / ry) ** 2 + (zz / rz) ** 2 <= 1.0
    out[ellipsoid_mask] = True
    return out, ellipsoid_mask


def insert_spherical_vug(
    matrix_void: np.ndarray,
    *,
    radius_vox: int,
    center: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a spherical vug into a 3D void image.

    Parameters
    ----------
    matrix_void :
        Input 3D binary void mask.
    radius_vox :
        Sphere radius in voxels.
    center :
        Optional center index. Defaults to image center.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated void image and spherical support mask.
    """

    radius = int(radius_vox)
    if radius <= 0:
        raise ValueError("radius_vox must be positive")
    return insert_ellipsoidal_vug(
        matrix_void,
        radii_vox=(radius, radius, radius),
        center=center,
    )


def insert_elliptical_vug_2d(
    matrix_void: np.ndarray,
    *,
    radii_vox: tuple[int, int],
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert an axis-aligned elliptical vug into a 2D binary void image.

    Parameters
    ----------
    matrix_void :
        Input 2D boolean array where ``True`` denotes void.
    radii_vox :
        Ellipse semi-axes ``(rx, ry)`` in voxels.
    center :
        Optional center index. Defaults to image center.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated void image and ellipse mask.
    """

    arr = np.asarray(matrix_void, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("matrix_void must be a 2D array")

    out = arr.copy()
    nx, ny = out.shape
    if center is None:
        cx, cy = nx // 2, ny // 2
    else:
        if len(center) != 2:
            raise ValueError("center must have length 2")
        cx, cy = center
    rx, ry = (float(radii_vox[0]), float(radii_vox[1]))
    if min(rx, ry) <= 0:
        raise ValueError("All ellipse radii must be positive")

    x = np.arange(nx, dtype=float) - float(cx)
    y = np.arange(ny, dtype=float) - float(cy)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ellipse_mask = (xx / rx) ** 2 + (yy / ry) ** 2 <= 1.0
    out[ellipse_mask] = True
    return out, ellipse_mask


def insert_circular_vug_2d(
    matrix_void: np.ndarray,
    *,
    radius_vox: int,
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a circular vug into a 2D binary void image.

    Parameters
    ----------
    matrix_void :
        Input 2D void mask.
    radius_vox :
        Circle radius in voxels.
    center :
        Optional center index.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated image and inserted circular mask.
    """

    radius = int(radius_vox)
    if radius <= 0:
        raise ValueError("radius_vox must be positive")
    return insert_elliptical_vug_2d(
        matrix_void,
        radii_vox=(radius, radius),
        center=center,
    )


def make_synthetic_grayscale(
    binary_void: np.ndarray,
    *,
    seed: int,
    void_mean: float = 65.0,
    solid_mean: float = 185.0,
    noise_std: float = 8.0,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> np.ndarray:
    """Generate synthetic grayscale contrast from binary void/solid phases.

    Parameters
    ----------
    binary_void :
        2D or 3D binary mask where ``True`` is void.
    seed :
        Seed for reproducible random noise.
    void_mean, solid_mean :
        Mean gray levels assigned to void and solid phases before noise.
    noise_std :
        Standard deviation of additive Gaussian noise.
    clip_min, clip_max :
        Output clipping range.

    Returns
    -------
    numpy.ndarray
        Floating-point grayscale image with same shape as ``binary_void``.

    Scientific interpretation
    -------------------------
    This is a synthetic observation model useful for controlled algorithm
    benchmarking. It does not represent scanner-specific physics (beam hardening,
    ring artifacts, partial-volume effects, etc.).
    """

    phase = np.asarray(binary_void, dtype=bool)
    if phase.ndim not in {2, 3}:
        raise ValueError("binary_void must be 2D or 3D")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if clip_max <= clip_min:
        raise ValueError("clip_max must be larger than clip_min")

    rng = np.random.default_rng(seed)
    base = np.where(phase, float(void_mean), float(solid_mean))
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=phase.shape)
    gray = np.clip(base + noise, float(clip_min), float(clip_max))
    return gray.astype(float)


def make_synthetic_grayscale_2d(
    binary_void: np.ndarray,
    seed: int,
    *,
    void_mean: float = 70.0,
    solid_mean: float = 185.0,
    noise_std: float = 8.0,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> np.ndarray:
    """Backward-compatible 2D wrapper around :func:`make_synthetic_grayscale`.

    Parameters
    ----------
    binary_void :
        2D binary void mask.
    seed :
        Random seed for noise realization.
    void_mean, solid_mean, noise_std, clip_min, clip_max :
        Same meaning as in :func:`make_synthetic_grayscale`.

    Returns
    -------
    numpy.ndarray
        2D floating-point grayscale image.
    """

    phase = np.asarray(binary_void, dtype=bool)
    if phase.ndim != 2:
        raise ValueError("binary_void must be 2D for make_synthetic_grayscale_2d")
    return make_synthetic_grayscale(
        phase,
        seed=seed,
        void_mean=void_mean,
        solid_mean=solid_mean,
        noise_std=noise_std,
        clip_min=clip_min,
        clip_max=clip_max,
    )


__all__ = [
    "MacroMicroPorousImage",
    "generate_macro_micro_blobs_matrix",
    "generate_spanning_multiscale_blobs_matrix",
    "generate_spanning_blobs_matrix",
    "generate_connected_matrix",
    "estimate_voronoi_ncells_for_porosity_2d",
    "generate_spanning_voronoi_matrix_2d",
    "generate_spanning_matrix_2d",
    "generate_connected_matrix_2d",
    "insert_ellipsoidal_vug",
    "insert_spherical_vug",
    "insert_elliptical_vug_2d",
    "insert_circular_vug_2d",
    "make_synthetic_grayscale",
    "make_synthetic_grayscale_2d",
]
