from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voids.examples.mms._core import (
    BrinkmanMMSCase,
    MMSConvergenceResult,
    MMSFacetLaw,
    MMSFacetSizeMode,
    MMSMethod,
)
from voids.examples.mms._runner import run_mms_convergence
from voids.examples.mms.cases_2d import boundary_layer_case_2d
from voids.examples.mms.cases_3d import bubble_case_3d
from voids.examples.mms.vug import (
    CenteredVugBenchmark,
    run_centered_vug_benchmark,
)
from voids.fem.singlephase import (
    FEMSinglePhaseResult,
    FEniCSSolverOptions,
    USFEMFacetLaw,
    USFEMFacetSizeMode,
)


@dataclass(frozen=True, slots=True)
class ReferenceQuantity:
    """One reported scalar value and its replication tolerance."""

    metric: str
    expected: float
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if not np.isfinite(self.expected):
            raise ValueError("expected must be finite")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.absolute_tolerance == 0.0 and self.relative_tolerance == 0.0:
            raise ValueError("at least one comparison tolerance must be positive")


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """Observed-versus-reported comparison for one scalar quantity."""

    metric: str
    observed: float
    expected: float
    absolute_tolerance: float
    relative_tolerance: float

    @property
    def absolute_error(self) -> float:
        """Return ``abs(observed - expected)``."""

        return abs(self.observed - self.expected)

    @property
    def relative_error(self) -> float:
        """Return the error relative to the reported magnitude."""

        if self.expected == 0.0:
            return 0.0 if self.observed == 0.0 else float("inf")
        return self.absolute_error / abs(self.expected)

    @property
    def passed(self) -> bool:
        """Return whether the observation lies within the stored tolerance."""

        return bool(
            np.isclose(
                self.observed,
                self.expected,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )
        )

    def as_dict(self) -> dict[str, str | float | bool]:
        """Return a table-ready representation."""

        return {
            "metric": self.metric,
            "observed": self.observed,
            "expected": self.expected,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PresentationComparison:
    """Comparison of one live solve with a supplied presentation baseline."""

    reference_name: str
    quantities: tuple[ReferenceComparison, ...]

    @property
    def passed(self) -> bool:
        """Return whether every reported quantity was reproduced."""

        return all(quantity.passed for quantity in self.quantities)

    def as_dicts(self) -> list[dict[str, str | float | bool]]:
        """Return comparison rows without requiring pandas."""

        return [quantity.as_dict() for quantity in self.quantities]

    def assert_matches(self) -> None:
        """Raise when any reported scalar falls outside its tolerance."""

        failures = [
            (
                f"{quantity.metric}: observed {quantity.observed:.6g}, "
                f"expected {quantity.expected:.6g}, "
                f"absolute error {quantity.absolute_error:.3g}"
            )
            for quantity in self.quantities
            if not quantity.passed
        ]
        if failures:
            raise AssertionError(
                f"{self.reference_name} did not reproduce the supplied baseline: "
                + "; ".join(failures)
            )


@dataclass(frozen=True, slots=True)
class MMSPresentationReference:
    """Exact configuration and reported values for one MMS presentation row."""

    name: str
    case: BrinkmanMMSCase
    method: MMSMethod
    resolutions: tuple[int, ...]
    facet_law: MMSFacetLaw
    facet_size_mode: MMSFacetSizeMode
    quantities: tuple[ReferenceQuantity, ...]
    source: str
    face_refinement: int = 24

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if len(self.resolutions) < 2:
            raise ValueError("resolutions must contain at least two mesh levels")
        if any(value < 1 for value in self.resolutions):
            raise ValueError("all resolutions must be positive")
        if any(
            current <= previous for previous, current in zip(self.resolutions, self.resolutions[1:])
        ):
            raise ValueError("resolutions must be strictly increasing")
        if not self.quantities:
            raise ValueError("quantities must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.face_refinement < 2:
            raise ValueError("face_refinement must be at least 2")
        if self.facet_size_mode not in {
            "cell_diameter",
            "facet_diameter",
            "representative",
        }:
            raise ValueError("facet_size_mode is not supported")


@dataclass(frozen=True, slots=True)
class VugPresentationReference:
    """Configuration and reported flux for a centered-vug presentation row."""

    name: str
    benchmark: CenteredVugBenchmark
    method: MMSMethod
    facet_law: USFEMFacetLaw | None
    facet_size_mode: USFEMFacetSizeMode | None
    quantities: tuple[ReferenceQuantity, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.benchmark.mesh_representation != "body_fitted":
            raise ValueError("presentation vug references require a body-fitted mesh")
        if not self.quantities:
            raise ValueError("quantities must not be empty")
        if self.method != "taylor_hood" and self.facet_size_mode is None:
            raise ValueError("USFEM vug references require a facet_size_mode")
        if not self.source.strip():
            raise ValueError("source must not be empty")


@dataclass(slots=True)
class MMSPresentationRun:
    """Live MMS refinement result paired with its baseline comparison."""

    reference: MMSPresentationReference
    result: MMSConvergenceResult
    comparison: PresentationComparison

    def assert_matches(self) -> None:
        """Raise unless the live result reproduces every stored target."""

        self.comparison.assert_matches()


@dataclass(slots=True)
class VugPresentationRun:
    """Live centered-vug result paired with its baseline comparison."""

    reference: VugPresentationReference
    result: FEMSinglePhaseResult
    comparison: PresentationComparison

    def assert_matches(self) -> None:
        """Raise unless the live result reproduces every stored target."""

        self.comparison.assert_matches()


def _rates(
    velocity_l2: float,
    velocity_h1: float,
    pressure_l2: float,
    divergence_l2: float,
    *,
    tolerance: float = 0.01,
) -> tuple[ReferenceQuantity, ...]:
    return (
        ReferenceQuantity("velocity_l2_rate", velocity_l2, absolute_tolerance=tolerance),
        ReferenceQuantity("velocity_h1_rate", velocity_h1, absolute_tolerance=tolerance),
        ReferenceQuantity("pressure_l2_rate", pressure_l2, absolute_tolerance=tolerance),
        ReferenceQuantity("divergence_l2_rate", divergence_l2, absolute_tolerance=tolerance),
    )


_SOURCE_2D = "Locally Conservative USFEM presentation and two-dimensional results report"
_SOURCE_3D = "Three-dimensional USFEM MMS and vug assessment report"

_REFERENCE_2D_P1DG0 = MMSPresentationReference(
    name="2d_boundary_layer_p1dg0",
    case=boundary_layer_case_2d(viscosity=1.0e-2, reaction=1.0),
    method="usfem_p1dg0",
    resolutions=(4, 8, 16, 32, 64, 128, 256),
    facet_law="reaction_diffusion",
    facet_size_mode="facet_diameter",
    quantities=(
        ReferenceQuantity("velocity_l2_error", 1.384e-3, relative_tolerance=0.005),
        ReferenceQuantity("velocity_h1_error", 1.119, relative_tolerance=0.005),
        ReferenceQuantity("pressure_l2_error", 9.821e-4, relative_tolerance=0.005),
        ReferenceQuantity("divergence_l2", 2.041e-3, relative_tolerance=0.005),
        ReferenceQuantity("velocity_l2_rate", 1.96, absolute_tolerance=0.01),
        ReferenceQuantity("velocity_h1_rate", 0.97, absolute_tolerance=0.01),
        ReferenceQuantity("pressure_l2_rate", 1.02, absolute_tolerance=0.01),
    ),
    source=_SOURCE_2D,
)

_REFERENCE_2D_P1DG1 = MMSPresentationReference(
    name="2d_boundary_layer_p1dg1",
    case=boundary_layer_case_2d(viscosity=1.0e-2, reaction=1.0),
    method="usfem_p1dg1",
    resolutions=(4, 8, 16, 32, 64, 128),
    facet_law="reaction_diffusion",
    facet_size_mode="facet_diameter",
    quantities=(
        ReferenceQuantity("velocity_l2_error", 5.577e-3, relative_tolerance=0.005),
        ReferenceQuantity("velocity_h1_error", 2.191, relative_tolerance=0.005),
        ReferenceQuantity("pressure_l2_error", 1.085e-2, relative_tolerance=0.005),
        ReferenceQuantity("divergence_l2", 6.246e-2, relative_tolerance=0.005),
        ReferenceQuantity("velocity_l2_rate", 1.866, absolute_tolerance=0.01),
        ReferenceQuantity("pressure_l2_rate", 1.019, absolute_tolerance=0.01),
    ),
    source=_SOURCE_2D,
)

_P1DG0_3D_RATES = {
    "classic": (1.724, 1.478, 1.340, 1.471),
    "shifted": (1.044, 0.875, 1.202, 0.801),
    "reaction_diffusion": (1.620, 1.381, 1.313, 1.369),
    "face3d": (1.968, 1.082, 1.174, 1.020),
}
_P1DG1_3D_RATES = {
    "classic": (2.033, 0.990, 1.787, 0.980),
    "shifted": (2.074, 0.991, 1.890, 0.976),
    "reaction_diffusion": (2.030, 0.990, 1.795, 0.980),
    "face3d": (1.976, 1.012, 1.099, 0.919),
}


def _three_dimensional_references() -> tuple[MMSPresentationReference, ...]:
    case = bubble_case_3d(viscosity=1.0e-2, reaction=1.0)
    resolutions = (4, 6, 8, 10, 12, 16, 20)
    references: list[MMSPresentationReference] = []
    rate_tables: tuple[
        tuple[MMSMethod, dict[str, tuple[float, float, float, float]]],
        ...,
    ] = (
        ("usfem_p1dg0", _P1DG0_3D_RATES),
        ("usfem_p1dg1", _P1DG1_3D_RATES),
    )
    for method, rate_table in rate_tables:
        for facet_law, rate_values in rate_table.items():
            references.append(
                MMSPresentationReference(
                    name=f"3d_bubble_{method}_{facet_law}",
                    case=case,
                    method=method,
                    resolutions=resolutions,
                    facet_law=facet_law,  # type: ignore[arg-type]
                    facet_size_mode="representative",
                    quantities=_rates(*rate_values),
                    source=_SOURCE_3D,
                )
            )
    return tuple(references)


_MMS_REFERENCES = (
    _REFERENCE_2D_P1DG0,
    _REFERENCE_2D_P1DG1,
    *_three_dimensional_references(),
)
_MMS_REFERENCE_BY_NAME = {reference.name: reference for reference in _MMS_REFERENCES}

_VUG_BENCHMARK_3D = CenteredVugBenchmark(
    dimension=3,
    resolution=30,
    radius=0.25,
    viscosity=1.0e-2,
    matrix_drag=1.0e7,
    vug_drag=1.0,
    pressure_inlet=1.0,
    pressure_outlet=-1.0,
    mesh_representation="body_fitted",
)
_VUG_REFERENCES = (
    VugPresentationReference(
        name="3d_centered_vug_p1dg1",
        benchmark=_VUG_BENCHMARK_3D,
        method="usfem_p1dg1",
        facet_law="shifted",
        facet_size_mode="facet_measure",
        quantities=(
            ReferenceQuantity("flow_rate", 2.413e-7, relative_tolerance=0.01),
            ReferenceQuantity("represented_vug_fraction", 0.0643, relative_tolerance=0.03),
        ),
        source=_SOURCE_3D,
    ),
    VugPresentationReference(
        name="3d_centered_vug_taylor_hood",
        benchmark=_VUG_BENCHMARK_3D,
        method="taylor_hood",
        facet_law=None,
        facet_size_mode=None,
        quantities=(
            ReferenceQuantity("flow_rate", 2.42167e-7, relative_tolerance=0.01),
            ReferenceQuantity("represented_vug_fraction", 0.0643, relative_tolerance=0.03),
        ),
        source=_SOURCE_3D,
    ),
)
_VUG_REFERENCE_BY_NAME = {reference.name: reference for reference in _VUG_REFERENCES}


def presentation_mms_references() -> tuple[MMSPresentationReference, ...]:
    """Return the shipped MMS presentation-replication profiles."""

    return _MMS_REFERENCES


def presentation_vug_references() -> tuple[VugPresentationReference, ...]:
    """Return the shipped centered-vug presentation-replication profiles."""

    return _VUG_REFERENCES


def _resolve_mms_reference(
    reference: str | MMSPresentationReference,
) -> MMSPresentationReference:
    if isinstance(reference, MMSPresentationReference):
        return reference
    try:
        return _MMS_REFERENCE_BY_NAME[reference]
    except KeyError as exc:
        names = ", ".join(_MMS_REFERENCE_BY_NAME)
        raise ValueError(
            f"unknown MMS presentation reference {reference!r}; choose from {names}"
        ) from exc


def _resolve_vug_reference(
    reference: str | VugPresentationReference,
) -> VugPresentationReference:
    if isinstance(reference, VugPresentationReference):
        return reference
    try:
        return _VUG_REFERENCE_BY_NAME[reference]
    except KeyError as exc:
        names = ", ".join(_VUG_REFERENCE_BY_NAME)
        raise ValueError(
            f"unknown vug presentation reference {reference!r}; choose from {names}"
        ) from exc


def _compare_quantities(
    reference_name: str,
    quantities: tuple[ReferenceQuantity, ...],
    observed: dict[str, float],
) -> PresentationComparison:
    missing = [quantity.metric for quantity in quantities if quantity.metric not in observed]
    if missing:
        raise ValueError(f"missing observed presentation metrics: {', '.join(missing)}")
    comparisons = tuple(
        ReferenceComparison(
            metric=quantity.metric,
            observed=float(observed[quantity.metric]),
            expected=quantity.expected,
            absolute_tolerance=quantity.absolute_tolerance,
            relative_tolerance=quantity.relative_tolerance,
        )
        for quantity in quantities
    )
    return PresentationComparison(reference_name, comparisons)


def compare_mms_with_presentation(
    result: MMSConvergenceResult,
    reference: str | MMSPresentationReference,
) -> PresentationComparison:
    """Compare a live MMS result with the exact supplied configuration and values."""

    resolved = _resolve_mms_reference(reference)
    if result.method != resolved.method:
        raise ValueError(
            f"method mismatch: result uses {result.method}, reference uses {resolved.method}"
        )
    if result.case.name != resolved.case.name:
        raise ValueError(
            f"case mismatch: result uses {result.case.name}, reference uses {resolved.case.name}"
        )
    if not np.isclose(result.case.viscosity, resolved.case.viscosity) or not np.isclose(
        result.case.reaction,
        resolved.case.reaction,
    ):
        raise ValueError("case coefficient mismatch with presentation reference")
    result_resolutions = tuple(level.resolution for level in result.levels)
    if result_resolutions[-2:] != resolved.resolutions[-2:]:
        raise ValueError(
            "the result must end with the presentation's finest mesh pair "
            f"{resolved.resolutions[-2:]}; received {result_resolutions[-2:]}"
        )
    if result.method != "taylor_hood" and result.metadata.get("facet_law") != resolved.facet_law:
        raise ValueError(
            "facet-law mismatch: result uses "
            f"{result.metadata.get('facet_law')}, reference uses {resolved.facet_law}"
        )
    if (
        result.method != "taylor_hood"
        and result.metadata.get("facet_size_mode") != resolved.facet_size_mode
    ):
        raise ValueError(
            "facet-size mismatch: result uses "
            f"{result.metadata.get('facet_size_mode')}, "
            f"reference uses {resolved.facet_size_mode}"
        )

    finest = result.levels[-1]
    observed = {
        "velocity_l2_error": finest.velocity_l2_error,
        "velocity_h1_error": finest.velocity_h1_error,
        "pressure_l2_error": finest.pressure_l2_error,
        "divergence_l2": finest.divergence_l2,
        **{f"{name}_rate": value for name, value in result.last_rates.items()},
    }
    return _compare_quantities(resolved.name, resolved.quantities, observed)


def compare_vug_with_presentation(
    result: FEMSinglePhaseResult,
    reference: str | VugPresentationReference,
) -> PresentationComparison:
    """Compare a live centered-vug result with the supplied report-scale values."""

    resolved = _resolve_vug_reference(reference)
    metadata = result.metadata
    for name, expected in (
        ("dimension", resolved.benchmark.dimension),
        ("resolution", resolved.benchmark.resolution),
        ("radius", resolved.benchmark.radius),
        ("matrix_drag", resolved.benchmark.matrix_drag),
        ("vug_drag", resolved.benchmark.vug_drag),
    ):
        actual = metadata.get(name)
        if actual is None or not np.isclose(
            float(actual),
            float(expected),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(f"vug configuration mismatch for {name}: {actual!r} != {expected!r}")
    if resolved.method == "taylor_hood":
        if result.formulation != "brinkman_taylor_hood_p2p1":
            raise ValueError("vug formulation does not match the Taylor-Hood reference")
    else:
        if result.formulation != "brinkman_usfem_p1dg1":
            raise ValueError("vug formulation does not match the P1/DG1 reference")
        if metadata.get("facet_law") != resolved.facet_law:
            raise ValueError("vug facet law does not match the presentation reference")
        if metadata.get("facet_size_mode") != resolved.facet_size_mode:
            raise ValueError("vug facet size does not match the presentation reference")

    observed = {
        "flow_rate": result.flow_rate,
        "represented_vug_fraction": float(metadata["represented_vug_fraction"]),
    }
    return _compare_quantities(resolved.name, resolved.quantities, observed)


def run_presentation_mms(
    reference: str | MMSPresentationReference,
    *,
    options: FEniCSSolverOptions | None = None,
    keep_solution: bool = False,
    callback: Any | None = None,
) -> MMSPresentationRun:
    """Run a full supplied MMS mesh sequence and compare its reported values."""

    resolved = _resolve_mms_reference(reference)
    result = run_mms_convergence(
        resolved.case,
        method=resolved.method,
        resolutions=resolved.resolutions,
        options=options,
        facet_law=resolved.facet_law,
        facet_size_mode=resolved.facet_size_mode,
        face_refinement=resolved.face_refinement,
        keep_solution=keep_solution,
        callback=callback,
    )
    comparison = compare_mms_with_presentation(result, resolved)
    return MMSPresentationRun(resolved, result, comparison)


def run_presentation_vug(
    reference: str | VugPresentationReference,
    *,
    options: FEniCSSolverOptions | None = None,
) -> VugPresentationRun:
    """Run a report-scale body-fitted vug case and compare its reported values."""

    resolved = _resolve_vug_reference(reference)
    result = run_centered_vug_benchmark(
        resolved.benchmark,
        method=resolved.method,
        options=options,
        facet_law=resolved.facet_law,
        facet_size_mode=(
            "facet_measure" if resolved.facet_size_mode is None else resolved.facet_size_mode
        ),
    )
    comparison = compare_vug_with_presentation(result, resolved)
    return VugPresentationRun(resolved, result, comparison)


__all__ = [
    "MMSPresentationReference",
    "MMSPresentationRun",
    "PresentationComparison",
    "ReferenceComparison",
    "ReferenceQuantity",
    "VugPresentationReference",
    "VugPresentationRun",
    "compare_mms_with_presentation",
    "compare_vug_with_presentation",
    "presentation_mms_references",
    "presentation_vug_references",
    "run_presentation_mms",
    "run_presentation_vug",
]
