from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from voids.examples.mms import (
    BrinkmanMMSCase,
    CenteredVugBenchmark,
    CenteredVugFlowCase2D,
    M2_PER_MILLIDARCY,
    MMSConvergenceLevel,
    MMSConvergenceResult,
    MMSPresentationReference,
    PresentationComparison,
    ReferenceComparison,
    ReferenceQuantity,
    VugPresentationReference,
    available_mms_methods,
    boundary_layer_case_2d,
    bubble_case_3d,
    compare_mms_with_presentation,
    compare_vug_with_presentation,
    face3d_pressure_jump_coefficient,
    make_body_fitted_centered_vug_mesh,
    observed_rate,
    presentation_mms_references,
    presentation_vug_references,
    run_centered_vug_benchmark,
    run_centered_vug_flow_case,
    run_mms_convergence,
    run_presentation_mms,
    run_presentation_vug,
)
from voids.examples.mms._core import ConvergenceExpectation
from voids.examples.mms import replication
from voids.examples.mms._runner import _safe_sqrt
from voids.examples.mms import vug as vug_module
from voids.examples.mms.vug import _classify_volume_entities
from voids.fem.singlephase import FEMSinglePhaseResult, FEniCSSolverOptions
from voids.fem.singlephase._common import _require_dolfinx_core
from voids.mesh.gmsh import (
    add_physical_group,
    axis_aligned_boundary_entities,
    configure_uniform_mesh_size,
    generate_dolfinx_gmsh_mesh,
    require_gmsh,
)


try:
    _require_dolfinx_core()
except ImportError as exc:
    requires_fem_stack = pytest.mark.skip(reason=str(exc))
else:
    requires_fem_stack = pytest.mark.skipif(False, reason="")


def _skip_native_gmsh_on_windows() -> None:
    if sys.platform == "win32":
        pytest.skip(
            "Gmsh 4.15 native initialization aborts the Windows CI process; "
            "native Gmsh coverage runs on Linux and macOS"
        )


def _dummy_solution_factory(_ufl: object, _domain: object) -> tuple[int, int]:
    return 0, 0


def test_mms_case_validates_definition_and_point_evaluation() -> None:
    case = boundary_layer_case_2d(viscosity=0.1)
    points = np.array([[0.0, 0.25, 1.0], [0.0, 0.75, 1.0]])
    velocity, pressure = case.evaluate(points)

    assert velocity.shape == (2, 3)
    assert pressure.shape == (3,)
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()
    assert np.allclose(case.evaluate(points[:, 0])[0], velocity[:, :1])

    with pytest.raises(ValueError, match="name"):
        replace(case, name="")
    with pytest.raises(ValueError, match="dimension"):
        BrinkmanMMSCase("x", 4, 1.0, 1.0, _dummy_solution_factory)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="viscosity"):
        replace(case, viscosity=0.0)
    with pytest.raises(ValueError, match="reaction"):
        replace(case, reaction=-1.0)
    with pytest.raises(TypeError, match="exact_solution_factory"):
        replace(case, exact_solution_factory=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="point_evaluator"):
        replace(case, point_evaluator=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coordinates"):
        case.evaluate(np.ones((1, 3)))


def test_mms_case_reports_missing_or_invalid_point_evaluator() -> None:
    case = BrinkmanMMSCase(
        name="symbolic_only",
        dimension=2,
        viscosity=1.0,
        reaction=0.0,
        exact_solution_factory=_dummy_solution_factory,
    )
    with pytest.raises(NotImplementedError, match="point evaluator"):
        case.evaluate(np.zeros((2, 1)))

    bad_velocity = replace(
        case,
        point_evaluator=lambda points: (
            np.zeros((1, points.shape[1])),
            np.zeros(points.shape[1]),
        ),
    )
    with pytest.raises(ValueError, match="velocity with shape"):
        bad_velocity.evaluate(np.zeros((2, 2)))

    bad_pressure = replace(
        case,
        point_evaluator=lambda points: (
            np.zeros((2, points.shape[1])),
            np.zeros((1, points.shape[1])),
        ),
    )
    with pytest.raises(ValueError, match="pressure with shape"):
        bad_pressure.evaluate(np.zeros((2, 2)))


def test_builtin_3d_case_is_zero_on_boundary_and_divergence_free_numerically() -> None:
    case = bubble_case_3d()
    boundary_points = np.array(
        [
            [0.0, 1.0, 0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.0, 1.0, 0.4, 0.4],
            [0.6, 0.6, 0.6, 0.6, 0.0, 1.0],
        ]
    )
    velocity, _ = case.evaluate(boundary_points)
    assert np.allclose(velocity, 0.0, atol=1.0e-14)

    point = np.array([[0.31], [0.43], [0.57]])
    epsilon = 1.0e-6
    divergence = 0.0
    for axis in range(3):
        offset = np.zeros_like(point)
        offset[axis] = epsilon
        u_plus, _ = case.evaluate(point + offset)
        u_minus, _ = case.evaluate(point - offset)
        divergence += (u_plus[axis, 0] - u_minus[axis, 0]) / (2.0 * epsilon)
    assert divergence == pytest.approx(0.0, abs=2.0e-8)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.25, 0.25, 0.25, 0.125), "previous_h"),
        ((0.5, 0.0, 0.25, 0.125), None),
        ((0.5, 0.25, 0.25, 0.0), None),
    ],
)
def test_observed_rate_validation(
    arguments: tuple[float, float, float, float],
    message: str | None,
) -> None:
    if message is None:
        assert np.isnan(observed_rate(*arguments))
    else:
        with pytest.raises(ValueError, match=message):
            observed_rate(*arguments)
    assert observed_rate(0.5, 0.25, 0.25, 0.0625) == pytest.approx(2.0)


def test_convergence_result_asserts_nominal_rates() -> None:
    first = MMSConvergenceLevel(2, 0.5, 8, 20, 0.1, 1.0, 1.0, 1.0, 1.0)
    second = MMSConvergenceLevel(
        4,
        0.25,
        32,
        80,
        0.2,
        0.25,
        0.5,
        0.5,
        0.5,
        rates={
            "velocity_l2": 2.0,
            "velocity_h1": 1.0,
            "pressure_l2": 1.0,
            "divergence_l2": 1.0,
        },
    )
    result = MMSConvergenceResult(
        boundary_layer_case_2d(),
        "usfem_p1dg0",
        (first, second),
        ConvergenceExpectation(2.0, 1.0, 1.0),
    )
    assert result.last_rates["velocity_l2"] == 2.0
    assert result.as_dicts()[-1]["pressure_l2_rate"] == 1.0
    result.assert_expected_rates()
    with pytest.raises(ValueError, match="absolute_tolerance"):
        result.assert_expected_rates(absolute_tolerance=-1.0)
    failing = replace(
        result,
        levels=(
            first,
            replace(second, rates={"velocity_l2": 0.5}),
        ),
    )
    with pytest.raises(AssertionError, match="velocity_l2"):
        failing.assert_expected_rates()


def _reported_mms_result(
    reference: MMSPresentationReference,
) -> MMSConvergenceResult:
    expected = {quantity.metric: quantity.expected for quantity in reference.quantities}
    rates = {
        name: expected.get(f"{name}_rate", 1.0)
        for name in (
            "velocity_l2",
            "velocity_h1",
            "pressure_l2",
            "divergence_l2",
        )
    }
    first = MMSConvergenceLevel(
        reference.resolutions[-2],
        1.0 / reference.resolutions[-2],
        1,
        1,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    finest = MMSConvergenceLevel(
        reference.resolutions[-1],
        1.0 / reference.resolutions[-1],
        1,
        1,
        0.0,
        expected.get("velocity_l2_error", 1.0),
        expected.get("velocity_h1_error", 1.0),
        expected.get("pressure_l2_error", 1.0),
        expected.get("divergence_l2", 1.0),
        rates=rates,
    )
    return MMSConvergenceResult(
        case=reference.case,
        method=reference.method,
        levels=(first, finest),
        expected_rates=ConvergenceExpectation(2.0, 1.0, 1.0),
        metadata={
            "facet_law": reference.facet_law,
            "facet_size_mode": reference.facet_size_mode,
        },
    )


def _reported_vug_result(
    reference: VugPresentationReference,
) -> FEMSinglePhaseResult:
    expected = {quantity.metric: quantity.expected for quantity in reference.quantities}
    taylor_hood = reference.method == "taylor_hood"
    return FEMSinglePhaseResult(
        method="test",
        formulation=("brinkman_taylor_hood_p2p1" if taylor_hood else "brinkman_usfem_p1dg1"),
        flow_axis="x",
        permeability=1.0,
        flow_rate=expected["flow_rate"],
        pressure_inlet=1.0,
        pressure_outlet=-1.0,
        pressure_drop=2.0,
        viscosity=1.0e-2,
        domain_length=1.0,
        cross_section_area=1.0,
        solve_seconds=0.0,
        velocity=None,
        pressure=None,
        metadata={
            "dimension": reference.benchmark.dimension,
            "resolution": reference.benchmark.resolution,
            "radius": reference.benchmark.radius,
            "matrix_drag": reference.benchmark.matrix_drag,
            "vug_drag": reference.benchmark.vug_drag,
            "facet_law": reference.facet_law,
            "facet_size_mode": reference.facet_size_mode,
            "represented_vug_fraction": expected["represented_vug_fraction"],
        },
    )


def test_presentation_reference_catalogues_and_comparisons() -> None:
    mms_references = presentation_mms_references()
    assert len(mms_references) == 10
    assert {reference.name for reference in mms_references} >= {
        "2d_boundary_layer_p1dg0",
        "3d_bubble_usfem_p1dg0_face3d",
        "3d_bubble_usfem_p1dg1_shifted",
    }
    for reference in mms_references:
        comparison = compare_mms_with_presentation(
            _reported_mms_result(reference),
            reference.name,
        )
        assert comparison.passed
        comparison.assert_matches()
        assert all(row["passed"] for row in comparison.as_dicts())

    vug_references = presentation_vug_references()
    assert len(vug_references) == 2
    for reference in vug_references:
        comparison = compare_vug_with_presentation(
            _reported_vug_result(reference),
            reference.name,
        )
        assert comparison.passed
        comparison.assert_matches()


def test_reference_quantity_and_failure_reporting_validation() -> None:
    quantity = ReferenceQuantity("zero", 0.0, absolute_tolerance=0.1)
    exact = ReferenceComparison("zero", 0.0, 0.0, 0.1, 0.0)
    failure = ReferenceComparison("zero", 1.0, 0.0, 0.1, 0.0)
    assert exact.relative_error == 0.0
    assert failure.relative_error == float("inf")
    assert not failure.passed
    comparison = PresentationComparison("failure", (failure,))
    assert not comparison.passed
    with pytest.raises(AssertionError, match="did not reproduce"):
        comparison.assert_matches()
    assert quantity.metric == "zero"

    for kwargs, message in (
        ({"metric": ""}, "metric"),
        ({"expected": float("nan")}, "expected"),
        ({"absolute_tolerance": -1.0}, "absolute_tolerance"),
        ({"relative_tolerance": -1.0}, "relative_tolerance"),
        ({"absolute_tolerance": 0.0, "relative_tolerance": 0.0}, "tolerance"),
    ):
        values = {
            "metric": "x",
            "expected": 1.0,
            "absolute_tolerance": 0.1,
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            ReferenceQuantity(**values)


def test_presentation_reference_configuration_validation() -> None:
    reference = presentation_mms_references()[0]
    for kwargs, message in (
        ({"name": ""}, "name"),
        ({"resolutions": (4,)}, "at least two"),
        ({"resolutions": (0, 1)}, "positive"),
        ({"resolutions": (4, 4)}, "increasing"),
        ({"quantities": ()}, "quantities"),
        ({"source": ""}, "source"),
        ({"face_refinement": 1}, "face_refinement"),
        ({"facet_size_mode": "other"}, "facet_size_mode"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(reference, **kwargs)

    vug_reference = presentation_vug_references()[0]
    for kwargs, message in (
        ({"name": ""}, "name"),
        (
            {
                "benchmark": replace(
                    vug_reference.benchmark,
                    mesh_representation="structured",
                )
            },
            "body-fitted",
        ),
        ({"quantities": ()}, "quantities"),
        ({"source": ""}, "source"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(vug_reference, **kwargs)


def test_presentation_comparison_rejects_mismatched_runs() -> None:
    reference = presentation_mms_references()[0]
    result = _reported_mms_result(reference)
    mismatches = (
        (replace(result, method="taylor_hood"), "method mismatch"),
        (
            replace(result, case=replace(result.case, name="other")),
            "case mismatch",
        ),
        (
            replace(result, case=replace(result.case, viscosity=0.02)),
            "coefficient mismatch",
        ),
        (
            replace(
                result,
                levels=(
                    replace(result.levels[0], resolution=12),
                    replace(result.levels[1], resolution=24),
                ),
            ),
            "finest mesh pair",
        ),
        (replace(result, metadata={"facet_law": "classic"}), "facet-law mismatch"),
        (
            replace(
                result,
                metadata={
                    "facet_law": reference.facet_law,
                    "facet_size_mode": "representative",
                },
            ),
            "facet-size mismatch",
        ),
    )
    for mismatched, message in mismatches:
        with pytest.raises(ValueError, match=message):
            compare_mms_with_presentation(mismatched, reference)

    with pytest.raises(ValueError, match="unknown MMS"):
        compare_mms_with_presentation(result, "missing")
    missing_metric_reference = replace(
        reference,
        quantities=(ReferenceQuantity("missing", 1.0, absolute_tolerance=0.1),),
    )
    with pytest.raises(ValueError, match="missing observed"):
        compare_mms_with_presentation(result, missing_metric_reference)


def test_vug_presentation_comparison_rejects_mismatched_runs() -> None:
    p1_reference, taylor_reference = presentation_vug_references()
    p1_result = _reported_vug_result(p1_reference)
    for key in ("dimension", "resolution", "radius", "matrix_drag", "vug_drag"):
        metadata = dict(p1_result.metadata)
        metadata[key] = float(metadata[key]) + 1.0
        with pytest.raises(ValueError, match=key):
            compare_vug_with_presentation(
                replace(p1_result, metadata=metadata),
                p1_reference,
            )

    with pytest.raises(ValueError, match="P1/DG1"):
        compare_vug_with_presentation(
            replace(p1_result, formulation="other"),
            p1_reference,
        )
    with pytest.raises(ValueError, match="facet law"):
        compare_vug_with_presentation(
            replace(p1_result, metadata={**p1_result.metadata, "facet_law": "classic"}),
            p1_reference,
        )
    with pytest.raises(ValueError, match="facet size"):
        compare_vug_with_presentation(
            replace(
                p1_result,
                metadata={**p1_result.metadata, "facet_size_mode": "cell_diameter"},
            ),
            p1_reference,
        )
    taylor_result = _reported_vug_result(taylor_reference)
    with pytest.raises(ValueError, match="Taylor-Hood"):
        compare_vug_with_presentation(
            replace(taylor_result, formulation="other"),
            taylor_reference,
        )
    with pytest.raises(ValueError, match="unknown vug"):
        compare_vug_with_presentation(p1_result, "missing")


def test_presentation_run_helpers_pair_live_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mms_reference = presentation_mms_references()[0]
    mms_result = _reported_mms_result(mms_reference)
    monkeypatch.setattr(replication, "run_mms_convergence", lambda *args, **kwargs: mms_result)
    mms_run = run_presentation_mms(mms_reference.name)
    assert mms_run.result is mms_result
    mms_run.assert_matches()

    vug_reference = presentation_vug_references()[0]
    vug_result = _reported_vug_result(vug_reference)
    monkeypatch.setattr(
        replication,
        "run_centered_vug_benchmark",
        lambda *args, **kwargs: vug_result,
    )
    vug_run = run_presentation_vug(vug_reference.name)
    assert vug_run.result is vug_result
    vug_run.assert_matches()


def test_face3d_pressure_jump_coefficient_is_positive_and_validated() -> None:
    coarse = face3d_pressure_jump_coefficient(
        viscosity=1.0e-2,
        reaction=1.0,
        resolution=8,
        face_refinement=12,
    )
    refined = face3d_pressure_jump_coefficient(
        viscosity=1.0e-2,
        reaction=1.0,
        resolution=8,
        face_refinement=24,
    )

    assert coarse > 0.0
    assert refined == pytest.approx(coarse, rel=5.0e-2)
    for kwargs, message in (
        ({"viscosity": 0.0}, "viscosity"),
        ({"reaction": -1.0}, "reaction"),
        ({"resolution": 0}, "resolution"),
        ({"face_refinement": 1}, "face_refinement"),
    ):
        with pytest.raises(ValueError, match=message):
            face3d_pressure_jump_coefficient(
                viscosity=kwargs.get("viscosity", 1.0e-2),
                reaction=kwargs.get("reaction", 1.0),
                resolution=kwargs.get("resolution", 8),
                face_refinement=kwargs.get("face_refinement", 8),
            )


def test_centered_vug_configuration_and_structured_problem() -> None:
    benchmark = CenteredVugBenchmark(
        dimension=2,
        resolution=16,
        mesh_representation="structured",
    )
    problem = benchmark.make_problem()

    assert problem.permeability_map.shape == (16, 16)
    assert 0.0 < benchmark.represented_fraction < 1.0
    assert benchmark.analytic_fraction == pytest.approx(np.pi * 0.25**2)
    assert benchmark.target_mesh_size == pytest.approx(np.sqrt(2.0) / 16.0)
    assert np.unique(problem.permeability_map.values).size == 2
    assert benchmark.matrix_nu == pytest.approx(benchmark.viscosity)
    assert benchmark.vug_nu == pytest.approx(benchmark.viscosity)

    matrix_only = CenteredVugBenchmark(
        radius=0.0,
        resolution=8,
        mesh_representation="structured",
        matrix_effective_viscosity=2.0,
        vug_effective_viscosity=1.0,
    )
    matrix_problem = matrix_only.make_problem()
    assert matrix_only.analytic_fraction == 0.0
    assert np.unique(matrix_problem.permeability_map.values).size == 1
    assert np.allclose(matrix_problem.porosity_map.values, 0.005)
    with pytest.raises(ValueError, match="positive vug_drag"):
        CenteredVugBenchmark(
            vug_drag=0.0,
            mesh_representation="structured",
        ).make_problem()

    for kwargs, message in (
        ({"dimension": 4}, "dimension"),
        ({"resolution": 1}, "resolution"),
        ({"radius": 0.5}, "radius"),
        ({"viscosity": 0.0}, "viscosity"),
        ({"vug_drag": -1.0}, "vug_drag"),
        ({"matrix_effective_viscosity": 0.0}, "matrix_effective_viscosity"),
        ({"pressure_inlet": -1.0}, "pressure_inlet"),
        ({"mesh_representation": "other"}, "mesh_representation"),
    ):
        with pytest.raises(ValueError, match=message):
            CenteredVugBenchmark(**kwargs)  # type: ignore[arg-type]


def test_physical_centered_vug_flow_case_parameters_and_validation() -> None:
    case = CenteredVugFlowCase2D(area_fraction=0.7)

    assert case.image_shape == (500, 500)
    assert case.side_length_m == pytest.approx(7.5e-3)
    assert case.voxel_size_m == pytest.approx(15.0e-6)
    assert case.radius_fraction == pytest.approx(np.sqrt(0.7 / np.pi))
    assert case.radius_m < 0.5 * case.side_length_m
    assert case.matrix_permeability_m2 == pytest.approx(200.0 * M2_PER_MILLIDARCY)
    assert case.vug_permeability_m2 == pytest.approx(1.0e-8)
    assert case.pressure_drop_pa == pytest.approx(1.0)
    assert case.permeability_contrast > 5.0e4
    assert case.mesh_resolution == 100
    assert case.base_target_mesh_size_fraction == pytest.approx(np.sqrt(2.0) / 100.0)
    brinkman_benchmark = case.make_benchmark()
    assert brinkman_benchmark.radius == pytest.approx(case.radius_fraction)
    assert brinkman_benchmark.matrix_drag == pytest.approx(1.0)
    assert brinkman_benchmark.vug_drag == 0.0
    darcy_benchmark = case.make_benchmark(model="darcy_darcy")
    assert darcy_benchmark.vug_drag == pytest.approx(
        case.matrix_permeability_m2 / case.vug_permeability_m2
    )
    with pytest.raises(ValueError, match="model"):
        case.make_benchmark(model="bad")  # type: ignore[arg-type]

    for kwargs, message in (
        ({"area_fraction": np.pi / 4.0}, "area_fraction"),
        ({"image_shape": (500, 400)}, "square"),
        ({"image_shape": (1, 1)}, "at least 2"),
        ({"voxel_size_m": 0.0}, "voxel_size"),
        ({"matrix_porosity": 1.1}, "matrix_porosity"),
        ({"matrix_permeability_md": 0.0}, "matrix_permeability"),
        ({"vug_permeability_m2": 0.0}, "vug_permeability"),
        ({"pressure_inlet_pa": 0.0}, "pressure_inlet"),
        ({"mesh_resolution": 7}, "mesh_resolution"),
    ):
        with pytest.raises(ValueError, match=message):
            parameters = {"area_fraction": 0.1, **kwargs}
            CenteredVugFlowCase2D(**parameters)  # type: ignore[arg-type]


@requires_fem_stack
@pytest.mark.parametrize(
    "method",
    available_mms_methods(),
)
def test_2d_mms_all_methods_reduce_errors(method: str) -> None:
    study = run_mms_convergence(
        boundary_layer_case_2d(viscosity=0.1),
        method=method,  # type: ignore[arg-type]
        resolutions=(2, 4),
        options=FEniCSSolverOptions.superlu_direct(),
        keep_solution=False,
    )

    assert len(study.levels) == 2
    assert study.finest_solution is None
    assert study.levels[-1].velocity_l2_error < study.levels[0].velocity_l2_error
    assert study.levels[-1].velocity_h1_error < study.levels[0].velocity_h1_error
    assert study.levels[-1].pressure_l2_error < study.levels[0].pressure_l2_error
    assert study.metadata["linear_backend"] == "superlu"


@requires_fem_stack
@pytest.mark.parametrize("method", available_mms_methods())
def test_3d_mms_builds_and_reduces_pressure_error(method: str) -> None:
    levels_seen: list[int] = []
    study = run_mms_convergence(
        bubble_case_3d(),
        method=method,  # type: ignore[arg-type]
        resolutions=(2, 3),
        options=FEniCSSolverOptions.superlu_direct(),
        callback=lambda level: levels_seen.append(level.resolution),
    )

    assert levels_seen == [2, 3]
    assert study.finest_solution is not None
    assert study.levels[-1].pressure_l2_error < study.levels[0].pressure_l2_error


@requires_fem_stack
def test_mms_runner_validates_controls() -> None:
    case = boundary_layer_case_2d()
    for kwargs, message in (
        ({"resolutions": (2,)}, "at least two"),
        ({"resolutions": (0, 2)}, "positive"),
        ({"resolutions": (2, 2)}, "strictly increasing"),
        ({"tau_factor": -1.0}, "tau_factor"),
        ({"tau_gamma_cap": 0.0}, "tau_gamma_cap"),
        ({"tau_gamma_cap": 1.0}, "tau_gamma_cap"),
        ({"tau_gamma_cap": 1.1}, "tau_gamma_cap"),
        ({"m_t": 0.0}, "m_t"),
        ({"alpha_edge": 0.0}, "alpha_edge"),
        ({"facet_law": "other"}, "facet_law"),
        ({"facet_size_mode": "other"}, "facet_size_mode"),
        ({"face_refinement": 1}, "face_refinement"),
        ({"method": "other"}, "method"),
    ):
        with pytest.raises(ValueError, match=message):
            run_mms_convergence(case, **kwargs)  # type: ignore[arg-type]

    without_cell_stabilization = run_mms_convergence(
        case,
        method="usfem_p1dg0",
        resolutions=(2, 3),
        tau_factor=0.0,
        options=FEniCSSolverOptions.superlu_direct(),
        keep_solution=False,
    )
    assert without_cell_stabilization.metadata["tau_factor"] == 0.0

    with pytest.raises(ValueError, match="only for 3D"):
        run_mms_convergence(
            case,
            method="usfem_p1dg0",
            resolutions=(2, 3),
            facet_law="face3d",
            options=FEniCSSolverOptions.superlu_direct(),
        )

    for facet_size_mode in ("facet_diameter", "representative"):
        study = run_mms_convergence(
            case,
            method="usfem_p1dg1",
            resolutions=(2, 3),
            facet_size_mode=facet_size_mode,  # type: ignore[arg-type]
            options=FEniCSSolverOptions.superlu_direct(),
            keep_solution=False,
        )
        assert study.metadata["facet_size_mode"] == facet_size_mode

    three_dimensional = run_mms_convergence(
        bubble_case_3d(),
        method="usfem_p1dg0",
        resolutions=(2, 3),
        facet_law="classic",
        facet_size_mode="representative",
        options=FEniCSSolverOptions.superlu_direct(),
        keep_solution=False,
    )
    assert three_dimensional.metadata["facet_size_mode"] == "representative"

    with pytest.warns(RuntimeWarning, match="alpha_edge is ignored"):
        parameter_free_face3d = run_mms_convergence(
            bubble_case_3d(),
            method="usfem_p1dg0",
            resolutions=(2, 3),
            facet_law="face3d",
            alpha_edge=2.0,
            options=FEniCSSolverOptions.superlu_direct(),
            keep_solution=False,
        )
    baseline_face3d = run_mms_convergence(
        bubble_case_3d(),
        method="usfem_p1dg0",
        resolutions=(2, 3),
        facet_law="face3d",
        options=FEniCSSolverOptions.superlu_direct(),
        keep_solution=False,
    )
    assert parameter_free_face3d.metadata["alpha_edge_active"] is False
    assert parameter_free_face3d.levels[-1].errors() == pytest.approx(
        baseline_face3d.levels[-1].errors(),
        rel=1.0e-13,
    )

    with pytest.raises(ValueError, match="defined only in 2D"):
        run_mms_convergence(
            bubble_case_3d(),
            method="usfem_p1dg0",
            resolutions=(2, 3),
            facet_law="classic",
            facet_size_mode="facet_diameter",
            options=FEniCSSolverOptions.superlu_direct(),
        )

    with pytest.raises(RuntimeError, match="negative"):
        _safe_sqrt(-1.0, name="test")


def test_gmsh_helpers_validate_and_classify_boundaries() -> None:
    physical_calls: list[tuple[object, ...]] = []
    physical_names: list[tuple[object, ...]] = []
    boxes = {
        1: (2.0, 4.0, 0.0, 2.0, 5.0, 0.0),
        2: (3.0, 4.0, 0.0, 3.0, 5.0, 0.0),
        3: (2.0, 4.0, 0.0, 3.0, 4.0, 0.0),
        4: (2.0, 5.0, 0.0, 3.0, 5.0, 0.0),
        5: (2.25, 4.25, 0.0, 2.75, 4.75, 0.0),
    }

    class Model:
        def addPhysicalGroup(
            self,
            dimension: int,
            entities: list[int],
            tag: int,
        ) -> int:
            physical_calls.append((dimension, entities, tag))
            return tag

        def setPhysicalName(self, dimension: int, tag: int, name: str) -> None:
            physical_names.append((dimension, tag, name))

        def getEntities(self, dimension: int) -> list[tuple[int, int]]:
            assert dimension == 1
            return [(dimension, entity) for entity in boxes]

        def getBoundingBox(self, dimension: int, entity: int) -> tuple[float, ...]:
            assert dimension == 1
            return boxes[entity]

    gmsh = SimpleNamespace(model=Model())
    assert add_physical_group(gmsh, 2, [7, 8], tag=3, name="fluid") == 3
    assert physical_calls == [(2, [7, 8], 3)]
    assert physical_names == [(2, 3, "fluid")]

    groups, unclassified = axis_aligned_boundary_entities(
        gmsh,
        2,
        bounds=((2.0, 3.0), (4.0, 5.0)),
    )
    assert groups == {
        "x_min": [1],
        "x_max": [2],
        "y_min": [3],
        "y_max": [4],
    }
    assert unclassified == [5]

    for arguments, message in (
        ((-1, [1]), "dimension"),
        ((1, []), "entities"),
    ):
        with pytest.raises(ValueError, match=message):
            add_physical_group(
                gmsh,
                arguments[0],
                arguments[1],
                tag=1,
                name="x",
            )
    with pytest.raises(ValueError, match="tag"):
        add_physical_group(gmsh, 1, [1], tag=0, name="x")
    with pytest.raises(ValueError, match="name"):
        add_physical_group(gmsh, 1, [1], tag=1, name=" ")

    for kwargs, message in (
        ({"geometric_dimension": 1}, "geometric_dimension"),
        ({"geometric_dimension": 2, "tolerance": 0.0}, "tolerance"),
        ({"geometric_dimension": 2, "bounds": ((0.0, 1.0),)}, "bounds"),
        (
            {
                "geometric_dimension": 2,
                "bounds": ((0.0, 0.0), (0.0, 1.0)),
            },
            "strictly increasing",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            axis_aligned_boundary_entities(gmsh, **kwargs)


def test_gmsh_mesh_size_and_conversion_validation() -> None:
    option_calls: list[tuple[str, float]] = []
    gmsh = SimpleNamespace(
        option=SimpleNamespace(setNumber=lambda name, value: option_calls.append((name, value)))
    )
    configure_uniform_mesh_size(
        gmsh,
        0.25,
        element_order=2,
        size_from_curvature=False,
    )
    assert option_calls == [
        ("Mesh.MeshSizeMin", 0.25),
        ("Mesh.MeshSizeMax", 0.25),
        ("Mesh.MeshSizeFromCurvature", 0),
        ("Mesh.ElementOrder", 2),
    ]
    with pytest.raises(ValueError, match="target_size"):
        configure_uniform_mesh_size(gmsh, 0.0)
    with pytest.raises(ValueError, match="element_order"):
        configure_uniform_mesh_size(gmsh, 0.25, element_order=0)

    assert require_gmsh().__version__
    with pytest.raises(TypeError, match="callable"):
        generate_dolfinx_gmsh_mesh(1, name="x", geometric_dimension=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        generate_dolfinx_gmsh_mesh(lambda _gmsh: None, name="", geometric_dimension=2)
    with pytest.raises(ValueError, match="geometric_dimension"):
        generate_dolfinx_gmsh_mesh(lambda _gmsh: None, name="x", geometric_dimension=1)
    with pytest.raises(ValueError, match="model_rank"):
        generate_dolfinx_gmsh_mesh(
            lambda _gmsh: None,
            name="x",
            geometric_dimension=2,
            comm=SimpleNamespace(rank=0, size=1),
            model_rank=1,
        )


@requires_fem_stack
def test_generate_dolfinx_gmsh_mesh_finalizes_after_model_failure() -> None:
    _skip_native_gmsh_on_windows()
    gmsh_module = require_gmsh()
    if gmsh_module.isInitialized():
        gmsh_module.finalize()

    def fail_model_build(_gmsh: object) -> None:
        raise RuntimeError("model build failed")

    with pytest.raises(RuntimeError, match="model build failed"):
        generate_dolfinx_gmsh_mesh(
            fail_model_build,
            name="failing_model",
            geometric_dimension=2,
        )
    assert not gmsh_module.isInitialized()


def test_vug_entity_classification_and_public_validation() -> None:
    benchmark = CenteredVugBenchmark()
    bad_gmsh = SimpleNamespace(
        model=SimpleNamespace(
            getEntities=lambda _dimension: [(2, 1)],
        )
    )
    with pytest.raises(RuntimeError, match="exactly two"):
        _classify_volume_entities(bad_gmsh, benchmark)
    with pytest.raises(ValueError, match="body_fitted"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(mesh_representation="structured"))
    with pytest.raises(ValueError, match="method"):
        run_centered_vug_benchmark(benchmark, method="invalid")  # type: ignore[arg-type]


def test_body_fitted_vug_mesh_rejects_incomplete_gmsh_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete_outer_groups = {
        "x_min": [1],
        "x_max": [2],
        "y_min": [3],
        "y_max": [4],
    }
    state: dict[str, object] = {
        "entities": [(2, 1)],
        "outer_groups": complete_outer_groups,
        "interface": [],
    }
    occ = SimpleNamespace(
        addRectangle=lambda *_args: 1,
        addDisk=lambda *_args: 2,
        fragment=lambda *_args: None,
        synchronize=lambda: None,
        getMass=lambda _dimension, tag: 0.03 if tag == 2 else 0.97,
    )
    fake_gmsh = SimpleNamespace(
        model=SimpleNamespace(
            occ=occ,
            getEntities=lambda _dimension: state["entities"],
            mesh=SimpleNamespace(generate=lambda _dimension: None),
        )
    )

    def fake_generate(build_model: object, **_kwargs: object) -> SimpleNamespace:
        build_model(fake_gmsh)  # type: ignore[operator]
        return SimpleNamespace(
            mesh=object(),
            cell_tags=object(),
            facet_tags=object(),
            physical_groups={},
        )

    monkeypatch.setattr(vug_module, "generate_dolfinx_gmsh_mesh", fake_generate)
    monkeypatch.setattr(vug_module, "add_physical_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vug_module,
        "configure_uniform_mesh_size",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        vug_module,
        "axis_aligned_boundary_entities",
        lambda *_args, **_kwargs: (
            state["outer_groups"],
            state["interface"],
        ),
    )

    state["entities"] = [(2, 1), (2, 2)]
    with pytest.raises(RuntimeError, match="exactly one volume"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(radius=0.0))

    state["entities"] = [(2, 1)]
    state["outer_groups"] = {**complete_outer_groups, "x_min": []}
    with pytest.raises(RuntimeError, match="expected outer facets"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(radius=0.0))

    state["entities"] = [(2, 1), (2, 2)]
    state["outer_groups"] = complete_outer_groups
    state["interface"] = []
    with pytest.raises(RuntimeError, match="vug interface"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(radius=0.1))

    state["entities"] = [(2, 1)]
    state["interface"] = [7]
    with pytest.raises(RuntimeError, match="unexpectedly contains interior"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(radius=0.0))

    monkeypatch.setattr(
        vug_module,
        "generate_dolfinx_gmsh_mesh",
        lambda *_args, **_kwargs: SimpleNamespace(
            mesh=object(),
            cell_tags=None,
            facet_tags=object(),
            physical_groups={},
        ),
    )
    with pytest.raises(RuntimeError, match="physical tags"):
        make_body_fitted_centered_vug_mesh(CenteredVugBenchmark(radius=0.0))


@requires_fem_stack
@pytest.mark.parametrize("dimension", [2, 3])
def test_body_fitted_vug_mesh_preserves_physical_tags(dimension: int) -> None:
    _skip_native_gmsh_on_windows()
    pytest.importorskip("gmsh")
    benchmark = CenteredVugBenchmark(
        dimension=dimension,  # type: ignore[arg-type]
        resolution=4,
    )
    tagged = make_body_fitted_centered_vug_mesh(benchmark)

    assert set(tagged.cell_tags.values.tolist()) == {1, 2}
    expected_facets = {1, 2, 3, 4, 7} if dimension == 2 else set(range(1, 8))
    assert set(tagged.facet_tags.values.tolist()) == expected_facets
    assert {"matrix", "vug", "vug_interface"}.issubset(tagged.physical_groups)


@requires_fem_stack
def test_matrix_only_body_fitted_vug_mesh_has_no_internal_interface() -> None:
    _skip_native_gmsh_on_windows()
    pytest.importorskip("gmsh")
    tagged = make_body_fitted_centered_vug_mesh(
        CenteredVugBenchmark(dimension=2, resolution=4, radius=0.0)
    )

    assert set(tagged.cell_tags.values.tolist()) == {1}
    assert set(tagged.facet_tags.values.tolist()) == {1, 2, 3, 4}
    assert "vug" not in tagged.physical_groups
    assert "vug_interface" not in tagged.physical_groups


@requires_fem_stack
def test_physical_centered_vug_flow_models_recover_matrix_baseline() -> None:
    _skip_native_gmsh_on_windows()
    pytest.importorskip("gmsh")
    options = FEniCSSolverOptions.superlu_direct()
    matrix_case = CenteredVugFlowCase2D(area_fraction=0.0, mesh_resolution=8)

    for model in ("darcy_brinkman", "darcy_darcy"):
        result = run_centered_vug_flow_case(
            matrix_case,
            model=model,  # type: ignore[arg-type]
            options=options,
        )
        assert result.permeability == pytest.approx(
            matrix_case.matrix_permeability_m2,
            rel=1.0e-10,
        )
        assert result.flow_rate > 0.0
        assert result.cross_section_area == pytest.approx(matrix_case.side_length_m)
        assert result.metadata["velocity_degree"] == 2
        assert result.metadata["pressure_degree"] == 1
        assert result.metadata["mesh_size_policy"] == "nearly_uniform_body_fitted"
        expected_vug_drag = (
            0.0
            if model == "darcy_brinkman"
            else matrix_case.matrix_permeability_m2 / matrix_case.vug_permeability_m2
        )
        assert result.metadata["vug_drag_coefficient_dimensionless"] == pytest.approx(
            expected_vug_drag
        )
        assert result.metadata["vug_permeability_m2"] == (
            None if model == "darcy_brinkman" else matrix_case.vug_permeability_m2
        )

    vug_case = CenteredVugFlowCase2D(area_fraction=0.7, mesh_resolution=8)
    brinkman = run_centered_vug_flow_case(
        vug_case,
        model="darcy_brinkman",
        options=options,
    )
    darcy = run_centered_vug_flow_case(
        vug_case,
        model="darcy_darcy",
        options=options,
    )
    assert brinkman.permeability > vug_case.matrix_permeability_m2
    assert darcy.permeability > vug_case.matrix_permeability_m2
    assert darcy.permeability == pytest.approx(brinkman.permeability, rel=0.2)
    assert brinkman.metadata["vug_drag_pa_s_per_m2"] == 0.0
    assert darcy.metadata["vug_drag_pa_s_per_m2"] == pytest.approx(
        vug_case.dynamic_viscosity_pa_s / vug_case.vug_permeability_m2
    )
    with pytest.raises(ValueError, match="model"):
        run_centered_vug_flow_case(vug_case, model="bad", options=options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vms_constant"):
        run_centered_vug_flow_case(
            vug_case,
            model="darcy_darcy",
            options=options,
            vms_constant=0.0,
        )


@requires_fem_stack
def test_body_fitted_vug_usfem_matches_taylor_hood_flux_in_2d() -> None:
    _skip_native_gmsh_on_windows()
    pytest.importorskip("gmsh")
    benchmark = CenteredVugBenchmark(dimension=2, resolution=8)
    options = FEniCSSolverOptions.superlu_direct()
    reference = run_centered_vug_benchmark(
        benchmark,
        method="taylor_hood",
        options=options,
    )
    usfem = run_centered_vug_benchmark(
        benchmark,
        method="usfem_p1dg1",
        options=options,
    )

    assert reference.flow_rate > 0.0
    assert usfem.flow_rate == pytest.approx(reference.flow_rate, rel=5.0e-3)
    assert usfem.metadata["geometry_representation"] == "body_fitted"
    assert usfem.metadata["facet_law"] == "reaction_diffusion"
    assert usfem.metadata["facet_size_mode"] == "facet_measure"
    assert usfem.metadata["pressure_constraint"] == "natural_traction"


@requires_fem_stack
def test_petsc_paths_and_float32_rejections() -> None:
    petsc_options = FEniCSSolverOptions.direct_reference()
    mms_result = run_mms_convergence(
        boundary_layer_case_2d(viscosity=0.1),
        method="taylor_hood",
        resolutions=(2, 3),
        options=petsc_options,
        keep_solution=False,
    )
    assert mms_result.metadata["linear_backend"] == "petsc"

    float32_petsc = FEniCSSolverOptions(
        linear_backend="petsc",
        linear_system_dtype="float32",
    )
    with pytest.raises(ValueError, match="float32"):
        run_mms_convergence(
            boundary_layer_case_2d(viscosity=0.1),
            resolutions=(2, 3),
            options=float32_petsc,
        )
    with pytest.raises(ValueError, match="float32"):
        run_centered_vug_benchmark(
            CenteredVugBenchmark(resolution=4),
            options=float32_petsc,
        )
    with pytest.raises(ValueError, match="float32"):
        run_centered_vug_flow_case(
            CenteredVugFlowCase2D(area_fraction=0.0, mesh_resolution=8),
            model="darcy_brinkman",
            options=float32_petsc,
        )


@requires_fem_stack
def test_petsc_body_fitted_vug_paths() -> None:
    _skip_native_gmsh_on_windows()
    pytest.importorskip("gmsh")
    petsc_options = FEniCSSolverOptions.direct_reference()
    physical_flow_result = run_centered_vug_flow_case(
        CenteredVugFlowCase2D(area_fraction=0.0, mesh_resolution=8),
        model="darcy_brinkman",
        options=petsc_options,
    )
    assert physical_flow_result.metadata["linear_backend"] == "petsc"

    gmsh = require_gmsh()
    gmsh.initialize()
    try:
        with pytest.warns(RuntimeWarning, match="nearly cancel matrix drag"):
            vug_result = run_centered_vug_benchmark(
                CenteredVugBenchmark(resolution=4),
                method="usfem_p1dg0",
                options=petsc_options,
            )
    finally:
        gmsh.finalize()
    assert vug_result.metadata["linear_backend"] == "petsc"
    assert vug_result.metadata["pressure_degree"] == 0


@requires_fem_stack
@pytest.mark.parametrize("method", available_mms_methods())
def test_structured_vug_benchmark_runs_all_methods(method: str) -> None:
    benchmark = CenteredVugBenchmark(
        dimension=2,
        resolution=3,
        mesh_representation="structured",
    )
    result = run_centered_vug_benchmark(
        benchmark,
        method=method,  # type: ignore[arg-type]
        tau_gamma_cap=0.5 if method == "usfem_p1dg0" else None,
        options=FEniCSSolverOptions.superlu_direct(),
    )

    assert result.flow_rate > 0.0
    assert result.metadata["geometry_representation"] == "structured"
    assert result.metadata["represented_vug_fraction"] == benchmark.represented_fraction
