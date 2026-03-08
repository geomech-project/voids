from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
import warnings

import numpy as np
import pytest

from voids.benchmarks import (
    XLBOptions,
    benchmark_segmented_volume_with_xlb,
    solve_binary_volume_with_xlb,
)
import voids.benchmarks.xlb as xlb_mod
from voids.physics.singlephase import FluidSinglePhase


def _make_fake_xlb_api(
    velocity_samples: list[float | np.ndarray],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build a deterministic fake XLB API for unit tests."""

    capture: dict[str, object] = {"velocity_samples": list(velocity_samples)}

    class _FakeJax:
        @staticmethod
        def block_until_ready(value):
            return value

    fake_precision = SimpleNamespace(compute_precision=SimpleNamespace(jax_dtype=np.float64))

    class _FakePrecisionPolicy:
        FP32FP32 = fake_precision

    class _FakeComputeBackend:
        JAX = "JAX"

    class _FakeVelocitySet:
        def __init__(self, precision_policy, compute_backend):
            self.precision_policy = precision_policy
            self.compute_backend = compute_backend
            self.cs2 = np.asarray(1.0 / 3.0)

    class _FakeGridModule:
        @staticmethod
        def grid_factory(shape, compute_backend=None):
            capture["grid_shape"] = tuple(shape)
            capture["grid_backend"] = compute_backend
            return SimpleNamespace(shape=tuple(shape), compute_backend=compute_backend)

    class _FakeXLB:
        __version__ = "fake-xlb"
        grid = _FakeGridModule()

        @staticmethod
        def init(velocity_set, compute_backend, precision_policy):
            capture["init_velocity_set"] = velocity_set
            capture["init_backend"] = compute_backend
            capture["init_precision_policy"] = precision_policy

    class _FakeRegularizedBC:
        def __init__(self, kind, prescribed_value, indices):
            self.kind = kind
            self.prescribed_value = prescribed_value
            self.indices = indices

    class _FakeHalfwayBounceBackBC:
        def __init__(self, indices):
            self.indices = indices

    class _FakeStepper:
        def __init__(self, grid, boundary_conditions, collision_type, streaming_scheme):
            self.grid = grid
            self.boundary_conditions = boundary_conditions
            self.collision_type = collision_type
            self.streaming_scheme = streaming_scheme
            self.last_step = 0
            capture["boundary_conditions"] = boundary_conditions
            capture["collision_type"] = collision_type
            capture["streaming_scheme"] = streaming_scheme

        def prepare_fields(self):
            return object(), object(), "bc_mask", "missing_mask"

        def __call__(self, f_0, f_1, bc_mask, missing_mask, omega, step):
            self.last_step = step + 1
            capture.setdefault("omegas", []).append(float(np.asarray(omega)))
            return object(), object()

        def macroscopic(self, f_current):
            index = min(max(self.last_step - 1, 0), len(velocity_samples) - 1)
            value = velocity_samples[index]
            if np.isscalar(value):
                axial = np.full(self.grid.shape, float(value), dtype=float)
            else:
                axial = np.asarray(value, dtype=float)
                assert axial.shape == self.grid.shape
            velocity = np.zeros((len(self.grid.shape), *self.grid.shape), dtype=float)
            velocity[0] = axial
            return np.ones(self.grid.shape, dtype=float), velocity

    api = {
        "jax": _FakeJax,
        "xlb": _FakeXLB,
        "ComputeBackend": _FakeComputeBackend,
        "HalfwayBounceBackBC": _FakeHalfwayBounceBackBC,
        "RegularizedBC": _FakeRegularizedBC,
        "IncompressibleNavierStokesStepper": _FakeStepper,
        "PrecisionPolicy": _FakePrecisionPolicy,
        "D2Q9": _FakeVelocitySet,
        "D3Q19": _FakeVelocitySet,
    }
    return capture, api


def test_xlb_helpers_cover_edge_cases() -> None:
    """Test helper functions with deterministic small arrays."""

    assert xlb_mod._rel_diff(3.0, 1.0) == pytest.approx(2.0 / 3.0)
    assert xlb_mod._axis_to_index("y", 2) == 1

    with pytest.raises(ValueError, match="flow_axis must be one of"):
        xlb_mod._axis_to_index("q", 3)
    with pytest.raises(ValueError, match="not compatible with a 2D volume"):
        xlb_mod._axis_to_index("z", 2)

    mask = np.array([[False, True], [True, False]])
    assert xlb_mod._mask_to_indices(mask) == [[0, 1], [1, 0]]
    assert xlb_mod._mask_to_indices(np.zeros((2, 2), dtype=bool)) is None

    axial = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    void_mask = np.array(
        [
            [[True, False], [False, True]],
            [[False, False], [False, False]],
        ]
    )
    profile = xlb_mod._superficial_velocity_profile(axial, void_mask)
    assert profile.tolist() == pytest.approx([1.25, 0.0])


def test_import_xlb_success_with_stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test optional XLB import wiring without the real dependency."""

    fake_jax = ModuleType("jax")
    fake_xlb = ModuleType("xlb")
    fake_xlb.__path__ = []
    fake_xlb_operator = ModuleType("xlb.operator")
    fake_xlb_operator.__path__ = []
    fake_compute_backend = ModuleType("xlb.compute_backend")
    fake_boundary_condition = ModuleType("xlb.operator.boundary_condition")
    fake_stepper = ModuleType("xlb.operator.stepper")
    fake_precision_policy = ModuleType("xlb.precision_policy")
    fake_velocity_set = ModuleType("xlb.velocity_set")

    fake_compute_backend.ComputeBackend = object()
    fake_boundary_condition.HalfwayBounceBackBC = object()
    fake_boundary_condition.RegularizedBC = object()
    fake_stepper.IncompressibleNavierStokesStepper = object()
    fake_precision_policy.PrecisionPolicy = object()
    fake_velocity_set.D2Q9 = object()
    fake_velocity_set.D3Q19 = object()

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "xlb", fake_xlb)
    monkeypatch.setitem(sys.modules, "xlb.operator", fake_xlb_operator)
    monkeypatch.setitem(sys.modules, "xlb.compute_backend", fake_compute_backend)
    monkeypatch.setitem(sys.modules, "xlb.operator.boundary_condition", fake_boundary_condition)
    monkeypatch.setitem(sys.modules, "xlb.operator.stepper", fake_stepper)
    monkeypatch.setitem(sys.modules, "xlb.precision_policy", fake_precision_policy)
    monkeypatch.setitem(sys.modules, "xlb.velocity_set", fake_velocity_set)

    api = xlb_mod._import_xlb()

    assert api["jax"] is fake_jax
    assert api["xlb"] is fake_xlb
    assert api["ComputeBackend"] is fake_compute_backend.ComputeBackend
    assert api["RegularizedBC"] is fake_boundary_condition.RegularizedBC
    assert api["D3Q19"] is fake_velocity_set.D3Q19


def test_import_xlb_raises_clean_error_on_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional XLB import failure message."""

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax":
            raise ImportError("missing jax")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="optional 'xlb' dependency"):
        xlb_mod._import_xlb()


def test_benchmark_segmented_volume_with_xlb_rejects_nonbinary_inputs() -> None:
    """Test binary-volume validation before optional XLB imports."""

    phases = np.array([[[0, 2], [1, 0]], [[1, 0], [0, 1]]], dtype=int)

    with pytest.raises(ValueError, match="phases must be binary with void=1 and solid=0"):
        benchmark_segmented_volume_with_xlb(phases, voxel_size=1.0)


def test_benchmark_segmented_volume_with_xlb_rejects_invalid_rank() -> None:
    """Test rank validation before extraction or optional XLB imports."""

    phases = np.array([0, 1, 0, 1], dtype=int)

    with pytest.raises(ValueError, match="phases must be a 2D or 3D binary segmented volume"):
        benchmark_segmented_volume_with_xlb(phases, voxel_size=1.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (XLBOptions(backend="warp"), "supports only backend='jax'"),
        (XLBOptions(max_steps=0), "max_steps must be positive"),
        (XLBOptions(min_steps=-1), "min_steps must be non-negative"),
        (XLBOptions(check_interval=0), "check_interval must be positive"),
        (XLBOptions(steady_rtol=0.0), "steady_rtol must be positive"),
        (XLBOptions(lattice_viscosity=0.0), "lattice_viscosity must be positive"),
        (
            XLBOptions(rho_inlet=1.0, rho_outlet=1.0),
            "rho_inlet must be greater than rho_outlet",
        ),
        (
            XLBOptions(inlet_outlet_buffer_cells=-1),
            "inlet_outlet_buffer_cells must be non-negative",
        ),
    ],
)
def test_xlb_direct_solver_rejects_invalid_numerical_options(
    options: XLBOptions,
    message: str,
) -> None:
    """Test numerical validation branches that happen before XLB import."""

    phases = np.ones((4, 5), dtype=int)

    with pytest.raises(ValueError, match=message):
        solve_binary_volume_with_xlb(phases, voxel_size=1.0, options=options)


def test_xlb_direct_solver_rejects_incompatible_flow_axis() -> None:
    """Test early axis validation before any XLB runtime work."""

    phases = np.ones((4, 5), dtype=int)

    with pytest.raises(ValueError, match="flow_axis 'z' is not compatible with a 2D volume"):
        solve_binary_volume_with_xlb(phases, voxel_size=1.0, flow_axis="z")


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        (
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
            "The inlet plane contains no void voxels",
        ),
        (
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                ],
                dtype=int,
            ),
            "The outlet plane contains no void voxels",
        ),
    ],
)
def test_xlb_direct_solver_rejects_closed_inlet_or_outlet(
    phases: np.ndarray,
    message: str,
) -> None:
    """Test inlet and outlet connectivity checks."""

    with pytest.raises(ValueError, match=message):
        solve_binary_volume_with_xlb(phases, voxel_size=1.0, flow_axis="x")


def test_xlb_direct_solver_api_available_or_clean_import_error() -> None:
    """Test that the direct XLB solve either runs or fails with a clean import error."""

    phases = np.zeros((8, 10, 10), dtype=int)
    phases[:, 3:7, 3:7] = 1

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            solve_binary_volume_with_xlb(
                phases,
                voxel_size=1.0,
                flow_axis="x",
                options=XLBOptions(max_steps=10, min_steps=0, check_interval=5),
            )
    except ImportError:
        return


def test_xlb_direct_solver_rejects_unknown_precision_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test precision-policy validation after importing the XLB adapter."""

    _, fake_api = _make_fake_xlb_api([0.2, 0.2])
    monkeypatch.setattr(xlb_mod, "_import_xlb", lambda: fake_api)

    with pytest.raises(ValueError, match="Unknown XLB precision policy"):
        solve_binary_volume_with_xlb(
            np.ones((4, 5), dtype=int),
            voxel_size=1.0,
            options=XLBOptions(
                precision_policy="UNKNOWN",
                max_steps=2,
                min_steps=1,
                check_interval=1,
            ),
        )


def test_xlb_direct_solver_2d_no_buffer_flow_axis_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a deterministic 2D solve path with no inlet/outlet reservoir padding."""

    capture, fake_api = _make_fake_xlb_api([0.2, 0.2])
    monkeypatch.setattr(xlb_mod, "_import_xlb", lambda: fake_api)

    phases = np.ones((4, 2), dtype=int)
    result = solve_binary_volume_with_xlb(
        phases,
        voxel_size=2.0,
        flow_axis="y",
        options=XLBOptions(
            max_steps=2,
            min_steps=1,
            check_interval=1,
            steady_rtol=1.0e-12,
            inlet_outlet_buffer_cells=0,
            lattice_viscosity=0.1,
            rho_inlet=1.001,
            rho_outlet=1.0,
        ),
    )

    assert capture["grid_shape"] == (2, 4)
    assert capture["grid_backend"] == "JAX"
    assert len(capture["boundary_conditions"]) == 3
    assert result.flow_axis == "y"
    assert result.inlet_outlet_buffer_cells == 0
    assert result.converged is True
    assert result.n_steps == 2
    assert result.axial_velocity_lattice.shape == phases.shape
    assert np.all(result.axial_velocity_lattice > 0.0)
    assert result.permeability > 0.0
    assert result.backend_version == "fake-xlb"


def test_xlb_inlet_outlet_masks_exclude_solid_voxels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: solid voxels on inlet/outlet planes must not receive pressure BCs.

    When ``inlet_outlet_buffer_cells=0`` the inlet and outlet planes of the
    sample are used directly as the BC planes.  Before the fix, any interior
    voxel on those planes – including solids – was assigned a pressure BC.
    This test verifies that solid cells are excluded from the pressure BC
    indices and are instead covered by the bounce-back BC.
    """

    capture, fake_api = _make_fake_xlb_api([0.2, 0.2])
    monkeypatch.setattr(xlb_mod, "_import_xlb", lambda: fake_api)

    # Flow along x (axis 0).  Shape (3 rows, 5 cols).
    # Row 0 (inlet):  solid at column 1.
    # Row 2 (outlet): solid at column 3.
    # Columns 0 and 4 are sealed side-wall edges.
    phases = np.array(
        [
            [1, 0, 1, 1, 1],  # inlet row: solid at col 1
            [1, 1, 1, 1, 1],  # interior: all void
            [1, 1, 1, 0, 1],  # outlet row: solid at col 3
        ],
        dtype=int,
    )
    solve_binary_volume_with_xlb(
        phases,
        voxel_size=1.0,
        flow_axis="x",
        options=XLBOptions(
            max_steps=2,
            min_steps=1,
            check_interval=1,
            steady_rtol=1.0e-12,
            inlet_outlet_buffer_cells=0,
        ),
    )

    bcs = capture["boundary_conditions"]
    # Expect: inlet pressure BC, outlet pressure BC, bounce-back BC.
    assert len(bcs) == 3

    # Unpack the per-dimension index lists returned by _mask_to_indices.
    inlet_cells = set(zip(bcs[0].indices[0], bcs[0].indices[1]))
    outlet_cells = set(zip(bcs[1].indices[0], bcs[1].indices[1]))
    bounceback_cells = set(zip(bcs[2].indices[0], bcs[2].indices[1]))

    # Solid cell on the inlet plane must not be assigned a pressure BC …
    assert (0, 1) not in inlet_cells
    # … but must be covered by the bounce-back BC.
    assert (0, 1) in bounceback_cells

    # Solid cell on the outlet plane must not be assigned a pressure BC …
    assert (2, 3) not in outlet_cells
    # … but must be covered by the bounce-back BC.
    assert (2, 3) in bounceback_cells

    # Sanity: interior void cell on the inlet plane IS in the pressure BC.
    assert (0, 2) in inlet_cells
    # Sanity: interior void cell on the outlet plane IS in the pressure BC.
    assert (2, 1) in outlet_cells


def test_xlb_direct_solver_warns_when_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that under-converged runs surface a warning instead of failing silently."""

    _, fake_api = _make_fake_xlb_api([0.1, 0.2, 0.4])
    monkeypatch.setattr(xlb_mod, "_import_xlb", lambda: fake_api)

    phases = np.ones((4, 5), dtype=int)
    with pytest.warns(RuntimeWarning, match="did not satisfy the steady-state tolerance"):
        result = solve_binary_volume_with_xlb(
            phases,
            voxel_size=1.0,
            options=XLBOptions(
                max_steps=3,
                min_steps=1,
                check_interval=1,
                steady_rtol=1.0e-9,
                inlet_outlet_buffer_cells=1,
            ),
        )

    assert result.converged is False
    assert result.n_steps == 3
    assert result.permeability > 0.0


def test_xlb_direct_solver_raises_on_nonphysical_permeability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that zero or negative flux does not propagate as a benchmark result."""

    _, fake_api = _make_fake_xlb_api([0.0])
    monkeypatch.setattr(xlb_mod, "_import_xlb", lambda: fake_api)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="non-physical permeability estimate"):
            solve_binary_volume_with_xlb(
                np.ones((4, 5), dtype=int),
                voxel_size=1.0,
                options=XLBOptions(
                    max_steps=1,
                    min_steps=0,
                    check_interval=1,
                    inlet_outlet_buffer_cells=1,
                ),
            )


def test_segmented_volume_xlb_to_record_handles_missing_voids_permeability() -> None:
    """Test tabulation fallback when the `voids` solve did not populate permeability."""

    extract = SimpleNamespace(
        flow_axis="x",
        net=SimpleNamespace(Np=3, Nt=2),
        backend="stub",
        backend_version="0.1",
    )
    xlb_result = xlb_mod.XLBDirectSimulationResult(
        flow_axis="x",
        voxel_size=1.0,
        image_porosity=0.5,
        sample_lengths={"x": 4.0},
        sample_cross_sections={"x": 6.0},
        lattice_viscosity=0.1,
        lattice_density_inlet=1.001,
        lattice_density_outlet=1.0,
        lattice_pressure_drop=1.0e-3,
        inlet_outlet_buffer_cells=2,
        omega=1.25,
        superficial_velocity_lattice=0.2,
        superficial_velocity_profile_lattice=np.array([0.2, 0.2]),
        axial_velocity_lattice=np.ones((2, 2), dtype=float),
        converged=True,
        n_steps=20,
        convergence_metric=1.0e-4,
        permeability=3.0,
        backend="jax",
        backend_version="fake-xlb",
    )
    result = xlb_mod.SegmentedVolumeXLBResult(
        extract=extract,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=SimpleNamespace(pin=2.0e5, pout=1.0e5),
        options=SimpleNamespace(conductance_model="stub", solver="direct"),
        xlb_options=XLBOptions(),
        image_porosity=0.5,
        absolute_porosity=0.45,
        effective_porosity=0.40,
        voids_result=SimpleNamespace(permeability=None, mass_balance_error=0.0),
        xlb_result=xlb_result,
        permeability_abs_diff=np.nan,
        permeability_rel_diff=np.nan,
    )

    record = result.to_record()

    assert np.isnan(record["k_voids"])
    assert record["k_xlb"] == pytest.approx(3.0)
    assert record["xlb_buffer_cells"] == 2


def test_benchmark_segmented_volume_with_xlb_uses_defaults_and_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the high-level benchmark wrapper without depending on the real XLB runtime."""

    phases = np.ones((4, 5, 6), dtype=int)
    extract_net = SimpleNamespace(
        Np=4,
        Nt=3,
        pore_labels={"inlet_xmin": [True, False], "outlet_xmax": [False, True]},
    )
    extract = SimpleNamespace(
        flow_axis="x",
        net=extract_net,
        backend="porespy",
        backend_version="1.0",
    )
    captured: dict[str, object] = {}

    def _fake_extract(
        phases_arg,
        *,
        voxel_size,
        flow_axis,
        length_unit,
        pressure_unit,
        extraction_kwargs,
        provenance_notes,
        strict,
    ):
        captured["extract_args"] = {
            "phases": np.asarray(phases_arg),
            "voxel_size": voxel_size,
            "flow_axis": flow_axis,
            "length_unit": length_unit,
            "pressure_unit": pressure_unit,
            "extraction_kwargs": extraction_kwargs,
            "provenance_notes": provenance_notes,
            "strict": strict,
        }
        return extract

    def _fake_solve(net, *, fluid, bc, axis, options):
        captured["solve_args"] = {
            "net": net,
            "fluid": fluid,
            "bc": bc,
            "axis": axis,
            "options": options,
        }
        return SimpleNamespace(permeability={"x": 8.0}, mass_balance_error=0.125)

    def _fake_xlb_solver(phases_arg, *, voxel_size, flow_axis, options):
        captured["xlb_args"] = {
            "phases": np.asarray(phases_arg),
            "voxel_size": voxel_size,
            "flow_axis": flow_axis,
            "options": options,
        }
        return xlb_mod.XLBDirectSimulationResult(
            flow_axis="x",
            voxel_size=float(voxel_size),
            image_porosity=float(np.asarray(phases_arg).mean()),
            sample_lengths={"x": 4.0},
            sample_cross_sections={"x": 30.0},
            lattice_viscosity=0.1,
            lattice_density_inlet=1.001,
            lattice_density_outlet=1.0,
            lattice_pressure_drop=1.0e-3,
            inlet_outlet_buffer_cells=6,
            omega=1.25,
            superficial_velocity_lattice=0.2,
            superficial_velocity_profile_lattice=np.array([0.2, 0.2, 0.2]),
            axial_velocity_lattice=np.ones((4, 5, 6), dtype=float),
            converged=True,
            n_steps=100,
            convergence_metric=5.0e-4,
            permeability=5.0,
            backend="jax",
            backend_version="fake-xlb",
        )

    monkeypatch.setattr(xlb_mod, "extract_spanning_pore_network", _fake_extract)
    monkeypatch.setattr(xlb_mod, "solve", _fake_solve)
    monkeypatch.setattr(xlb_mod, "solve_binary_volume_with_xlb", _fake_xlb_solver)
    monkeypatch.setattr(xlb_mod, "absolute_porosity", lambda net: 0.55)
    monkeypatch.setattr(xlb_mod, "effective_porosity", lambda net, axis: 0.45)

    result = benchmark_segmented_volume_with_xlb(
        phases,
        voxel_size=2.5,
        extraction_kwargs={"sigma": 1.2},
        strict=False,
    )
    record = result.to_record()

    extract_args = captured["extract_args"]
    assert np.array_equal(extract_args["phases"], phases)
    assert extract_args["voxel_size"] == pytest.approx(2.5)
    assert extract_args["flow_axis"] is None
    assert extract_args["length_unit"] == "m"
    assert extract_args["pressure_unit"] == "Pa"
    assert extract_args["extraction_kwargs"] == {"sigma": 1.2}
    assert extract_args["strict"] is False
    assert extract_args["provenance_notes"]["benchmark_kind"] == "segmented_volume_xlb"

    solve_args = captured["solve_args"]
    assert solve_args["net"] is extract_net
    assert solve_args["axis"] == "x"
    assert solve_args["bc"].inlet_label == "inlet_xmin"
    assert solve_args["bc"].outlet_label == "outlet_xmax"
    assert solve_args["bc"].pin == pytest.approx(2.0e5)
    assert solve_args["bc"].pout == pytest.approx(1.0e5)
    assert solve_args["fluid"].viscosity == pytest.approx(1.0e-3)
    assert solve_args["options"].conductance_model == "valvatne_blunt_baseline"
    assert solve_args["options"].solver == "direct"

    xlb_args = captured["xlb_args"]
    assert np.array_equal(xlb_args["phases"], phases)
    assert xlb_args["voxel_size"] == pytest.approx(2.5)
    assert xlb_args["flow_axis"] == "x"
    assert isinstance(xlb_args["options"], xlb_mod.XLBOptions)

    assert result.image_porosity == pytest.approx(1.0)
    assert result.absolute_porosity == pytest.approx(0.55)
    assert result.effective_porosity == pytest.approx(0.45)
    assert result.permeability_abs_diff == pytest.approx(3.0)
    assert result.permeability_rel_diff == pytest.approx(3.0 / 8.0)
    assert record["k_voids"] == pytest.approx(8.0)
    assert record["k_xlb"] == pytest.approx(5.0)
    assert record["voids_mass_balance_error"] == pytest.approx(0.125)
    assert record["extract_backend"] == "porespy"
    assert record["xlb_backend_version"] == "fake-xlb"


def test_benchmark_segmented_volume_with_xlb_rejects_empty_extracted_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the benchmark guard when extraction yields no valid boundary pores."""

    extract = SimpleNamespace(
        flow_axis="x",
        net=SimpleNamespace(Np=0, Nt=0, pore_labels={}),
        backend="porespy",
        backend_version="1.0",
    )
    monkeypatch.setattr(xlb_mod, "extract_spanning_pore_network", lambda *args, **kwargs: extract)

    with pytest.raises(ValueError, match="lacks non-empty inlet/outlet pore labels"):
        benchmark_segmented_volume_with_xlb(np.ones((4, 5, 6), dtype=int), voxel_size=1.0)


def test_benchmark_segmented_volume_with_xlb_returns_positive_permeabilities() -> None:
    """Test end-to-end extraction plus direct-image XLB comparison on a tiny segmented volume."""

    pytest.importorskip("xlb")

    phases = np.zeros((12, 16, 16), dtype=int)
    phases[:, 5:11, 5:11] = 1
    phases[2:4, 1:3, 1:3] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = benchmark_segmented_volume_with_xlb(
            phases,
            voxel_size=1.0,
            flow_axis="x",
            length_unit="voxel",
            fluid=FluidSinglePhase(viscosity=1.0),
            provenance_notes={"case": "tiny_xlb"},
            xlb_options=XLBOptions(
                max_steps=120,
                min_steps=60,
                check_interval=20,
                lattice_viscosity=0.1,
                rho_inlet=1.001,
                rho_outlet=1.0,
            ),
        )
    record = result.to_record()

    assert result.extract.flow_axis == "x"
    assert result.extract.provenance.user_notes["case"] == "tiny_xlb"
    assert result.image_porosity == pytest.approx(float(phases.mean()))
    assert record["Np"] == result.extract.net.Np
    assert record["Nt"] == result.extract.net.Nt
    assert record["phi_abs"] == pytest.approx(result.absolute_porosity)
    assert record["phi_eff"] == pytest.approx(result.effective_porosity)
    assert record["k_voids"] > 0.0
    assert record["k_xlb"] > 0.0
    assert record["xlb_steps"] > 0
    assert record["xlb_backend"] == "jax"


def test_xlb_direct_solver_open_duct_returns_finite_positive_permeability() -> None:
    """Test that an all-void duct does not produce NaNs or negative permeability."""

    pytest.importorskip("xlb")

    phases = np.ones((16, 8, 8), dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = solve_binary_volume_with_xlb(
            phases,
            voxel_size=1.0,
            flow_axis="x",
            options=XLBOptions(
                max_steps=160,
                min_steps=80,
                check_interval=20,
                steady_rtol=1.0e-4,
                lattice_viscosity=0.1,
                rho_inlet=1.001,
                rho_outlet=1.0,
            ),
        )

    assert np.isfinite(result.superficial_velocity_lattice)
    assert np.isfinite(result.permeability)
    assert result.superficial_velocity_lattice > 0.0
    assert result.permeability > 0.0
