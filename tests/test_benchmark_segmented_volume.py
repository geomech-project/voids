from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import voids.benchmarks._shared as shared_benchmark_mod
import voids.benchmarks.segmented_volume as segmented_volume_mod
from voids.benchmarks import benchmark_segmented_volume_with_openpnm
from voids.physics.singlephase import FluidSinglePhase


def test_benchmark_segmented_volume_with_openpnm_returns_consistent_scalars() -> None:
    """Test end-to-end extraction plus OpenPNM comparison on a tiny segmented volume."""

    phases = np.zeros((12, 16, 16), dtype=int)
    phases[:, 5:11, 5:11] = 1
    phases[2:4, 1:3, 1:3] = 1

    pytest.importorskip("openpnm")

    result = benchmark_segmented_volume_with_openpnm(
        phases,
        voxel_size=1.0,
        flow_axis="x",
        length_unit="voxel",
        fluid=FluidSinglePhase(viscosity=1.0),
        pin=2.0,
        pout=1.0,
        provenance_notes={"case": "tiny"},
    )
    record = result.to_record()

    assert result.extract.flow_axis == "x"
    assert result.extract.provenance.user_notes["case"] == "tiny"
    assert result.summary.reference == "openpnm_stokesflow"
    assert result.image_porosity == pytest.approx(float(phases.mean()))
    assert record["Np"] == result.extract.net.Np
    assert record["Nt"] == result.extract.net.Nt
    assert record["phi_abs"] == pytest.approx(result.absolute_porosity)
    assert record["phi_eff"] == pytest.approx(result.effective_porosity)
    assert record["conductance_model"] == "valvatne_blunt"
    assert record["solver_voids"] == "direct"
    assert record["dp_physical"] == pytest.approx(1.0)
    assert record["k_rel_diff"] < 1.0e-10
    assert record["Q_rel_diff"] < 1.0e-10


def test_benchmark_segmented_volume_with_openpnm_rejects_nonbinary_inputs() -> None:
    """Test binary-volume validation before extraction or optional imports."""

    phases = np.array([[[0, 2], [1, 0]], [[1, 0], [0, 1]]], dtype=int)

    with pytest.raises(ValueError, match="phases must be binary with void=1 and solid=0"):
        benchmark_segmented_volume_with_openpnm(phases, voxel_size=1.0)


def test_benchmark_segmented_volume_with_openpnm_rejects_invalid_rank() -> None:
    """Test rank validation before binary-value checks or optional imports."""

    phases = np.array([0, 1, 0, 1], dtype=int)

    with pytest.raises(ValueError, match="phases must be a 2D or 3D binary segmented volume"):
        benchmark_segmented_volume_with_openpnm(phases, voxel_size=1.0)


def test_resolve_benchmark_pressures_supports_delta_p_and_pressure_gauge() -> None:
    """High-level benchmark pressure inputs should be resolved from `delta_p` consistently."""

    pin, pout, delta_p = shared_benchmark_mod.resolve_benchmark_pressures(delta_p=1.0)
    assert pin == pytest.approx(1.0)
    assert pout == pytest.approx(0.0)
    assert delta_p == pytest.approx(1.0)

    pin, pout, delta_p = shared_benchmark_mod.resolve_benchmark_pressures(
        delta_p=1.0,
        pout=101325.0,
    )
    assert pin == pytest.approx(101326.0)
    assert pout == pytest.approx(101325.0)
    assert delta_p == pytest.approx(1.0)

    pin, pout, delta_p = shared_benchmark_mod.resolve_benchmark_pressures(
        delta_p=1.0,
        pin=101326.0,
    )
    assert pin == pytest.approx(101326.0)
    assert pout == pytest.approx(101325.0)
    assert delta_p == pytest.approx(1.0)

    pin, pout, delta_p = shared_benchmark_mod.resolve_benchmark_pressures(
        pin=101326.0,
        pout=101325.0,
    )
    assert pin == pytest.approx(101326.0)
    assert pout == pytest.approx(101325.0)
    assert delta_p == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Provide either `delta_p`, or both `pin` and `pout`"):
        shared_benchmark_mod.resolve_benchmark_pressures(pin=1.0)

    with pytest.raises(ValueError, match="`delta_p` must equal `pin - pout`"):
        shared_benchmark_mod.resolve_benchmark_pressures(
            delta_p=2.0,
            pin=101326.0,
            pout=101325.0,
        )

    with pytest.raises(ValueError, match="must be finite"):
        shared_benchmark_mod.resolve_benchmark_pressures(delta_p=float("nan"))


def test_benchmark_segmented_volume_with_openpnm_uses_harmonized_pressure_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenPNM wrapper should mirror the shared high-level benchmark BC convention."""

    phases = np.ones((4, 5, 6), dtype=int)
    extract_net = SimpleNamespace(Np=4, Nt=3)
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

    def _fake_crosscheck(net, fluid, bc, *, axis, options):
        captured["crosscheck_args"] = {
            "net": net,
            "fluid": fluid,
            "bc": bc,
            "axis": axis,
            "options": options,
        }
        return SimpleNamespace(
            axis="x",
            reference="openpnm_stokesflow",
            permeability_abs_diff=0.0,
            permeability_rel_diff=0.0,
            total_flow_abs_diff=0.0,
            total_flow_rel_diff=0.0,
            details={
                "k_voids": 8.0,
                "k_ref": 8.0,
                "Q_voids": 4.0,
                "Q_ref": 4.0,
                "n_inlet_pores": 2,
                "n_outlet_pores": 2,
                "conductance_model": options.conductance_model,
                "solver_voids": options.solver,
                "openpnm_version": "fake-openpnm",
            },
        )

    monkeypatch.setattr(segmented_volume_mod, "extract_spanning_pore_network", _fake_extract)
    monkeypatch.setattr(
        segmented_volume_mod, "crosscheck_singlephase_with_openpnm", _fake_crosscheck
    )
    monkeypatch.setattr(segmented_volume_mod, "absolute_porosity", lambda net: 0.55)
    monkeypatch.setattr(segmented_volume_mod, "effective_porosity", lambda net, axis: 0.45)

    result = benchmark_segmented_volume_with_openpnm(
        phases,
        voxel_size=2.5e-6,
        extraction_kwargs={"sigma": 1.2},
        provenance_notes={"case": "tiny"},
        strict=False,
    )
    record = result.to_record()

    extract_args = captured["extract_args"]
    assert np.array_equal(extract_args["phases"], phases)
    assert extract_args["voxel_size"] == pytest.approx(2.5e-6)
    assert extract_args["flow_axis"] is None
    assert extract_args["length_unit"] == "m"
    assert extract_args["pressure_unit"] == "Pa"
    assert extract_args["extraction_kwargs"] == {"sigma": 1.2}
    assert extract_args["strict"] is False
    assert extract_args["provenance_notes"]["benchmark_kind"] == "segmented_volume_openpnm"
    assert extract_args["provenance_notes"]["case"] == "tiny"

    crosscheck_args = captured["crosscheck_args"]
    assert crosscheck_args["net"] is extract_net
    assert crosscheck_args["axis"] == "x"
    assert crosscheck_args["bc"].inlet_label == "inlet_xmin"
    assert crosscheck_args["bc"].outlet_label == "outlet_xmax"
    assert crosscheck_args["bc"].pin == pytest.approx(1.0)
    assert crosscheck_args["bc"].pout == pytest.approx(0.0)
    assert crosscheck_args["fluid"].viscosity == pytest.approx(1.0e-3)
    assert crosscheck_args["options"].conductance_model == "valvatne_blunt"
    assert crosscheck_args["options"].solver == "direct"

    assert result.image_porosity == pytest.approx(1.0)
    assert result.absolute_porosity == pytest.approx(0.55)
    assert result.effective_porosity == pytest.approx(0.45)
    assert record["k_voids"] == pytest.approx(8.0)
    assert record["k_openpnm"] == pytest.approx(8.0)
    assert record["p_inlet_physical"] == pytest.approx(1.0)
    assert record["p_outlet_physical"] == pytest.approx(0.0)
    assert record["dp_physical"] == pytest.approx(1.0)
    assert record["backend"] == "porespy"
    assert record["openpnm_version"] == "fake-openpnm"


def test_benchmark_segmented_volume_with_openpnm_rejects_nonpositive_pressure_drop() -> None:
    """High-level benchmark wrappers should reject zero or negative imposed pressure drops."""

    phases = np.ones((4, 5, 6), dtype=int)

    with pytest.raises(
        ValueError,
        match="positive physical pressure drop",
    ):
        benchmark_segmented_volume_with_openpnm(
            phases,
            voxel_size=1.0,
            pin=1.0,
            pout=1.0,
        )
