from __future__ import annotations

import numpy as np
import pytest

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
    assert record["conductance_model"] == "valvatne_blunt_baseline"
    assert record["solver_voids"] == "direct"
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
