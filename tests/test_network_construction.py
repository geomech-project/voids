from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voids.core.provenance import Provenance
from voids.examples.demo import make_linear_chain_network
from voids.image import network_extraction as nex
from voids.paths import data_path


def test_construct_spanning_network_dispatches_to_image_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image backends should normalize aliases and lift extraction results."""

    net = make_linear_chain_network(num_pores=3)
    captured: dict[str, object] = {}

    def _fake_extract(
        phases: np.ndarray,
        *,
        voxel_size: float,
        backend: str,
        flow_axis: str | None,
        length_unit: str,
        pressure_unit: str,
        extraction_kwargs: dict[str, object] | None,
        provenance_notes: dict[str, object] | None,
        strict: bool,
        geometry_repairs: str | None,
        repair_seed: int | None,
    ) -> nex.NetworkExtractionResult:
        captured["shape"] = tuple(phases.shape)
        captured["voxel_size"] = voxel_size
        captured["backend"] = backend
        captured["flow_axis"] = flow_axis
        captured["provenance_notes"] = provenance_notes
        return nex.NetworkExtractionResult(
            image=np.asarray(phases),
            voxel_size=float(voxel_size),
            axis_lengths={"x": 3.0},
            axis_areas={"x": 1.0},
            flow_axis="x",
            network_dict={"throat.conns": np.array([[0, 1]])},
            sample=net.sample,
            provenance=Provenance(source_kind="test", extraction_method="fake_image"),
            net_full=net,
            net=net,
            pore_indices=np.arange(net.Np, dtype=np.int64),
            throat_mask=np.ones(net.Nt, dtype=bool),
            backend=backend,
            backend_version="test-version",
        )

    monkeypatch.setattr(nex, "extract_spanning_pore_network", _fake_extract)

    result = nex.construct_spanning_network(
        backend="porespy",
        phases=np.ones((3, 3, 3), dtype=int),
        voxel_size=2.5,
        provenance_notes={"campaign": "unit-test"},
    )

    assert captured["shape"] == (3, 3, 3)
    assert captured["voxel_size"] == pytest.approx(2.5)
    assert captured["backend"] == "porespy_snow2"
    assert captured["provenance_notes"] == {"campaign": "unit-test"}
    assert result.backend == "porespy_snow2"
    assert result.net.Np == net.Np
    assert result.backend_version == "test-version"

    result = nex.construct_spanning_network(
        backend="porespy_imperial",
        phases=np.ones((3, 3, 3), dtype=int),
        voxel_size=2.5,
    )
    assert captured["backend"] == "porespy_snow2_imperial"
    assert result.backend == "porespy_snow2_imperial"

    result = nex.construct_spanning_network(
        backend="maximal_ball",
        phases=np.ones((3, 3, 3), dtype=int),
        voxel_size=2.5,
    )
    assert captured["backend"] == "native_maximal_ball"
    assert result.backend == "native_maximal_ball"


def test_construct_spanning_network_supports_native_maximal_ball_backend() -> None:
    """The unified constructor should assemble a native maximal-ball network."""

    phases = np.zeros((7, 5, 5), dtype=int)
    phases[:, 1:4, 1:4] = 1

    result = nex.construct_spanning_network(
        backend="native_maximal_ball",
        phases=phases,
        voxel_size=1.0,
        extraction_kwargs={
            "distance_map_backend": "scipy",
            "apply_boundary_clipping": False,
            "settings": {"minimal_pore_radius_voxels": 1.0},
        },
    )

    assert result.backend == "native_maximal_ball"
    assert result.image is not None
    assert result.network_dict is not None
    assert result.net_full.Np >= 1
    assert result.net.pore_labels["inlet_xmin"].any()
    assert result.net.pore_labels["outlet_xmax"].any()


def test_construct_spanning_network_supports_imported_pnflow_cnm_backend() -> None:
    """Imperial CNM imports should be available through the unified constructor."""

    case = "phi038_b18"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    result = nex.construct_spanning_network(
        backend="imperial_cnm",
        pnflow_cnm_prefix=prefix,
        pnflow_solver_box_compat=True,
        provenance_notes={"campaign": "unit-test"},
    )

    assert result.backend == "pnflow_cnm"
    assert result.flow_axis == "x"
    assert result.image is None
    assert result.network_dict is None
    assert result.axis_lengths == {"x": 6.4e-05, "y": 6.4e-05, "z": 6.4e-05}
    assert result.net.Nt == 180
    assert result.backend_details["n_physical_pores"] == 64
    assert result.backend_details["n_boundary_mirror_pores"] == 34
    assert result.provenance.user_notes["campaign"] == "unit-test"


def test_construct_spanning_network_leaves_solver_box_compat_opt_in() -> None:
    """The unified constructor should keep the Imperial solver-box quirk explicit."""

    case = "phi035_b16"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    result = nex.construct_spanning_network(
        backend="pnflow_cnm",
        pnflow_cnm_prefix=prefix,
    )

    assert not result.net.pore_labels["inlet_xmin"][0]
    assert not result.net.pore_labels["outlet_xmax"][0]


def test_construct_spanning_network_rejects_missing_backend_inputs() -> None:
    """Each backend should require its own source inputs."""

    with pytest.raises(ValueError, match="phases is required"):
        nex.construct_spanning_network(backend="porespy", voxel_size=1.0)

    with pytest.raises(ValueError, match="voxel_size is required"):
        nex.construct_spanning_network(backend="porespy", phases=np.ones((2, 2), dtype=int))

    with pytest.raises(ValueError, match="pnflow_cnm_prefix is required"):
        nex.construct_spanning_network(backend="pnflow_cnm")

    prefix = Path("examples/data/external_pnflow_benchmark/phi032_b14/phi032_b14")
    with pytest.raises(ValueError, match="flow_axis='x'"):
        nex.construct_spanning_network(
            backend="pnflow_cnm",
            pnflow_cnm_prefix=prefix,
            flow_axis="y",
        )
