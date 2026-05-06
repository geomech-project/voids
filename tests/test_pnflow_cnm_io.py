from __future__ import annotations

from pathlib import Path

import pytest

from voids.io import load_pnflow_cnm
from voids.paths import data_path
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve


def _write_text(path: Path, text: str) -> None:
    """Write a small CNM fixture file."""

    path.write_text(text, encoding="utf-8")


def test_load_pnflow_cnm_parses_boundary_connections_with_mirror_pores(tmp_path: Path) -> None:
    """Import should add one mirrored pseudo-pore per reservoir connection."""

    prefix = tmp_path / "toy"
    _write_text(
        prefix.with_name("toy_node1.dat"),
        "\n".join(
            [
                "2 1.0 2.0 3.0",
                "1 0.2 0.5 0.7 2 -1 2 1 0 1 2",
                "2 0.8 1.5 2.2 2 1 0 0 1 2 3",
            ]
        )
        + "\n",
    )
    _write_text(
        prefix.with_name("toy_node2.dat"),
        "\n".join(
            [
                "1 0.05 0.10 0.04 0.0",
                "2 0.06 0.12 0.03 0.0",
            ]
        )
        + "\n",
    )
    _write_text(
        prefix.with_name("toy_link1.dat"),
        "\n".join(
            [
                "3",
                "1 -1 1 0.08 0.04 0.55",
                "2 1 2 0.07 0.03 0.60",
                "3 2 0 0.09 0.05 0.45",
            ]
        )
        + "\n",
    )
    _write_text(
        prefix.with_name("toy_link2.dat"),
        "\n".join(
            [
                "1 -1 1 0.20 0.10 0.25 0.01 0.0",
                "2 1 2 0.15 0.15 0.30 0.02 0.0",
                "3 2 0 0.12 0.18 0.15 0.03 0.0",
            ]
        )
        + "\n",
    )

    imported = load_pnflow_cnm(prefix)
    net = imported.net

    assert imported.n_physical_pores == 2
    assert imported.n_boundary_mirror_pores == 2
    assert imported.box_lengths == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert net.Np == 4
    assert net.Nt == 3
    assert net.pore_labels["inlet_xmin"].sum() == 1
    assert net.pore_labels["outlet_xmax"].sum() == 1
    assert net.pore_labels["boundary_connected_inlet_xmin"].tolist() == [True, False, False, False]
    assert net.pore_labels["boundary_connected_outlet_xmax"].tolist() == [False, True, False, False]
    assert net.throat["pore1_length"][0] == pytest.approx(1.0e-300)
    assert net.throat["pore2_length"][2] == pytest.approx(1.0e-300)
    assert net.pore["volume"][2] == pytest.approx(0.0)
    assert net.pore["volume"][3] == pytest.approx(0.0)
    assert net.pore_coords[2, 0] == pytest.approx(0.0)
    assert net.pore_coords[3, 0] == pytest.approx(1.0)


def test_load_pnflow_cnm_supports_tight_singlephase_comparison_on_saved_benchmark_case() -> None:
    """Imported Imperial CNM data should solve close to the saved `pnflow` reference."""

    case = "phi038_b18"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    imported = load_pnflow_cnm(prefix)
    result = solve(
        imported.net,
        fluid=FluidSinglePhase(viscosity=1.0e-3),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=2.0e5, pout=0.0),
        axis="x",
        options=SinglePhaseOptions(conductance_model="valvatne_blunt", solver="direct"),
    )

    k_ref = 1.19927e-14
    rel_err = abs(result.permeability["x"] - k_ref) / k_ref

    assert imported.n_physical_pores == 64
    assert imported.net.Nt == 180
    assert rel_err < 0.08
