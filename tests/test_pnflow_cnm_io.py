from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import voids.io.pnflow_cnm as pnflow_cnm_module
from voids.io import load_pnflow_cnm
from voids.paths import data_path
from voids.physics.singlephase import FluidSinglePhase, PressureBC, SinglePhaseOptions, solve


def _write_text(path: Path, text: str) -> None:
    """Write a small CNM fixture file."""

    path.write_text(text, encoding="utf-8")


def _write_minimal_cnm_fixture(
    prefix: Path,
    *,
    node1_text: str | None = None,
    node2_text: str | None = None,
    link1_text: str | None = None,
    link2_text: str | None = None,
) -> None:
    """Write a minimal one-pore CNM fixture, optionally overriding any file."""

    _write_text(
        prefix.with_name(f"{prefix.name}_node1.dat"),
        node1_text if node1_text is not None else "1 1.0 1.0 1.0\n1 0.5 0.5 0.5 1 2 0 0 1\n",
    )
    _write_text(
        prefix.with_name(f"{prefix.name}_node2.dat"),
        node2_text if node2_text is not None else "1 0.05 0.10 0.04 0.0\n",
    )
    _write_text(
        prefix.with_name(f"{prefix.name}_link1.dat"),
        link1_text if link1_text is not None else "1\n1 1 1 0.08 0.04 0.55\n",
    )
    _write_text(
        prefix.with_name(f"{prefix.name}_link2.dat"),
        link2_text if link2_text is not None else "1 1 1 0.20 0.10 0.25 0.01 0.0\n",
    )


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

    imported = load_pnflow_cnm(prefix, pnflow_solver_box_compat=False)
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


def test_pnflow_shape_factor_and_line_parsing_helpers_cover_all_element_classes() -> None:
    """CNM helper functions should cover malformed rows and shape-factor classes."""

    shape_factors = pnflow_cnm_module._pnflow_effective_shape_factor(
        np.array([0.01, 0.05, 0.08], dtype=float)
    )

    assert shape_factors[0] == pytest.approx(min(0.01, pnflow_cnm_module._TRIANGLE_MAX_G - 5.0e-5))
    assert shape_factors[1] == pytest.approx(pnflow_cnm_module._SQUARE_G)
    assert shape_factors[2] == pytest.approx(pnflow_cnm_module._CIRCLE_G)
    with pytest.raises(ValueError, match="Malformed toy"):
        pnflow_cnm_module._split_numeric_line("1 2", expected_min_tokens=3, label="toy")


def test_load_pnflow_cnm_rejects_invalid_import_options(tmp_path: Path) -> None:
    """Importer options should reject unsupported or nonphysical boundary controls."""

    prefix = tmp_path / "bad_options"

    with pytest.raises(ValueError, match="boundary_axis"):
        load_pnflow_cnm(prefix, boundary_axis="y")
    with pytest.raises(ValueError, match="boundary_length_epsilon"):
        load_pnflow_cnm(prefix, boundary_length_epsilon=0.0)
    with pytest.raises(ValueError, match="boundary_radius_scale"):
        load_pnflow_cnm(prefix, boundary_radius_scale=0.0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"node1_text": ""}, "CNM file is empty"),
        (
            {
                "node1_text": "2 1.0 1.0 1.0\n1 0.5 0.5 0.5 1 2 0 0 1\n",
                "node2_text": ("1 0.05 0.10 0.04 0.0\n2 0.05 0.10 0.04 0.0\n"),
            },
            "header declares 2 pores",
        ),
        ({"node2_text": ""}, "should contain 1 pore rows"),
        ({"link1_text": ""}, "CNM file is empty"),
        ({"link1_text": "2\n1 1 1 0.08 0.04 0.55\n"}, "Throat-row mismatch"),
        (
            {"node1_text": "1 1.0 1.0 1.0\n1 0.5 0.5 0.5 2 1\n"},
            "Malformed pore-connectivity row",
        ),
        (
            {"link1_text": "1\n1 -1 0 0.08 0.04 0.55\n"},
            "Boundary throat without an internal neighbor",
        ),
        (
            {"link1_text": "1\n1 -2 -1 0.08 0.04 0.55\n"},
            "Boundary throat without an internal neighbor",
        ),
        (
            {"link1_text": "1\n1 -2 -3 0.08 0.04 0.55\n"},
            "Unresolved throat endpoints",
        ),
    ],
)
def test_load_pnflow_cnm_reports_malformed_fixture_errors(
    tmp_path: Path,
    overrides: dict[str, str],
    match: str,
) -> None:
    """Malformed CNM text fixtures should fail at the intended parser guard."""

    prefix = tmp_path / "malformed"
    _write_minimal_cnm_fixture(prefix, **overrides)

    with pytest.raises(ValueError, match=match):
        load_pnflow_cnm(prefix)


def test_load_pnflow_cnm_defaults_to_generic_import_without_solver_box_compat() -> None:
    """Generic CNM imports should not silently apply the Imperial solver-box quirk."""

    case = "phi035_b16"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    imported = load_pnflow_cnm(prefix)

    assert not imported.net.pore_labels["inlet_xmin"][0]
    assert not imported.net.pore_labels["outlet_xmax"][0]


def test_load_pnflow_cnm_supports_tight_singlephase_comparison_on_saved_benchmark_case() -> None:
    """Imported Imperial CNM data should solve close to the saved `pnflow` reference."""

    case = "phi038_b18"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    imported = load_pnflow_cnm(prefix, pnflow_solver_box_compat=True)
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
    assert rel_err < 1.0e-5


def test_load_pnflow_cnm_solver_box_compatibility_matches_hard_case() -> None:
    """The Imperial solver-box compatibility mode should reproduce the saved hard case."""

    case = "phi035_b16"
    prefix = data_path() / "external_pnflow_benchmark" / case / case
    imported = load_pnflow_cnm(prefix, pnflow_solver_box_compat=True)
    result = solve(
        imported.net,
        fluid=FluidSinglePhase(viscosity=1.0e-3),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=2.0e5, pout=0.0),
        axis="x",
        options=SinglePhaseOptions(conductance_model="valvatne_blunt", solver="direct"),
    )

    k_ref = 1.33185e-14
    rel_err = abs(result.permeability["x"] - k_ref) / k_ref

    assert imported.net.pore_labels["outlet_xmax"][0]
    assert rel_err < 1.0e-5
