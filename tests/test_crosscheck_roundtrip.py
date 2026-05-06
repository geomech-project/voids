from __future__ import annotations

import numpy as np
import pytest

from voids.benchmarks.crosscheck import (
    audit_singlephase_conduit_conductance,
    compare_network_geometry,
    crosscheck_singlephase_roundtrip_openpnm_dict,
    summarize_network_geometry,
)
from voids.core.network import Network
from voids.examples.mesh import make_cartesian_mesh_network
from voids.graph.connectivity import induced_subnetwork
from voids.io import load_pnflow_cnm
from voids.paths import data_path
from voids.geom.hydraulic import throat_conductance
from voids.physics.singlephase import FluidSinglePhase, PressureBC


def test_singlephase_roundtrip_openpnm_dict_crosscheck(line_network: Network) -> None:
    """Test that the OpenPNM-style dict roundtrip preserves single-phase results."""

    s = crosscheck_singlephase_roundtrip_openpnm_dict(
        line_network,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )
    assert s.reference == "openpnm_dict_roundtrip"
    assert s.permeability_abs_diff < 1e-14
    assert s.total_flow_abs_diff < 1e-14


def test_audit_singlephase_conduit_conductance_matches_valvatne_blunt_reference() -> None:
    """The conduit audit should reproduce the active throat conductance field."""

    case = "phi038_b18"
    imported = load_pnflow_cnm(
        data_path() / "external_pnflow_benchmark" / case / case,
        pnflow_solver_box_compat=True,
    )
    audit = audit_singlephase_conduit_conductance(
        imported.net,
        viscosity=1.0e-3,
        model="valvatne_blunt",
    )
    g_ref = throat_conductance(imported.net, viscosity=1.0e-3, model="valvatne_blunt")

    assert audit.model == "valvatne_blunt"
    assert audit.throat_index.shape == (imported.net.Nt,)
    assert audit.pore1_index.shape == (imported.net.Nt,)
    assert audit.equivalent_conductance.shape == (imported.net.Nt,)
    assert int(audit.pore1_is_boundary.sum() + audit.pore2_is_boundary.sum()) == 37
    assert np.allclose(audit.equivalent_conductance, g_ref)


def test_summarize_network_geometry_on_cartesian_lattice() -> None:
    """Geometry summary should recover expected lattice counts and boundary structure."""

    net = make_cartesian_mesh_network((2, 2, 2), spacing=1.0)

    summary = summarize_network_geometry(net, axis="x")

    assert summary.n_pores == 8
    assert summary.n_throats == 12
    assert summary.n_components == 1
    assert summary.mean_coordination == 3.0
    assert summary.inlet_pore_count == 4
    assert summary.outlet_pore_count == 4
    assert summary.overlapping_boundary_count == 0
    assert summary.boundary_pore_count == 8
    assert summary.throat_core_length_mean == pytest.approx(
        float(np.median(net.throat["core_length"]))
    )


def test_compare_network_geometry_supports_reference_pore_mask() -> None:
    """Masked geometry comparison should match an explicitly induced subnetwork."""

    full_net = make_cartesian_mesh_network((3, 2, 2), spacing=1.0)
    reference_pore_mask = full_net.pore_coords[:, 0] <= 1.5
    candidate_net, _, _ = induced_subnetwork(full_net, reference_pore_mask)

    comparison = compare_network_geometry(
        full_net,
        candidate_net,
        axis="x",
        reference_pore_mask=reference_pore_mask,
        reference_name="reference",
        candidate_name="candidate",
    )

    assert comparison.pore_count_rel_diff == 0.0
    assert comparison.throat_count_rel_diff == 0.0
    assert comparison.mean_coordination_rel_diff == 0.0
    assert comparison.pore_radius_ks == 0.0
    assert comparison.throat_radius_ks == 0.0
