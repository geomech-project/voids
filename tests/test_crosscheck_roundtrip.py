from __future__ import annotations

import numpy as np

from voids.benchmarks.crosscheck import (
    audit_singlephase_conduit_conductance,
    crosscheck_singlephase_roundtrip_openpnm_dict,
)
from voids.core.network import Network
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
