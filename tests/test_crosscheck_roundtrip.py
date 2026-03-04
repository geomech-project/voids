from __future__ import annotations

from voids.benchmarks.crosscheck import crosscheck_singlephase_roundtrip_openpnm_dict
from voids.core.network import Network
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
