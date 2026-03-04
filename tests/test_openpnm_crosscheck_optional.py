from __future__ import annotations

import pytest

from voids.benchmarks.crosscheck import crosscheck_singlephase_with_openpnm
from voids.core.network import Network
from voids.physics.singlephase import FluidSinglePhase, PressureBC


def test_openpnm_crosscheck_api_available_or_clean_import_error(line_network: Network) -> None:
    """Test that the OpenPNM crosscheck either runs or fails with a clean import error."""

    try:
        crosscheck_singlephase_with_openpnm(
            line_network,
            fluid=FluidSinglePhase(viscosity=1.0),
            bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
            axis="x",
        )
    except ImportError:
        # Expected outside the full test environment with optional OpenPNM deps.
        return
    except NotImplementedError:
        pytest.skip("OpenPNM cross-check adapter placeholder present; enable when adapter is wired")
