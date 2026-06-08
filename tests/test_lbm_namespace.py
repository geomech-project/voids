from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import voids.lbm.singlephase.stokes as stokes
from voids.lbm.singlephase.xlb import XLBOptions


def test_lbm_stokes_namespace_forwards_to_xlb_with_steady_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}

    def fake_solve_binary_volume_with_xlb(phases: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        capture["phases"] = phases
        capture.update(kwargs)
        return SimpleNamespace(permeability=1.23)

    monkeypatch.setattr(
        stokes,
        "solve_binary_volume_with_xlb",
        fake_solve_binary_volume_with_xlb,
    )

    phases = np.zeros((2, 2), dtype=bool)
    result = stokes.solve_binary_volume_stokes(phases, voxel_size=2.0, flow_axis="x")

    assert result.permeability == 1.23
    assert capture["phases"] is phases
    assert capture["voxel_size"] == 2.0
    assert capture["flow_axis"] == "x"
    assert capture["options"].formulation == "steady_stokes_limit"


def test_lbm_stokes_namespace_preserves_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}

    def fake_solve_binary_volume_with_xlb(phases: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        capture.update(kwargs)
        return SimpleNamespace(permeability=4.56)

    monkeypatch.setattr(
        stokes,
        "solve_binary_volume_with_xlb",
        fake_solve_binary_volume_with_xlb,
    )
    options = XLBOptions(max_steps=12)

    result = stokes.solve_binary_volume_stokes(
        np.zeros((2, 2), dtype=bool),
        voxel_size=1.0,
        options=options,
    )

    assert result.permeability == 4.56
    assert capture["options"] is options
