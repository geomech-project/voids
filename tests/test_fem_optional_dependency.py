from __future__ import annotations

import builtins
from typing import Any

import pytest

from voids.fem.singlephase import _common


def test_fem_backend_reports_clean_missing_dolfinx_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dolfinx" or name.startswith("dolfinx."):
            raise ImportError("simulated missing dolfinx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="FEniCSx FEM backends require DOLFINx"):
        _common._require_dolfinx()
