from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sync_deps_module():
    """Load the sync-deps script as an importable module for testing."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_deps.py"
    spec = importlib.util.spec_from_file_location("voids_sync_deps", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_requirement_list_updates_versions_and_preserves_unknowns() -> None:
    module = _load_sync_deps_module()

    requirements = [
        "numpy>=1.20",
        "mkdocstrings[python]>=0.20",
        "unknownpkg>=0.1",
    ]
    pixi_specs = {
        module._canonicalize_name("numpy"): ">=1.26,<2.2",
        module._canonicalize_name("mkdocstrings"): ">=0.25",
    }

    synced, missing = module._sync_requirement_list(requirements, pixi_specs=pixi_specs)

    assert synced == [
        "numpy>=1.26,<2.2",
        "mkdocstrings[python]>=0.25",
        "unknownpkg>=0.1",
    ]
    assert missing == ["unknownpkg"]


def test_replace_toml_array_section_key_handles_brackets_inside_strings() -> None:
    module = _load_sync_deps_module()

    text = (
        "[project.optional-dependencies]\n"
        "docs = [\n"
        '  "mkdocstrings[python]>=0.25",\n'
        '  "ruff>=0.6",\n'
        "]\n"
    )

    updated = module._replace_toml_array_section_key(
        text,
        section="project.optional-dependencies",
        key="docs",
        new_items=["mkdocstrings[python]>=0.30", "ruff>=0.7"],
    )

    assert '"mkdocstrings[python]>=0.30",\n' in updated
    assert '"ruff>=0.7",\n' in updated
    assert '"mkdocstrings[python]>=0.25",\n' not in updated
