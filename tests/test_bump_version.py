from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_bump_version_module():
    """Load the bump-version script as an importable module for testing."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bump_version.py"
    spec = importlib.util.spec_from_file_location("voids_bump_version", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replace_toml_section_version_preserves_line_break_after_version() -> None:
    """Test that TOML version replacement preserves the following line break."""

    module = _load_bump_version_module()
    text = (
        "[project]\n"
        'name = "voids"\n'
        'version = "0.1.2" # inline comment\n'
        'description = "Scientific Python package for pore network modeling (PNM)"\n'
    )

    updated, current = module._replace_toml_section_version(
        text, section="project", new_version="0.1.3"
    )

    assert current == "0.1.2"
    assert 'version = "0.1.3" # inline comment\n' in updated
    assert (
        '\ndescription = "Scientific Python package for pore network modeling (PNM)"\n' in updated
    )
    assert 'version = "0.1.3" # inline commentdescription' not in updated
