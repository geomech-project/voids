from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import tomllib


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


def test_pep508_specifier_from_pixi_specifier_handles_pixi_forms() -> None:
    module = _load_sync_deps_module()

    assert module._pep508_specifier_from_pixi_specifier("*") == ""
    assert module._pep508_specifier_from_pixi_specifier("9.*") == "==9.*"
    assert module._pep508_specifier_from_pixi_specifier(">=1.26,<2.2") == ">=1.26,<2.2"


def test_project_targets_are_generated_from_core_feature_and_platform_tables() -> None:
    module = _load_sync_deps_module()

    pixi_data = {
        "dependencies": {
            "python": ">=3.11,<3.13",
        },
        "feature": {
            "core": {
                "dependencies": {
                    "numpy": ">=1.26,<2.2",
                    "scipy": ">=1.11",
                    "ipykernel": ">=7.2.0,<8",
                },
                "pypi-dependencies": {
                    "voids": {"path": ".", "editable": True},
                    "porespy": ">=2.4",
                },
            },
        },
        "target": {
            "linux-64": {
                "pypi-dependencies": {"pypardiso": ">=0.4"},
            },
        },
    }

    targets, empty_features = module._sync_targets_from_pixi(pixi_data)

    assert empty_features == ["core"]
    assert targets == [
        module.SyncTarget(
            section="project",
            key="dependencies",
            requirements=(
                "numpy>=1.26,<2.2",
                "scipy>=1.11",
                "porespy>=2.4",
                "pypardiso>=0.4; sys_platform == 'linux'",
            ),
        )
    ]


def test_feature_targets_are_generated_from_pixi_features_with_policy_exceptions() -> None:
    module = _load_sync_deps_module()

    pixi_data = {
        "feature": {
            "core": {
                "dependencies": {
                    "numpy": ">=1.26,<2.2",
                },
            },
            "viz": {
                "dependencies": {
                    "vtk": "9.*",
                    "matplotlib-base": "*",
                },
            },
            "docs": {
                "pypi-dependencies": {
                    "mkdocstrings": {"version": ">=0.25", "extras": ["python"]},
                },
            },
            "fem": {
                "dependencies": {
                    "fenics-dolfinx": ">=0.9,<0.11",
                },
            },
            "solvers": {
                "dependencies": {
                    "suitesparse": "*",
                    "libumfpack": "*",
                    "scikit-umfpack": "*",
                },
                "pypi-dependencies": {
                    "pyamg": ">=5.3",
                },
            },
        }
    }

    targets, empty_features = module._sync_targets_from_pixi(pixi_data)

    rendered = {(target.section, target.key): target.requirements for target in targets}
    assert rendered[("project.optional-dependencies", "viz")] == (
        "vtk==9.*",
        "matplotlib>=3.8",
    )
    assert ("project.optional-dependencies", "core") not in rendered
    assert ("project.optional-dependencies", "docs") not in rendered
    assert ("project.optional-dependencies", "fem") not in rendered
    assert rendered[("project.optional-dependencies", "solvers")] == (
        "scikit-umfpack",
        "pyamg>=5.3",
    )
    assert empty_features == ["core", "docs", "fem"]


def test_sync_pyproject_text_removes_empty_conda_only_feature_extras() -> None:
    module = _load_sync_deps_module()
    repo_root = Path(__file__).resolve().parents[1]

    pixi_data = module._read_toml(repo_root / "pixi.toml")
    synced = module._sync_pyproject_text(
        pixi_data,
        (repo_root / "pyproject.toml").read_text(encoding="utf-8"),
    )
    pyproject_data = tomllib.loads(synced)

    optional_dependencies = pyproject_data["project"]["optional-dependencies"]
    assert set(optional_dependencies) >= {
        "dev",
        "notebooks",
        "viz",
        "test",
        "lbm",
        "thermo",
        "solvers",
    }
    assert "core" not in optional_dependencies
    assert "docs" not in optional_dependencies
    assert "fem" not in optional_dependencies
    all_optional_requirements = [
        requirement
        for requirements in optional_dependencies.values()
        for requirement in requirements
    ]
    assert not any("fenics-dolfinx" in requirement for requirement in all_optional_requirements)
    assert not any("suitesparse" in requirement for requirement in all_optional_requirements)
    assert not any("libumfpack" in requirement for requirement in all_optional_requirements)


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
