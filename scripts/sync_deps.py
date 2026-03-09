from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PIXI_PATH = REPO_ROOT / "pixi.toml"


@dataclass(frozen=True)
class SyncTarget:
    """Describe one dependency list in ``pyproject.toml`` to synchronize."""

    section: str
    key: str


TARGETS = (
    SyncTarget(section="project", key="dependencies"),
    SyncTarget(section="project.optional-dependencies", key="dev"),
    SyncTarget(section="project.optional-dependencies", key="viz"),
    SyncTarget(section="project.optional-dependencies", key="test"),
    SyncTarget(section="project.optional-dependencies", key="lbm"),
    SyncTarget(section="project.optional-dependencies", key="docs"),
)


def _canonicalize_name(name: str) -> str:
    """Normalize a package name using PEP 503 normalization rules."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _split_requirement(req: str) -> tuple[str, str, str]:
    """Split requirement into ``name``, ``name_with_extras``, and ``specifier``."""

    match = re.match(r"^\s*([A-Za-z0-9_.-]+)(\[[^\]]+\])?\s*(.*)$", req)
    if match is None:
        raise RuntimeError(f"Unsupported requirement entry format: {req!r}")
    base_name = match.group(1)
    extras = match.group(2) or ""
    spec = match.group(3).strip()
    return base_name, f"{base_name}{extras}", spec


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _collect_pixi_specs(pixi_data: dict[str, object]) -> dict[str, str]:
    """Collect dependency version specifiers from pixi dependency tables."""

    specs: dict[str, str] = {}

    def merge_table(raw: object) -> None:
        if not isinstance(raw, dict):
            return
        for raw_name, raw_value in raw.items():
            if not isinstance(raw_name, str):
                continue
            value: str | None = None
            if isinstance(raw_value, str):
                value = raw_value.strip()
            elif isinstance(raw_value, dict):
                # Ignore local path/editable entries, e.g. voids = { path = ".", editable = true }
                if "version" in raw_value and isinstance(raw_value["version"], str):
                    value = raw_value["version"].strip()
            if value is None:
                continue
            specs[_canonicalize_name(raw_name)] = value

    merge_table(pixi_data.get("dependencies"))
    merge_table(pixi_data.get("pypi-dependencies"))

    features = pixi_data.get("feature")
    if isinstance(features, dict):
        for feature_data in features.values():
            if not isinstance(feature_data, dict):
                continue
            merge_table(feature_data.get("dependencies"))
            merge_table(feature_data.get("pypi-dependencies"))

    return specs


def _get_pyproject_list(pyproject: dict[str, object], target: SyncTarget) -> list[str]:
    if target.section == "project":
        project = pyproject.get("project")
        if not isinstance(project, dict):
            raise RuntimeError("Missing [project] section in pyproject.toml")
        values = project.get(target.key)
    elif target.section == "project.optional-dependencies":
        project = pyproject.get("project")
        if not isinstance(project, dict):
            raise RuntimeError("Missing [project] section in pyproject.toml")
        optional = project.get("optional-dependencies")
        if not isinstance(optional, dict):
            raise RuntimeError("Missing [project.optional-dependencies] section in pyproject.toml")
        values = optional.get(target.key)
    else:
        raise RuntimeError(f"Unsupported target section: {target.section}")

    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise RuntimeError(f"Expected a string list for [{target.section}].{target.key}")

    return list(values)


def _sync_requirement_list(
    requirements: list[str], *, pixi_specs: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Update requirement specifiers in place based on pixi specs."""

    updated: list[str] = []
    missing: list[str] = []

    for req in requirements:
        base_name, name_with_extras, old_spec = _split_requirement(req)
        pixi_spec = pixi_specs.get(_canonicalize_name(base_name))
        if pixi_spec is None:
            missing.append(base_name)
            updated.append(req)
            continue
        new_req = f"{name_with_extras}{pixi_spec}" if pixi_spec else name_with_extras
        if old_spec.startswith(";"):
            new_req = f"{new_req} {old_spec}"
        updated.append(new_req)

    return updated, missing


def _is_section_header(line_body: str) -> bool:
    stripped = line_body.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def _bracket_delta_outside_quotes(text: str) -> int:
    """Return net square-bracket delta, ignoring brackets inside strings."""

    delta = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            delta += 1
        elif ch == "]":
            delta -= 1
    return delta


def _replace_toml_array_section_key(
    text: str,
    *,
    section: str,
    key: str,
    new_items: list[str],
) -> str:
    """Replace a TOML array value in the selected section."""

    lines = text.splitlines(keepends=True)
    in_section = False

    for idx, line in enumerate(lines):
        if line.endswith("\r\n"):
            body, line_ending = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            body, line_ending = line[:-1], line[-1]
        else:
            body, line_ending = line, ""

        stripped = body.strip()
        if _is_section_header(body):
            in_section = stripped == f"[{section}]"
            continue

        if not in_section:
            continue

        match = re.match(rf"^(\s*{re.escape(key)}\s*=\s*\[)(.*)$", body)
        if match is None:
            continue

        indent_match = re.match(r"^(\s*)", body)
        indent = indent_match.group(1) if indent_match else ""
        item_indent = f"{indent}  "

        start_idx = idx
        bracket_balance = _bracket_delta_outside_quotes(body)
        end_idx = idx
        while bracket_balance > 0:
            end_idx += 1
            if end_idx >= len(lines):
                raise RuntimeError(f"Unterminated array for [{section}].{key}")
            next_line = lines[end_idx]
            if next_line.endswith("\r\n"):
                next_body = next_line[:-2]
            elif next_line.endswith("\n") or next_line.endswith("\r"):
                next_body = next_line[:-1]
            else:
                next_body = next_line
            bracket_balance += _bracket_delta_outside_quotes(next_body)

        replacement = [f"{indent}{key} = [{line_ending}"]
        replacement.extend(f'{item_indent}"{item}",{line_ending}' for item in new_items)
        replacement.append(f"{indent}]{line_ending}")

        lines[start_idx : end_idx + 1] = replacement
        return "".join(lines)

    raise RuntimeError(f"Could not find array for [{section}].{key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize dependency specifiers in pyproject.toml from pixi.toml "
            "for existing dependency entries."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned updates without writing pyproject.toml.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    pixi_data = _read_toml(PIXI_PATH)
    pyproject_data = _read_toml(PYPROJECT_PATH)
    pixi_specs = _collect_pixi_specs(pixi_data)

    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    missing_by_target: dict[SyncTarget, list[str]] = {}

    for target in TARGETS:
        current = _get_pyproject_list(pyproject_data, target)
        synced, missing = _sync_requirement_list(current, pixi_specs=pixi_specs)
        missing_by_target[target] = missing
        pyproject_text = _replace_toml_array_section_key(
            pyproject_text,
            section=target.section,
            key=target.key,
            new_items=synced,
        )

    if args.dry_run:
        print("Would update pyproject.toml dependency specifiers from pixi.toml")
    else:
        PYPROJECT_PATH.write_text(pyproject_text, encoding="utf-8")
        print("Updated pyproject.toml dependency specifiers from pixi.toml")

    for target in TARGETS:
        missing = sorted(set(missing_by_target[target]))
        if missing:
            print(
                f"Warning: [{target.section}].{target.key} entries not found in pixi.toml: "
                + ", ".join(missing)
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
