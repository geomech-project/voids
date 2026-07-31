from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT_ENV = "VOIDS_PROJECT_ROOT"
PIXI_PROJECT_ROOT_ENV = "PIXI_PROJECT_ROOT"
NOTEBOOKS_PATH_ENV = "VOIDS_NOTEBOOKS_PATH"
EXAMPLES_PATH_ENV = "VOIDS_EXAMPLES_PATH"
DATA_PATH_ENV = "VOIDS_DATA_PATH"


def _repo_root_from_source_tree() -> Path:
    """Return the repository root for an editable source checkout.

    Returns
    -------
    pathlib.Path
        Absolute repository root.

    Raises
    ------
    RuntimeError
        If the installed package layout does not match ``<repo>/src/voids``.

    Notes
    -----
    The fallback is intentionally narrow. It avoids guessing from the current
    working directory and only succeeds when the package is imported from the
    editable source tree.
    """

    root = Path(__file__).resolve().parents[2]
    if (root / "src" / "voids").exists() and (root / "pixi.toml").exists():
        return root
    raise RuntimeError(
        "Could not resolve the project paths from the installed package layout. "
        "Run inside a Pixi environment with VOIDS_* path variables set."
    )


def _resolve_path(env_name: str, relative_to_root: str) -> Path:
    """Resolve a consumer project path without requiring the ``voids`` checkout.

    Parameters
    ----------
    env_name :
        Name of the environment variable to inspect.
    relative_to_root :
        Relative path from the project root used when the environment variable is
        unset.

    Returns
    -------
    pathlib.Path
        Resolved absolute path.
    """

    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser().resolve()

    pixi_project_root = os.getenv(PIXI_PROJECT_ROOT_ENV)
    if pixi_project_root:
        return (Path(pixi_project_root).expanduser() / relative_to_root).resolve()

    return (_repo_root_from_source_tree() / relative_to_root).resolve()


def project_root() -> Path:
    """Return the active consumer project root directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the active Pixi project, an explicit
        ``VOIDS_PROJECT_ROOT``, or the editable ``voids`` checkout.
    """

    return _resolve_path(PROJECT_ROOT_ENV, ".")


def notebooks_path() -> Path:
    """Return the active consumer project's notebooks directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the notebooks directory.
    """

    return _resolve_path(NOTEBOOKS_PATH_ENV, "notebooks")


def examples_path() -> Path:
    """Return the examples directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the examples directory.
    """

    return _resolve_path(EXAMPLES_PATH_ENV, "examples")


def data_path() -> Path:
    """Return the canonical examples-data directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the examples data directory.
    """

    return _resolve_path(DATA_PATH_ENV, "examples/data")
