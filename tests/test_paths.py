from __future__ import annotations

from pathlib import Path

from voids.paths import DATA_PATH_ENV, NOTEBOOKS_PATH_ENV, data_path, notebooks_path


def test_data_path_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    """Test environment override for the example-data path."""

    target = tmp_path / "custom-data"
    monkeypatch.setenv(DATA_PATH_ENV, str(target))

    assert data_path() == target.resolve()


def test_data_path_falls_back_to_repo_layout(monkeypatch) -> None:
    """Test default resolution of the example-data path."""

    monkeypatch.delenv(DATA_PATH_ENV, raising=False)

    resolved = data_path()

    assert resolved.name == "data"
    assert (resolved / "manufactured_void_image.npy").exists()


def test_notebooks_path_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    """Test environment override for the notebooks path."""

    target = tmp_path / "custom-notebooks"
    monkeypatch.setenv(NOTEBOOKS_PATH_ENV, str(target))

    assert notebooks_path() == target.resolve()
