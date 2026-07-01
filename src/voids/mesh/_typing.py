from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class MeshIOModule(Protocol):
    def Mesh(
        self,
        *,
        points: np.ndarray,
        cells: list[tuple[str, np.ndarray]],
        cell_data: dict[str, list[np.ndarray]],
    ) -> object: ...

    def write(self, path: Path, mesh: object, *, file_format: str | None = None) -> None: ...
