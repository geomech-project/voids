from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
NetworkExtraValue: TypeAlias = JsonValue | np.ndarray
