from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np


SignedIntegerDType: TypeAlias = type[np.int16] | type[np.int32] | type[np.int64]


class PoreSpyFilters(Protocol):
    def find_peaks(self, *args: object, **kwargs: object) -> np.ndarray: ...
    def trim_saddle_points(self, *args: object, **kwargs: object) -> np.ndarray: ...
    def trim_nearby_peaks(self, *args: object, **kwargs: object) -> np.ndarray: ...


class PoreSpyNetworks(Protocol):
    def snow2(self, *args: object, **kwargs: object) -> object: ...
    def regions_to_network(self, regions: object, /, **kwargs: object) -> object: ...


class PoreSpyNetworkModule(Protocol):
    networks: PoreSpyNetworks


class PoreSpyModule(PoreSpyNetworkModule, Protocol):
    filters: PoreSpyFilters
