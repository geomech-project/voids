from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


@dataclass(slots=True)
class Network:
    """Store pore-network topology, geometry, labels, and metadata.

    Parameters
    ----------
    throat_conns :
        Integer array with shape ``(Nt, 2)``. Each row stores the two pore
        indices connected by one throat.
    pore_coords :
        Floating-point array with shape ``(Np, 3)`` containing pore centroid
        coordinates in physical units.
    sample :
        Sample-scale geometry used by porosity and permeability calculations.
    provenance :
        Metadata describing how the network was created or imported.
    schema_version :
        Version tag used by the serialized network schema.
    pore, throat :
        Dictionaries mapping field names to pore-wise and throat-wise arrays.
    pore_labels, throat_labels :
        Dictionaries of boolean masks selecting pore and throat subsets.
    extra :
        Additional metadata not yet promoted to the formal schema.

    Notes
    -----
    The class represents a pore network as a graph

    ``G = (V, E)``

    where pores are vertices ``V`` and throats are edges ``E``. The array
    ``throat_conns`` is the primary topological object used to construct
    adjacency matrices, incidence matrices, and transport operators.
    """

    throat_conns: np.ndarray
    pore_coords: np.ndarray
    sample: SampleGeometry
    provenance: Provenance = field(default_factory=Provenance)
    schema_version: str = "0.1.0"
    pore: dict[str, np.ndarray] = field(default_factory=dict)
    throat: dict[str, np.ndarray] = field(default_factory=dict)
    pore_labels: dict[str, np.ndarray] = field(default_factory=dict)
    throat_labels: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize arrays immediately after initialization.

        Notes
        -----
        Topology is converted to ``int64``, coordinates to floating point, pore
        and throat fields to NumPy arrays, and label dictionaries to boolean
        arrays. The method performs coercion only; semantic validation is left to
        :func:`voids.core.validation.validate_network`.
        """

        self.throat_conns = np.asarray(self.throat_conns, dtype=np.int64)
        self.pore_coords = np.asarray(self.pore_coords, dtype=float)
        for d in (self.pore, self.throat):
            for k, v in list(d.items()):
                d[k] = np.asarray(v)
        for d in (self.pore_labels, self.throat_labels):
            for k, v in list(d.items()):
                d[k] = np.asarray(v, dtype=bool)

    @property
    def Np(self) -> int:
        """Return the number of pores.

        Returns
        -------
        int
            Number of rows in :attr:`pore_coords`.
        """

        return int(self.pore_coords.shape[0])

    @property
    def Nt(self) -> int:
        """Return the number of throats.

        Returns
        -------
        int
            Number of rows in :attr:`throat_conns`.
        """

        return int(self.throat_conns.shape[0])

    def get_pore_array(self, name: str) -> np.ndarray:
        """Return a pore field by name.

        Parameters
        ----------
        name :
            Key in :attr:`pore`.

        Returns
        -------
        numpy.ndarray
            Requested pore-wise array.

        Raises
        ------
        KeyError
            If the field is not present.
        """

        if name not in self.pore:
            raise KeyError(f"Missing pore field '{name}'")
        return self.pore[name]

    def get_throat_array(self, name: str) -> np.ndarray:
        """Return a throat field by name.

        Parameters
        ----------
        name :
            Key in :attr:`throat`.

        Returns
        -------
        numpy.ndarray
            Requested throat-wise array.

        Raises
        ------
        KeyError
            If the field is not present.
        """

        if name not in self.throat:
            raise KeyError(f"Missing throat field '{name}'")
        return self.throat[name]

    def copy(self) -> "Network":
        """Return a deep-array copy of the network.

        Returns
        -------
        Network
            New network instance whose topology, coordinates, field arrays, and
            labels are copied.

        Notes
        -----
        Array-valued data are copied to avoid in-place aliasing. Metadata
        containers such as :class:`SampleGeometry` and :class:`Provenance` are
        reused because they are treated as immutable records in current usage.
        """

        return Network(
            throat_conns=self.throat_conns.copy(),
            pore_coords=self.pore_coords.copy(),
            sample=self.sample,
            provenance=self.provenance,
            schema_version=self.schema_version,
            pore={k: v.copy() for k, v in self.pore.items()},
            throat={k: v.copy() for k, v in self.throat.items()},
            pore_labels={k: v.copy() for k, v in self.pore_labels.items()},
            throat_labels={k: v.copy() for k, v in self.throat_labels.items()},
            extra={**self.extra},
        )
