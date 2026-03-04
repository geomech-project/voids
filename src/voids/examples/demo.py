from __future__ import annotations

import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def make_linear_chain_network(
    num_pores: int = 3,
    *,
    axis: str = "x",
    length: float = 1.0,
    cross_section: float = 1.0,
    bulk_volume: float = 10.0,
    pore_volume: float = 1.0,
    throat_volume: float = 0.5,
    throat_length: float = 1.0,
    hydraulic_conductance: float = 1.0,
) -> Network:
    """Build a deterministic one-dimensional pore-throat chain.

    Parameters
    ----------
    num_pores :
        Number of pores in the chain. The number of throats is
        ``num_pores - 1``.
    axis :
        Axis along which the chain is embedded.
    length :
        Sample length along the chosen axis.
    cross_section :
        Cross-sectional area normal to the flow axis.
    bulk_volume :
        Bulk sample volume associated with the toy problem.
    pore_volume, throat_volume :
        Synthetic pore and throat void volumes.
    throat_length :
        Length assigned to each throat.
    hydraulic_conductance :
        Precomputed throat hydraulic conductance.

    Returns
    -------
    Network
        Synthetic line network with canonical inlet and outlet labels.

    Raises
    ------
    ValueError
        If the number of pores, axis, or geometric parameters are invalid.

    Notes
    -----
    The pore coordinates are uniformly spaced so that the pore positions satisfy

    ``x_k = k * length / (num_pores - 1)``

    along the selected axis. The function is intended for solver smoke tests,
    tutorials, and regression examples rather than realistic porous-media
    reconstruction.
    """

    if num_pores < 2:
        raise ValueError("num_pores must be >= 2")
    if axis not in _AXIS_TO_INDEX:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    if length <= 0 or cross_section <= 0 or bulk_volume <= 0:
        raise ValueError("length, cross_section, and bulk_volume must be positive")
    if pore_volume < 0 or throat_volume < 0 or throat_length < 0 or hydraulic_conductance < 0:
        raise ValueError("pore/throat properties must be nonnegative")

    coords = np.zeros((num_pores, 3), dtype=float)
    coords[:, _AXIS_TO_INDEX[axis]] = np.linspace(0.0, float(length), num_pores)
    throat_conns = np.column_stack(
        [np.arange(num_pores - 1, dtype=np.int64), np.arange(1, num_pores, dtype=np.int64)]
    )

    pore_labels: dict[str, np.ndarray] = {
        f"inlet_{axis}min": np.zeros(num_pores, dtype=bool),
        f"outlet_{axis}max": np.zeros(num_pores, dtype=bool),
        "boundary": np.zeros(num_pores, dtype=bool),
    }
    pore_labels[f"inlet_{axis}min"][0] = True
    pore_labels[f"outlet_{axis}max"][-1] = True
    pore_labels["boundary"][[0, -1]] = True

    sample = SampleGeometry(
        bulk_volume=float(bulk_volume),
        lengths={axis: float(length)},
        cross_sections={axis: float(cross_section)},
    )
    provenance = Provenance(
        source_kind="synthetic_demo",
        extraction_method="linear_chain",
        user_notes={"num_pores": int(num_pores), "axis": axis},
    )

    return Network(
        throat_conns=throat_conns,
        pore_coords=coords,
        sample=sample,
        provenance=provenance,
        pore={"volume": np.full(num_pores, float(pore_volume), dtype=float)},
        throat={
            "volume": np.full(num_pores - 1, float(throat_volume), dtype=float),
            "length": np.full(num_pores - 1, float(throat_length), dtype=float),
            "hydraulic_conductance": np.full(
                num_pores - 1, float(hydraulic_conductance), dtype=float
            ),
        },
        pore_labels=pore_labels,
    )
