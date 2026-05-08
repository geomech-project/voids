from __future__ import annotations

import numpy as np
import pytest

from voids.io.hdf5 import load_hdf5, save_hdf5


def test_hdf5_roundtrip(tmp_path, line_network):
    """Test HDF5 save/load roundtrip for a small network."""

    p = tmp_path / "net.h5"
    save_hdf5(line_network, p)
    net2 = load_hdf5(p)
    assert net2.Np == line_network.Np
    assert net2.Nt == line_network.Nt
    assert np.array_equal(net2.throat_conns, line_network.throat_conns)
    assert np.allclose(net2.pore_coords, line_network.pore_coords)
    assert np.allclose(net2.pore["volume"], line_network.pore["volume"])
    assert np.array_equal(net2.pore_labels["inlet_xmin"], line_network.pore_labels["inlet_xmin"])


def test_hdf5_extra_roundtrip_accepts_numpy_metadata(tmp_path, line_network):
    """Network metadata may include NumPy values produced by import adapters."""

    line_network.extra["throat.hydraulic_size_factors"] = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    line_network.extra["numpy_scalars"] = {
        "int": np.int64(7),
        "float": np.float32(1.25),
        "bool": np.bool_(True),
    }

    p = tmp_path / "net_numpy_extra.h5"
    save_hdf5(line_network, p)
    net2 = load_hdf5(p)

    assert np.allclose(
        np.asarray(net2.extra["throat.hydraulic_size_factors"]),
        line_network.extra["throat.hydraulic_size_factors"],
    )
    assert net2.extra["numpy_scalars"] == {
        "int": 7,
        "float": 1.25,
        "bool": True,
    }


def test_hdf5_extra_rejects_unsupported_metadata(tmp_path, line_network):
    """Unsupported metadata should fail loudly instead of being stringified."""

    line_network.extra["unsupported"] = object()

    with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
        save_hdf5(line_network, tmp_path / "net_unsupported_extra.h5")
