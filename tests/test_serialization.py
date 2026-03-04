from __future__ import annotations

import numpy as np

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
