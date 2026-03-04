from __future__ import annotations

import numpy as np

from voids.core.sample import SampleGeometry
from voids.io.porespy import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry


def test_from_porespy_maps_openpnm_aliases_and_derives_fields() -> None:
    """Test alias normalization and derived-field construction during PoreSpy import."""

    d = {
        "pore.coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.volume": np.array([1.0, 1.0]),
        "throat.volume": np.array([0.1]),
        "throat.cross_sectional_area": np.array([4.0]),
        "throat.total_length": np.array([0.7]),
        "throat.conduit_lengths.pore1": np.array([0.1]),
        "throat.conduit_lengths.throat": np.array([0.5]),
        "throat.conduit_lengths.pore2": np.array([0.1]),
        "throat.perimeter": np.array([8.0]),
        "pore.radius_inscribed": np.array([1.0, 1.0]),
        "pore.left": np.array([True, False]),
        "pore.right": np.array([False, True]),
    }
    net = from_porespy(d, sample=SampleGeometry(bulk_volume=10.0))

    assert np.allclose(net.throat["area"], [4.0])
    assert np.allclose(net.throat["length"], [0.7])
    assert np.allclose(net.throat["core_length"], [0.5])
    assert np.allclose(net.throat["pore1_length"], [0.1])
    assert np.allclose(net.throat["pore2_length"], [0.1])
    # Derived from A/P^2 = 4/64 = 1/16
    assert np.allclose(net.throat["shape_factor"], [1 / 16])
    # Derived from radius -> diameter and area
    assert np.allclose(net.pore["diameter_inscribed"], [2.0, 2.0])
    assert np.allclose(net.pore["area"], [np.pi, np.pi])
    assert net.pore_labels["inlet_xmin"].sum() == 1
    assert net.pore_labels["outlet_xmax"].sum() == 1


def test_scale_porespy_geometry_and_infer_boundaries() -> None:
    """Test voxel-to-physical scaling and simple Cartesian boundary inference."""

    d = {
        "pore.coords": np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.region_volume": np.array([10.0, 12.0]),
        "throat.cross_sectional_area": np.array([3.0]),
        "throat.total_length": np.array([2.0]),
    }
    scaled = scale_porespy_geometry(d, voxel_size=2.0)
    labeled = ensure_cartesian_boundary_labels(scaled, axes=("x",))

    assert np.allclose(scaled["pore.coords"], [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert np.allclose(scaled["pore.volume"], [80.0, 96.0])
    assert np.allclose(scaled["throat.volume"], [48.0])
    assert labeled["pore.inlet_xmin"].tolist() == [True, False]
    assert labeled["pore.outlet_xmax"].tolist() == [False, True]
    assert labeled["pore.boundary"].tolist() == [True, True]
