from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voids.core.network import Network
from voids.core.sample import SampleGeometry
from voids.core.validation import assert_finite, validate_network
from voids.examples.demo import make_linear_chain_network
from voids.examples.manufactured import (
    make_manufactured_void_image,
    save_default_manufactured_void_image,
)
from voids.examples.mesh import make_cartesian_mesh_network


def test_assert_finite_rejects_nonfinite_values() -> None:
    """Test rejection of arrays containing non-finite values."""

    with pytest.raises(ValueError, match="contains non-finite"):
        assert_finite("pressure", np.array([1.0, np.nan]))


def test_validate_network_rejects_parallel_throats_when_disallowed() -> None:
    """Test strict rejection of parallel throats."""

    net = Network(
        throat_conns=np.array([[0, 1], [1, 0]], dtype=int),
        pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sample=SampleGeometry(bulk_volume=1.0),
        pore={"volume": np.array([1.0, 1.0])},
        throat={"volume": np.array([0.1, 0.1]), "length": np.array([1.0, 1.0])},
    )

    with pytest.raises(ValueError, match="parallel throats found"):
        validate_network(net, allow_parallel_throats=False)


def test_validate_network_warns_for_parallel_throats_and_missing_recommended_fields() -> None:
    """Test validation warnings for parallel throats and missing recommended fields."""

    net = Network(
        throat_conns=np.array([[0, 1], [1, 0]], dtype=int),
        pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sample=SampleGeometry(bulk_volume=1.0),
        throat={"volume": np.array([0.1, 0.1]), "length": np.array([1.0, 1.0])},
    )

    with pytest.warns(RuntimeWarning, match="parallel throats detected"):
        with pytest.warns(RuntimeWarning, match="Recommended pore field missing"):
            validate_network(net)


def test_validate_network_rejects_negative_and_nonpositive_geometry(line_network) -> None:
    """Test validation of negative pore volumes and nonpositive throat lengths."""

    bad_pore = line_network.copy()
    bad_pore.pore["volume"][0] = -1.0
    with pytest.raises(ValueError, match="contains negative values"):
        validate_network(bad_pore)

    bad_throat = line_network.copy()
    bad_throat.throat["length"][0] = 0.0
    with pytest.raises(ValueError, match="contains nonpositive values"):
        validate_network(bad_throat)


def test_validate_network_covers_structural_error_branches() -> None:
    """Test structural validation failures for malformed topology and coordinates."""

    sample = SampleGeometry(bulk_volume=1.0)

    with pytest.raises(ValueError, match="throat_conns must have shape"):
        validate_network(
            Network(
                throat_conns=np.array([0, 1]),
                pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                sample=sample,
            )
        )

    with pytest.raises(ValueError, match="pore_coords must have shape"):
        validate_network(
            Network(
                throat_conns=np.array([[0, 1]]),
                pore_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
                sample=sample,
            )
        )

    with pytest.raises(ValueError, match="pore_coords contains NaNs"):
        validate_network(
            Network(
                throat_conns=np.array([[0, 1]]),
                pore_coords=np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
                sample=sample,
            )
        )

    with pytest.raises(ValueError, match="out-of-range pore indices"):
        validate_network(
            Network(
                throat_conns=np.array([[0, 2]]),
                pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                sample=sample,
            )
        )


def test_validate_network_covers_field_and_label_shape_errors(line_network) -> None:
    """Test validation failures for malformed field and label arrays."""

    bad_pore_dim = line_network.copy()
    bad_pore_dim.pore["volume"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="wrong first dimension"):
        validate_network(bad_pore_dim)

    bad_pore_nan = line_network.copy()
    bad_pore_nan.pore["volume"][0] = np.nan
    with pytest.raises(ValueError, match="contains NaNs"):
        validate_network(bad_pore_nan)

    bad_throat_dim = line_network.copy()
    bad_throat_dim.throat["volume"] = np.array([0.5])
    with pytest.raises(ValueError, match="wrong first dimension"):
        validate_network(bad_throat_dim)

    bad_throat_nan = line_network.copy()
    bad_throat_nan.throat["volume"][0] = np.nan
    with pytest.raises(ValueError, match="contains NaNs"):
        validate_network(bad_throat_nan)

    bad_throat_negative = line_network.copy()
    bad_throat_negative.throat["volume"][0] = -0.1
    with pytest.raises(ValueError, match="contains negative values"):
        validate_network(bad_throat_negative)

    bad_throat_label = line_network.copy()
    bad_throat_label.throat_labels["bad"] = np.array([True])
    with pytest.raises(ValueError, match="wrong shape"):
        validate_network(bad_throat_label)


def test_validate_network_ignores_nonpositive_sample_bulk_volume_if_sample_raises() -> None:
    """Test that sample-volume validation is skipped when the sample object raises."""

    net = Network(
        throat_conns=np.array([[0, 1]], dtype=int),
        pore_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sample=SampleGeometry(bulk_volume=0.0),
        pore={"volume": np.array([1.0, 1.0])},
        throat={"volume": np.array([0.1]), "length": np.array([1.0])},
    )

    validate_network(net)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_pores": 1}, "num_pores must be >= 2"),
        ({"axis": "q"}, "axis must be one of"),
        ({"length": 0.0}, "must be positive"),
        ({"pore_volume": -1.0}, "must be nonnegative"),
    ],
)
def test_make_linear_chain_network_rejects_invalid_arguments(
    kwargs: dict[str, float | int | str], message: str
) -> None:
    """Test invalid argument combinations for the linear-chain example generator."""

    with pytest.raises(ValueError, match=message):
        make_linear_chain_network(**kwargs)


def test_save_default_manufactured_void_image_creates_parent_dirs(tmp_path: Path) -> None:
    """Test creation of parent directories when saving the default manufactured image."""

    target = tmp_path / "nested" / "manufactured.npy"

    returned = save_default_manufactured_void_image(target)

    assert returned == target
    restored = np.load(target)
    assert restored.dtype == bool
    assert restored.shape == make_manufactured_void_image().shape


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (20,)}, "shape must have length 2 or 3"),
        ({"shape": (1, 20)}, "each entry in shape must be >= 2"),
        ({"shape": (4, 4), "spacing": 0.0}, "spacing must be positive"),
        ({"shape": (4, 4), "pore_radius": 0.0}, "pore_radius and throat_radius must be positive"),
        ({"shape": (4, 4), "pore_radius": 0.5, "spacing": 1.0}, "pore_radius must be smaller"),
        ({"shape": (4, 4), "throat_radius": 0.5, "spacing": 1.0}, "throat_radius must be smaller"),
        ({"shape": (4, 4), "thickness": 0.0}, "thickness must be positive"),
        (
            {"shape": (4, 4), "spacing": 1.0, "pore_radius": 0.6, "throat_radius": 0.1},
            "pore_radius must be smaller",
        ),
    ],
)
def test_make_cartesian_mesh_network_rejects_invalid_geometry(
    kwargs: dict[str, object], message: str
) -> None:
    """Test invalid geometry inputs for the Cartesian mesh example generator."""

    with pytest.raises(ValueError, match=message):
        make_cartesian_mesh_network(**kwargs)
