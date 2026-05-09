from __future__ import annotations

import numpy as np
import pytest
import h5py

from voids.image import porosity as pmap
from voids.image.porosity import (
    PermeabilityMap,
    PorosityMap,
    calibrated_porosity_from_grayscale,
    kozeny_carman_inverse_permeability,
    kozeny_carman_permeability,
    load_permeability_map_hdf5,
    load_porosity_map_hdf5,
    permeability_map_from_porosity,
    porosity_map_from_binary,
    porosity_map_from_grayscale,
    save_permeability_map_hdf5,
    save_porosity_map_hdf5,
)


def test_porosity_map_from_binary_2d_block_average() -> None:
    """A coarse cell stores the void fraction in its fine-voxel block."""

    void = np.array(
        [
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    porosity = porosity_map_from_binary(
        void,
        block_shape=(2, 2),
        voxel_size=(2.0e-6, 3.0e-6),
        metadata={"sample": "toy_2d"},
    )

    assert porosity.shape == (2, 2)
    assert porosity.cell_size == (4.0e-6, 6.0e-6)
    assert np.allclose(porosity.values, [[0.75, 0.5], [0.0, 0.75]])
    assert np.isclose(porosity.mean_porosity, 0.5)
    assert np.isclose(porosity.bulk_volume, 4 * 24.0e-12)
    assert np.isclose(porosity.void_volume, 2.0 * 24.0e-12)
    assert porosity.metadata["sample"] == "toy_2d"


def test_porosity_map_from_binary_defaults_to_voxel_cells() -> None:
    """Omitting block shape keeps one porosity cell per fine voxel."""

    porosity = porosity_map_from_binary(np.array([[True, False]]))

    assert porosity.ndim == 2
    assert porosity.shape == (1, 2)
    assert porosity.cell_size == (1.0, 1.0)
    assert np.allclose(porosity.values, [[1.0, 0.0]])


def test_porosity_map_from_binary_3d_block_average_and_solid_polarity() -> None:
    """The binary path supports 3D arrays and explicit phase polarity."""

    solid = np.ones((4, 2, 2), dtype=bool)
    solid[:2, :, :] = False
    solid[2:, 0, :] = False

    porosity = porosity_map_from_binary(
        solid,
        block_shape=(2, 1, 2),
        image_is_void=False,
        strict=True,
    )

    assert porosity.shape == (2, 2, 1)
    assert np.allclose(porosity.values[:, :, 0], [[1.0, 1.0], [1.0, 0.0]])


def test_porosity_map_strict_block_shape_rejects_partial_cells() -> None:
    """Strict mode fails rather than silently changing the represented domain."""

    with pytest.raises(ValueError, match="exactly divisible"):
        porosity_map_from_binary(np.ones((5, 4), dtype=bool), block_shape=(2, 2))


def test_porosity_map_rejects_block_shape_larger_than_image() -> None:
    """A coarse-cell block cannot be larger than the input image domain."""

    with pytest.raises(ValueError, match="must not exceed"):
        porosity_map_from_binary(np.ones((2, 2), dtype=bool), block_shape=(3, 1))


def test_porosity_map_non_strict_trims_partial_cells() -> None:
    """Optional trimming is explicit in metadata when exact blocks are impossible."""

    porosity = porosity_map_from_binary(
        np.ones((5, 4), dtype=bool),
        block_shape=(2, 2),
        strict=False,
    )

    assert porosity.shape == (2, 2)
    assert porosity.metadata["fine_shape"] == (5, 4)
    assert porosity.metadata["trimmed_shape"] == (4, 4)


def test_binary_porosity_map_rejects_nonbinary_inputs() -> None:
    """Segmented-image inputs must be boolean or explicit 0/1 labels."""

    with pytest.raises(ValueError, match="finite"):
        porosity_map_from_binary(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="0/1"):
        porosity_map_from_binary(np.array([[0, 2]]))


def test_calibrated_porosity_from_grayscale_clips_to_background_porosity() -> None:
    """Two-point grayscale calibration preserves the background-porosity floor."""

    gray = np.array([[0.0, 5.0, 10.0, 15.0]])

    phi = calibrated_porosity_from_grayscale(
        gray,
        pore_gray=0.0,
        solid_gray=10.0,
        background_porosity=0.2,
    )

    assert np.allclose(phi, [[1.0, 0.6, 0.2, 0.2]])


def test_calibrated_porosity_from_grayscale_can_extrapolate_when_requested() -> None:
    """The calibration can expose out-of-range values for diagnostic use."""

    gray = np.array([[15.0]])

    phi = calibrated_porosity_from_grayscale(
        gray,
        pore_gray=0.0,
        solid_gray=10.0,
        background_porosity=0.2,
        clip=False,
    )

    assert np.allclose(phi, [[-0.2]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"solid_gray": 1.0, "pore_gray": 0.0, "grayscale": np.array([[np.nan]])}, "finite"),
        ({"solid_gray": np.nan, "pore_gray": 0.0, "grayscale": np.array([[0.0]])}, "finite"),
        ({"solid_gray": 1.0, "pore_gray": 1.0, "grayscale": np.array([[0.0]])}, "differ"),
        (
            {
                "solid_gray": 1.0,
                "pore_gray": 0.0,
                "background_porosity": -0.1,
                "grayscale": np.array([[0.0]]),
            },
            r"\[0, 1\]",
        ),
    ],
)
def test_calibrated_porosity_from_grayscale_rejects_invalid_calibration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Calibration inputs fail loudly when the science metadata is invalid."""

    with pytest.raises(ValueError, match=message):
        calibrated_porosity_from_grayscale(**kwargs)


def test_porosity_map_from_grayscale_blocks_and_metadata() -> None:
    """Grayscale calibration can be coarsened directly to FEM-style cells."""

    gray = np.array(
        [
            [0.0, 10.0, 5.0, 5.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )

    porosity = porosity_map_from_grayscale(
        gray,
        pore_gray=0.0,
        solid_gray=10.0,
        background_porosity=0.0,
        block_shape=(1, 2),
        voxel_size=1.5,
        metadata={"sample": "toy_grayscale"},
    )

    assert porosity.shape == (2, 2)
    assert porosity.cell_size == (1.5, 3.0)
    assert np.allclose(porosity.values, [[0.5, 0.5], [1.0, 0.0]])
    assert porosity.metadata["source_kind"] == "grayscale_linear_calibration"
    assert porosity.metadata["background_porosity"] == 0.0
    assert porosity.metadata["sample"] == "toy_grayscale"


def test_porosity_map_hdf5_roundtrip(tmp_path) -> None:
    """HDF5 export preserves values and calibration metadata."""

    porosity = PorosityMap(
        values=np.array([[0.25, 0.75], [0.5, 1.0]]),
        cell_size=(2.0, 3.0),
        origin=(10.0, 20.0),
        units={"length": "um"},
        metadata={
            "source_kind": "synthetic_test",
            "seed": np.int64(11),
            "scale": np.float32(2.5),
            "active": np.bool_(True),
            "raw": np.array([1, 2]),
        },
    )
    path = tmp_path / "toy_porosity.h5"

    save_porosity_map_hdf5(porosity, path)
    loaded = load_porosity_map_hdf5(path)

    assert np.allclose(loaded.values, porosity.values)
    assert loaded.cell_size == (2.0, 3.0)
    assert loaded.origin == (10.0, 20.0)
    assert loaded.units == {"length": "um"}
    assert loaded.metadata == {
        "source_kind": "synthetic_test",
        "seed": 11,
        "scale": 2.5,
        "active": True,
        "raw": [1, 2],
    }


def test_porosity_map_validates_physical_values() -> None:
    """Porosity maps reject nonphysical cell-average values."""

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PorosityMap(values=np.array([[1.1]]))

    with pytest.raises(ValueError, match="finite"):
        PorosityMap(values=np.array([[np.nan]]))


def test_porosity_map_validates_physical_metadata() -> None:
    """Cell sizes and origins are part of the physical model metadata."""

    with pytest.raises(ValueError, match="cell_size must"):
        PorosityMap(values=np.array([[0.5]]), cell_size=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="cell_size entries"):
        PorosityMap(values=np.array([[0.5]]), cell_size=(1.0, 0.0))
    with pytest.raises(ValueError, match="origin must"):
        PorosityMap(values=np.array([[0.5]]), origin=(0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="origin entries"):
        PorosityMap(values=np.array([[0.5]]), origin=(0.0, np.nan))


def test_porosity_map_hdf5_rejects_unsupported_metadata(tmp_path) -> None:
    """Unsupported metadata fails rather than being stringified silently."""

    porosity = PorosityMap(values=np.array([[0.5]]), metadata={"bad": object()})

    with pytest.raises(TypeError, match="not JSON serializable"):
        save_porosity_map_hdf5(porosity, tmp_path / "bad_metadata.h5")


def test_load_porosity_map_hdf5_rejects_unknown_schema(tmp_path) -> None:
    """HDF5 files must advertise the porosity-map schema."""

    path = tmp_path / "bad_schema.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = np.bytes_("unknown")
        f.create_dataset("porosity", data=np.array([[0.5]]))

    with pytest.raises(ValueError, match="Unsupported"):
        load_porosity_map_hdf5(path)


def test_private_metadata_and_block_helpers_cover_defensive_branches(tmp_path) -> None:
    """Exercise defensive helpers that are normally guarded by public APIs."""

    path = tmp_path / "metadata.h5"
    with h5py.File(path, "w") as f:
        assert pmap._read_json_attr(f, "missing", default={"ok": True}) == {"ok": True}
        f.attrs["payload"] = np.bytes_('{"ok": true}')
        assert pmap._read_json_attr(f, "payload") == {"ok": True}

    with pytest.raises(ValueError, match="length"):
        pmap._block_mean(np.ones((2, 2)), block_shape=(1,), strict=True)


def test_kozeny_carman_permeability_handles_endpoint_limits() -> None:
    """The permeability closure preserves the solid and free-flow limits."""

    phi = np.array([[0.0, 0.5, 1.0]])

    permeability = kozeny_carman_permeability(
        phi,
        characteristic_length=2.0,
        kozeny_constant=180.0,
    )

    assert permeability[0, 0] == 0.0
    assert np.isclose(permeability[0, 1], 2.0**2 * 0.5**3 / (180.0 * 0.5**2))
    assert np.isposinf(permeability[0, 2])


def test_kozeny_carman_permeability_supports_caps_and_custom_endpoints() -> None:
    """Solver-facing permeability caps are explicit closure parameters."""

    phi = np.array([[0.0, 0.5, 1.0]])

    permeability = kozeny_carman_permeability(
        phi,
        characteristic_length=2.0e-6,
        solid_permeability=1.0e-18,
        free_flow_permeability=1.0e-6,
        max_permeability=1.0e-8,
    )

    assert permeability[0, 0] == 1.0e-18
    assert np.isclose(permeability[0, 1], (2.0e-6) ** 2 * 0.5**3 / (180.0 * 0.5**2))
    assert permeability[0, 2] == 1.0e-8


def test_kozeny_carman_inverse_permeability_matches_reciprocal_limits() -> None:
    """The inverse closure is the Darcy-Brinkman resistance form."""

    phi = np.array([[0.0, 0.5, 1.0]])

    inverse = kozeny_carman_inverse_permeability(
        phi,
        characteristic_length=2.0,
        kozeny_constant=180.0,
    )

    assert np.isposinf(inverse[0, 0])
    assert np.isclose(inverse[0, 1], 180.0 * 0.5**2 / (2.0**2 * 0.5**3))
    assert inverse[0, 2] == 0.0

    capped = kozeny_carman_inverse_permeability(
        phi,
        characteristic_length=2.0,
        kozeny_constant=180.0,
        max_inverse_permeability=50.0,
    )

    assert np.allclose(capped, [[50.0, 50.0, 0.0]])


def test_kozeny_carman_closure_rejects_invalid_parameters() -> None:
    """Closure constants and endpoint caps are part of the physical model."""

    phi = np.array([[0.5]])

    with pytest.raises(ValueError, match="characteristic_length"):
        kozeny_carman_permeability(phi, characteristic_length=0.0)
    with pytest.raises(ValueError, match="kozeny_constant"):
        kozeny_carman_permeability(phi, characteristic_length=1.0, kozeny_constant=0.0)
    with pytest.raises(ValueError, match="solid_permeability"):
        kozeny_carman_permeability(
            phi,
            characteristic_length=1.0,
            solid_permeability=-1.0,
        )
    with pytest.raises(ValueError, match="free_flow_permeability"):
        kozeny_carman_permeability(
            phi,
            characteristic_length=1.0,
            free_flow_permeability=np.nan,
        )
    with pytest.raises(ValueError, match="max_permeability"):
        kozeny_carman_permeability(
            phi,
            characteristic_length=1.0,
            max_permeability=np.inf,
        )
    with pytest.raises(ValueError, match="solid_inverse_permeability"):
        kozeny_carman_inverse_permeability(
            phi,
            characteristic_length=1.0,
            solid_inverse_permeability=np.nan,
        )
    with pytest.raises(ValueError, match="free_flow_inverse_permeability"):
        kozeny_carman_inverse_permeability(
            phi,
            characteristic_length=1.0,
            free_flow_inverse_permeability=np.inf,
        )
    with pytest.raises(ValueError, match="max_inverse_permeability"):
        kozeny_carman_inverse_permeability(
            phi,
            characteristic_length=1.0,
            max_inverse_permeability=0.0,
        )


def test_permeability_map_from_porosity_preserves_grid_metadata() -> None:
    """A permeability map is tied to the porosity-map grid metadata."""

    porosity = PorosityMap(
        values=np.array([[0.25, 0.75]]),
        cell_size=(2.0e-6, 3.0e-6),
        origin=(1.0e-6, 2.0e-6),
        metadata={"case": "toy"},
    )

    permeability = permeability_map_from_porosity(
        porosity,
        characteristic_length=2.0e-6,
        max_permeability=1.0e-9,
        metadata={"closure_note": "synthetic"},
    )

    expected = kozeny_carman_permeability(
        porosity.values,
        characteristic_length=2.0e-6,
        max_permeability=1.0e-9,
    )
    assert np.allclose(permeability.values, expected)
    assert permeability.cell_size == porosity.cell_size
    assert permeability.origin == porosity.origin
    assert permeability.units == {"length": "m", "permeability": "m^2"}
    assert permeability.metadata["source_kind"] == "kozeny_carman_permeability"
    assert permeability.metadata["porosity_source_metadata"] == {"case": "toy"}
    assert permeability.metadata["closure_note"] == "synthetic"


def test_permeability_map_inverse_values_and_finite_mean() -> None:
    """The map wrapper exposes reciprocal resistance values for DBS solvers."""

    permeability = PermeabilityMap(values=np.array([[0.0, 2.0], [np.inf, 4.0]]))

    assert np.allclose(permeability.inverse_values, [[np.inf, 0.5], [0.0, 0.25]])
    assert np.isclose(permeability.finite_mean_permeability, 2.0)
    assert permeability.shape == (2, 2)
    assert permeability.ndim == 2

    all_free = PermeabilityMap(values=np.array([[np.inf]]))
    assert np.isnan(all_free.finite_mean_permeability)


def test_permeability_map_hdf5_roundtrip(tmp_path) -> None:
    """HDF5 export preserves Kozeny-Carman field values and metadata."""

    permeability = PermeabilityMap(
        values=np.array([[1.0e-15, 2.0e-15]]),
        cell_size=(4.0e-4, 4.0e-4),
        origin=(0.0, 1.0),
        metadata={"source_kind": "kozeny_carman_permeability", "raw": np.array([1])},
    )
    path = tmp_path / "toy_permeability.h5"

    save_permeability_map_hdf5(permeability, path)
    loaded = load_permeability_map_hdf5(path)

    assert np.allclose(loaded.values, permeability.values)
    assert loaded.cell_size == permeability.cell_size
    assert loaded.origin == permeability.origin
    assert loaded.metadata == {"source_kind": "kozeny_carman_permeability", "raw": [1]}


def test_permeability_map_validation_and_schema_errors(tmp_path) -> None:
    """Permeability fields reject negative/NaN values and unknown schemas."""

    with pytest.raises(ValueError, match="nonnegative"):
        PermeabilityMap(values=np.array([[-1.0]]))
    with pytest.raises(ValueError, match="NaN"):
        PermeabilityMap(values=np.array([[np.nan]]))

    path = tmp_path / "bad_permeability_schema.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = np.bytes_("unknown")
        f.create_dataset("permeability", data=np.array([[1.0e-15]]))

    with pytest.raises(ValueError, match="Unsupported"):
        load_permeability_map_hdf5(path)
