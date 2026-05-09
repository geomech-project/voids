from __future__ import annotations

import json
import struct

import h5py
import numpy as np
import pytest
from scipy.io import netcdf_file

from voids.io import volume as vol
from voids.io.volume import (
    SurfaceMesh,
    VolumeData,
    load_surface_mesh,
    load_volume,
    load_volume_data,
    save_surface_mesh,
    save_volume,
    save_volume_bundle,
    surface_mesh_from_binary_volume,
)


def test_volume_roundtrip_common_array_formats(tmp_path) -> None:
    """Synthetic image volumes should round-trip through common voxel formats."""

    volume = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    voxel_size = (1.0e-6, 2.0e-6, 3.0e-6)

    expected = {
        "raw": tmp_path / "case.raw",
        "npy": tmp_path / "case.npy",
        "h5": tmp_path / "case.h5",
        "nc": tmp_path / "case.nc",
        "tiff": tmp_path / "case.tiff",
    }
    for fmt, path in expected.items():
        save_volume(
            volume,
            path,
            file_format=fmt,
            metadata={"case": fmt},
            voxel_size=voxel_size,
            units={"length": "m"},
        )
        loaded = load_volume(path, file_format=fmt)
        assert np.array_equal(loaded, volume)
        loaded_data = load_volume_data(path, file_format=fmt)
        assert np.array_equal(loaded_data.values, volume)
        assert loaded_data.values.dtype == volume.dtype
        assert loaded_data.voxel_size == voxel_size
        assert loaded_data.units == {"length": "m"}
        assert loaded_data.metadata == {"case": fmt}

    raw_metadata = json.loads((tmp_path / "case.raw.json").read_text(encoding="utf-8"))
    assert raw_metadata["shape"] == [2, 3, 4]
    assert raw_metadata["stored_dtype"] == "uint8"
    assert raw_metadata["voxel_size"] == [1.0e-6, 2.0e-6, 3.0e-6]
    assert raw_metadata["metadata"] == {"case": "raw"}
    assert (tmp_path / "case.npy.json").exists()
    assert (tmp_path / "case.tiff.json").exists()


def test_volume_data_restores_bool_dtype_from_integer_storage(tmp_path) -> None:
    """Boolean phase fields should reload with their semantic dtype."""

    volume = np.zeros((3, 3, 3), dtype=bool)
    volume[1, 1, 1] = True

    for fmt in ("raw", "npy", "h5", "nc", "tiff"):
        path = save_volume(volume, tmp_path / f"case.{fmt}", file_format=fmt)
        loaded = load_volume_data(path, file_format=fmt)
        assert loaded.values.dtype == np.dtype(bool)
        assert np.array_equal(loaded.values, volume)


def test_volume_data_preserves_and_overrides_voxel_size(tmp_path) -> None:
    """External image stacks should accept explicit physical resolution."""

    values = np.arange(16, dtype=np.uint8).reshape(4, 4)
    data = VolumeData(
        values=values,
        voxel_size=(4.0, 5.0),
        units={"length": "um"},
        metadata={"sample": "synthetic_2d"},
    )

    path = save_volume(data, tmp_path / "slice.tiff")
    loaded = load_volume_data(path)

    assert np.array_equal(loaded.values, values)
    assert loaded.voxel_size == (4.0, 5.0)
    assert loaded.units == {"length": "um"}
    assert loaded.metadata == {"sample": "synthetic_2d"}

    (tmp_path / "slice.tiff.json").unlink()
    overridden = load_volume_data(
        path,
        voxel_size=(40.0e-6, 50.0e-6),
        units={"length": "m"},
        metadata={"source": "external_tiff"},
    )

    assert overridden.voxel_size == (40.0e-6, 50.0e-6)
    assert overridden.units == {"length": "m"}
    assert overridden.metadata == {"source": "external_tiff"}


def test_volume_data_validation() -> None:
    """Voxel spacing must match dimensionality and remain physically positive."""

    data = VolumeData(np.zeros((2, 3), dtype=np.uint8), voxel_size=2.0)
    assert data.voxel_size == (2.0, 2.0)
    assert data.ndim == 2
    assert data.shape == (2, 3)

    with pytest.raises(ValueError, match="length 2"):
        VolumeData(np.zeros((2, 3), dtype=np.uint8), voxel_size=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="positive"):
        VolumeData(np.zeros((2, 3, 4), dtype=np.uint8), voxel_size=(1.0, -1.0, 1.0))


def test_raw_volume_requires_shape_when_sidecar_is_missing(tmp_path) -> None:
    """Raw binary files need explicit shape metadata without the JSON sidecar."""

    path = tmp_path / "bare.raw"
    np.arange(6, dtype=np.uint8).tofile(path)

    with pytest.raises(ValueError, match="shape is required"):
        load_volume(path)

    loaded = load_volume(path, shape=(1, 2, 3), dtype=np.uint8)
    assert np.array_equal(loaded, np.arange(6, dtype=np.uint8).reshape(1, 2, 3))

    with pytest.raises(ValueError, match="shape requires"):
        load_volume(path, shape=(2, 2, 2), dtype=np.uint8)


def test_surface_mesh_from_binary_volume_and_obj_stl_roundtrip(tmp_path) -> None:
    """Binary 3D cases should export to CAD/printing-oriented surface formats."""

    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[1:4, 1:4, 1:4] = True

    mesh = surface_mesh_from_binary_volume(volume, voxel_size=(1.0, 2.0, 3.0))
    assert isinstance(mesh, SurfaceMesh)
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.metadata["source_kind"] == "binary_volume_marching_cubes"

    integer_volume = volume.astype(np.uint8)
    integer_mesh = surface_mesh_from_binary_volume(integer_volume)
    assert integer_mesh.faces.shape[1] == 3

    obj_path = save_surface_mesh(mesh, tmp_path / "case.obj")
    stl_path = save_surface_mesh(mesh, tmp_path / "case.stl")
    obj_loaded = load_surface_mesh(obj_path)
    stl_loaded = load_surface_mesh(stl_path)

    assert obj_loaded.vertices.shape[1] == 3
    assert obj_loaded.faces.shape[1] == 3
    assert stl_loaded.vertices.shape[1] == 3
    assert stl_loaded.faces.shape[1] == 3


def test_volume_bundle_exports_voxels_and_surfaces(tmp_path) -> None:
    """One call should write the branch-requested synthetic-case artifacts."""

    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[1:4, 1:4, 1:4] = True

    written = save_volume_bundle(
        volume,
        tmp_path,
        stem="toy",
        formats=("raw", "npy", "h5", "nc", "stl", "obj"),
        metadata={"kind": "macro_micro_test"},
    )

    assert set(written) == {"raw", "npy", "h5", "nc", "stl", "obj"}
    assert all(path.exists() for path in written.values())
    assert (tmp_path / "toy.raw.json").exists()
    assert np.array_equal(load_volume(written["npy"]), volume)
    assert np.array_equal(load_volume(written["nc"]).astype(bool), volume)
    assert load_surface_mesh(written["obj"]).faces.size > 0


def test_surface_mesh_validation() -> None:
    """Surface export should reject non-interface volumes and bad connectivity."""

    with pytest.raises(ValueError, match="both void and solid"):
        surface_mesh_from_binary_volume(np.ones((3, 3, 3), dtype=bool))

    grayscale = np.zeros((3, 3, 3), dtype=np.uint8)
    grayscale[1, 1, 1] = 128
    with pytest.raises(ValueError, match="binary volume"):
        surface_mesh_from_binary_volume(grayscale)

    with pytest.raises(ValueError, match="outside vertices"):
        SurfaceMesh(vertices=np.zeros((1, 3)), faces=np.array([[0, 1, 2]]))

    with pytest.raises(ValueError, match="vertices must"):
        SurfaceMesh(vertices=np.zeros((3, 2)), faces=np.array([[0, 1, 2]]))

    with pytest.raises(ValueError, match="faces must"):
        SurfaceMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1]]))

    with pytest.raises(ValueError, match="finite"):
        SurfaceMesh(vertices=np.array([[0.0, 0.0, np.nan]]), faces=np.zeros((0, 3)))

    with pytest.raises(ValueError, match="nonnegative"):
        SurfaceMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, -1, 2]]))

    with pytest.raises(ValueError, match="Unsupported"):
        save_surface_mesh(
            SurfaceMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]])),
            "bad.ply",
        )


def test_volume_private_validation_branches(tmp_path) -> None:
    """Exercise format, JSON, and voxel-size validation branches."""

    assert json.loads(vol._json_dumps({"arr": np.array([1]), "flag": np.bool_(True)})) == {
        "arr": [1],
        "flag": True,
    }
    assert json.loads(vol._json_dumps({"i": np.int64(2), "x": np.float32(3.5)})) == {
        "i": 2,
        "x": 3.5,
    }
    with pytest.raises(TypeError, match="not JSON serializable"):
        vol._json_dumps({"bad": object()})

    with pytest.raises(ValueError, match="could not be inferred"):
        vol._normalize_format(tmp_path / "nosuffix", None)
    with pytest.raises(ValueError, match="2D or 3D"):
        vol._normalize_voxel_size(1.0, ndim=4)
    with pytest.raises(ValueError, match="length 3"):
        vol._normalize_voxel_size((1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        vol._normalize_voxel_size((1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="binary volume"):
        vol._binary_volume_mask(np.full((3, 3, 3), "void", dtype=object))

    assert vol._metadata_from_json(None) == {}
    assert vol._metadata_from_json(b'{"a": 1}') == {"a": 1}
    assert vol._metadata_from_json(np.asarray('{"a": 2}')) == {"a": 2}
    assert vol._metadata_from_json("") == {}
    with pytest.raises(ValueError, match="scalar JSON"):
        vol._metadata_from_json(np.asarray(["{}"]))
    with pytest.raises(ValueError, match="JSON object"):
        vol._metadata_from_json("[1, 2]")
    assert vol._load_volume_metadata(tmp_path / "ignored.mesh", "obj") == {}

    empty_h5 = tmp_path / "empty.h5"
    with h5py.File(empty_h5, "w"):
        pass
    with pytest.raises(KeyError, match="not found"):
        vol._load_volume_metadata(empty_h5, "h5", hdf5_dataset="volume")

    empty_nc = tmp_path / "empty.nc"
    with netcdf_file(empty_nc, "w"):
        pass
    with pytest.raises(KeyError, match="not found"):
        vol._load_volume_metadata(empty_nc, "nc", netcdf_variable="volume")

    with pytest.raises(ValueError, match="Use load_surface_mesh"):
        load_volume(tmp_path / "mesh.stl", file_format="stl")


def test_volume_metadata_dtype_restore_validation_branches() -> None:
    """Semantic dtype restoration should fail before lossy or invalid casts."""

    explicit = np.asarray([1], dtype=np.int16)
    assert (
        vol._restore_metadata_dtype(
            explicit,
            {"dtype": "uint8"},
            dtype_was_explicit=True,
        )
        is explicit
    )

    float_values = vol._restore_metadata_dtype(
        np.asarray([1], dtype=np.int16),
        {"dtype": "float32"},
        dtype_was_explicit=False,
    )
    assert float_values.dtype == np.dtype("float32")

    with pytest.raises(ValueError, match="cannot be restored"):
        vol._restore_metadata_dtype(
            np.asarray(["1"]),
            {"dtype": "uint8"},
            dtype_was_explicit=False,
        )

    for invalid in (
        np.asarray([np.nan], dtype=float),
        np.asarray([1.5], dtype=float),
        np.asarray([300], dtype=np.int16),
    ):
        with pytest.raises(ValueError, match="cannot be restored"):
            vol._restore_metadata_dtype(
                invalid,
                {"dtype": "uint8"},
                dtype_was_explicit=False,
            )

    with pytest.raises(ValueError, match="cannot be restored"):
        vol._restore_metadata_dtype(
            np.asarray([1.0 + 0.0j]),
            {"dtype": "float64"},
            dtype_was_explicit=False,
        )


def test_surface_mesh_loader_and_writer_edge_branches(tmp_path) -> None:
    """Cover ndarray surface export, unsupported mesh formats, and binary STL."""

    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[1:4, 1:4, 1:4] = True
    obj_path = save_surface_mesh(volume, tmp_path / "from_array.obj")
    assert load_surface_mesh(obj_path).faces.size > 0

    mesh = SurfaceMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="only be saved"):
        save_surface_mesh(mesh, tmp_path / "bad.h5")
    with pytest.raises(ValueError, match="only be loaded"):
        load_surface_mesh(tmp_path / "bad.h5")

    assert np.allclose(vol._face_normal(np.zeros((3, 3))), np.zeros(3))

    binary_stl = tmp_path / "binary.stl"
    header = b"binary stl".ljust(80, b" ")
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    binary_stl.write_bytes(header + struct.pack("<I", 1) + triangle)
    loaded = load_surface_mesh(binary_stl)
    assert loaded.vertices.shape == (3, 3)
    assert loaded.faces.shape == (1, 3)

    with pytest.raises(ValueError, match="too small"):
        vol._read_binary_stl(b"short")
    with pytest.raises(ValueError, match="truncated"):
        vol._read_binary_stl(header + struct.pack("<I", 1))
