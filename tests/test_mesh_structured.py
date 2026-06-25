from __future__ import annotations

import numpy as np
import pytest

import voids.mesh.structured as structured_mod
from voids.image.porosity import PermeabilityMap, PorosityMap
from voids.mesh import (
    mesh_format_extension,
    structured_map_mesh,
    write_structured_map_mesh,
    write_structured_map_meshes,
)


def test_structured_map_mesh_2d_uses_map_cell_order_and_coordinates() -> None:
    """A 2D porosity map becomes a quad mesh with C-order cell data."""

    porosity = PorosityMap(
        values=np.array([[0.10, 0.20], [0.30, 0.40]]),
        cell_size=(2.0, 3.0),
        origin=(10.0, 20.0),
    )
    permeability = PermeabilityMap(
        values=np.array([[1.0e-15, 2.0e-15], [3.0e-15, 4.0e-15]]),
        cell_size=porosity.cell_size,
        origin=porosity.origin,
    )

    mesh = structured_map_mesh(porosity, permeability_map=permeability)

    assert mesh.cell_type == "quad"
    assert mesh.cell_count == 4
    assert mesh.points.shape == (9, 3)
    assert np.allclose(mesh.points[0], [10.0, 20.0, 0.0])
    assert np.allclose(mesh.points[-1], [14.0, 26.0, 0.0])
    assert np.array_equal(mesh.cells[0][1][0], [0, 3, 4, 1])
    assert np.allclose(mesh.cell_data["porosity"][0], [0.10, 0.20, 0.30, 0.40])
    assert np.allclose(
        mesh.cell_data["permeability"][0],
        [1.0e-15, 2.0e-15, 3.0e-15, 4.0e-15],
    )
    assert np.array_equal(mesh.cell_data["cell_index"][0], [0, 1, 2, 3])


def test_structured_map_mesh_3d_uses_hexahedra() -> None:
    """A 3D map becomes a hexahedral mesh on the same regular grid."""

    porosity = PorosityMap(
        values=np.ones((1, 1, 2)),
        cell_size=(1.0, 2.0, 3.0),
        origin=(0.5, 1.5, 2.5),
    )

    mesh = structured_map_mesh(porosity)

    assert mesh.cell_type == "hexahedron"
    assert mesh.cell_count == 2
    assert mesh.points.shape == (12, 3)
    assert np.allclose(mesh.points[0], [0.5, 1.5, 2.5])
    assert np.allclose(mesh.points[-1], [1.5, 3.5, 8.5])
    assert mesh.cells[0][1].shape == (2, 8)


def test_structured_map_mesh_2d_can_split_quads_to_triangles() -> None:
    """A 2D map can be exported as triangles with duplicated parent data."""

    porosity = PorosityMap(values=np.array([[0.25]]), cell_size=(2.0, 3.0))
    permeability = PermeabilityMap(
        values=np.array([[1.0e-15]]),
        cell_size=porosity.cell_size,
    )

    mesh = structured_map_mesh(
        porosity,
        permeability_map=permeability,
        element_type="triangle",
    )

    assert mesh.cell_type == "triangle"
    assert mesh.cell_count == 2
    assert mesh.points.shape == (4, 3)
    assert np.array_equal(mesh.cells[0][1], [[0, 2, 3], [0, 3, 1]])
    assert np.allclose(mesh.cell_data["porosity"][0], [0.25, 0.25])
    assert np.allclose(mesh.cell_data["permeability"][0], [1.0e-15, 1.0e-15])
    assert np.array_equal(mesh.cell_data["cell_index"][0], [0, 0])


def test_structured_map_mesh_3d_can_split_hexahedra_to_tetrahedra() -> None:
    """A 3D map can be exported as tetrahedra with duplicated parent data."""

    porosity = PorosityMap(values=np.array([[[0.5]]]), cell_size=(1.0, 2.0, 3.0))

    mesh = structured_map_mesh(porosity, element_type="tetrahedron")

    assert mesh.cell_type == "tetra"
    assert mesh.cell_count == 6
    assert mesh.points.shape == (8, 3)
    assert np.array_equal(
        mesh.cells[0][1],
        [
            [0, 4, 6, 7],
            [0, 6, 2, 7],
            [0, 2, 3, 7],
            [0, 3, 1, 7],
            [0, 1, 5, 7],
            [0, 5, 4, 7],
        ],
    )
    assert np.allclose(mesh.cell_data["porosity"][0], np.full(6, 0.5))
    assert np.array_equal(mesh.cell_data["cell_index"][0], np.zeros(6, dtype=np.int64))


def test_structured_map_mesh_rejects_incompatible_element_types() -> None:
    """Simplex and tensor-product element choices must match map dimensionality."""

    porosity_2d = PorosityMap(values=np.ones((1, 1)))
    porosity_3d = PorosityMap(values=np.ones((1, 1, 1)))

    with pytest.raises(ValueError, match="2D.*quad.*triangle"):
        structured_map_mesh(porosity_2d, element_type="tetra")
    with pytest.raises(ValueError, match="3D.*hexahedron.*tetra"):
        structured_map_mesh(porosity_3d, element_type="triangle")
    with pytest.raises(ValueError, match="element_type must be"):
        structured_map_mesh(porosity_2d, element_type="polygon")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quad_cells"):
        structured_mod._split_quads_to_triangles(np.ones((1, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="hexahedron_cells"):
        structured_mod._split_hexahedra_to_tetrahedra(np.ones((1, 7), dtype=np.int64))


def test_structured_map_mesh_rejects_mismatched_or_nonfinite_fields() -> None:
    """Grid metadata and finite cell data are part of the export contract."""

    porosity = PorosityMap(values=np.array([[0.5, 0.6]]), cell_size=(1.0, 1.0))
    bad_grid = PermeabilityMap(values=np.array([[1.0e-15]]), cell_size=(1.0, 1.0))
    with pytest.raises(ValueError, match="same shape"):
        structured_map_mesh(porosity, permeability_map=bad_grid)

    bad_cell_size = PermeabilityMap(
        values=np.array([[1.0e-15, 2.0e-15]]),
        cell_size=(2.0, 1.0),
    )
    with pytest.raises(ValueError, match="same cell_size"):
        structured_map_mesh(porosity, permeability_map=bad_cell_size)

    bad_origin = PermeabilityMap(
        values=np.array([[1.0e-15, 2.0e-15]]),
        cell_size=porosity.cell_size,
        origin=(1.0, 0.0),
    )
    with pytest.raises(ValueError, match="same origin"):
        structured_map_mesh(porosity, permeability_map=bad_origin)

    permeability = PermeabilityMap(values=np.array([[np.inf, 1.0e-15]]))
    with pytest.raises(ValueError, match="finite"):
        structured_map_mesh(porosity, permeability_map=permeability)

    with pytest.raises(ValueError, match="shape"):
        structured_map_mesh(porosity, extra_cell_data={"bad": np.array([1.0])})
    with pytest.raises(ValueError, match="numeric"):
        structured_map_mesh(porosity, extra_cell_data={"bad": np.array([["x", "y"]])})
    with pytest.raises(ValueError, match="non-empty"):
        structured_map_mesh(porosity, extra_cell_data={"": np.ones(porosity.shape)})
    with pytest.raises(ValueError, match="Duplicate"):
        structured_map_mesh(porosity, extra_cell_data={"porosity": np.ones(porosity.shape)})

    mesh = structured_map_mesh(
        porosity,
        permeability_map=permeability,
        require_finite_cell_data=False,
    )
    assert np.isposinf(mesh.cell_data["permeability"][0][0])


def test_private_grid_validation_defensive_branches() -> None:
    """Exercise defensive branches normally guarded by map dataclasses."""

    class DummyMap:
        def __init__(
            self,
            *,
            ndim: int,
            cell_size: float | tuple[float, ...],
            origin: tuple[float, ...] | None,
        ) -> None:
            self.ndim = ndim
            self.cell_size = cell_size
            self.origin = origin

    with pytest.raises(ValueError, match="only 2D or 3D"):
        structured_mod._validate_map_grid(DummyMap(ndim=1, cell_size=(1.0,), origin=(0.0,)))
    with pytest.raises(ValueError, match="cell_size length"):
        structured_mod._validate_map_grid(DummyMap(ndim=2, cell_size=(1.0,), origin=(0.0, 0.0)))
    with pytest.raises(ValueError, match="origin length"):
        structured_mod._validate_map_grid(DummyMap(ndim=2, cell_size=(1.0, 1.0), origin=(0.0,)))

    assert structured_mod._map_cell_size(DummyMap(ndim=2, cell_size=2.0, origin=None)) == (
        2.0,
        2.0,
    )
    assert structured_mod._map_origin(DummyMap(ndim=2, cell_size=(1.0, 1.0), origin=None)) == (
        0.0,
        0.0,
    )

    with pytest.raises(ValueError, match="only 2D or 3D"):
        structured_mod._structured_points_and_cells(
            shape=(1,),
            cell_size=(1.0,),
            origin=(0.0,),
        )
    with pytest.raises(ValueError, match="flattened cell data length"):
        structured_mod._expand_cell_data_to_mesh_cells(
            np.ones(2),
            base_cell_count=1,
            mesh_cell_count=1,
        )
    with pytest.raises(RuntimeError, match="unsupported structured map mesh cell expansion"):
        structured_mod._expand_cell_data_to_mesh_cells(
            np.ones(1),
            base_cell_count=1,
            mesh_cell_count=3,
        )


def test_mesh_format_extension_and_invalid_format(tmp_path) -> None:
    """Format labels map to the filename extensions used by meshio."""

    assert mesh_format_extension("gmsh") == ".msh"
    assert mesh_format_extension("vtk") == ".vtk"
    assert mesh_format_extension("vtu") == ".vtu"
    assert mesh_format_extension("netgen") == ".vol"

    with pytest.raises(ValueError, match="Unsupported"):
        mesh_format_extension("unknown")

    with pytest.raises(ValueError, match="non-empty"):
        write_structured_map_meshes(
            PorosityMap(values=np.array([[0.25]])),
            tmp_path,
            stem=" ",
        )


def test_write_structured_map_mesh_roundtrips_vtu_cell_data(tmp_path) -> None:
    """VTU export preserves floating cell-data fields through meshio."""

    import meshio

    porosity = PorosityMap(values=np.array([[0.25, 0.75]]), cell_size=(2.0, 3.0))
    permeability = PermeabilityMap(
        values=np.array([[1.0e-15, 2.0e-15]]),
        cell_size=porosity.cell_size,
    )
    path = write_structured_map_mesh(
        porosity,
        tmp_path / "toy.vtu",
        permeability_map=permeability,
    )

    loaded = meshio.read(path)

    assert path.exists()
    assert loaded.cells[0].type == "quad"
    assert np.allclose(loaded.cell_data["porosity"][0], [0.25, 0.75])
    assert np.allclose(loaded.cell_data["permeability"][0], [1.0e-15, 2.0e-15])

    triangle_path = write_structured_map_mesh(
        porosity,
        tmp_path / "toy-triangle.vtu",
        permeability_map=permeability,
        element_type="triangle",
    )
    loaded_triangles = meshio.read(triangle_path)
    assert loaded_triangles.cells[0].type == "triangle"
    assert np.allclose(loaded_triangles.cell_data["porosity"][0], [0.25, 0.25, 0.75, 0.75])


def test_write_structured_map_meshes_writes_requested_formats(tmp_path) -> None:
    """Batch export writes mesh paths for selected format labels."""

    import meshio

    porosity = PorosityMap(values=np.array([[0.25, 0.75]]), cell_size=(2.0, 3.0))

    paths = write_structured_map_meshes(
        porosity,
        tmp_path,
        stem="toy",
        formats=("gmsh", "vtk", "netgen"),
    )

    assert paths["gmsh"] == tmp_path / "toy.msh"
    assert paths["vtk"] == tmp_path / "toy.vtk"
    assert paths["netgen"] == tmp_path / "toy.vol"
    assert paths["gmsh"].exists()
    assert paths["vtk"].exists()
    assert paths["netgen"].exists()
    assert meshio.read(paths["gmsh"]).cells[0].type == "quad"
    assert "porosity" in meshio.read(paths["gmsh"]).cell_data
    assert meshio.read(paths["vtk"]).cells[0].type == "quad"
    assert meshio.read(paths["netgen"]).cells[0].type == "quad"


def test_write_structured_tetra_meshes_writes_requested_formats(tmp_path) -> None:
    """Batch export can write the 3D simplex counterpart of triangle meshes."""

    import meshio

    porosity = PorosityMap(values=np.array([[[0.5]]]), cell_size=(1.0, 1.0, 1.0))

    paths = write_structured_map_meshes(
        porosity,
        tmp_path,
        stem="toy_tetra",
        formats=("gmsh", "vtk", "vtu", "netgen"),
        element_type="tetra",
    )

    for path in paths.values():
        loaded = meshio.read(path)
        assert loaded.cells[0].type == "tetra"
        assert len(loaded.cells[0].data) == 6
