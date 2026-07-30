from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


GmshModelBuilder = Callable[[Any], None]


@dataclass(slots=True)
class DolfinxGmshMesh:
    """DOLFINx mesh and physical tags converted from an in-memory Gmsh model."""

    mesh: Any
    cell_tags: Any | None
    facet_tags: Any | None
    ridge_tags: Any | None
    peak_tags: Any | None
    physical_groups: dict[str, Any]


def require_gmsh() -> Any:
    """Import and return the optional Gmsh Python module."""

    try:
        import gmsh  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError(
            "Gmsh mesh generation requires the Gmsh executable and Python "
            "bindings. In Pixi, install both 'gmsh' and 'python-gmsh'."
        ) from exc
    return gmsh


def add_physical_group(
    gmsh: Any,
    dimension: int,
    entities: Sequence[int],
    *,
    tag: int,
    name: str,
) -> int:
    """Add a nonempty named physical group and return its integer tag."""

    entity_tags = [int(entity) for entity in entities]
    if dimension < 0:
        raise ValueError("dimension must be nonnegative")
    if not entity_tags:
        raise ValueError("entities must not be empty")
    if tag <= 0:
        raise ValueError("tag must be positive")
    if not name.strip():
        raise ValueError("name must not be empty")
    created = int(gmsh.model.addPhysicalGroup(dimension, entity_tags, tag))
    gmsh.model.setPhysicalName(dimension, created, name)
    return created


def axis_aligned_boundary_entities(
    gmsh: Any,
    geometric_dimension: int,
    *,
    bounds: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 1.0e-10,
) -> tuple[dict[str, list[int]], list[int]]:
    """Classify axis-aligned facets by their Gmsh bounding boxes.

    Curved and interior facets are returned in the second tuple item.
    """

    if geometric_dimension not in {2, 3}:
        raise ValueError("geometric_dimension must be either 2 or 3")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    resolved_bounds = tuple(bounds or ((0.0, 1.0),) * geometric_dimension)
    if len(resolved_bounds) != geometric_dimension:
        raise ValueError("bounds must contain one (minimum, maximum) pair per axis")
    for lower, upper in resolved_bounds:
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("each bounds pair must be finite and strictly increasing")

    axis_names = ("x", "y", "z")[:geometric_dimension]
    groups: dict[str, list[int]] = {
        f"{axis_name}_{side}": [] for axis_name in axis_names for side in ("min", "max")
    }
    unclassified: list[int] = []
    facet_dimension = geometric_dimension - 1
    for _, entity in gmsh.model.getEntities(facet_dimension):
        bounding_box = gmsh.model.getBoundingBox(facet_dimension, entity)
        lower_coordinates = bounding_box[:3]
        upper_coordinates = bounding_box[3:]
        matched_name: str | None = None
        for axis, axis_name in enumerate(axis_names):
            lower, upper = resolved_bounds[axis]
            if (
                abs(lower_coordinates[axis] - lower) <= tolerance
                and abs(upper_coordinates[axis] - lower) <= tolerance
            ):
                matched_name = f"{axis_name}_min"
                break
            if (
                abs(lower_coordinates[axis] - upper) <= tolerance
                and abs(upper_coordinates[axis] - upper) <= tolerance
            ):
                matched_name = f"{axis_name}_max"
                break
        if matched_name is None:
            unclassified.append(int(entity))
        else:
            groups[matched_name].append(int(entity))
    return groups, unclassified


def configure_uniform_mesh_size(
    gmsh: Any,
    target_size: float,
    *,
    element_order: int = 1,
    size_from_curvature: bool = True,
) -> None:
    """Configure a uniform target size for subsequent Gmsh generation."""

    if target_size <= 0.0 or not np.isfinite(target_size):
        raise ValueError("target_size must be positive and finite")
    if element_order < 1:
        raise ValueError("element_order must be at least 1")
    gmsh.option.setNumber("Mesh.MeshSizeMin", float(target_size))
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_size))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(size_from_curvature))
    gmsh.option.setNumber("Mesh.ElementOrder", int(element_order))


def generate_dolfinx_gmsh_mesh(
    build_model: GmshModelBuilder,
    *,
    name: str,
    geometric_dimension: int,
    comm: Any | None = None,
    model_rank: int = 0,
    terminal_output: bool = False,
) -> DolfinxGmshMesh:
    """Build a Gmsh model on one MPI rank and convert it to DOLFINx.

    ``build_model`` receives Gmsh after a fresh model has been created. It must
    synchronize the CAD kernel, define physical groups, and generate the mesh.
    """

    if not callable(build_model):
        raise TypeError("build_model must be callable")
    if not name.strip():
        raise ValueError("name must not be empty")
    if geometric_dimension not in {2, 3}:
        raise ValueError("geometric_dimension must be either 2 or 3")
    gmsh = require_gmsh()
    try:
        from dolfinx.io import gmsh as dolfinx_gmsh
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError(
            "DOLFINx Gmsh conversion requires dolfinx, mpi4py, and python-gmsh"
        ) from exc

    communicator = MPI.COMM_WORLD if comm is None else comm
    if not 0 <= model_rank < communicator.size:
        raise ValueError("model_rank must identify a rank in comm")
    initialized_here = False
    try:
        if communicator.rank == model_rank:
            if not gmsh.isInitialized():
                gmsh.initialize()
                initialized_here = True
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", int(terminal_output))
            gmsh.model.add(name)
            build_model(gmsh)
        mesh_data = dolfinx_gmsh.model_to_mesh(
            gmsh.model,
            communicator,
            model_rank,
            gdim=geometric_dimension,
        )
    finally:
        if communicator.rank == model_rank:
            if initialized_here:
                gmsh.finalize()
            else:
                gmsh.clear()
    return DolfinxGmshMesh(
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags,
        ridge_tags=getattr(mesh_data, "ridge_tags", getattr(mesh_data, "edge_tags", None)),
        peak_tags=getattr(mesh_data, "peak_tags", getattr(mesh_data, "vertex_tags", None)),
        physical_groups=dict(mesh_data.physical_groups),
    )


__all__ = [
    "DolfinxGmshMesh",
    "GmshModelBuilder",
    "add_physical_group",
    "axis_aligned_boundary_entities",
    "configure_uniform_mesh_size",
    "generate_dolfinx_gmsh_mesh",
    "require_gmsh",
]
