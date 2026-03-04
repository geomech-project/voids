from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleGeometry:
    """Store sample-scale geometry needed for bulk property calculations.

    Attributes
    ----------
    voxel_size :
        Scalar or anisotropic voxel spacing in physical units.
    bulk_shape_voxels :
        Image-domain shape used to derive bulk volume when a direct value is not
        available.
    bulk_volume :
        Total bulk volume in physical units.
    lengths :
        Representative sample lengths by axis.
    cross_sections :
        Cross-sectional areas normal to each flow axis.
    axis_map :
        Optional mapping from custom axis names to canonical identifiers.
    units :
        Unit metadata used for reporting and serialization.
    """

    voxel_size: float | tuple[float, float, float] | None = None
    bulk_shape_voxels: tuple[int, int, int] | None = None
    bulk_volume: float | None = None
    lengths: dict[str, float] = field(default_factory=dict)
    cross_sections: dict[str, float] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=lambda: {"length": "m", "pressure": "Pa"})

    def resolved_bulk_volume(self) -> float:
        """Return the bulk volume, deriving it from voxel metadata when needed.

        Returns
        -------
        float
            Bulk volume of the sample.

        Raises
        ------
        ValueError
            If ``bulk_volume`` is unavailable and the voxel-based metadata is
            incomplete.

        Notes
        -----
        When ``bulk_volume`` is not explicitly stored, the method computes

        ``V_bulk = nx * ny * nz * vx * vy * vz``

        using either an isotropic scalar voxel size or an anisotropic voxel-size
        tuple ``(vx, vy, vz)``.
        """

        if self.bulk_volume is not None:
            return float(self.bulk_volume)
        if self.bulk_shape_voxels is None or self.voxel_size is None:
            raise ValueError("bulk_volume is unavailable and cannot be derived")
        if isinstance(self.voxel_size, tuple):
            vx, vy, vz = self.voxel_size
        else:
            vx = vy = vz = float(self.voxel_size)
        nx, ny, nz = self.bulk_shape_voxels
        return float(nx * ny * nz * vx * vy * vz)

    def length_for_axis(self, axis: str) -> float:
        """Return the representative sample length for one axis.

        Parameters
        ----------
        axis :
            Axis key such as ``"x"``, ``"y"``, or ``"z"``.

        Returns
        -------
        float
            Length associated with the requested axis.

        Raises
        ------
        KeyError
            If no length is registered for the requested axis.
        """

        if axis not in self.lengths:
            raise KeyError(f"Missing sample length for axis '{axis}'")
        return float(self.lengths[axis])

    def area_for_axis(self, axis: str) -> float:
        """Return the sample cross-section normal to one axis.

        Parameters
        ----------
        axis :
            Axis key such as ``"x"``, ``"y"``, or ``"z"``.

        Returns
        -------
        float
            Cross-sectional area used in Darcy-type calculations.

        Raises
        ------
        KeyError
            If no cross-section is registered for the requested axis.
        """

        if axis not in self.cross_sections:
            raise KeyError(f"Missing sample cross-section for axis '{axis}'")
        return float(self.cross_sections[axis])

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the sample geometry to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Mapping suitable for HDF5 or JSON serialization.
        """

        return {
            "voxel_size": self.voxel_size,
            "bulk_shape_voxels": self.bulk_shape_voxels,
            "bulk_volume": self.bulk_volume,
            "lengths": self.lengths,
            "cross_sections": self.cross_sections,
            "axis_map": self.axis_map,
            "units": self.units,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "SampleGeometry":
        """Reconstruct sample geometry from serialized metadata.

        Parameters
        ----------
        data :
            Metadata dictionary previously produced by :meth:`to_metadata`.

        Returns
        -------
        SampleGeometry
            Reconstructed sample-geometry record.
        """

        return cls(
            voxel_size=data.get("voxel_size"),
            bulk_shape_voxels=tuple(data["bulk_shape_voxels"])
            if data.get("bulk_shape_voxels") is not None
            else None,
            bulk_volume=data.get("bulk_volume"),
            lengths={str(k): float(v) for k, v in (data.get("lengths") or {}).items()},
            cross_sections={
                str(k): float(v) for k, v in (data.get("cross_sections") or {}).items()
            },
            axis_map={str(k): str(v) for k, v in (data.get("axis_map") or {}).items()},
            units={str(k): str(v) for k, v in (data.get("units") or {}).items()},
        )
