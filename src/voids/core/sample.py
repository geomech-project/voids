from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from voids._typing import JsonObject, JsonValue


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

    def to_metadata(self) -> JsonObject:
        """Serialize the sample geometry to a JSON-friendly dictionary.

        Returns
        -------
        JsonObject
            Mapping suitable for HDF5 or JSON serialization.
        """

        return {
            "voxel_size": self.voxel_size,
            "bulk_shape_voxels": self.bulk_shape_voxels,
            "bulk_volume": self.bulk_volume,
            "lengths": dict(self.lengths),
            "cross_sections": dict(self.cross_sections),
            "axis_map": dict(self.axis_map),
            "units": dict(self.units),
        }

    @classmethod
    def from_metadata(cls, data: JsonObject) -> "SampleGeometry":
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
            voxel_size=_scalar_or_tuple(data.get("voxel_size")),
            bulk_shape_voxels=_int_tuple(data.get("bulk_shape_voxels")),
            bulk_volume=_optional_float(data.get("bulk_volume")),
            lengths={str(k): _float_value(v) for k, v in _mapping(data.get("lengths")).items()},
            cross_sections={
                str(k): _float_value(v) for k, v in _mapping(data.get("cross_sections")).items()
            },
            axis_map={str(k): str(v) for k, v in _mapping(data.get("axis_map")).items()},
            units={str(k): str(v) for k, v in _mapping(data.get("units")).items()},
        )


def _mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: JsonValue) -> float | None:
    return float(value) if isinstance(value, (int, float, str)) else None


def _float_value(value: JsonValue) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric metadata value, got {type(value).__name__}")


def _scalar_or_tuple(value: JsonValue) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return (
            _float_value(value[0]),
            _float_value(value[1]),
            _float_value(value[2]),
        )
    return None


def _int_tuple(value: JsonValue) -> tuple[int, int, int] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return (
            int(_float_value(value[0])),
            int(_float_value(value[1])),
            int(_float_value(value[2])),
        )
    return None
