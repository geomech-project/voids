from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from voids._typing import JsonObject, JsonValue


@dataclass(slots=True)
class Provenance:
    """Store metadata describing the origin of a network.

    Attributes
    ----------
    source_kind :
        Broad category of origin, such as ``"porespy"`` or
        ``"synthetic_mesh"``.
    source_version :
        Version string of the generating package or workflow, when known.
    extraction_method :
        Short description of the extraction or construction procedure.
    segmentation_notes :
        Free-form notes about segmentation or preprocessing assumptions.
    voxel_size_original :
        Original voxel spacing before any physical-unit conversion.
    image_hash, preprocessing_hash :
        Optional hashes identifying input images or preprocessing recipes.
    random_seed :
        Seed used by any stochastic preprocessing or synthetic generator.
    created_at :
        UTC timestamp encoded as an ISO 8601 string.
    user_notes :
        Additional JSON-serializable metadata.
    """

    source_kind: str = "custom"
    source_version: str | None = None
    extraction_method: str | None = None
    segmentation_notes: str | None = None
    voxel_size_original: float | tuple[float, float, float] | None = None
    image_hash: str | None = None
    preprocessing_hash: str | None = None
    random_seed: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_notes: JsonObject = field(default_factory=dict)

    def to_metadata(self) -> JsonObject:
        """Serialize the provenance record to a JSON-friendly mapping.

        Returns
        -------
        JsonObject
            Dictionary suitable for storage in HDF5 attributes or JSON payloads.
        """

        return {
            "source_kind": self.source_kind,
            "source_version": self.source_version,
            "extraction_method": self.extraction_method,
            "segmentation_notes": self.segmentation_notes,
            "voxel_size_original": self.voxel_size_original,
            "image_hash": self.image_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_metadata(cls, data: JsonObject) -> "Provenance":
        """Construct a provenance record from serialized metadata.

        Parameters
        ----------
        data :
            Metadata dictionary previously produced by :meth:`to_metadata`.

        Returns
        -------
        Provenance
            Reconstructed provenance record.
        """

        return cls(
            source_kind=str(data.get("source_kind", "custom")),
            source_version=_optional_str(data.get("source_version")),
            extraction_method=_optional_str(data.get("extraction_method")),
            segmentation_notes=_optional_str(data.get("segmentation_notes")),
            voxel_size_original=_voxel_size_original(data.get("voxel_size_original")),
            image_hash=_optional_str(data.get("image_hash")),
            preprocessing_hash=_optional_str(data.get("preprocessing_hash")),
            random_seed=_optional_int(data.get("random_seed")),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            user_notes=_json_object(data.get("user_notes")),
        )


def _optional_str(value: JsonValue) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: JsonValue) -> int | None:
    return None if value is None else int(value) if isinstance(value, (int, float, str)) else None


def _voxel_size_original(value: JsonValue) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return (
            _float_value(value[0]),
            _float_value(value[1]),
            _float_value(value[2]),
        )
    return None


def _json_object(value: JsonValue) -> JsonObject:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_value(value: JsonValue) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric metadata value, got {type(value).__name__}")
