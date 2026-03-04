from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
    user_notes: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the provenance record to a JSON-friendly mapping.

        Returns
        -------
        dict[str, Any]
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
    def from_metadata(cls, data: dict[str, Any]) -> "Provenance":
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

        return cls(**data)
