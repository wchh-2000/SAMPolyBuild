"""Filesystem and result data foundations for per-tile processing.

This module deliberately contains no image tiling, model inference, prompting,
or polygon refinement. It only defines stable contracts for later stages.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .manifest import SCHEMA_VERSION, read_json, write_json


POLYGON_STATUSES = {"active", "deleted", "hidden"}
POLYGON_SOURCES = {"auto", "click", "refine"}


def _normalize_point(point: Any, field_name: str) -> list[float]:
    """Validate one polygon point and return a JSON-compatible pair."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{field_name} points must be [x, y] pairs.")
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} points must be numeric.") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field_name} points must be finite.")
    return [x, y]


def _normalize_polygon(points: Any, field_name: str, required: bool) -> list[list[float]]:
    """Validate a polygon ring and close it when the final point is omitted."""
    if points is None or points == []:
        if required:
            raise ValueError(f"{field_name} must contain at least three points.")
        return []
    if not isinstance(points, list):
        raise ValueError(f"{field_name} must be a list of [x, y] points.")
    normalized = [_normalize_point(point, field_name) for point in points]
    unique_vertices = normalized[:-1] if normalized[0] == normalized[-1] else normalized
    distinct_vertices = {tuple(point) for point in unique_vertices}
    if len(distinct_vertices) < 3:
        raise ValueError(f"{field_name} must contain at least three distinct vertices.")
    if normalized[0] != normalized[-1]:
        normalized.append(normalized[0].copy())
    return normalized


@dataclass
class PolygonRecord:
    """Canonical polygon exchanged by auto, click, and refinement stages."""

    id: str
    tile_id: str
    status: str
    source: str
    score: float | None
    points_tile: list[list[float]]
    bbox_tile: list[float] = field(default_factory=list)
    points_global: list[list[float]] = field(default_factory=list)
    click_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate lifecycle fields, score, point rings, and reserved objects."""
        self.id = str(self.id).strip()
        self.tile_id = str(self.tile_id).strip()
        if not self.id or not self.tile_id:
            raise ValueError("Polygon id and tile_id must not be empty.")
        if self.status not in POLYGON_STATUSES:
            raise ValueError(f"Unsupported polygon status: {self.status}")
        if self.source not in POLYGON_SOURCES:
            raise ValueError(f"Unsupported polygon source: {self.source}")
        if self.score is not None:
            self.score = float(self.score)
            if not math.isfinite(self.score):
                raise ValueError("Polygon score must be finite or null.")
        self.points_tile = _normalize_polygon(self.points_tile, "points_tile", required=True)
        if self.bbox_tile:
            if not isinstance(self.bbox_tile, (list, tuple)) or len(self.bbox_tile) != 4:
                raise ValueError("bbox_tile must be [x_min, y_min, x_max, y_max].")
            try:
                self.bbox_tile = [float(value) for value in self.bbox_tile]
            except (TypeError, ValueError) as exc:
                raise ValueError("bbox_tile values must be numeric.") from exc
            if not all(math.isfinite(value) for value in self.bbox_tile):
                raise ValueError("bbox_tile values must be finite.")
            if self.bbox_tile[2] < self.bbox_tile[0] or self.bbox_tile[3] < self.bbox_tile[1]:
                raise ValueError("bbox_tile maximum coordinates must not be below minima.")
        self.points_global = _normalize_polygon(
            self.points_global, "points_global", required=False
        )
        if not isinstance(self.click_history, list) or not all(
            isinstance(item, dict) for item in self.click_history
        ):
            raise ValueError("click_history must be a list of JSON objects.")
        if not isinstance(self.metadata, dict):
            raise ValueError("polygon metadata must be an object.")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation of this polygon."""
        return {
            "id": self.id,
            "tile_id": self.tile_id,
            "status": self.status,
            "source": self.source,
            "score": self.score,
            "points_tile": self.points_tile,
            "bbox_tile": self.bbox_tile,
            "points_global": self.points_global,
            "click_history": self.click_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolygonRecord":
        """Build and validate a polygon from its canonical JSON object."""
        if not isinstance(payload, dict):
            raise ValueError("Each polygon must be a JSON object.")
        required = ("id", "tile_id", "status", "source", "score", "points_tile")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Polygon is missing fields: {missing}")
        return cls(
            id=payload["id"],
            tile_id=payload["tile_id"],
            status=payload["status"],
            source=payload["source"],
            score=payload["score"],
            points_tile=payload["points_tile"],
            bbox_tile=payload.get("bbox_tile", []),
            points_global=payload.get("points_global", []),
            click_history=payload.get("click_history", []),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class TileResult:
    """Versioned collection of all polygon records belonging to one tile."""

    tile_id: str
    offset: list[float]
    polygons: list[PolygonRecord] = field(default_factory=list)
    image_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identity, offset, schema version, and polygon ownership."""
        self.tile_id = str(self.tile_id).strip()
        if not self.tile_id:
            raise ValueError("TileResult tile_id must not be empty.")
        if not isinstance(self.offset, (list, tuple)) or len(self.offset) != 2:
            raise ValueError("TileResult offset must be [x_offset, y_offset].")
        try:
            self.offset = [float(self.offset[0]), float(self.offset[1])]
        except (TypeError, ValueError) as exc:
            raise ValueError("TileResult offset values must be numeric.") from exc
        if not all(math.isfinite(value) for value in self.offset):
            raise ValueError("TileResult offset values must be finite.")
        if any(value < 0 for value in self.offset):
            raise ValueError("TileResult offset values must be non-negative.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported TileResult schema_version={self.schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("TileResult metadata must be an object.")
        if self.image_path is not None:
            self.image_path = Path(self.image_path)
            if self.image_path.is_absolute() or self.image_path.as_posix() in ("", "."):
                raise ValueError("TileResult image_path must be a non-empty relative path.")
        if self.error is not None:
            self.error = str(self.error).strip()
            if not self.error:
                self.error = None
        self.polygons = [
            polygon if isinstance(polygon, PolygonRecord) else PolygonRecord.from_dict(polygon)
            for polygon in self.polygons
        ]
        polygon_ids = [polygon.id for polygon in self.polygons]
        if len(polygon_ids) != len(set(polygon_ids)):
            raise ValueError("Polygon ids must be unique within one TileResult.")
        mismatched = [polygon.id for polygon in self.polygons if polygon.tile_id != self.tile_id]
        if mismatched:
            raise ValueError(f"Polygons reference a different tile_id: {mismatched}")

    @classmethod
    def create_empty(
        cls,
        tile_id: str,
        offset: list[float],
        image_path: Path | None = None,
    ) -> "TileResult":
        """Create a valid tile result before inference or editing has produced polygons."""
        return cls(tile_id=tile_id, offset=offset, polygons=[], image_path=image_path)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical versioned TileResult JSON object."""
        return {
            "schema_version": self.schema_version,
            "tile_id": self.tile_id,
            "offset": self.offset,
            "image_path": self.image_path.as_posix() if self.image_path is not None else None,
            "polygons": [polygon.to_dict() for polygon in self.polygons],
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TileResult":
        """Build and validate a TileResult from its canonical JSON object."""
        if not isinstance(payload, dict):
            raise ValueError("TileResult JSON root must be an object.")
        required = ("tile_id", "offset", "polygons")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"TileResult is missing fields: {missing}")
        if not isinstance(payload["polygons"], list):
            raise ValueError("TileResult polygons must be a list.")
        return cls(
            tile_id=payload["tile_id"],
            offset=payload["offset"],
            polygons=[PolygonRecord.from_dict(item) for item in payload["polygons"]],
            image_path=(
                Path(payload["image_path"])
                if payload.get("image_path") is not None else None
            ),
            error=payload.get("error"),
            metadata=payload.get("metadata", {}),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path: Path) -> "TileResult":
        """Load and validate one tile result from disk."""
        return cls.from_dict(read_json(Path(path)))

    def save(self, path: Path) -> Path:
        """Validate and atomically save one tile result to disk."""
        self.__post_init__()
        return write_json(Path(path), self.to_dict())


@dataclass
class TileManager:
    """Resolve and create the standard directories used by the tile workflow."""

    data_root: Path = Path("data")

    DIRECTORY_NAMES: ClassVar[dict[str, str]] = {
        "manifests": "manifests",
        "auto": "auto_results",
        "edited": "edited_results",
        "final": "final_results",
    }

    def __post_init__(self) -> None:
        """Normalize the configured root without requiring it to exist yet."""
        self.data_root = Path(self.data_root).expanduser()

    def ensure_directories(self) -> dict[str, Path]:
        """Create standard workflow directories and return their resolved mapping."""
        directories = {
            name: self.data_root / directory_name
            for name, directory_name in self.DIRECTORY_NAMES.items()
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def manifest_path(self, image_name: str) -> Path:
        """Return the canonical manifest path for an image identifier."""
        safe_name = self._safe_identifier(image_name, "image_name")
        return self.data_root / self.DIRECTORY_NAMES["manifests"] / f"{safe_name}.json"

    def tile_result_path(self, tile_id: str, stage: str = "auto") -> Path:
        """Return the canonical per-tile JSON path for a workflow stage."""
        if stage not in ("auto", "edited", "final"):
            raise ValueError("stage must be one of: auto, edited, final.")
        safe_tile_id = self._safe_identifier(tile_id, "tile_id")
        return self.data_root / self.DIRECTORY_NAMES[stage] / f"{safe_tile_id}.json"

    @staticmethod
    def _safe_identifier(value: str, name: str) -> str:
        """Reject path traversal while allowing Unicode tile and image names."""
        identifier = str(value).strip()
        if (
            not identifier
            or Path(identifier).name != identifier
            or "/" in identifier
            or "\\" in identifier
            or identifier in (".", "..")
        ):
            raise ValueError(f"{name} must be a file-name-safe identifier.")
        return identifier
