"""Versioned manifest data structures and JSON persistence helpers."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON from a :class:`pathlib.Path` with explicit errors."""
    json_path = Path(path).expanduser()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> Path:
    """Atomically write UTF-8 JSON using a temporary sibling file."""
    json_path = Path(path).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary_path.replace(json_path)
    return json_path


def _positive_integer(value: Any, name: str) -> int:
    """Normalize a positive integer while rejecting booleans and fractions."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    """Normalize a non-negative integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


@dataclass
class TileManifestEntry:
    """Location and extent of one tile in source-image pixel coordinates."""

    tile_id: str
    image_path: Path
    x_offset: int
    y_offset: int
    width: int
    height: int
    is_edge_tile: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields at construction so invalid manifests fail early."""
        self.tile_id = str(self.tile_id).strip()
        if not self.tile_id:
            raise ValueError("tile_id must not be empty.")
        self.image_path = Path(self.image_path)
        if self.image_path.as_posix() in ("", ".") or self.image_path.is_absolute():
            raise ValueError("image_path must be a non-empty path relative to the manifest.")
        self.x_offset = _non_negative_integer(self.x_offset, "x_offset")
        self.y_offset = _non_negative_integer(self.y_offset, "y_offset")
        self.width = _positive_integer(self.width, "width")
        self.height = _positive_integer(self.height, "height")
        if not isinstance(self.is_edge_tile, bool):
            raise ValueError("is_edge_tile must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise ValueError("tile metadata must be an object.")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation of this tile."""
        return {
            "tile_id": self.tile_id,
            "image_path": self.image_path.as_posix(),
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "width": self.width,
            "height": self.height,
            "is_edge_tile": self.is_edge_tile,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TileManifestEntry":
        """Build one entry from its canonical JSON object."""
        if not isinstance(payload, dict):
            raise ValueError("Each manifest tile must be a JSON object.")
        required = (
            "tile_id", "image_path", "x_offset", "y_offset",
            "width", "height", "is_edge_tile",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Manifest tile is missing fields: {missing}")
        return cls(
            tile_id=payload["tile_id"],
            image_path=Path(payload["image_path"]),
            x_offset=payload["x_offset"],
            y_offset=payload["y_offset"],
            width=payload["width"],
            height=payload["height"],
            is_edge_tile=payload["is_edge_tile"],
            metadata=payload.get("metadata", {}),
        )


@dataclass
class TileManifest:
    """Canonical, versioned description of an image and all retained tiles."""

    image_name: str
    image_width: int
    image_height: int
    tile_size: int
    overlap: int
    tiles: list[TileManifestEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate image dimensions, tiling parameters, and tile uniqueness."""
        self.image_name = str(self.image_name).strip()
        if not self.image_name:
            raise ValueError("image_name must not be empty.")
        self.image_width = _positive_integer(self.image_width, "image_width")
        self.image_height = _positive_integer(self.image_height, "image_height")
        self.tile_size = _positive_integer(self.tile_size, "tile_size")
        self.overlap = _non_negative_integer(self.overlap, "overlap")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version={self.schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("manifest metadata must be an object.")
        normalized_tiles = [
            tile if isinstance(tile, TileManifestEntry) else TileManifestEntry.from_dict(tile)
            for tile in self.tiles
        ]
        tile_ids = [tile.tile_id for tile in normalized_tiles]
        if len(tile_ids) != len(set(tile_ids)):
            raise ValueError("Manifest tile_id values must be unique.")
        outside_image = [
            tile.tile_id for tile in normalized_tiles
            if tile.x_offset + tile.width > self.image_width
            or tile.y_offset + tile.height > self.image_height
        ]
        if outside_image:
            raise ValueError(f"Manifest tiles exceed source-image bounds: {outside_image}")
        self.tiles = normalized_tiles

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical versioned manifest JSON object."""
        return {
            "schema_version": self.schema_version,
            "image_name": self.image_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TileManifest":
        """Build and validate a manifest from a canonical JSON object."""
        if not isinstance(payload, dict):
            raise ValueError("Manifest JSON root must be an object.")
        required = (
            "image_name", "image_width", "image_height",
            "tile_size", "overlap", "tiles",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Manifest is missing fields: {missing}")
        if not isinstance(payload["tiles"], list):
            raise ValueError("Manifest tiles must be a list.")
        return cls(
            image_name=payload["image_name"],
            image_width=payload["image_width"],
            image_height=payload["image_height"],
            tile_size=payload["tile_size"],
            overlap=payload["overlap"],
            tiles=[TileManifestEntry.from_dict(tile) for tile in payload["tiles"]],
            metadata=payload.get("metadata", {}),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path: Path) -> "TileManifest":
        """Load and validate a manifest from disk."""
        return cls.from_dict(read_json(Path(path)))

    def save(self, path: Path) -> Path:
        """Validate and atomically save this manifest to disk."""
        self.__post_init__()
        return write_json(Path(path), self.to_dict())
