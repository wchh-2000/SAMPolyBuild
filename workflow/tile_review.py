"""Model-free state management and geometry helpers for tile review."""

import math
from dataclasses import dataclass
from pathlib import Path

from .manifest import TileManifestEntry
from .tile_manager import PolygonRecord, TileResult


@dataclass(frozen=True)
class ReviewPaths:
    """Resolved directories used by one review session."""

    workspace: Path
    auto_results: Path
    edited_results: Path


def resolve_review_paths(
    manifest_path: Path,
    auto_results: Path | None = None,
    edited_results: Path | None = None,
) -> ReviewPaths:
    """Resolve default result directories relative to the manifest workspace."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    workspace = manifest_path.parent.parent if manifest_path.parent.name == "manifests" \
        else manifest_path.parent
    auto_path = Path(auto_results).expanduser().resolve() if auto_results else workspace / "auto_results"
    edited_path = Path(edited_results).expanduser().resolve() \
        if edited_results else workspace / "edited_results"
    edited_path.mkdir(parents=True, exist_ok=True)
    return ReviewPaths(
        workspace=workspace.resolve(),
        auto_results=auto_path,
        edited_results=edited_path,
    )


def resolve_tile_image(workspace: Path, image_path: Path) -> Path:
    """Resolve a manifest image path while preventing workspace traversal."""
    workspace = Path(workspace).resolve()
    relative_path = Path(image_path)
    if relative_path.is_absolute():
        raise ValueError(f"Manifest image_path must be relative: {relative_path}")
    resolved = (workspace / relative_path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Manifest image_path escapes workspace: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Tile image not found: {resolved}")
    return resolved


def _validate_result_for_tile(result: TileResult, tile: TileManifestEntry) -> TileResult:
    """Reject stale or misrouted result files before they are edited."""
    if result.tile_id != tile.tile_id:
        raise ValueError(
            f"Result tile_id={result.tile_id!r} does not match manifest tile {tile.tile_id!r}."
        )
    expected_offset = [float(tile.x_offset), float(tile.y_offset)]
    if result.offset != expected_offset:
        raise ValueError(
            f"Result offset={result.offset} does not match manifest offset={expected_offset}."
        )
    if result.image_path is not None and result.image_path != tile.image_path:
        raise ValueError(
            f"Result image_path={result.image_path} does not match manifest path={tile.image_path}."
        )
    if result.image_path is None:
        result.image_path = tile.image_path
    return result


def load_review_result(
    tile: TileManifestEntry,
    auto_results_dir: Path,
    edited_results_dir: Path,
) -> tuple[TileResult, bool]:
    """Load edited first, then auto, otherwise create a valid empty TileResult."""
    edited_path = Path(edited_results_dir) / f"{tile.tile_id}.json"
    auto_path = Path(auto_results_dir) / f"{tile.tile_id}.json"
    if edited_path.is_file():
        return _validate_result_for_tile(TileResult.load(edited_path), tile), True
    if auto_path.is_file():
        return _validate_result_for_tile(TileResult.load(auto_path), tile), False
    return (
        TileResult.create_empty(
            tile_id=tile.tile_id,
            offset=[tile.x_offset, tile.y_offset],
            image_path=tile.image_path,
        ),
        False,
    )


def save_review_result(result: TileResult, edited_results_dir: Path) -> Path:
    """Atomically save one reviewed tile without dropping polygon fields."""
    destination = Path(edited_results_dir) / f"{result.tile_id}.json"
    return result.save(destination)


def set_polygon_status(result: TileResult, polygon_id: str, status: str) -> PolygonRecord:
    """Set one polygon to active or deleted without physically removing it."""
    if status not in ("active", "deleted"):
        raise ValueError("Review status must be active or deleted.")
    for polygon in result.polygons:
        if polygon.id == polygon_id:
            polygon.status = status
            return polygon
    raise KeyError(f"Polygon not found in {result.tile_id}: {polygon_id}")


def polygon_status_counts(result: TileResult) -> tuple[int, int]:
    """Return active and deleted polygon counts for the review overlay."""
    active = sum(polygon.status == "active" for polygon in result.polygons)
    deleted = sum(polygon.status == "deleted" for polygon in result.polygons)
    return active, deleted


def first_review_index(
    tiles: list[TileManifestEntry],
    edited_results_dir: Path,
    start_tile: str | None = None,
) -> int:
    """Resolve an explicit tile or the first tile without an edited result."""
    if not tiles:
        raise ValueError("Manifest contains no tiles to review.")
    if start_tile:
        for index, tile in enumerate(tiles):
            if tile.tile_id == start_tile:
                return index
        raise ValueError(f"--start-tile was not found in manifest: {start_tile}")
    edited_results_dir = Path(edited_results_dir)
    for index, tile in enumerate(tiles):
        if not (edited_results_dir / f"{tile.tile_id}.json").is_file():
            return index
    return 0


def _point_segment_distance(point, start, end) -> float:
    """Return Euclidean distance from a point to a finite line segment."""
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - x1, py - y1)
    projection = ((px - x1) * dx + (py - y1) * dy) / length_squared
    projection = min(max(projection, 0.0), 1.0)
    return math.hypot(px - (x1 + projection * dx), py - (y1 + projection * dy))


def polygon_boundary_distance(point, polygon: PolygonRecord) -> float:
    """Return minimum distance from a click to a closed polygon boundary."""
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(polygon.points_tile, polygon.points_tile[1:])
    )


def _contains_point(point, polygon: PolygonRecord) -> bool:
    """Test polygon containment, treating boundary clicks as contained."""
    if polygon_boundary_distance(point, polygon) <= 1e-9:
        return True
    x, y = float(point[0]), float(point[1])
    inside = False
    for start, end in zip(polygon.points_tile, polygon.points_tile[1:]):
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _polygon_area(polygon: PolygonRecord) -> float:
    """Return absolute shoelace area for deterministic overlap selection."""
    doubled_area = sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon.points_tile, polygon.points_tile[1:])
    )
    return abs(doubled_area) * 0.5


def select_polygon_at_point(
    result: TileResult,
    point: tuple[float, float],
    max_distance: float = 10.0,
) -> PolygonRecord | None:
    """Prefer containing polygons, then the nearest boundary within threshold."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    containing = [polygon for polygon in result.polygons if _contains_point(point, polygon)]
    if containing:
        return min(containing, key=lambda polygon: (_polygon_area(polygon), polygon.id))
    if not result.polygons:
        return None
    nearest = min(
        result.polygons,
        key=lambda polygon: (polygon_boundary_distance(point, polygon), polygon.id),
    )
    return nearest if polygon_boundary_distance(point, nearest) <= max_distance else None
