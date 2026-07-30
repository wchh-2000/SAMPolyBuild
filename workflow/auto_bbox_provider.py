"""Select tile-local Prompt bboxes from canonical Auto TileResult polygons."""

import math
from typing import Any


def _normalize_bbox(raw_bbox: Any) -> list[float] | None:
    """Convert one candidate bbox to finite XYXY floats when possible."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox


def _bbox_from_points(points: Any) -> list[float] | None:
    """Derive one XYXY bbox from polygon points when bbox_tile is unavailable."""
    if not isinstance(points, list) or len(points) < 3:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys or not all(math.isfinite(value) for value in [*xs, *ys]):
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _candidate_bbox(polygon: Any) -> list[float] | None:
    """Prefer bbox_tile and fall back to one derived from points_tile."""
    bbox = _normalize_bbox(getattr(polygon, "bbox_tile", None))
    if bbox is not None:
        return bbox
    return _bbox_from_points(getattr(polygon, "points_tile", None))


def _clip_bbox(bbox: list[float], tile_width: int, tile_height: int) -> list[float] | None:
    """Clip one bbox to tile bounds and reject empty results."""
    x1, y1, x2, y2 = bbox
    clipped = [
        min(max(x1, 0.0), float(tile_width)),
        min(max(y1, 0.0), float(tile_height)),
        min(max(x2, 0.0), float(tile_width)),
        min(max(y2, 0.0), float(tile_height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _distance_to_bbox(click: tuple[int, int], bbox: list[float]) -> float:
    """Return Euclidean distance from one point to the nearest bbox edge."""
    px, py = float(click[0]), float(click[1])
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _bbox_area(bbox: list[float]) -> float:
    """Return bbox area for deterministic tie-breaking."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def select_bbox_for_click(
    polygons: list[Any],
    click: tuple[int, int],
    nearest_threshold: float,
    tile_width: int,
    tile_height: int,
    min_score: float | None = None,
) -> dict[str, Any] | None:
    """Select one tile-local bbox from active Auto polygons for a Prompt click."""
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_width and tile_height must be positive.")
    if nearest_threshold < 0:
        raise ValueError("nearest_threshold must be non-negative.")
    if not isinstance(click, (list, tuple)) or len(click) != 2:
        raise ValueError("click must be an [x, y] pair.")

    px, py = int(click[0]), int(click[1])
    candidates = []
    for polygon in polygons:
        if getattr(polygon, "status", None) != "active":
            continue
        score = getattr(polygon, "score", None)
        score_value = float(score) if score is not None else 0.0
        if min_score is not None and score_value < float(min_score):
            continue
        bbox = _candidate_bbox(polygon)
        if bbox is None:
            continue
        clipped_bbox = _clip_bbox(bbox, tile_width, tile_height)
        if clipped_bbox is None:
            continue
        contains = (
            clipped_bbox[0] <= px <= clipped_bbox[2]
            and clipped_bbox[1] <= py <= clipped_bbox[3]
        )
        candidates.append(
            {
                "bbox": clipped_bbox,
                "polygon_id": getattr(polygon, "id", None),
                "score": score_value,
                "distance": _distance_to_bbox((px, py), clipped_bbox),
                "contains": contains,
                "area": _bbox_area(clipped_bbox),
            }
        )

    if not candidates:
        return None

    contains_candidates = [item for item in candidates if item["contains"]]
    if contains_candidates:
        selected = max(
            contains_candidates,
            key=lambda item: (item["score"], -item["area"], item["polygon_id"] or ""),
        )
        return {
            "bbox": selected["bbox"],
            "source": "auto_contains",
            "polygon_id": selected["polygon_id"],
            "score": selected["score"],
            "distance": 0.0,
        }

    nearest = min(
        candidates,
        key=lambda item: (item["distance"], -item["score"], item["area"], item["polygon_id"] or ""),
    )
    if nearest["distance"] > float(nearest_threshold):
        return None
    return {
        "bbox": nearest["bbox"],
        "source": "auto_nearest",
        "polygon_id": nearest["polygon_id"],
        "score": nearest["score"],
        "distance": nearest["distance"],
    }
