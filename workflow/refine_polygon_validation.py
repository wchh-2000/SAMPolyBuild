"""Geometry quality checks for refined Prompt polygons."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class RefinePolygonValidationConfig:
    """Configurable thresholds for accepting one refined polygon update."""

    max_polygon_points: int = 80
    min_polygon_area: float = 50.0
    max_area_change_ratio: float = 1.5
    min_area_change_ratio: float = 0.3
    reject_invalid_refine: bool = True
    min_bbox_overlap_ratio: float = 0.35


def validate_refined_polygon(
    candidate_points,
    previous_points,
    session_bbox,
    config: RefinePolygonValidationConfig,
) -> dict[str, Any]:
    """Validate one candidate polygon against geometry and session-bbox constraints."""
    result = {
        "accepted": True,
        "reason": "accepted",
        "candidate_area": None,
        "previous_area": None,
        "area_ratio": None,
        "candidate_bbox": None,
        "session_bbox": [float(value) for value in session_bbox] if session_bbox else None,
        "candidate_point_count": 0,
    }
    polygon = _normalize_polygon(candidate_points)
    if polygon is None:
        result["accepted"] = False
        result["reason"] = "invalid_shape"
        return result
    if not np.isfinite(polygon).all():
        result["accepted"] = False
        result["reason"] = "non_finite_coordinates"
        return result

    simplified = _simplify_polygon(polygon)
    result["candidate_point_count"] = int(len(simplified))
    if len(simplified) < 3:
        result["accepted"] = False
        result["reason"] = "too_few_points"
        return result
    if len(simplified) > int(config.max_polygon_points):
        result["accepted"] = False
        result["reason"] = "too_many_points"
        return result

    area = _polygon_area(simplified)
    result["candidate_area"] = float(area)
    result["candidate_bbox"] = _polygon_bbox(simplified)
    if area < float(config.min_polygon_area):
        result["accepted"] = False
        result["reason"] = "area_too_small"
        return result
    if _has_self_intersection(simplified):
        result["accepted"] = False
        result["reason"] = "self_intersection"
        return result
    if session_bbox:
        overlap_ratio = _bbox_overlap_ratio(result["candidate_bbox"], session_bbox)
        result["bbox_overlap_ratio"] = float(overlap_ratio)
        if overlap_ratio < float(config.min_bbox_overlap_ratio):
            result["accepted"] = False
            result["reason"] = "bbox_overlap_too_low"
            return result
    previous = _normalize_polygon(previous_points)
    if previous is not None and len(previous) >= 3:
        previous_area = _polygon_area(_simplify_polygon(previous))
        result["previous_area"] = float(previous_area)
        if previous_area > 0:
            area_ratio = float(area) / float(previous_area)
            result["area_ratio"] = area_ratio
            if area_ratio < float(config.min_area_change_ratio):
                result["accepted"] = False
                result["reason"] = "area_shrank_too_much"
                return result
            if area_ratio > float(config.max_area_change_ratio):
                result["accepted"] = False
                result["reason"] = "area_grew_too_much"
                return result
    result["simplified_points"] = simplified.tolist()
    return result


def _normalize_polygon(points) -> np.ndarray | None:
    """Convert one candidate polygon-like value to an Nx2 float array."""
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        return None
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    return polygon


def _simplify_polygon(polygon: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
    """Simplify one polygon contour with approxPolyDP while keeping closure semantics implicit."""
    contour = polygon.astype(np.float32).reshape((-1, 1, 2))
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(perimeter * float(epsilon_ratio), 1.0)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
    return simplified.astype(np.float64)


def _polygon_area(polygon: np.ndarray) -> float:
    """Return the absolute shoelace area of one polygon."""
    closed = np.vstack([polygon, polygon[0]])
    return float(
        0.5 * abs(
            np.dot(closed[:-1, 0], closed[1:, 1]) -
            np.dot(closed[:-1, 1], closed[1:, 0])
        )
    )


def _polygon_bbox(polygon: np.ndarray) -> list[float]:
    """Return one polygon bounding box as [x1, y1, x2, y2]."""
    return [
        float(np.min(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 0])),
        float(np.max(polygon[:, 1])),
    ]


def _bbox_overlap_ratio(candidate_bbox, session_bbox) -> float:
    """Measure how much of the candidate bbox lies inside the session bbox."""
    x1 = max(float(candidate_bbox[0]), float(session_bbox[0]))
    y1 = max(float(candidate_bbox[1]), float(session_bbox[1]))
    x2 = min(float(candidate_bbox[2]), float(session_bbox[2]))
    y2 = min(float(candidate_bbox[3]), float(session_bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    candidate_area = max(
        (float(candidate_bbox[2]) - float(candidate_bbox[0])) *
        (float(candidate_bbox[3]) - float(candidate_bbox[1])),
        1e-6,
    )
    return float(intersection / candidate_area)


def _has_self_intersection(polygon: np.ndarray) -> bool:
    """Return True when non-adjacent polygon edges intersect."""
    total = len(polygon)
    if total < 4:
        return False
    for index in range(total):
        a1 = polygon[index]
        a2 = polygon[(index + 1) % total]
        for other in range(index + 1, total):
            if other in (index, (index + 1) % total, (index - 1) % total):
                continue
            if index == 0 and other == total - 1:
                continue
            b1 = polygon[other]
            b2 = polygon[(other + 1) % total]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect(a1, a2, b1, b2) -> bool:
    """Check segment intersection with orientation tests."""
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-7:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-7 <= q[0] <= max(p[0], r[0]) + 1e-7 and
            min(p[1], r[1]) - 1e-7 <= q[1] <= max(p[1], r[1]) + 1e-7
        )

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a1, b1, a2):
        return True
    if o2 == 0 and on_segment(a1, b2, a2):
        return True
    if o3 == 0 and on_segment(b1, a1, b2):
        return True
    if o4 == 0 and on_segment(b1, a2, b2):
        return True
    return False
