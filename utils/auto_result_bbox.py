import json
import math
import os


def load_auto_results(results_path):
    if not results_path:
        raise ValueError("results_path is required.")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Auto result file not found: {results_path}")
    with open(results_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Auto result file must contain a list of predictions.")
    return data


def infer_image_id(image_path):
    if not image_path:
        return None
    return os.path.splitext(os.path.basename(image_path))[0]


def select_bbox_from_auto_results(
    results,
    click_point,
    image_id=None,
    min_score=0.0,
    max_center_distance=None,
):
    """
    Select one auto-mode bbox for a click point.

    Auto results use COCO-style bbox [x, y, w, h]. This function returns
    prompt-mode xyxy bbox [x1, y1, x2, y2].
    """
    point = _normalize_point(click_point)
    candidates = _filter_candidates(results, image_id, min_score)
    if not candidates:
        raise ValueError("No auto bbox candidates matched the requested image_id and score.")

    selected = _select_candidate(candidates, point)
    if max_center_distance is not None and selected["center_distance"] > max_center_distance:
        raise ValueError(
            "Nearest auto bbox is farther than max_center_distance: "
            f"{selected['center_distance']:.2f} > {max_center_distance}"
        )

    return selected["xyxy"], selected


def load_and_select_bbox(
    results_path,
    click_point,
    image_id=None,
    min_score=0.0,
    max_center_distance=None,
):
    results = load_auto_results(results_path)
    return select_bbox_from_auto_results(
        results,
        click_point,
        image_id=image_id,
        min_score=min_score,
        max_center_distance=max_center_distance,
    )


def _normalize_point(click_point):
    if click_point is None or len(click_point) != 2:
        raise ValueError("click_point must be a pair of x, y coordinates.")
    try:
        return float(click_point[0]), float(click_point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("click_point must contain numeric x, y values.") from exc


def _filter_candidates(results, image_id, min_score):
    candidates = []
    image_id_str = str(image_id) if image_id is not None else None
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        if image_id_str is not None and str(item.get("image_id")) != image_id_str:
            continue
        score = float(item.get("score_cls", item.get("score", 0.0)))
        if score < min_score:
            continue
        bbox = item.get("bbox")
        xyxy = _xywh_to_xyxy(bbox)
        if xyxy is None:
            continue
        candidates.append(
            {
                "index": idx,
                "image_id": item.get("image_id"),
                "score": float(item.get("score", 0.0)),
                "score_cls": float(item.get("score_cls", score)),
                "bbox_xywh": bbox,
                "xyxy": xyxy,
                "raw": item,
            }
        )
    return candidates


def _xywh_to_xyxy(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _select_candidate(candidates, point):
    px, py = point
    scored = []
    for item in candidates:
        x1, y1, x2, y2 = item["xyxy"]
        contains = x1 <= px <= x2 and y1 <= py <= y2
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_distance = math.hypot(px - cx, py - cy)
        edge_distance = _distance_to_box(point, item["xyxy"])
        ranked = dict(item)
        ranked.update(
            {
                "contains_click": contains,
                "center": [cx, cy],
                "center_distance": center_distance,
                "edge_distance": edge_distance,
            }
        )
        scored.append(ranked)

    scored.sort(
        key=lambda item: (
            0 if item["contains_click"] else 1,
            item["center_distance"] if item["contains_click"] else item["edge_distance"],
            -item["score_cls"],
            -item["score"],
        )
    )
    return scored[0]


def _distance_to_box(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)
