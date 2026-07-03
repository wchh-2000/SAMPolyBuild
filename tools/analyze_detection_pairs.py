import argparse
import itertools
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.evaluate_tile_detections import (
    compact_number,
    is_finite_number,
    load_json,
    load_tile_index,
    normalize_polygon_rings,
    require_finite_number,
    validate_bbox,
    write_json,
)


DEFAULT_IOU_THRESHOLD = 0.2
DEFAULT_CENTER_THRESHOLD = 50.0
DEFAULT_MAX_CENTER_THRESHOLD = 100.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze likely duplicate detection pairs before merge or NMS."
    )
    parser.add_argument("--manifest", required=True, help="Path to tiles_manifest.json.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results_global_evaluated.json.",
    )
    parser.add_argument("--output", required=True, help="Output candidate_pairs.json path.")
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Candidate threshold for bbox or polygon IoU (greater than or equal).",
    )
    parser.add_argument(
        "--center_thr",
        type=float,
        default=DEFAULT_CENTER_THRESHOLD,
        help="Candidate threshold for center distance in pixels (less than or equal).",
    )
    parser.add_argument(
        "--max_center_thr",
        type=float,
        default=DEFAULT_MAX_CENTER_THRESHOLD,
        help="Safety upper bound allowed for --center_thr.",
    )
    parser.add_argument(
        "--allow_same_tile_pairs",
        action="store_true",
        help="Include same-tile pairs for duplicate-detection debugging.",
    )
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--require_tile_overlap",
        dest="require_tile_overlap",
        action="store_true",
        default=True,
        help="Compare cross-tile pairs only when their tile bounds overlap (default).",
    )
    overlap_group.add_argument(
        "--allow_non_overlap_tiles",
        dest="require_tile_overlap",
        action="store_false",
        help="Also compare detections from non-overlapping tiles.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def bbox_xyxy(bbox, detection_index):
    x, y, width, height = validate_bbox(bbox, detection_index)
    return x, y, x + width, y + height


def bbox_iou(box_a, box_b):
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def bbox_center(box):
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def center_distance(center_a, center_b):
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def rectangle_intersection(rect_a, rect_b):
    left = max(rect_a[0], rect_b[0])
    top = max(rect_a[1], rect_b[1])
    right = min(rect_a[2], rect_b[2])
    bottom = min(rect_a[3], rect_b[3])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def tiles_overlap(rect_a, rect_b):
    return rectangle_intersection(rect_a, rect_b) is not None


def polygon_array(detection, detection_index):
    if "segmentation" in detection:
        field_name = "segmentation"
    elif "polygon" in detection:
        field_name = "polygon"
    else:
        raise ValueError(
            f"Detection {detection_index} has neither segmentation nor polygon coordinates."
        )

    rings = normalize_polygon_rings(
        detection.get(field_name), detection_index, field_name
    )
    if not rings:
        raise ValueError(f"Detection {detection_index} {field_name} is empty.")
    if len(rings) != 1:
        raise ValueError(
            f"Detection {detection_index} has {len(rings)} polygon rings; "
            "the existing iou_poly utility supports one ring per detection."
        )
    ring = rings[0]
    return list(zip(ring[0::2], ring[1::2]))


def polygon_iou(polygon_a, polygon_b, det_a, det_b):
    try:
        from utils.metrics.iou import iou_poly

        value = float(iou_poly(polygon_a, polygon_b))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Polygon IoU requires the project's Shapely, OpenCV, and NumPy dependencies. "
            "Run this script in the sampoly environment."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Unable to calculate polygon IoU for detections {det_a} and {det_b}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Polygon IoU for detections {det_a} and {det_b} is not finite: {value!r}."
        )
    if value < -1e-9 or value > 1.0 + 1e-9:
        raise ValueError(
            f"Polygon IoU for detections {det_a} and {det_b} is outside [0, 1]: {value}."
        )
    return min(1.0, max(0.0, value))


def tile_rectangle(tile):
    return (
        tile["x_offset"],
        tile["y_offset"],
        tile["x_offset"] + tile["width"],
        tile["y_offset"] + tile["height"],
    )


def merge_priority(detection, detection_index):
    quality = detection.get("tile_quality")
    if not isinstance(quality, dict):
        raise ValueError(f"Detection {detection_index} has no valid tile_quality object.")
    return require_finite_number(
        quality.get("merge_priority"),
        f"Detection {detection_index} tile_quality.merge_priority",
    )


def prepare_detection(detection, detection_index, tile_index):
    if not isinstance(detection, dict):
        raise ValueError(f"Detection {detection_index} must be an object.")
    tile_id = detection.get("tile_id")
    if tile_id is None or str(tile_id) == "":
        raise ValueError(f"Detection {detection_index} has no valid tile_id.")
    tile_key = str(tile_id)
    tile = tile_index.get(tile_key)
    if tile is None:
        raise ValueError(
            f"Detection {detection_index} tile_id={tile_id!r} was not found in manifest tiles."
        )
    box = bbox_xyxy(detection.get("bbox"), detection_index)
    return {
        "index": detection_index,
        "bbox": box,
        "center": bbox_center(box),
        "polygon": polygon_array(detection, detection_index),
        "priority": merge_priority(detection, detection_index),
        "tile_id": tile_key,
        "tile_rect": tile_rectangle(tile),
    }


def analyze_pairs(
    results,
    tile_index,
    iou_threshold,
    center_threshold,
    allow_same_tile_pairs=False,
    require_tile_overlap=True,
):
    if not isinstance(results, list):
        raise ValueError("Evaluated results must contain a JSON list of detections.")
    detections = [
        prepare_detection(detection, index, tile_index)
        for index, detection in enumerate(results)
    ]

    candidate_pairs = []
    skipped_same_tile_pairs = 0
    skipped_non_overlap_tile_pairs = 0
    compared_cross_tile_overlap_pairs = 0
    for item_a, item_b in itertools.combinations(detections, 2):
        same_tile = item_a["tile_id"] == item_b["tile_id"]
        pair_tiles_overlap = tiles_overlap(item_a["tile_rect"], item_b["tile_rect"])
        if same_tile and not allow_same_tile_pairs:
            skipped_same_tile_pairs += 1
            continue
        if not same_tile and require_tile_overlap and not pair_tiles_overlap:
            skipped_non_overlap_tile_pairs += 1
            continue
        if not same_tile and pair_tiles_overlap:
            compared_cross_tile_overlap_pairs += 1

        pair_bbox_iou = bbox_iou(item_a["bbox"], item_b["bbox"])
        pair_center_distance = center_distance(item_a["center"], item_b["center"])
        pair_polygon_iou = polygon_iou(
            item_a["polygon"],
            item_b["polygon"],
            item_a["index"],
            item_b["index"],
        )
        candidate = (
            pair_bbox_iou >= iou_threshold
            or pair_polygon_iou >= iou_threshold
            or pair_center_distance <= center_threshold
        )
        if not candidate:
            continue

        candidate_pairs.append({
            "det_a": item_a["index"],
            "det_b": item_b["index"],
            "tile_a": item_a["tile_id"],
            "tile_b": item_b["tile_id"],
            "bbox_iou": compact_number(pair_bbox_iou),
            "polygon_iou": compact_number(pair_polygon_iou),
            "center_distance": compact_number(pair_center_distance),
            "priority_a": compact_number(item_a["priority"]),
            "priority_b": compact_number(item_b["priority"]),
            "same_tile": same_tile,
            "tiles_overlap": pair_tiles_overlap,
            "in_overlap": (not same_tile) and pair_tiles_overlap,
            "candidate": True,
        })

    pair_count = len(candidate_pairs)
    same_tile_candidates = sum(pair["same_tile"] for pair in candidate_pairs)
    cross_tile_candidates = pair_count - same_tile_candidates
    overlap_candidates = sum(pair["in_overlap"] for pair in candidate_pairs)
    stats = {
        "detection_count": len(detections),
        "all_raw_pair_count": len(detections) * (len(detections) - 1) // 2,
        "skipped_same_tile_pair_count": skipped_same_tile_pairs,
        "skipped_non_overlap_tile_pair_count": skipped_non_overlap_tile_pairs,
        "compared_cross_tile_overlap_pair_count": compared_cross_tile_overlap_pairs,
        "candidate_pair_count": pair_count,
        "average_iou": (
            sum(float(pair["bbox_iou"]) for pair in candidate_pairs) / pair_count
            if pair_count else 0.0
        ),
        "average_center_distance": (
            sum(float(pair["center_distance"]) for pair in candidate_pairs) / pair_count
            if pair_count else 0.0
        ),
        "same_tile_candidate_count": same_tile_candidates,
        "cross_tile_candidate_count": cross_tile_candidates,
        "overlap_candidate_count": overlap_candidates,
    }
    return candidate_pairs, stats


def validate_thresholds(iou_threshold, center_threshold, max_center_threshold):
    if not is_finite_number(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("--iou_thr must be a finite number between 0 and 1.")
    if not is_finite_number(center_threshold) or center_threshold < 0:
        raise ValueError("--center_thr must be a finite, non-negative number.")
    if not is_finite_number(max_center_threshold) or max_center_threshold < 0:
        raise ValueError("--max_center_thr must be a finite, non-negative number.")
    if center_threshold > max_center_threshold:
        raise ValueError(
            f"--center_thr ({center_threshold}) cannot exceed "
            f"--max_center_thr ({max_center_threshold})."
        )


def run(args):
    validate_thresholds(args.iou_thr, args.center_thr, args.max_center_thr)
    manifest_path = Path(args.manifest).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in ((manifest_path, "manifest"), (results_path, "evaluated results")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    tile_index = load_tile_index(manifest_path)
    results = load_json(results_path)
    candidate_pairs, stats = analyze_pairs(
        results,
        tile_index,
        float(args.iou_thr),
        float(args.center_thr),
        allow_same_tile_pairs=bool(args.allow_same_tile_pairs),
        require_tile_overlap=bool(args.require_tile_overlap),
    )
    write_json(output_path, candidate_pairs)

    print(f"Detection count: {stats['detection_count']}")
    print(f"All raw pair count: {stats['all_raw_pair_count']}")
    print(f"Skipped same-tile pair count: {stats['skipped_same_tile_pair_count']}")
    print(
        "Skipped non-overlap tile pair count: "
        f"{stats['skipped_non_overlap_tile_pair_count']}"
    )
    print(
        "Compared cross-tile overlap pair count: "
        f"{stats['compared_cross_tile_overlap_pair_count']}"
    )
    print(f"Candidate Pair count: {stats['candidate_pair_count']}")
    print(f"Same-tile candidate count: {stats['same_tile_candidate_count']}")
    print(f"Cross-tile candidate count: {stats['cross_tile_candidate_count']}")
    print(f"Overlap candidate count: {stats['overlap_candidate_count']}")
    print(f"Average IoU: {stats['average_iou']:.6f}")
    print(f"Average center distance: {stats['average_center_distance']:.6f}")
    if stats["cross_tile_candidate_count"] == 0:
        print(
            "[warning] no cross-tile candidate pairs found. This may mean tile_limit "
            "is too small, overlap too small, or detections are not near tile boundaries.",
            file=sys.stderr,
        )
    print(f"output path: {output_path}")
    return stats


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Detection candidate pair analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
