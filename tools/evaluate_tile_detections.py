import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_EDGE_MARGIN = 32.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tile-edge quality for globally remapped detections."
    )
    parser.add_argument("--manifest", required=True, help="Path to tiles_manifest.json.")
    parser.add_argument("--results", required=True, help="Path to results_global.json.")
    parser.add_argument("--output", required=True, help="Output evaluated JSON path.")
    parser.add_argument(
        "--edge_margin",
        type=float,
        default=DEFAULT_EDGE_MARGIN,
        help="Distance in pixels used to classify contact with each tile edge.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_finite_number(value, label):
    if not is_finite_number(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return float(value)


def compact_number(value):
    rounded = round(float(value), 6)
    return int(rounded) if rounded.is_integer() else rounded


def load_tile_index(manifest_path):
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        raise ValueError("Manifest must be an object containing a 'tiles' list.")

    tile_index = {}
    for index, tile in enumerate(manifest["tiles"]):
        if not isinstance(tile, dict):
            raise ValueError(f"Manifest tile entry {index} must be an object.")
        tile_id = tile.get("tile_id")
        if tile_id is None or str(tile_id) == "":
            raise ValueError(f"Manifest tile entry {index} has no valid tile_id.")
        tile_key = str(tile_id)
        if tile_key in tile_index:
            raise ValueError(f"Duplicate tile_id in manifest: {tile_key!r}.")

        values = {}
        for field in ("x_offset", "y_offset", "width", "height"):
            values[field] = require_finite_number(
                tile.get(field), f"Manifest tile {tile_key!r} field {field!r}"
            )
        if values["width"] <= 0 or values["height"] <= 0:
            raise ValueError(f"Manifest tile {tile_key!r} must have positive width and height.")
        tile_index[tile_key] = values

    if not tile_index:
        raise ValueError("Manifest contains no written tiles.")
    return tile_index


def validate_bbox(bbox, detection_index):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"Detection {detection_index} bbox must be [x, y, width, height]."
        )
    x, y, width, height = (
        require_finite_number(value, f"Detection {detection_index} bbox[{offset}]")
        for offset, value in enumerate(bbox)
    )
    if width < 0 or height < 0:
        raise ValueError(f"Detection {detection_index} bbox width and height cannot be negative.")
    return x, y, width, height


def normalize_polygon_rings(polygon, detection_index, field_name):
    if polygon in (None, []):
        return []
    if not isinstance(polygon, (list, tuple)):
        raise ValueError(f"Detection {detection_index} {field_name} must be a polygon list.")

    rings = polygon
    if rings and all(is_finite_number(value) for value in rings):
        rings = [rings]
    elif rings and all(
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(is_finite_number(value) for value in point)
        for point in rings
    ):
        rings = [[coordinate for point in rings for coordinate in point]]

    normalized = []
    for ring_index, ring in enumerate(rings):
        if not isinstance(ring, (list, tuple)):
            raise ValueError(
                f"Detection {detection_index} {field_name} ring {ring_index} must be a list."
            )
        if ring and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(is_finite_number(value) for value in point)
            for point in ring
        ):
            ring = [coordinate for point in ring for coordinate in point]
        if len(ring) < 6 or len(ring) % 2:
            raise ValueError(
                f"Detection {detection_index} {field_name} ring {ring_index} "
                "must contain at least three x/y coordinate pairs."
            )
        coordinates = [
            require_finite_number(
                value,
                f"Detection {detection_index} {field_name} ring {ring_index}[{offset}]",
            )
            for offset, value in enumerate(ring)
        ]
        normalized.append(coordinates)
    return normalized


def polygon_area(rings):
    total_area = 0.0
    for ring in rings:
        points = list(zip(ring[0::2], ring[1::2]))
        signed_twice_area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            signed_twice_area += x1 * y2 - x2 * y1
        total_area += abs(signed_twice_area) * 0.5
    return total_area


def polygon_bounds(rings):
    x_values = [ring[index] for ring in rings for index in range(0, len(ring), 2)]
    y_values = [ring[index] for ring in rings for index in range(1, len(ring), 2)]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def quality_for_detection(detection, detection_index, tile, edge_margin):
    x, y, width, height = validate_bbox(detection.get("bbox"), detection_index)
    bbox_area = width * height

    polygon_field = "segmentation" if "segmentation" in detection else "polygon"
    rings = normalize_polygon_rings(
        detection.get(polygon_field), detection_index, polygon_field
    )
    if rings:
        min_x, min_y, max_x, max_y = polygon_bounds(rings)
    else:
        min_x, min_y, max_x, max_y = x, y, x + width, y + height

    existing_area = detection.get("area")
    if existing_area is None:
        calculated_polygon_area = polygon_area(rings) if rings else bbox_area
    else:
        calculated_polygon_area = require_finite_number(
            existing_area, f"Detection {detection_index} area"
        )
        if calculated_polygon_area < 0:
            raise ValueError(f"Detection {detection_index} area cannot be negative.")

    tile_left = tile["x_offset"]
    tile_top = tile["y_offset"]
    tile_right = tile_left + tile["width"]
    tile_bottom = tile_top + tile["height"]
    side_distances = (
        min_x - tile_left,
        tile_right - max_x,
        min_y - tile_top,
        tile_bottom - max_y,
    )
    edge_distance = min(side_distances)
    edge_touch_count = sum(distance < edge_margin for distance in side_distances)
    edge_touch = edge_touch_count > 0

    score = require_finite_number(detection.get("score"), f"Detection {detection_index} score")
    if edge_touch_count >= 2:
        priority_factor = 0.4
    elif edge_touch:
        priority_factor = 0.6
    else:
        priority_factor = 1.0
    merge_priority = score * priority_factor
    polygon_bbox_ratio = calculated_polygon_area / bbox_area if bbox_area > 0 else 0.0

    return {
        "edge_distance": compact_number(edge_distance),
        "edge_touch": edge_touch,
        "edge_touch_count": edge_touch_count,
        "is_truncated": edge_touch,
        "bbox_area": compact_number(bbox_area),
        "polygon_area": compact_number(calculated_polygon_area),
        "polygon_bbox_ratio": compact_number(polygon_bbox_ratio),
        "merge_priority": compact_number(merge_priority),
    }


def evaluate_results(results, tile_index, edge_margin):
    if not isinstance(results, list):
        raise ValueError("Global results must contain a JSON list of detections.")

    evaluated = []
    edge_touch_count = 0
    truncated_count = 0
    edge_distances = []
    merge_priorities = []
    for index, detection in enumerate(results):
        if not isinstance(detection, dict):
            raise ValueError(f"Detection {index} must be an object.")
        tile_id = detection.get("tile_id")
        tile = tile_index.get(str(tile_id)) if tile_id is not None else None
        if tile is None:
            raise ValueError(
                f"Detection {index} tile_id={tile_id!r} was not found in manifest tiles."
            )

        quality = quality_for_detection(detection, index, tile, edge_margin)
        output_detection = dict(detection)
        output_detection["tile_quality"] = quality
        evaluated.append(output_detection)
        edge_touch_count += int(quality["edge_touch"])
        truncated_count += int(quality["is_truncated"])
        edge_distances.append(float(quality["edge_distance"]))
        merge_priorities.append(float(quality["merge_priority"]))

    count = len(evaluated)
    stats = {
        "detection_count": count,
        "edge_touch_count": edge_touch_count,
        "truncated_count": truncated_count,
        "mean_edge_distance": sum(edge_distances) / count if count else 0.0,
        "mean_merge_priority": sum(merge_priorities) / count if count else 0.0,
    }
    return evaluated, stats


def run(args):
    if not is_finite_number(args.edge_margin) or args.edge_margin < 0:
        raise ValueError("--edge_margin must be a finite, non-negative number.")

    manifest_path = Path(args.manifest).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in ((manifest_path, "manifest"), (results_path, "results")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    tile_index = load_tile_index(manifest_path)
    results = load_json(results_path)
    evaluated, stats = evaluate_results(results, tile_index, float(args.edge_margin))
    write_json(output_path, evaluated)

    print(f"Detection count: {stats['detection_count']}")
    print(f"edge touch count: {stats['edge_touch_count']}")
    print(f"truncated count: {stats['truncated_count']}")
    print(f"mean edge distance: {stats['mean_edge_distance']:.6f}")
    print(f"mean merge priority: {stats['mean_merge_priority']:.6f}")
    print(f"output path: {output_path}")
    return stats


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Tile detection quality evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
