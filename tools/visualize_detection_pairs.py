import argparse
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.evaluate_tile_detections import (
    load_json,
    normalize_polygon_rings,
    require_finite_number,
    validate_bbox,
)


DEFAULT_MAX_SIZE = 3000
DEFAULT_MAX_PAIRS_DRAW = 200
DEFAULT_MAX_DETECTIONS_DRAW = 300
HIGH_PRIORITY_THRESHOLD = 0.7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize candidate detection pairs on a downsampled raster preview."
    )
    parser.add_argument("--image", required=True, help="Original large raster path.")
    parser.add_argument("--results", required=True, help="Evaluated global detection JSON path.")
    parser.add_argument("--pairs", required=True, help="Candidate pair JSON path.")
    parser.add_argument("--output", required=True, help="Output visualization image path.")
    parser.add_argument("--manifest", default=None, help="Optional tiles_manifest.json path.")
    parser.add_argument(
        "--max_size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help="Maximum preview width or height in pixels.",
    )
    pair_group = parser.add_mutually_exclusive_group()
    pair_group.add_argument(
        "--candidate_only",
        dest="candidate_only",
        action="store_true",
        default=True,
        help="Draw only pairs whose candidate field is true (default).",
    )
    pair_group.add_argument(
        "--draw_all_pairs",
        dest="candidate_only",
        action="store_false",
        help="Draw pairs regardless of their candidate field.",
    )
    parser.add_argument("--draw_tiles", action="store_true", help="Draw tile boundaries from manifest.")
    parser.add_argument(
        "--max_pairs_draw",
        type=int,
        default=DEFAULT_MAX_PAIRS_DRAW,
        help="Maximum number of pair connections to draw.",
    )
    parser.add_argument(
        "--max_detections_draw",
        type=int,
        default=DEFAULT_MAX_DETECTIONS_DRAW,
        help="Maximum number of detections to draw and label.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output image.")
    return parser.parse_args()


def require_visual_dependencies():
    try:
        import cv2
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise ImportError(
            "Visualization requires rasterio, numpy, and opencv-python. "
            "Run this script in the sampoly environment."
        ) from exc
    return cv2, np, rasterio, Resampling


def validate_args(args):
    if args.max_size <= 0:
        raise ValueError("--max_size must be positive.")
    if args.max_pairs_draw < 0:
        raise ValueError("--max_pairs_draw cannot be negative.")
    if args.max_detections_draw < 0:
        raise ValueError("--max_detections_draw cannot be negative.")
    if args.draw_tiles and not args.manifest:
        raise ValueError("--draw_tiles requires --manifest.")


def resolve_input(path_value, label):
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def stretch_preview(data, np):
    bands = []
    for band in data:
        band_array = np.ma.asarray(band, dtype=np.float32)
        values = band_array.compressed()
        values = values[np.isfinite(values)]
        if values.size == 0:
            bands.append(np.zeros(band_array.shape, dtype=np.uint8))
            continue
        low, high = np.percentile(values, (2.0, 98.0))
        if not math.isfinite(float(low)) or not math.isfinite(float(high)):
            raise ValueError("Raster preview contains non-finite percentile bounds.")
        if high <= low:
            low = float(values.min())
            high = float(values.max())
        if high <= low:
            stretched = np.zeros(band_array.shape, dtype=np.uint8)
        else:
            filled = np.ma.filled(band_array, low)
            scaled = (filled - low) * (255.0 / (high - low))
            stretched = np.clip(scaled, 0, 255).astype(np.uint8)
        bands.append(stretched)

    if len(bands) == 1:
        return np.stack((bands[0], bands[0], bands[0]), axis=2)
    return np.stack(bands[:3], axis=2)


def read_downsampled_preview(image_path, max_size):
    cv2, np, rasterio, Resampling = require_visual_dependencies()
    with rasterio.open(image_path, mode="r") as dataset:
        source_width = int(dataset.width)
        source_height = int(dataset.height)
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"Raster has invalid dimensions: {source_width} x {source_height}."
            )
        scale = min(1.0, max_size / max(source_width, source_height))
        preview_width = max(1, int(round(source_width * scale)))
        preview_height = max(1, int(round(source_height * scale)))
        indexes = (1, 2, 3) if dataset.count >= 3 else (1,)
        preview_data = dataset.read(
            indexes=indexes,
            out_shape=(len(indexes), preview_height, preview_width),
            resampling=Resampling.bilinear,
            masked=True,
        )

    preview_rgb = stretch_preview(preview_data, np)
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    scale_x = preview_width / source_width
    scale_y = preview_height / source_height
    return preview_bgr, {
        "source_width": source_width,
        "source_height": source_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def validate_results(results):
    if not isinstance(results, list):
        raise ValueError("Evaluated results must contain a JSON list of detections.")
    for index, detection in enumerate(results):
        if not isinstance(detection, dict):
            raise ValueError(f"Detection {index} must be an object.")
        validate_bbox(detection.get("bbox"), index)
        require_finite_number(detection.get("score"), f"Detection {index} score")
        quality = detection.get("tile_quality")
        if not isinstance(quality, dict):
            raise ValueError(f"Detection {index} has no valid tile_quality object.")
        require_finite_number(
            quality.get("merge_priority"),
            f"Detection {index} tile_quality.merge_priority",
        )


def validate_pairs(pairs, detection_count):
    if not isinstance(pairs, list):
        raise ValueError("Candidate pairs must contain a JSON list.")
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair {pair_index} must be an object.")
        for field in ("det_a", "det_b"):
            value = pair.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Pair {pair_index} field {field!r} must be an integer.")
            if value < 0 or value >= detection_count:
                raise ValueError(
                    f"Pair {pair_index} field {field!r}={value} is outside detection range."
                )
        require_finite_number(pair.get("bbox_iou"), f"Pair {pair_index} bbox_iou")
        require_finite_number(
            pair.get("center_distance"), f"Pair {pair_index} center_distance"
        )


def scaled_point(x, y, preview_info):
    return (
        int(round(float(x) * preview_info["scale_x"])),
        int(round(float(y) * preview_info["scale_y"])),
    )


def detection_center(detection):
    x, y, width, height = detection["bbox"]
    return float(x) + float(width) * 0.5, float(y) + float(height) * 0.5


def detection_color(detection):
    quality = detection["tile_quality"]
    if bool(quality.get("edge_touch")) or bool(quality.get("is_truncated")):
        return 0, 128, 255
    if float(quality["merge_priority"]) >= HIGH_PRIORITY_THRESHOLD:
        return 0, 200, 0
    return 255, 100, 0


def detection_polygon_rings(detection, detection_index):
    if "segmentation" in detection:
        field_name = "segmentation"
    elif "polygon" in detection:
        field_name = "polygon"
    else:
        return []
    return normalize_polygon_rings(
        detection.get(field_name), detection_index, field_name
    )


def draw_text(canvas, text, origin, cv2, font_scale=0.38):
    x, y = origin
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_detection(canvas, detection, detection_index, preview_info, cv2, np):
    color = detection_color(detection)
    rings = detection_polygon_rings(detection, detection_index)
    if rings:
        for ring in rings:
            points = [
                scaled_point(ring[offset], ring[offset + 1], preview_info)
                for offset in range(0, len(ring), 2)
            ]
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)],
                True,
                color,
                2,
                cv2.LINE_AA,
            )
    else:
        x, y, width, height = detection["bbox"]
        top_left = scaled_point(x, y, preview_info)
        bottom_right = scaled_point(x + width, y + height, preview_info)
        cv2.rectangle(canvas, top_left, bottom_right, color, 2, cv2.LINE_AA)

    x, y, _, _ = detection["bbox"]
    label_origin = scaled_point(x, y, preview_info)
    label_origin = (max(0, label_origin[0]), max(12, label_origin[1] - 3))
    quality = detection["tile_quality"]
    label = (
        f"#{detection_index} s={float(detection['score']):.2f} "
        f"p={float(quality['merge_priority']):.2f} "
        f"e={int(bool(quality.get('edge_touch')))}"
    )
    draw_text(canvas, label, label_origin, cv2)


def load_manifest_tiles(manifest_path, preview_info):
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        raise ValueError("Manifest must be an object containing a tiles list.")
    source = manifest.get("source")
    if isinstance(source, dict):
        manifest_width = source.get("width")
        manifest_height = source.get("height")
        if manifest_width is not None and int(manifest_width) != preview_info["source_width"]:
            raise ValueError("Manifest source width does not match the input raster.")
        if manifest_height is not None and int(manifest_height) != preview_info["source_height"]:
            raise ValueError("Manifest source height does not match the input raster.")
    return manifest["tiles"]


def draw_tiles(canvas, tiles, preview_info, cv2):
    for tile_index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"Manifest tile {tile_index} must be an object.")
        x = require_finite_number(tile.get("x_offset"), f"Manifest tile {tile_index} x_offset")
        y = require_finite_number(tile.get("y_offset"), f"Manifest tile {tile_index} y_offset")
        width = require_finite_number(tile.get("width"), f"Manifest tile {tile_index} width")
        height = require_finite_number(tile.get("height"), f"Manifest tile {tile_index} height")
        if width <= 0 or height <= 0:
            raise ValueError(f"Manifest tile {tile_index} must have positive dimensions.")
        cv2.rectangle(
            canvas,
            scaled_point(x, y, preview_info),
            scaled_point(x + width, y + height, preview_info),
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def draw_pairs(canvas, pairs, results, preview_info, cv2):
    for pair in pairs:
        det_a = pair["det_a"]
        det_b = pair["det_b"]
        point_a = scaled_point(*detection_center(results[det_a]), preview_info)
        point_b = scaled_point(*detection_center(results[det_b]), preview_info)
        cv2.line(canvas, point_a, point_b, (255, 0, 255), 1, cv2.LINE_AA)
        print(
            f"pair det_a={det_a} det_b={det_b} "
            f"bbox_iou={float(pair['bbox_iou']):.6f} "
            f"center_distance={float(pair['center_distance']):.3f}"
        )


def draw_legend(canvas, cv2):
    entries = (
        ("high priority", (0, 200, 0)),
        ("edge/truncated", (0, 128, 255)),
        ("normal detection", (255, 100, 0)),
        ("candidate pair", (255, 0, 255)),
        ("tile boundary", (180, 180, 180)),
    )
    for index, (label, color) in enumerate(entries):
        y = 18 + index * 18
        cv2.line(canvas, (8, y), (28, y), color, 2, cv2.LINE_AA)
        draw_text(canvas, label, (34, y + 4), cv2)


def write_image(output_path, canvas, cv2):
    extension = output_path.suffix.lower()
    if not extension:
        raise ValueError("Output image path must include an extension such as .png.")
    success, encoded = cv2.imencode(extension, canvas)
    if not success:
        raise ValueError(f"OpenCV cannot encode output format: {extension}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{extension}")
    temporary_path.write_bytes(encoded.tobytes())
    temporary_path.replace(output_path)


def run(args):
    validate_args(args)
    image_path = resolve_input(args.image, "image")
    results_path = resolve_input(args.results, "results")
    pairs_path = resolve_input(args.pairs, "pairs")
    manifest_path = (
        resolve_input(args.manifest, "manifest") if args.manifest is not None else None
    )
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    canvas, preview_info = read_downsampled_preview(image_path, args.max_size)
    cv2, np, _, _ = require_visual_dependencies()
    results = load_json(results_path)
    pairs = load_json(pairs_path)
    validate_results(results)
    validate_pairs(pairs, len(results))

    if args.draw_tiles:
        tiles = load_manifest_tiles(manifest_path, preview_info)
        draw_tiles(canvas, tiles, preview_info, cv2)

    filtered_pairs = (
        [pair for pair in pairs if pair.get("candidate") is True]
        if args.candidate_only else list(pairs)
    )
    selected_pairs = filtered_pairs[:args.max_pairs_draw]
    draw_pairs(canvas, selected_pairs, results, preview_info, cv2)

    selected_detections = results[:args.max_detections_draw]
    for detection_index, detection in enumerate(selected_detections):
        draw_detection(canvas, detection, detection_index, preview_info, cv2, np)

    draw_legend(canvas, cv2)
    write_image(output_path, canvas, cv2)

    print(
        f"image size: {preview_info['source_width']} x {preview_info['source_height']}"
    )
    print(
        f"preview size: {preview_info['preview_width']} x {preview_info['preview_height']}"
    )
    print(
        f"scale: x={preview_info['scale_x']:.8f}, y={preview_info['scale_y']:.8f}"
    )
    print(f"detections loaded: {len(results)}")
    print(f"detections drawn: {len(selected_detections)}")
    print(f"pairs loaded: {len(pairs)}")
    print(f"candidate pairs drawn: {len(selected_pairs)}")
    print(f"output path: {output_path}")
    return {
        "detections_loaded": len(results),
        "detections_drawn": len(selected_detections),
        "pairs_loaded": len(pairs),
        "pairs_drawn": len(selected_pairs),
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Detection pair visualization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
