import argparse
import json
import math
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remap tile-local COCO polygon results to source-image pixel coordinates."
    )
    parser.add_argument("--manifest", required=True, help="Path to tiles_manifest.json.")
    parser.add_argument("--tile_results", required=True, help="Tile-local polygon results.json.")
    parser.add_argument("--output", required=True, help="Output global-coordinate results.json.")
    parser.add_argument(
        "--source_image_id",
        default=None,
        help="Override manifest source.image_id in output records.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_tile_index(manifest_path, source_image_id_override=None):
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Tile manifest must be a JSON object.")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Tile manifest must contain a 'source' object.")
    source_image_id = source_image_id_override or source.get("image_id")
    if source_image_id in (None, ""):
        raise ValueError(
            "Source image id is missing; provide --source_image_id or manifest source.image_id."
        )

    tiles = manifest.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("Tile manifest field 'tiles' must be a list.")
    tile_index = {}
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"Manifest tile entry {index} must be an object.")
        tile_id = tile.get("tile_id")
        if tile_id in (None, ""):
            raise ValueError(f"Manifest tile entry {index} is missing tile_id.")
        tile_key = str(tile_id)
        if tile_key in tile_index:
            raise ValueError(f"Duplicate tile_id in manifest: {tile_key}")
        required_fields = ("x_offset", "y_offset", "width", "height")
        invalid_fields = [
            field for field in required_fields
            if not is_finite_number(tile.get(field))
        ]
        if invalid_fields:
            raise ValueError(
                f"Manifest tile {tile_key} has invalid numeric fields: {invalid_fields}"
            )
        if tile["width"] <= 0 or tile["height"] <= 0:
            raise ValueError(f"Manifest tile {tile_key} must have positive width and height.")
        tile_index[tile_key] = tile
    return source_image_id, tile_index


def validate_polygon_results(results):
    if not isinstance(results, list):
        raise ValueError("Tile results must contain a JSON list of COCO polygon detections.")
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Tile result entry {index} must be an object.")
        if isinstance(item.get("segmentation"), dict):
            raise ValueError(
                "RLE segmentation is not supported. Use polygon results.json, "
                f"not results_mask.json (entry {index})."
            )


def remap_bbox(bbox, x_offset, y_offset, result_index):
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            f"Result entry {result_index} bbox must be COCO [x, y, w, h]."
        )
    if not all(is_finite_number(value) for value in bbox):
        raise ValueError(f"Result entry {result_index} bbox contains non-numeric values.")
    return [bbox[0] + x_offset, bbox[1] + y_offset, bbox[2], bbox[3]]


def remap_segmentation(segmentation, x_offset, y_offset, result_index, warn):
    if not isinstance(segmentation, list) or not segmentation:
        warn(
            f"result entry {result_index}: empty or invalid segmentation; "
            "the original field was preserved"
        )
        return segmentation

    remapped_polygons = []
    for polygon_index, polygon in enumerate(segmentation):
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2 != 0
            or not all(is_finite_number(value) for value in polygon)
        ):
            warn(
                f"result entry {result_index}, polygon {polygon_index}: invalid polygon; "
                "the original segmentation field was preserved"
            )
            return segmentation
        remapped_polygon = [
            value + (x_offset if coordinate_index % 2 == 0 else y_offset)
            for coordinate_index, value in enumerate(polygon)
        ]
        remapped_polygons.append(remapped_polygon)
    return remapped_polygons


def remap_results(results, tile_index, source_image_id):
    output_results = []
    unmatched_count = 0
    warning_count = 0

    def warn(message):
        nonlocal warning_count
        warning_count += 1
        print(f"Warning: {message}", file=sys.stderr)

    for result_index, item in enumerate(results):
        original_image_id = item.get("image_id")
        tile_key = str(original_image_id)
        tile = tile_index.get(tile_key)
        if tile is None:
            unmatched_count += 1
            warn(
                f"result entry {result_index} image_id={original_image_id!r} "
                "was not found in manifest tiles and was skipped"
            )
            continue

        x_offset = tile["x_offset"]
        y_offset = tile["y_offset"]
        remapped = dict(item)
        remapped["image_id"] = source_image_id
        remapped["tile_id"] = original_image_id
        remapped["bbox"] = remap_bbox(item.get("bbox"), x_offset, y_offset, result_index)
        if "segmentation" in item:
            remapped["segmentation"] = remap_segmentation(
                item["segmentation"],
                x_offset,
                y_offset,
                result_index,
                warn,
            )
        else:
            warn(f"result entry {result_index}: segmentation field is missing")
        remapped.update({
            "x_offset": x_offset,
            "y_offset": y_offset,
            "tile_width": tile["width"],
            "tile_height": tile["height"],
        })
        output_results.append(remapped)

    stats = {
        "input_detections": len(results),
        "remapped_detections": len(output_results),
        "unmatched_detections": unmatched_count,
        "warnings": warning_count,
    }
    return output_results, stats


def run(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    tile_results_path = Path(args.tile_results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in (
        (manifest_path, "manifest"),
        (tile_results_path, "tile results"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    source_image_id, tile_index = load_tile_index(
        manifest_path,
        source_image_id_override=args.source_image_id,
    )
    results = load_json(tile_results_path)
    validate_polygon_results(results)
    output_results, stats = remap_results(results, tile_index, source_image_id)
    write_json(output_path, output_results)

    print(f"input detection count: {stats['input_detections']}")
    print(f"successfully remapped: {stats['remapped_detections']}")
    print(f"unmatched image_id count: {stats['unmatched_detections']}")
    print(f"warning count: {stats['warnings']}")
    print(f"output path: {output_path}")
    return stats


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Tile result remapping failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
