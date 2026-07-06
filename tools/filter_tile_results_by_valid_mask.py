"""Filter canonical tile polygons that substantially overlap invalid imagery."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infer_tiles_auto import resolve_tile_path
from utils.image_tile_reader import ImageTileReader
from workflow.manifest import TileManifest, write_json
from workflow.tile_manager import TileResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove Auto polygons that mostly cover Alpha/NoData/background pixels."
    )
    parser.add_argument("--manifest", required=True, help="Canonical manifest.json path.")
    parser.add_argument("--input-results", required=True, help="Original per-tile Auto results.")
    parser.add_argument("--output-results", required=True, help="Filtered per-tile output directory.")
    parser.add_argument("--min-polygon-valid-ratio", type=float, default=0.9)
    parser.add_argument("--min-polygon-valid-pixels", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rasterize_polygon(points, width: int, height: int) -> np.ndarray:
    """Rasterize one tile-local polygon at pixel centers using a ray-crossing test."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 3:
        raise ValueError("Polygon points must contain at least three [x, y] pairs.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Polygon coordinates must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("Raster dimensions must be positive.")

    x_min = max(int(np.floor(coordinates[:, 0].min())), 0)
    y_min = max(int(np.floor(coordinates[:, 1].min())), 0)
    x_max = min(int(np.ceil(coordinates[:, 0].max())), width)
    y_max = min(int(np.ceil(coordinates[:, 1].max())), height)
    mask = np.zeros((height, width), dtype=bool)
    if x_min >= x_max or y_min >= y_max:
        return mask

    x_grid = np.arange(x_min, x_max, dtype=np.float64)[None, :] + 0.5
    y_grid = np.arange(y_min, y_max, dtype=np.float64)[:, None] + 0.5
    inside = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
    previous = coordinates[-1]
    for current in coordinates:
        x1, y1 = previous
        x2, y2 = current
        crosses_rows = (y1 > y_grid) != (y2 > y_grid)
        if y2 != y1:
            crossing_x = (x2 - x1) * (y_grid - y1) / (y2 - y1) + x1
            inside ^= crosses_rows & (x_grid < crossing_x)
        previous = current
    mask[y_min:y_max, x_min:x_max] = inside
    return mask


def polygon_validity(points, valid_mask: np.ndarray) -> dict:
    """Return rasterized polygon area and its overlap with one valid-pixel mask."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must be 2D, got {valid.shape}.")
    polygon_mask = rasterize_polygon(points, valid.shape[1], valid.shape[0])
    polygon_pixels = int(polygon_mask.sum())
    valid_pixels = int(np.count_nonzero(polygon_mask & valid))
    ratio = valid_pixels / polygon_pixels if polygon_pixels else 0.0
    return {
        "polygon_pixels": polygon_pixels,
        "valid_pixels": valid_pixels,
        "valid_ratio": ratio,
    }


def filter_result(result: TileResult, valid_mask, min_ratio: float, min_pixels: int):
    """Return a filtered result copy and audit records for rejected polygons."""
    filtered = TileResult.from_dict(copy.deepcopy(result.to_dict()))
    kept = []
    rejected = []
    for polygon in filtered.polygons:
        validity = polygon_validity(polygon.points_tile, valid_mask)
        polygon.metadata["valid_mask_filter"] = {
            "polygon_pixels": validity["polygon_pixels"],
            "valid_pixels": validity["valid_pixels"],
            "valid_ratio": round(validity["valid_ratio"], 6),
        }
        reasons = []
        if validity["polygon_pixels"] == 0:
            reasons.append("empty_rasterized_polygon")
        if validity["valid_pixels"] < min_pixels:
            reasons.append("valid_pixels_below_threshold")
        if validity["valid_ratio"] < min_ratio:
            reasons.append("valid_ratio_below_threshold")
        if reasons:
            rejected.append({
                "tile_id": result.tile_id,
                "polygon_id": polygon.id,
                **polygon.metadata["valid_mask_filter"],
                "reasons": reasons,
            })
        else:
            kept.append(polygon)
    filtered.polygons = kept
    filtered.metadata["valid_mask_filter"] = {
        "input_polygons": len(result.polygons),
        "kept_polygons": len(kept),
        "rejected_polygons": len(rejected),
        "min_polygon_valid_ratio": min_ratio,
        "min_polygon_valid_pixels": min_pixels,
    }
    return filtered, rejected


def run(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest).expanduser().resolve()
    input_dir = Path(args.input_results).expanduser().resolve()
    output_dir = Path(args.output_results).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input results directory not found: {input_dir}")
    if not 0.0 <= float(args.min_polygon_valid_ratio) <= 1.0:
        raise ValueError("--min-polygon-valid-ratio must be between 0 and 1.")
    if int(args.min_polygon_valid_pixels) < 0:
        raise ValueError("--min-polygon-valid-pixels must be non-negative.")
    if output_dir == input_dir:
        raise ValueError("--output-results must differ from --input-results.")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = TileManifest.load(manifest_path)
    rejected = []
    input_polygons = 0
    kept_polygons = 0
    for index, tile in enumerate(manifest.tiles, start=1):
        input_path = input_dir / f"{tile.tile_id}.json"
        if not input_path.is_file():
            raise FileNotFoundError(f"Tile result not found: {input_path}")
        result = TileResult.load(input_path)
        if result.tile_id != tile.tile_id:
            raise ValueError(f"Result tile_id mismatch: {result.tile_id} != {tile.tile_id}")
        tile_path = resolve_tile_path(manifest_path, tile.image_path)
        with ImageTileReader(tile_path, bands=(1, 2, 3)) as reader:
            _, valid_mask = reader.read_tile_with_valid_mask(
                0, 0, reader.width, reader.height
            )
        if valid_mask.shape != (tile.height, tile.width):
            raise ValueError(
                f"Valid mask shape mismatch for {tile.tile_id}: {valid_mask.shape}"
            )
        filtered, tile_rejected = filter_result(
            result,
            valid_mask,
            float(args.min_polygon_valid_ratio),
            int(args.min_polygon_valid_pixels),
        )
        filtered.save(output_dir / f"{tile.tile_id}.json")
        input_polygons += len(result.polygons)
        kept_polygons += len(filtered.polygons)
        rejected.extend(tile_rejected)
        print(
            f"[{index}/{len(manifest.tiles)}] {tile.tile_id}: "
            f"input={len(result.polygons)}, kept={len(filtered.polygons)}, "
            f"rejected={len(tile_rejected)}"
        )

    summary = {
        "tile_count": len(manifest.tiles),
        "input_polygons": input_polygons,
        "kept_polygons": kept_polygons,
        "rejected_polygons": len(rejected),
        "min_polygon_valid_ratio": float(args.min_polygon_valid_ratio),
        "min_polygon_valid_pixels": int(args.min_polygon_valid_pixels),
        "input_results": str(input_dir),
        "output_results": str(output_dir),
        "rejected": rejected,
    }
    write_json(output_dir / "filter_summary.json", summary)
    print(f"input polygons: {input_polygons}")
    print(f"kept polygons: {kept_polygons}")
    print(f"rejected polygons: {len(rejected)}")
    print(f"output results: {output_dir}")
    return summary


def main() -> int:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"Valid-mask polygon filter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
