import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.image_tile_reader import ImageTileReader


def parse_bands(value):
    try:
        bands = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bands must be comma-separated integers.") from exc
    if not bands:
        raise argparse.ArgumentTypeError("--bands must select at least one band.")
    return bands


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a large geospatial raster into window-read GeoTIFF tiles."
    )
    parser.add_argument("--input", required=True, help="Input GeoTIFF path.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile side length in pixels.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between adjacent tiles.")
    parser.add_argument("--prefix", default=None, help="Tile name prefix; defaults to input stem.")
    parser.add_argument("--bands", type=parse_bands, default=(1, 2, 3), help="Bands, e.g. 1,2,3.")
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="Compatibility flag for skipping tiles whose invalid ratio exceeds --empty_threshold.",
    )
    parser.add_argument(
        "--empty_threshold",
        type=float,
        default=0.98,
        help="Pixel ratio above which a tile is considered empty.",
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.25,
        help="Skip tiles with a lower valid-pixel ratio; use 0 to disable filtering.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tiles and manifest.")
    return parser.parse_args()


def axis_offsets(length, stride):
    return list(range(0, length, stride))


def iter_tile_windows(width, height, tile_size, stride):
    for y_offset in axis_offsets(height, stride):
        tile_height = min(tile_size, height - y_offset)
        for x_offset in axis_offsets(width, stride):
            tile_width = min(tile_size, width - x_offset)
            yield x_offset, y_offset, tile_width, tile_height


def affine_to_list(transform):
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def crs_to_string(crs):
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg is None:
        epsg = crs.to_epsg(confidence_threshold=0)
    return f"EPSG:{epsg}" if epsg is not None else crs.to_wkt()


def valid_pixel_stats(valid_mask):
    valid_array = np.asarray(valid_mask, dtype=bool)
    if valid_array.ndim != 2:
        raise ValueError(f"Expected valid mask shape (H, W), got {valid_array.shape}.")
    total_pixels = int(valid_array.size)
    valid_pixels = int(np.count_nonzero(valid_array))
    valid_ratio = valid_pixels / total_pixels if total_pixels else 0.0
    return valid_pixels, total_pixels, valid_ratio


def tile_skip_reason(valid_ratio, min_valid_ratio, skip_empty, empty_threshold):
    if min_valid_ratio > 0 and valid_ratio < min_valid_ratio:
        return "valid_ratio_below_threshold"
    if skip_empty and valid_ratio < (1.0 - empty_threshold):
        return "empty_tile"
    return None


def remove_existing_prefix_tiles(tiles_dir, prefix):
    name_prefix = f"{prefix}_x"
    for tile_path in tiles_dir.glob("*.tif"):
        if tile_path.name.startswith(name_prefix):
            tile_path.unlink()


def write_tile(tile_path, tile, profile):
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "tile_large_image.py requires rasterio. Install it with: "
            "conda install -c conda-forge rasterio"
        ) from exc

    write_profile = dict(profile)
    write_profile.update(compress="deflate")
    with rasterio.open(tile_path, mode="w", **write_profile) as dataset:
        dataset.write(np.moveaxis(tile, -1, 0))


def validate_args(args):
    if args.tile_size <= 0:
        raise ValueError("--tile_size must be positive.")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile_size.")
    if not 0.0 <= args.empty_threshold <= 1.0:
        raise ValueError("--empty_threshold must be between 0 and 1.")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min_valid_ratio must be between 0 and 1.")
    if not args.prefix:
        args.prefix = Path(args.input).stem
    if Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        raise ValueError("--prefix must be a file-name-safe value without path separators.")


def run(args):
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tiles_dir = output_dir / "tiles"
    manifest_path = output_dir / "tiles_manifest.json"
    stride = args.tile_size - args.overlap

    existing_tiles = [
        path for path in tiles_dir.glob("*.tif")
        if path.name.startswith(f"{args.prefix}_x")
    ] if tiles_dir.exists() else []
    if not args.overwrite and (manifest_path.exists() or existing_tiles):
        raise FileExistsError(
            f"Output already exists in {output_dir}. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_existing_prefix_tiles(tiles_dir, args.prefix)

    tiles = []
    skipped_tiles = []
    skip_reasons = Counter()
    candidate_valid_ratios = []
    written_valid_ratios = []
    candidate_edge_tile_count = 0
    written_edge_tile_count = 0
    with ImageTileReader(input_path, bands=args.bands) as reader:
        planned_windows = list(
            iter_tile_windows(reader.width, reader.height, args.tile_size, stride)
        )
        if not planned_windows:
            raise RuntimeError("No tile windows were generated.")
        right_coverage = max(x + width for x, _, width, _ in planned_windows)
        bottom_coverage = max(y + height for _, y, _, height in planned_windows)
        if right_coverage != reader.width or bottom_coverage != reader.height:
            raise RuntimeError(
                "Tile grid does not cover the source bounds: "
                f"coverage=({right_coverage}, {bottom_coverage}), "
                f"source=({reader.width}, {reader.height})."
            )

        source_crs = crs_to_string(reader.crs)
        source = {
            "path": str(input_path),
            "image_id": input_path.stem,
            "width": reader.width,
            "height": reader.height,
            "count": reader.count,
            "bands": list(reader.bands),
            "alpha_band": reader.alpha_band,
            "dtype": reader.dtype,
            "crs": source_crs,
            "transform": affine_to_list(reader.transform),
            "nodata": reader.nodata,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": stride,
        }

        print(f"source size: {reader.width} x {reader.height}")
        print(f"dtype: {reader.dtype}")
        print(f"crs: {source_crs}")
        print(f"tile_size: {args.tile_size}")
        print(f"overlap: {args.overlap}")
        print(f"stride: {stride}")

        for x_offset, y_offset, width, height in planned_windows:
            tile_id = f"{args.prefix}_x{x_offset:06d}_y{y_offset:06d}"
            file_name = f"{tile_id}.tif"
            is_edge = width < args.tile_size or height < args.tile_size
            candidate_edge_tile_count += int(is_edge)
            tile, valid_mask = reader.read_tile_with_valid_mask(
                x_offset,
                y_offset,
                width,
                height,
            )
            valid_pixels, total_pixels, valid_ratio = valid_pixel_stats(valid_mask)
            candidate_valid_ratios.append(valid_ratio)
            skip_reason = tile_skip_reason(
                valid_ratio,
                args.min_valid_ratio,
                args.skip_empty,
                args.empty_threshold,
            )
            if skip_reason is not None:
                skip_reasons[skip_reason] += 1
                skipped_tiles.append({
                    "tile_id": tile_id,
                    "file_name": file_name,
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "width": width,
                    "height": height,
                    "is_edge": is_edge,
                    "valid_pixels": valid_pixels,
                    "total_pixels": total_pixels,
                    "valid_ratio": round(valid_ratio, 6),
                    "skipped": True,
                    "skip_reason": skip_reason,
                })
                continue

            tile_path = tiles_dir / file_name
            profile = reader.tile_profile(x_offset, y_offset, width, height)
            write_tile(tile_path, tile, profile)
            written_edge_tile_count += int(is_edge)
            written_valid_ratios.append(valid_ratio)
            tiles.append({
                "tile_id": tile_id,
                "file_name": file_name,
                "path": (Path("tiles") / file_name).as_posix(),
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": width,
                "height": height,
                "is_edge": is_edge,
                "crs": source_crs,
                "transform": affine_to_list(profile["transform"]),
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "valid_ratio": round(valid_ratio, 6),
                "skipped": False,
                "skip_reason": None,
            })

    actual_tiles = [
        path for path in tiles_dir.glob("*.tif")
        if path.name.startswith(f"{args.prefix}_x")
    ]
    if len(actual_tiles) != len(tiles):
        raise RuntimeError(
            f"Manifest has {len(tiles)} tiles but found {len(actual_tiles)} tile files."
        )

    summary = {
        "candidate_tiles": len(planned_windows),
        "written_tiles": len(tiles),
        "skipped_tiles": len(skipped_tiles),
        "candidate_edge_tiles": candidate_edge_tile_count,
        "written_edge_tiles": written_edge_tile_count,
        "min_valid_ratio": args.min_valid_ratio,
        "mean_valid_ratio_candidate": round(float(np.mean(candidate_valid_ratios)), 6),
        "mean_valid_ratio_written": (
            round(float(np.mean(written_valid_ratios)), 6)
            if written_valid_ratios else 0.0
        ),
        "skip_reasons": dict(skip_reasons),
    }
    manifest = {
        "source": source,
        "summary": summary,
        "tiles": tiles,
        "skipped_tiles": skipped_tiles,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    temporary_manifest.replace(manifest_path)

    print(f"candidate tiles: {len(planned_windows)}")
    print(f"written tiles: {len(tiles)}")
    print(f"skipped tiles: {len(skipped_tiles)}")
    print(f"edge tiles: {written_edge_tile_count}")
    print(f"min_valid_ratio: {args.min_valid_ratio}")
    print(f"mean valid_ratio written: {summary['mean_valid_ratio_written']:.6f}")
    if skip_reasons:
        print("top skip reasons:")
        for reason, count in skip_reasons.most_common():
            print(f"  {reason}: {count}")
    print(f"manifest path: {manifest_path}")
    return manifest


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Tiling failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
