import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.image_tile_reader import ImageTileReader
from workflow.manifest import TileManifest, TileManifestEntry, write_json


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
        description="Split a large raster into window-read tiles and versioned manifests."
    )
    parser.add_argument("--input", required=True, help="Input GeoTIFF, TIFF, PNG, or JPEG path.")
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", required=True,
        help="Output directory. Both spellings are kept for backward compatibility.",
    )
    parser.add_argument(
        "--tile-size", "--tile_size", dest="tile_size", type=int, default=1024,
        help="Tile side length in pixels.",
    )
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between adjacent tiles.")
    parser.add_argument("--prefix", default=None, help="Tile name prefix; defaults to input stem.")
    parser.add_argument("--bands", type=parse_bands, default=(1, 2, 3), help="Bands, e.g. 1,2,3.")
    parser.add_argument(
        "--tile-format", "--tile_format", dest="tile_format",
        choices=("tif", "png", "jpg"), default="tif",
        help="Tile output format. GeoTIFF remains the default for backward compatibility.",
    )
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


def bounded_axis_offsets(length, tile_size, stride):
    """Return offsets until the first window that reaches the image boundary."""
    offsets = []
    for offset in axis_offsets(length, stride):
        offsets.append(offset)
        if offset + tile_size >= length:
            break
    return offsets


def iter_tile_windows(width, height, tile_size, stride):
    """Yield a non-redundant row-major grid that fully covers the source image."""
    for y_offset in bounded_axis_offsets(height, tile_size, stride):
        tile_height = min(tile_size, height - y_offset)
        for x_offset in bounded_axis_offsets(width, tile_size, stride):
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


def remove_existing_tiles(tiles_dir, prefix):
    """Remove only tile files owned by the legacy or unified naming schemes."""
    legacy_name_prefix = f"{prefix}_x"
    for tile_path in tiles_dir.iterdir() if tiles_dir.exists() else ():
        if tile_path.is_file() and (
            tile_path.name.startswith(legacy_name_prefix)
            or tile_path.stem.startswith("tile_")
        ) and tile_path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            tile_path.unlink()


def write_tile(tile_path, tile, profile, tile_format):
    """Write one HWC tile while preserving GeoTIFF output compatibility."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "tile_large_image.py requires rasterio. Install it with: "
            "conda install -c conda-forge rasterio"
        ) from exc

    if tile_format == "jpg" and str(tile.dtype) != "uint8":
        raise ValueError("JPEG tile output requires uint8 source bands.")

    if tile_format == "tif":
        write_profile = dict(profile)
        write_profile.update(driver="GTiff", compress="deflate")
    else:
        # PNG/JPEG do not carry the source affine transform. It remains available
        # in the canonical manifest tile metadata for later GIS-aware stages.
        write_profile = {
            "driver": "PNG" if tile_format == "png" else "JPEG",
            "height": profile["height"],
            "width": profile["width"],
            "count": profile["count"],
            "dtype": profile["dtype"],
        }
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


def build_manifest(input_path, args, source, tiles, summary, skipped_tiles):
    """Build the canonical manifest by reusing the workflow data contract."""
    entries = [
        TileManifestEntry(
            tile_id=tile["tile_id"],
            image_path=Path(tile["path"]),
            x_offset=tile["x_offset"],
            y_offset=tile["y_offset"],
            width=tile["width"],
            height=tile["height"],
            is_edge_tile=tile["is_edge"],
            metadata={
                "coordinate_id": tile["coordinate_id"],
                "file_name": tile["file_name"],
                "crs": tile["crs"],
                "transform": tile["transform"],
                "valid_pixels": tile["valid_pixels"],
                "total_pixels": tile["total_pixels"],
                "valid_ratio": tile["valid_ratio"],
            },
        )
        for tile in tiles
    ]
    return TileManifest(
        image_name=input_path.name,
        image_width=source["width"],
        image_height=source["height"],
        tile_size=args.tile_size,
        overlap=args.overlap,
        tiles=entries,
        metadata={
            "source": source,
            "summary": summary,
            "skipped_tiles": skipped_tiles,
        },
    )


def run(args):
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    tiles_dir = output_dir / "tiles"
    manifests_dir = output_dir / "manifests"
    manifest_path = manifests_dir / "manifest.json"
    legacy_manifest_path = output_dir / "tiles_manifest.json"
    stride = args.tile_size - args.overlap
    tile_format = getattr(args, "tile_format", "tif")

    existing_tiles = [
        path for path in tiles_dir.iterdir()
        if path.is_file() and (
            path.name.startswith(f"{args.prefix}_x") or path.stem.startswith("tile_")
        )
    ] if tiles_dir.exists() else []
    if not args.overwrite and (
        manifest_path.exists() or legacy_manifest_path.exists() or existing_tiles
    ):
        raise FileExistsError(
            f"Output already exists in {output_dir}. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_existing_tiles(tiles_dir, args.prefix)

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
            coordinate_id = f"{args.prefix}_x{x_offset:06d}_y{y_offset:06d}"
            is_edge = (
                x_offset + width == reader.width
                or y_offset + height == reader.height
            )
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
                    # Legacy fields retain the coordinate-based id because no
                    # sequential output tile is created for a skipped window.
                    "tile_id": coordinate_id,
                    "coordinate_id": coordinate_id,
                    "file_name": f"{coordinate_id}.{tile_format}",
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

            # Assign ids only after filtering so retained tiles remain continuous
            # while preserving row-major order from iter_tile_windows().
            tile_id = f"tile_{len(tiles) + 1:04d}"
            file_name = f"{tile_id}.{tile_format}"
            tile_path = tiles_dir / file_name
            profile = reader.tile_profile(x_offset, y_offset, width, height)
            write_tile(tile_path, tile, profile, tile_format)
            written_edge_tile_count += int(is_edge)
            written_valid_ratios.append(valid_ratio)
            tiles.append({
                "tile_id": tile_id,
                "coordinate_id": coordinate_id,
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
        path for path in tiles_dir.glob(f"tile_*.{tile_format}")
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
    legacy_manifest = {
        "source": source,
        "summary": summary,
        "tiles": tiles,
        "skipped_tiles": skipped_tiles,
    }
    canonical_manifest = build_manifest(
        input_path,
        args,
        source,
        tiles,
        summary,
        skipped_tiles,
    )
    canonical_manifest.save(manifest_path)
    write_json(legacy_manifest_path, legacy_manifest)

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
    print(f"legacy manifest path: {legacy_manifest_path}")
    return legacy_manifest


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
