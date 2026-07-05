"""Generate one canonical Auto ``TileResult`` JSON file per manifest tile."""

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from workflow.auto_infer_adapter import run_auto_inference_on_tiles
from workflow.manifest import TileManifest
from workflow.tile_manager import PolygonRecord, TileResult


def parse_args() -> argparse.Namespace:
    """Parse the canonical stage-2 tile inference command line."""
    parser = argparse.ArgumentParser(
        description="Run existing SAMPoly Auto inference and save one TileResult per tile."
    )
    parser.add_argument("--manifest", required=True, help="Canonical manifest.json path.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory; defaults to auto_results beside the manifest workspace.",
    )
    parser.add_argument("--config", required=True, help="Existing Auto model config path.")
    parser.add_argument("--checkpoint", required=True, help="Existing Auto checkpoint path.")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N.")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_dir(manifest_path: Path) -> Path:
    """Resolve ``auto_results`` from either the canonical or a direct manifest path."""
    manifest_parent = manifest_path.parent
    workspace = manifest_parent.parent if manifest_parent.name == "manifests" else manifest_parent
    return workspace / "auto_results"


def resolve_tile_path(manifest_path: Path, image_path: Path) -> Path:
    """Resolve an output-root-relative tile path without allowing path traversal."""
    relative_path = Path(image_path)
    if relative_path.is_absolute():
        raise ValueError(f"Manifest image_path must be relative: {relative_path}")
    workspace = manifest_path.parent.parent if manifest_path.parent.name == "manifests" \
        else manifest_path.parent
    workspace = workspace.resolve()
    resolved = (workspace / relative_path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Manifest image_path escapes its workspace: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Tile image not found: {resolved}")
    return resolved


def _flattened_ring(segmentation: Any) -> list[float]:
    """Extract one flattened ring from the existing COCO polygon representation."""
    candidate = segmentation
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
        candidate = max(candidate, key=len)
    if not isinstance(candidate, list) or len(candidate) < 6 or len(candidate) % 2:
        raise ValueError("Auto segmentation must contain at least three x/y pairs.")
    try:
        values = [float(value) for value in candidate]
    except (TypeError, ValueError) as exc:
        raise ValueError("Auto segmentation coordinates must be numeric.") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Auto segmentation coordinates must be finite.")
    return values


def _points_in_tile(segmentation: Any, width: int, height: int) -> list[list[float]]:
    """Convert a COCO ring to bounded tile-local point pairs."""
    values = _flattened_ring(segmentation)
    points = []
    for index in range(0, len(values), 2):
        x, y = values[index], values[index + 1]
        if x < -1.0 or y < -1.0 or x > width + 1.0 or y > height + 1.0:
            raise ValueError(
                f"Auto polygon coordinate ({x}, {y}) exceeds tile bounds ({width}, {height})."
            )
        points.append([min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))])
    return points


def _bbox_xyxy(
    raw_bbox: Any,
    points: list[list[float]],
    tile_width: int,
    tile_height: int,
) -> list[float]:
    """Convert the existing COCO xywh bbox, or derive one from polygon points."""
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            x, y, width, height = [float(value) for value in raw_bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError("Auto bbox values must be numeric.") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("Auto bbox values must be finite.")
        if width < 0 or height < 0:
            raise ValueError("Auto bbox width and height must be non-negative.")
        bbox = [x, y, x + width, y + height]
        if (
            bbox[0] < -1.0 or bbox[1] < -1.0
            or bbox[2] > tile_width + 1.0 or bbox[3] > tile_height + 1.0
        ):
            raise ValueError(
                f"Auto bbox {bbox} exceeds tile bounds ({tile_width}, {tile_height})."
            )
        return [
            min(max(bbox[0], 0.0), float(tile_width)),
            min(max(bbox[1], 0.0), float(tile_height)),
            min(max(bbox[2], 0.0), float(tile_width)),
            min(max(bbox[3], 0.0), float(tile_height)),
        ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def convert_raw_polygons(
    tile,
    raw_results: list[dict],
    score_threshold: float,
) -> list[PolygonRecord]:
    """Convert one tile's existing COCO records into canonical PolygonRecords."""
    polygons = []
    for raw_index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            raise ValueError(f"Raw Auto result {raw_index} must be an object.")
        threshold_score = float(raw.get("score_cls", raw.get("score", 0.0)))
        if threshold_score < score_threshold:
            continue
        points = _points_in_tile(raw.get("segmentation"), tile.width, tile.height)
        polygon_id = f"{tile.tile_id}_poly_{len(polygons) + 1:04d}"
        polygons.append(
            PolygonRecord(
                id=polygon_id,
                tile_id=tile.tile_id,
                status="active",
                source="auto",
                score=float(raw.get("score", threshold_score)),
                points_tile=points,
                bbox_tile=_bbox_xyxy(raw.get("bbox"), points, tile.width, tile.height),
                points_global=[],
                click_history=[],
                metadata={
                    "raw_index": raw_index,
                    "score_cls": raw.get("score_cls"),
                    "category_id": raw.get("category_id"),
                    "area": raw.get("area"),
                },
            )
        )
    return polygons


def save_tile_result(tile, output_path: Path, raw_results: list[dict], score_threshold: float) -> int:
    """Convert and save one successful tile result, returning polygon count."""
    polygons = convert_raw_polygons(tile, raw_results, score_threshold)
    TileResult(
        tile_id=tile.tile_id,
        offset=[tile.x_offset, tile.y_offset],
        image_path=tile.image_path,
        polygons=polygons,
    ).save(output_path)
    return len(polygons)


def save_tile_error(tile, output_path: Path, error: Exception) -> None:
    """Persist a failed tile without hiding the concrete exception type or message."""
    TileResult(
        tile_id=tile.tile_id,
        offset=[tile.x_offset, tile.y_offset],
        image_path=tile.image_path,
        polygons=[],
        error=f"{type(error).__name__}: {error}",
    ).save(output_path)


def group_raw_results(raw_results: list[dict], tile_ids: set[str]) -> dict[str, list[dict]]:
    """Group raw COCO records by tile id and reject unexpected model output ids."""
    grouped = {tile_id: [] for tile_id in tile_ids}
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            raise ValueError(f"Raw Auto result {index} must be an object.")
        tile_id = str(raw.get("image_id", ""))
        if tile_id not in grouped:
            raise ValueError(f"Unexpected Auto result image_id: {tile_id!r}")
        grouped[tile_id].append(raw)
    return grouped


def run(
    args: argparse.Namespace,
    infer_batch: Callable[..., list[dict]] = run_auto_inference_on_tiles,
) -> dict[str, int]:
    """Run batch Auto inference, split results, and isolate fallback failures per tile."""
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    for path, label in (
        (manifest_path, "manifest"),
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if not 0.0 <= float(args.score_threshold) <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")

    manifest = TileManifest.load(manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve() \
        if args.output_dir else default_output_dir(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = 0
    for index, tile in enumerate(manifest.tiles, start=1):
        output_path = output_dir / f"{tile.tile_id}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(manifest.tiles)}] skip {tile.tile_id} (result exists)")
            continue
        tile_path = resolve_tile_path(manifest_path, tile.image_path)
        pending.append((index, tile, tile_path, output_path))
        print(f"[{index}/{len(manifest.tiles)}] infer {tile.tile_id} ...")

    summary = {
        "total_tiles": len(manifest.tiles),
        "success_tiles": 0,
        "failed_tiles": 0,
        "skipped_tiles": skipped,
        "total_polygons": 0,
    }
    if not pending:
        print_summary(summary)
        return summary

    infer_kwargs = {
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "device": args.device,
        "score_threshold": float(args.score_threshold),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }
    try:
        raw_results = infer_batch([item[2] for item in pending], **infer_kwargs)
        grouped = group_raw_results(raw_results, {item[1].tile_id for item in pending})
        for _, tile, _, output_path in pending:
            try:
                count = save_tile_result(
                    tile, output_path, grouped[tile.tile_id], float(args.score_threshold)
                )
                summary["success_tiles"] += 1
                summary["total_polygons"] += count
            except Exception as exc:
                save_tile_error(tile, output_path, exc)
                summary["failed_tiles"] += 1
    except Exception as batch_error:
        print(f"Batch inference failed; retrying per tile: {type(batch_error).__name__}: {batch_error}")
        for _, tile, tile_path, output_path in pending:
            try:
                raw_results = infer_batch([tile_path], **infer_kwargs)
                grouped = group_raw_results(raw_results, {tile.tile_id})
                count = save_tile_result(
                    tile, output_path, grouped[tile.tile_id], float(args.score_threshold)
                )
                summary["success_tiles"] += 1
                summary["total_polygons"] += count
            except Exception as exc:
                save_tile_error(tile, output_path, exc)
                summary["failed_tiles"] += 1

    print_summary(summary)
    return summary


def print_summary(summary: dict[str, int]) -> None:
    """Print the stable stage summary expected by operators and tests."""
    print("Tile Auto inference summary:")
    for key in (
        "total_tiles", "success_tiles", "failed_tiles", "skipped_tiles", "total_polygons"
    ):
        print(f"  {key}: {summary[key]}")


def main() -> int:
    """CLI entry point with one explicit top-level failure message."""
    try:
        run(parse_args())
    except Exception as exc:
        print(f"Tile Auto inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
