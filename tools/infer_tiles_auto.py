import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run existing SAMPoly auto inference on tiles selected by a manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to tiles_manifest.json.")
    parser.add_argument("--config", default="configs/auto_whumix.py", help="Auto model config.")
    parser.add_argument("--ckpt_path", default="auto_whumix.pth", help="Auto model checkpoint.")
    parser.add_argument("--work_dir", required=True, help="Dedicated output directory for this run.")
    parser.add_argument("--score_thr", type=float, default=0.1, help="Detection score threshold.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--batch_size", type=int, default=2, help="Tile inference batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Predict dataloader workers.")
    parser.add_argument("--tile_limit", type=int, default=None, help="Infer only the first N tiles.")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs owned by this tool.")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_args(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.ckpt_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    for path, label in (
        (manifest_path, "manifest"),
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.tile_limit is not None and args.tile_limit <= 0:
        raise ValueError("--tile_limit must be positive when provided.")
    if work_dir == Path.cwd().resolve() or is_relative_to(manifest_path, work_dir):
        raise ValueError("--work_dir must be a dedicated output directory outside the tile source.")
    return manifest_path, config_path, checkpoint_path, work_dir


def load_selected_tiles(manifest_path, tile_limit=None):
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Tile manifest must be a JSON object.")
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Tile manifest must contain at least one retained tile in 'tiles'.")
    selected_tiles = tiles[:tile_limit] if tile_limit is not None else tiles

    manifest_dir = manifest_path.parent.resolve()
    seen_ids = set()
    resolved_tiles = []
    for index, tile in enumerate(selected_tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"Tile entry {index} must be an object.")
        tile_id = str(tile.get("tile_id", "")).strip()
        relative_path = tile.get("path")
        if not tile_id or not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Tile entry {index} must contain tile_id and path.")
        if tile_id in seen_ids:
            raise ValueError(f"Duplicate tile_id in manifest: {tile_id}")
        path_value = Path(relative_path)
        if path_value.is_absolute():
            raise ValueError(f"Tile path must be relative to the manifest: {relative_path}")
        source_path = (manifest_dir / path_value).resolve()
        if not is_relative_to(source_path, manifest_dir):
            raise ValueError(f"Tile path escapes the manifest directory: {relative_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Tile file not found: {source_path}")
        if source_path.stem != tile_id:
            raise ValueError(
                f"Tile image_id mismatch: tile_id={tile_id}, file stem={source_path.stem}"
            )
        seen_ids.add(tile_id)
        resolved_tiles.append({"tile": tile, "source_path": source_path})
    return manifest, resolved_tiles


def prepare_work_dir(work_dir, overwrite):
    owned_paths = (
        "input_tiles",
        "_runtime_auto",
        "auto_results",
        "logs",
        "input_manifest.json",
        "run_config.json",
    )
    if work_dir.exists() and not overwrite:
        raise FileExistsError(f"Work directory already exists: {work_dir}. Use --overwrite.")
    work_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in owned_paths:
            path = work_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def stage_tiles(resolved_tiles, input_dir):
    input_dir.mkdir(parents=True, exist_ok=False)
    link_count = 0
    copy_count = 0
    for item in resolved_tiles:
        source_path = item["source_path"]
        target_path = input_dir / source_path.name
        try:
            os.link(source_path, target_path)
            link_count += 1
        except OSError:
            shutil.copy2(source_path, target_path)
            copy_count += 1
    return {"hard_links": link_count, "copies": copy_count}


def archive_result(raw_path, output_path):
    raw_path = Path(raw_path)
    if raw_path.is_file():
        shutil.copy2(raw_path, output_path)
    else:
        write_json(output_path, [])


def validate_results(results_path, requested_tile_ids):
    results = load_json(results_path)
    if not isinstance(results, list):
        raise ValueError("Auto results.json must contain a list.")
    result_image_ids = set()
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Auto result entry {index} must be an object.")
        image_id = str(item.get("image_id", ""))
        if image_id not in requested_tile_ids:
            raise ValueError(f"Unexpected result image_id: {image_id}")
        result_image_ids.add(image_id)
    return results, result_image_ids


def run(args):
    manifest_path, config_path, checkpoint_path, work_dir = validate_args(args)
    manifest, resolved_tiles = load_selected_tiles(manifest_path, args.tile_limit)
    suffixes = {item["source_path"].suffix for item in resolved_tiles}
    if len(suffixes) != 1:
        raise ValueError(f"Selected tiles must share one image suffix, got {sorted(suffixes)}")
    img_suffix = suffixes.pop()
    prepare_work_dir(work_dir, args.overwrite)

    input_dir = work_dir / "input_tiles"
    runtime_dir = work_dir / "_runtime_auto"
    auto_results_dir = work_dir / "auto_results"
    logs_dir = work_dir / "logs"
    input_manifest_path = work_dir / "input_manifest.json"
    run_config_path = work_dir / "run_config.json"
    final_results_path = auto_results_dir / "results.json"
    final_results_mask_path = auto_results_dir / "results_mask.json"
    auto_results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    staging = stage_tiles(resolved_tiles, input_dir)
    requested_tile_ids = {item["tile"]["tile_id"] for item in resolved_tiles}

    selected_manifest = {
        "source_manifest": str(manifest_path),
        "source": manifest.get("source"),
        "summary": manifest.get("summary"),
        "tiles": [item["tile"] for item in resolved_tiles],
    }
    write_json(input_manifest_path, selected_manifest)
    run_config = {
        "status": "running",
        "started_at": utc_now(),
        "manifest": str(manifest_path),
        "config": str(config_path),
        "ckpt_path": str(checkpoint_path),
        "work_dir": str(work_dir),
        "score_thr": args.score_thr,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "tile_limit": args.tile_limit,
        "tile_count_requested": len(resolved_tiles),
        "img_suffix": img_suffix,
        "staging": staging,
    }
    write_json(run_config_path, run_config)

    print(f"tile count requested: {len(resolved_tiles)}")
    print(f"staged tiles: hard_links={staging['hard_links']}, copies={staging['copies']}")
    print("Starting SAMPoly auto inference...")
    try:
        from infer_auto import run_auto_inference

        artifacts = run_auto_inference(
            config=str(config_path),
            ckpt_path=str(checkpoint_path),
            img_dir=str(input_dir),
            work_dir=str(work_dir),
            img_suffix=img_suffix,
            score_thr=args.score_thr,
            gpu=args.gpu,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            status="predict",
            result_dir=runtime_dir,
            replace_predict_loader=True,
        )
        archive_result(artifacts["results_path"], final_results_path)
        archive_result(artifacts["results_mask_path"], final_results_mask_path)
        results, result_image_ids = validate_results(final_results_path, requested_tile_ids)
    except Exception as exc:
        run_config.update({
            "status": "failed",
            "finished_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        write_json(run_config_path, run_config)
        raise

    run_summary = {
        "tile_count_requested": len(resolved_tiles),
        "tile_count_inferred": len(resolved_tiles),
        "tiles_with_detections": len(result_image_ids),
        "detection_count": len(results),
        "results_path": str(final_results_path),
        "results_mask_path": str(final_results_mask_path),
    }
    run_config.update({
        "status": "completed",
        "finished_at": utc_now(),
        "summary": run_summary,
    })
    write_json(run_config_path, run_config)
    write_json(logs_dir / "inference_summary.json", run_summary)
    shutil.rmtree(runtime_dir, ignore_errors=True)

    print(f"tile count inferred: {len(resolved_tiles)}")
    print(f"tiles with detections: {len(result_image_ids)}")
    print(f"detection count: {len(results)}")
    print(f"results path: {final_results_path}")
    return run_summary


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Tile auto inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
