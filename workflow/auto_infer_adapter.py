"""Thin adapter around the existing SAMPoly Auto inference entry point.

No model construction or prediction logic is implemented here. The adapter only
stages selected tile files, calls :func:`infer_auto.run_auto_inference`, and
returns its existing COCO-style polygon records.
"""

import tempfile
from pathlib import Path

from tools.infer_tiles_auto import stage_tiles
from workflow.manifest import read_json


def normalize_device(device: str) -> tuple[str, int]:
    """Normalize ``cpu``, ``cuda``, or ``cuda:N`` into runner arguments."""
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu", 0
    if normalized == "cuda":
        return "cuda", 0
    if normalized.startswith("cuda:"):
        try:
            gpu_index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("--device must be cpu, cuda, or cuda:N.") from exc
        if gpu_index < 0:
            raise ValueError("CUDA device index must be non-negative.")
        return "cuda", gpu_index
    raise ValueError("--device must be cpu, cuda, or cuda:N.")


def run_auto_inference_on_tiles(
    tile_paths: list[Path],
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    score_threshold: float = 0.5,
    batch_size: int = 2,
    num_workers: int = 1,
) -> list[dict]:
    """Run the existing Auto pipeline for one or more same-suffix tile images."""
    normalized_paths = [Path(path).expanduser().resolve() for path in tile_paths]
    if not normalized_paths:
        return []
    missing = [path for path in normalized_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tile image not found: {missing[0]}")
    suffixes = {path.suffix.lower() for path in normalized_paths}
    if len(suffixes) != 1:
        raise ValueError(f"Auto inference tiles must share one suffix: {sorted(suffixes)}")
    runner_device, gpu_index = normalize_device(device)

    with tempfile.TemporaryDirectory(prefix="sampoly_tile_auto_") as directory:
        runtime_root = Path(directory)
        input_dir = runtime_root / "input_tiles"
        stage_tiles(
            [{"source_path": path} for path in normalized_paths],
            input_dir,
        )

        # Lazy import keeps mock tests independent from torch/mmengine and makes
        # CLI validation available even when the model environment is absent.
        from infer_auto import run_auto_inference

        artifacts = run_auto_inference(
            config=str(Path(config_path).expanduser().resolve()),
            ckpt_path=str(Path(checkpoint_path).expanduser().resolve()),
            img_dir=str(input_dir),
            work_dir=str(runtime_root / "work"),
            img_suffix=suffixes.pop(),
            score_thr=float(score_threshold),
            gpu=gpu_index,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            status="predict",
            result_dir=runtime_root / "results",
            replace_predict_loader=True,
            device=runner_device,
        )
        results_path = Path(artifacts["results_path"])
        if not results_path.is_file():
            # CocoMetric does not create results.json when every tile has zero
            # polygons, so absence represents a valid empty prediction set.
            return []
        results = read_json(results_path)
        if not isinstance(results, list):
            raise ValueError("Existing Auto inference results must be a JSON list.")
        return results
