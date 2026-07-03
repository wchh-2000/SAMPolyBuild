import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.prompt_inference_utils import (
    DEFAULT_PROMPT_CHECKPOINT,
    DEFAULT_PROMPT_CONFIG,
    build_prompt_predictor,
    infer_single_polygon,
    load_prompt_settings,
    load_rgb_image,
    select_bbox,
    validate_click,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one headless SAMPoly prompt inference from an image click."
    )
    parser.add_argument("--imgpth", required=True, help="Input image path.")
    parser.add_argument("--click_x", required=True, type=float, help="Click x coordinate in image pixels.")
    parser.add_argument("--click_y", required=True, type=float, help="Click y coordinate in image pixels.")
    parser.add_argument("--work_dir", default="work_dir", help="Working directory.")
    parser.add_argument("--output_json", required=True, help="Output JSON path.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_PROMPT_CHECKPOINT,
        help="Prompt model checkpoint path.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PROMPT_CONFIG,
        help="Prompt model JSON config path.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU.")
    parser.add_argument(
        "--bbox_mode",
        choices=["fixed", "auto"],
        default="auto",
        help="BBox source. Auto falls back to fixed when no containing detection is available.",
    )
    parser.add_argument("--bbox_size", type=int, default=256, help="Fixed fallback bbox side length.")
    parser.add_argument("--auto_results", default=None, help="Optional auto-mode results.json path.")
    parser.add_argument("--auto_image_id", default=None, help="Optional image id in auto results.")
    parser.add_argument("--auto_min_score", type=float, default=0.0, help="Minimum auto bbox score.")
    return parser.parse_args()


def make_payload(args, success, message, bbox=None, bbox_source=None, instances=None, bbox_info=None):
    payload = {
        "success": bool(success),
        "image_path": str(args.imgpth),
        "click": {"x": float(args.click_x), "y": float(args.click_y)},
        "instances": instances or [],
        "message": str(message),
    }
    if bbox is not None:
        payload["bbox"] = [int(value) for value in bbox]
    if bbox_source is not None:
        payload["bbox_source"] = bbox_source
    if bbox_info is not None:
        payload["bbox_info"] = json_safe(bbox_info)
    return payload


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != "raw"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run(args):
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Loading image: {args.imgpth}")
    image = load_rgb_image(args.imgpth)
    image_height, image_width = image.shape[:2]
    click = validate_click((args.click_x, args.click_y), image_width, image_height)

    print(f"[2/5] Selecting bbox using mode={args.bbox_mode}")
    bbox, bbox_source, bbox_info = select_bbox(
        mode=args.bbox_mode,
        click=click,
        image_width=image_width,
        image_height=image_height,
        bbox_size=args.bbox_size,
        auto_results=args.auto_results,
        auto_image_id=args.auto_image_id,
        auto_min_score=args.auto_min_score,
        image_path=args.imgpth,
    )
    if bbox_info.get("fallback_used"):
        print(f"Auto bbox fallback: {bbox_info.get('fallback_reason')}")
    print(f"Selected bbox: {bbox} ({bbox_source})")

    print(f"[3/5] Loading prompt config and checkpoint: {args.checkpoint}")
    settings = load_prompt_settings(args.config, checkpoint=args.checkpoint)
    predictor, device = build_prompt_predictor(settings, gpu=args.gpu)
    print(f"Model ready on {device}")

    print("[4/5] Running SAMPoly prompt inference")
    prediction = infer_single_polygon(
        predictor,
        image,
        bbox,
        click,
        multi_mask=bool(getattr(settings, "multi_mask", True)),
        max_distance=int(getattr(settings, "max_distance", 10)),
    )
    polygon = prediction["polygon"]
    instance = {
        "id": "building_0001",
        "score": prediction["score"],
        "model_score": prediction["model_score"],
        "latest_polygon": [[float(x), float(y)] for x, y in polygon],
    }

    print(f"[5/5] Writing polygon JSON: {args.output_json}")
    payload = make_payload(
        args,
        success=True,
        message="ok",
        bbox=bbox,
        bbox_source=bbox_source,
        instances=[instance],
        bbox_info=bbox_info,
    )
    write_json(args.output_json, payload)
    print(f"Inference completed: polygon_points={len(polygon)}")
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        try:
            payload = make_payload(args, success=False, message=message)
            write_json(args.output_json, payload)
            print(f"Failure JSON written to: {args.output_json}", file=sys.stderr)
        except Exception as write_exc:
            print(f"Unable to write failure JSON: {write_exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
