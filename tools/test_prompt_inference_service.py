import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.models import InferenceRequest, PromptPoint, PromptServiceConfig
from services.prompt_inference_service import PromptInferenceService
from tools.prompt_inference_utils import normalize_prompts, prepare_crop_prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a non-GUI validation of PromptInferenceService."
    )
    parser.add_argument("--imgpth", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="Prompt checkpoint path.")
    parser.add_argument(
        "--config",
        default="configs/prompt_instance_spacenet.json",
        help="Prompt model JSON config path.",
    )
    parser.add_argument("--click_x", required=True, type=float, help="Positive point x coordinate.")
    parser.add_argument("--click_y", required=True, type=float, help="Positive point y coordinate.")
    parser.add_argument("--neg_x", type=float, default=None, help="Optional negative point x coordinate.")
    parser.add_argument("--neg_y", type=float, default=None, help="Optional negative point y coordinate.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU.")
    parser.add_argument("--device", default=None, help="Optional explicit torch device.")
    parser.add_argument("--bbox_mode", choices=["auto", "fixed"], default="fixed")
    parser.add_argument("--bbox_size", type=int, default=256)
    parser.add_argument("--auto_results", default=None)
    parser.add_argument("--auto_image_id", default=None)
    parser.add_argument("--auto_min_score", type=float, default=0.0)
    parser.add_argument("--output_json", default=None, help="Optional response JSON output path.")
    parser.add_argument(
        "--no_bbox_center_prompt",
        action="store_true",
        help="Do not add the selected bbox center as an extra positive prompt.",
    )
    return parser.parse_args()


def build_prompts(args):
    prompts = [PromptPoint(x=args.click_x, y=args.click_y, label=1)]
    if (args.neg_x is None) != (args.neg_y is None):
        raise ValueError("--neg_x and --neg_y must be provided together.")
    if args.neg_x is not None:
        prompts.append(PromptPoint(x=args.neg_x, y=args.neg_y, label=0))
    return prompts


def save_response(path, response):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def assert_raises(expected_exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} from {function.__name__}")


def run_contract_checks():
    assert_raises(ValueError, PromptPoint, x=1, y=2, label=2)
    assert_raises(ValueError, PromptPoint, x=1, y=2, label=0.5)
    assert_raises(ValueError, normalize_prompts, [], 100, 100)
    assert_raises(ValueError, normalize_prompts, [(10, 10, 0)], 100, 100)
    assert_raises(ValueError, normalize_prompts, [(120, 10, 1)], 100, 100)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, _, point_coords, point_labels, _ = prepare_crop_prompts(
        image=image,
        bbox=[20, 20, 80, 80],
        prompts=[(30, 30, 1), (70, 40, 0)],
        include_bbox_center_prompt=True,
        crop_margin=0,
    )
    if point_coords.tolist() != [[30.0, 30.0], [10.0, 10.0], [50.0, 20.0]]:
        raise AssertionError(f"Unexpected crop prompt coordinates: {point_coords.tolist()}")
    if point_labels.tolist() != [1, 1, 0]:
        raise AssertionError(f"Unexpected prompt labels: {point_labels.tolist()}")
    uninitialized = PromptInferenceService(
        PromptServiceConfig(checkpoint="unused", model_config="unused")
    )
    response = uninitialized.infer(
        InferenceRequest(
            image_path="unused",
            prompts=[PromptPoint(x=10, y=10, label=1)],
        )
    )
    if response.success or "not initialized" not in response.message:
        raise AssertionError("Uninitialized service must return a failure response.")
    print("Service contract checks passed.")


def main():
    args = parse_args()
    run_contract_checks()
    service = PromptInferenceService(
        PromptServiceConfig(
            checkpoint=args.checkpoint,
            model_config=args.config,
            gpu=args.gpu,
            device=args.device,
            bbox_size=args.bbox_size,
            bbox_mode=args.bbox_mode,
            auto_min_score=args.auto_min_score,
            include_bbox_center_prompt=not args.no_bbox_center_prompt,
        )
    )

    try:
        prompts = build_prompts(args)
        print("Initializing PromptInferenceService...")
        service.initialize()
        print(f"Service initialized on device: {service.device}")
        request = InferenceRequest(
            image_path=args.imgpth,
            prompts=prompts,
            bbox_mode=args.bbox_mode,
            bbox_size=args.bbox_size,
            auto_results=args.auto_results,
            auto_image_id=args.auto_image_id,
            auto_min_score=args.auto_min_score,
        )
        response = service.infer(request)
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        if args.output_json:
            save_response(args.output_json, response)
            print(f"Response JSON saved to: {args.output_json}")
        return 0 if response.success else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
