import argparse
import json
import os
import sys
from dataclasses import asdict

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.auto_bbox import AutoBBoxConfig, generate_adaptive_bbox


def parse_args():
    default_cfg = AutoBBoxConfig()
    parser = argparse.ArgumentParser(description="Generate and visualize an adaptive bbox from one click point.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--x", required=True, type=int, help="Click x coordinate.")
    parser.add_argument("--y", required=True, type=int, help="Click y coordinate.")
    parser.add_argument("--out", required=True, help="Output directory for debug artifacts.")
    parser.add_argument("--debug", action="store_true", help="Save debug mask when available.")

    for name, value in asdict(default_cfg).items():
        arg_type = type(value)
        parser.add_argument(f"--{name.replace('_', '-')}", default=value, type=arg_type)
    return parser.parse_args()


def build_config(args):
    keys = asdict(AutoBBoxConfig()).keys()
    return AutoBBoxConfig(**{key: getattr(args, key) for key in keys})


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, f"Unable to read image: {path}"
    return image, ""


def to_bgr_for_overlay(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape for overlay: {image.shape}")


def draw_overlay(image, click_point, bbox, roi):
    overlay = to_bgr_for_overlay(image)
    x, y = click_point
    if roi is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(overlay, "ROI", (rx1, max(0, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, "bbox", (x1, min(overlay.shape[0] - 1, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.drawMarker(overlay, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    return overlay


def save_debug_mask(debug_mask, out_dir):
    if isinstance(debug_mask, np.ndarray):
        mask_path = os.path.join(out_dir, "mask.png")
        cv2.imwrite(mask_path, debug_mask)
        return {
            "path": "mask.png",
            "shape": list(debug_mask.shape),
            "dtype": str(debug_mask.dtype),
            "nonzero": int(np.count_nonzero(debug_mask)),
        }
    return debug_mask


def make_json_ready_info(info, out_dir):
    json_info = dict(info)
    json_info["debug_mask"] = save_debug_mask(json_info.get("debug_mask"), out_dir)
    return json_info


def save_outputs(image, click_point, bbox, info, out_dir, image_error=""):
    os.makedirs(out_dir, exist_ok=True)
    json_info = make_json_ready_info(info, out_dir)
    if image_error:
        json_info["image_error"] = image_error

    if image is not None:
        try:
            overlay = draw_overlay(image, click_point, bbox, info.get("roi"))
            cv2.imwrite(os.path.join(out_dir, "bbox_overlay.png"), overlay)
        except Exception as exc:
            json_info["overlay_error"] = str(exc)

    info_path = os.path.join(out_dir, "info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(json_info, f, indent=2, ensure_ascii=False)


def print_summary(bbox, info):
    print("bbox:", bbox)
    print("method:", info.get("method"))
    print("fallback_used:", info.get("fallback_used"))
    print("message:", info.get("message"))


def main():
    args = parse_args()
    click_point = (args.x, args.y)
    config = build_config(args)
    image, image_error = load_image(args.image)

    bbox, info = generate_adaptive_bbox(image, click_point, config=config, debug=args.debug)
    save_outputs(image, click_point, bbox, info, args.out, image_error=image_error)
    print_summary(bbox, info)

    if image is None:
        print(image_error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
