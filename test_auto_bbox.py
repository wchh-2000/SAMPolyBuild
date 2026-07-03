import argparse
import json

import cv2
import numpy as np

from utils.auto_bbox import AutoBBoxConfig, generate_adaptive_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Test adaptive bbox generation without GUI.")
    parser.add_argument("--image", default=None, help="Optional image path. Uses a synthetic image when omitted.")
    parser.add_argument("--click-x", type=int, default=80, help="Click x coordinate.")
    parser.add_argument("--click-y", type=int, default=80, help="Click y coordinate.")
    parser.add_argument("--debug", action="store_true", help="Include debug mask metadata in output.")
    return parser.parse_args()


def load_or_create_image(path):
    if path is None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 45), (115, 115), (180, 180, 180), thickness=-1)
        cv2.rectangle(image, (55, 55), (105, 105), (220, 220, 220), thickness=-1)
        return image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def summarize_info(info):
    summary = dict(info)
    debug_mask = summary.get("debug_mask")
    if isinstance(debug_mask, np.ndarray):
        summary["debug_mask"] = {
            "shape": list(debug_mask.shape),
            "dtype": str(debug_mask.dtype),
            "nonzero": int(np.count_nonzero(debug_mask)),
        }
    return summary


def assert_bbox_valid(bbox, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < x2 <= width, f"Invalid x range: {bbox}"
    assert 0 <= y1 < y2 <= height, f"Invalid y range: {bbox}"


def main():
    args = parse_args()
    image = load_or_create_image(args.image)
    config = AutoBBoxConfig()
    bbox, info = generate_adaptive_bbox(
        image,
        (args.click_x, args.click_y),
        config=config,
        debug=args.debug,
    )
    assert_bbox_valid(bbox, image.shape)
    print(json.dumps({"bbox": bbox, "info": summarize_info(info)}, indent=2))


if __name__ == "__main__":
    main()
