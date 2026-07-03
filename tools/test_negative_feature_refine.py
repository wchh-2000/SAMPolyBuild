import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.negative_feature_refine import refine_mask_by_negative_features, save_negative_feature_debug


def build_synthetic_case():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (35, 110, 45)
    image[15:60, 15:60] = (170, 60, 55)
    image[15:60, 48:72] = (40, 125, 45)

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[15:60, 15:72] = 1

    positive_points = [(30, 35)]
    negative_points = [(68, 35)]
    return image, mask, positive_points, negative_points


def main():
    parser = argparse.ArgumentParser(description="Non-GUI test for negative feature mask refinement.")
    parser.add_argument("--out_dir", default=None, help="Optional directory for debug images.")
    args = parser.parse_args()

    image, mask, positive_points, negative_points = build_synthetic_case()
    result = refine_mask_by_negative_features(
        image=image,
        mask=mask,
        positive_points=positive_points,
        negative_points=negative_points,
        patch_size=15,
        similarity_thr=0.35,
        protect_radius=8,
        spatial_sigma=80,
    )

    before_count = int(np.count_nonzero(mask))
    after_count = int(np.count_nonzero(result["mask"]))
    protected_kept = int(result["mask"][positive_points[0][1], positive_points[0][0]]) == 1
    negative_removed = int(result["mask"][negative_points[0][1], negative_points[0][0]]) == 0

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        save_negative_feature_debug(
            args.out_dir,
            mask,
            result["mask"],
            result["similarity_map"],
            result["prototypes"],
        )

    print(result["message"])
    print(f"before_pixels={before_count}")
    print(f"after_pixels={after_count}")
    print(f"protected_positive_kept={protected_kept}")
    print(f"negative_region_removed={negative_removed}")

    if not result["applied"]:
        raise AssertionError("Expected negative feature refinement to be applied.")
    if after_count >= before_count:
        raise AssertionError("Expected refined mask to remove some pixels.")
    if not protected_kept:
        raise AssertionError("Expected positive protection area to remain in the mask.")
    if not negative_removed:
        raise AssertionError("Expected negative feature area to be removed.")


if __name__ == "__main__":
    main()
