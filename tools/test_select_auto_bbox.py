import argparse
import json

from utils.auto_result_bbox import infer_image_id, load_and_select_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Select one bbox from auto-mode results by click point.")
    parser.add_argument("--results", required=True, help="Path to infer_auto.py results.json.")
    parser.add_argument("--image", default=None, help="Image path. Used to infer image_id from file stem.")
    parser.add_argument("--image-id", default=None, help="Image id in results.json. Overrides --image.")
    parser.add_argument("--x", type=float, required=True, help="Click x coordinate in original image.")
    parser.add_argument("--y", type=float, required=True, help="Click y coordinate in original image.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score_cls/score.")
    parser.add_argument("--max-center-distance", type=float, default=None,
                        help="Reject nearest bbox when center distance is larger than this value.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_id = args.image_id or infer_image_id(args.image)
    bbox, info = load_and_select_bbox(
        args.results,
        (args.x, args.y),
        image_id=image_id,
        min_score=args.min_score,
        max_center_distance=args.max_center_distance,
    )
    summary = {
        "image_id": info["image_id"],
        "selected_bbox_xyxy": [round(v, 2) for v in bbox],
        "source_bbox_xywh": [round(float(v), 2) for v in info["bbox_xywh"]],
        "contains_click": info["contains_click"],
        "center": [round(v, 2) for v in info["center"]],
        "center_distance": round(info["center_distance"], 2),
        "edge_distance": round(info["edge_distance"], 2),
        "score": round(info["score"], 6),
        "score_cls": round(info["score_cls"], 6),
        "result_index": info["index"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
