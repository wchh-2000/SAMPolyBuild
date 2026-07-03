import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.infer_single_click import make_payload, write_json
from tools.prompt_inference_utils import make_fixed_bbox, select_bbox, validate_polygon


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected}, actual={actual}")


def test_fixed_bbox():
    assert_equal(make_fixed_bbox((500, 400), 1000, 800, 256), [372, 272, 628, 528], "center bbox")
    assert_equal(make_fixed_bbox((5, 8), 1000, 800, 256), [0, 0, 256, 256], "edge bbox")
    assert_raises(ValueError, make_fixed_bbox, (500, 400), 1000, 800, 0)


def test_auto_bbox_and_fallback(temp_dir):
    results_path = Path(temp_dir) / "auto_results.json"
    results = [
        {
            "image_id": "test",
            "bbox": [450, 350, 120, 100],
            "score": 0.91,
            "score_cls": 0.95,
        }
    ]
    results_path.write_text(json.dumps(results), encoding="utf-8")

    bbox, source, info = select_bbox(
        mode="auto",
        click=(500, 400),
        image_width=1000,
        image_height=800,
        bbox_size=256,
        auto_results=str(results_path),
        auto_image_id="test",
        auto_min_score=0.1,
    )
    assert_equal(bbox, [450, 350, 570, 450], "auto bbox")
    assert_equal(source, "auto", "auto bbox source")
    assert_equal(info["contains_click"], True, "auto bbox contains click")

    bbox, source, info = select_bbox(
        mode="auto",
        click=(50, 50),
        image_width=1000,
        image_height=800,
        bbox_size=256,
        auto_results=str(results_path),
        auto_image_id="test",
        auto_min_score=0.1,
    )
    assert_equal(bbox, [0, 0, 256, 256], "fallback bbox")
    assert_equal(source, "fixed", "fallback bbox source")
    assert_equal(info["fallback_used"], True, "fallback flag")


def test_polygon_and_json(temp_dir):
    polygon = validate_polygon(
        [[-1, 2], [20, 3], [30, 40], [10, 60]],
        image_width=50,
        image_height=50,
    )
    assert_equal(
        polygon.tolist(),
        [[0.0, 2.0], [20.0, 3.0], [30.0, 40.0], [10.0, 49.0], [0.0, 2.0]],
        "polygon clip and closure",
    )

    args = Namespace(imgpth="test.png", click_x=10, click_y=20)
    payload = make_payload(
        args,
        success=True,
        message="ok",
        bbox=[0, 0, 50, 50],
        bbox_source="fixed",
        instances=[
            {
                "id": "building_0001",
                "score": 1.0,
                "latest_polygon": polygon.tolist(),
            }
        ],
    )
    output_path = Path(temp_dir) / "result.json"
    write_json(output_path, payload)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    for key in ("success", "image_path", "click", "instances", "message"):
        if key not in loaded:
            raise AssertionError(f"Missing JSON field: {key}")
    if "latest_polygon" not in loaded["instances"][0]:
        raise AssertionError("Missing instances[0].latest_polygon")
    assert_raises(ValueError, validate_polygon, [[0, 0], [1, 1]], 50, 50)


def assert_raises(expected_exception, function, *args):
    try:
        function(*args)
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} from {function.__name__}")


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_fixed_bbox()
        test_auto_bbox_and_fallback(temp_dir)
        test_polygon_and_json(temp_dir)
    print("All headless single-click utility tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
