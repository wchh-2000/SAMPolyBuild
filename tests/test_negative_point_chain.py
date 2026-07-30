"""Mock-driven tests for ordered negative-point Prompt review plumbing."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from workflow.prompt_infer_adapter import run_prompt_inference_on_tile
from workflow.prompt_review_session import PromptReviewSession
from workflow.prompt_infer_adapter import PromptSessionCache


class NegativePointChainTest(unittest.TestCase):
    """Verify label ordering, adapter contracts, and debug outputs without a real model."""

    def test_session_preserves_click_order_and_negative_label(self):
        """PromptReviewSession should keep user click order when exporting labels."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 32, 32],
            bbox_source="fixed_fallback",
        )

        session.add_positive_point(
            (5, 5),
            lambda positive_points, negative_points, **kwargs: {
                "points_tile": [[1, 1], [10, 1], [10, 10], [1, 10]],
                "score": 0.5,
                "bbox_tile": [0, 0, 16, 16],
            },
        )
        session.add_negative_point(
            (7, 8),
            lambda positive_points, negative_points, **kwargs: {
                "points_tile": [[1, 1], [12, 1], [12, 12], [1, 12]],
                "score": 0.6,
                "bbox_tile": [0, 0, 16, 16],
            },
        )

        points, labels = session.get_prompt_points_and_labels()
        self.assertEqual(points, [[5, 5], [7, 8]])
        self.assertEqual(labels, [1, 0])

    def test_adapter_receives_ordered_labels_and_writes_debug_json(self):
        """The adapter should preserve label=0 and emit one prompt_points debug JSON."""
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "tile.png"
            image_path.touch()
            predictor = SimpleNamespace(
                set_image_resize=lambda image_crop: None,
                transform=SimpleNamespace(apply_coords=lambda coords, shape: coords),
            )

            with patch("workflow.prompt_infer_adapter._prompt_runtime", return_value=(SimpleNamespace(multi_mask=True, max_distance=10), predictor, "cpu")), \
                    patch("tools.prompt_inference_utils.load_rgb_image", return_value=np.zeros((64, 64, 3), dtype=np.uint8)), \
                    patch(
                        "tools.prompt_inference_utils.prepare_crop_prompts",
                        return_value=(
                            np.zeros((20, 20, 3), dtype=np.uint8),
                            np.asarray([0, 0, 20, 20], dtype=np.float32),
                            np.asarray([[10.0, 10.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
                            np.asarray([1, 1, 0], dtype=np.int32),
                            [0, 0, 1, 1],
                        ),
                    ), \
                    patch(
                        "tools.prompt_inference_utils.infer_polygon_with_prepared_prompts",
                        return_value={
                            "polygon": [[1.0, 1.0], [15.0, 1.0], [15.0, 15.0], [1.0, 15.0]],
                            "score": 0.6,
                            "model_score": 0.7,
                            "mask": np.ones((1, 20, 20), dtype=np.uint8),
                            "point_coords": np.asarray([[10.0, 10.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
                            "point_labels": np.asarray([1, 1, 0], dtype=np.int32),
                            "pred_vmap": np.zeros((1, 1, 1), dtype=np.float32),
                            "pred_voff": np.zeros((1, 1, 1), dtype=np.float32),
                            "pos_transform": [0, 0, 1, 1],
                            "crop_bbox": np.asarray([0, 0, 20, 20], dtype=np.float32),
                            "image_crop": np.zeros((20, 20, 3), dtype=np.uint8),
                            "metadata": {},
                        },
                    ), \
                    patch(
                        "tools.prompt_inference_utils.polygon_from_prediction_mask",
                        return_value=(
                            [[1.0, 1.0], [14.0, 1.0], [14.0, 14.0], [1.0, 14.0]],
                            0.65,
                        ),
                    ):
                output = run_prompt_inference_on_tile(
                    image_path=image_path,
                    positive_points=[(10, 10)],
                    negative_points=[(20, 20)],
                    bbox=[0, 0, 64, 64],
                    config_path=Path("dummy.json"),
                    checkpoint_path=Path("dummy.pth"),
                    device="cpu",
                    session_cache=PromptSessionCache(),
                    prompt_points=[[10, 10], [20, 20]],
                    prompt_labels=[1, 0],
                    debug_prompt_points=True,
                    refine_by_points=True,
                    point_refine_mode="distance",
                    debug_refine_dir=Path(directory) / "debug_refine",
                    tile_id="tile_0001",
                    session_id="session_demo",
                    click_order=[
                        {"type": "positive", "point": [10.0, 10.0], "label": 1},
                        {"type": "negative", "point": [20.0, 20.0], "label": 0},
                    ],
                )
            self.assertTrue(output["success"])
            self.assertEqual(output["metadata"]["negative_point_count"], 1)
            debug_json = Path(output["debug_paths"]["prompt_points_json"]).read_text(encoding="utf-8")
            self.assertIn('"labels": [', debug_json)
            self.assertIn('0', debug_json)

    def test_adapter_rejects_mismatched_prompt_points_and_labels(self):
        """Prompt adapter input should reject mismatched ordered point metadata."""
        predictor = SimpleNamespace(set_image_resize=lambda image_crop: None)
        with patch("workflow.prompt_infer_adapter._prompt_runtime", return_value=(SimpleNamespace(multi_mask=True, max_distance=10), predictor, "cpu")), \
                patch("tools.prompt_inference_utils.load_rgb_image", return_value=np.zeros((16, 16, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                run_prompt_inference_on_tile(
                    image_path=Path("tile.png"),
                    positive_points=[(3, 3)],
                    negative_points=[],
                    bbox=[0, 0, 16, 16],
                    config_path=Path("dummy.json"),
                    checkpoint_path=Path("dummy.pth"),
                    device="cpu",
                    prompt_points=[[3, 3], [5, 5]],
                    prompt_labels=[1],
                )


if __name__ == "__main__":
    unittest.main()
