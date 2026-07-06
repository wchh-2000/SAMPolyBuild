"""GUI-free tests for click-based polygon additions in tile review."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from review_tiles import is_auto_repeat_key_event, navigation_step_for_key
from workflow.prompt_infer_adapter import make_add_bbox, run_mock_prompt_inference
from workflow.tile_manager import PolygonRecord, TileResult
from workflow.tile_review import add_click_polygon, next_polygon_id, save_review_result


class ReviewTilesAddModeTest(unittest.TestCase):
    """Verify Add-mode state mutations without opening Matplotlib."""

    @staticmethod
    def _polygon(tile_id, sequence):
        return PolygonRecord(
            id=f"{tile_id}_poly_{sequence:04d}",
            tile_id=tile_id,
            status="active",
            source="auto",
            score=0.9,
            points_tile=[[10, 10], [30, 10], [30, 30], [10, 30]],
            bbox_tile=[10, 10, 30, 30],
        )

    @classmethod
    def _result(cls):
        tile_id = "tile_0001"
        return TileResult(
            tile_id=tile_id,
            offset=[100, 200],
            image_path=Path("tiles/tile_0001.tif"),
            polygons=[cls._polygon(tile_id, 1), cls._polygon(tile_id, 3)],
        )

    def test_bbox_generation_clips_to_tile_bounds(self):
        """Centered Add bboxes must retain size without crossing tile edges."""
        self.assertEqual(make_add_bbox((2, 3), 100, 80, 50), [0, 0, 50, 50])
        self.assertEqual(make_add_bbox((98, 78), 100, 80, 50), [50, 30, 100, 80])

    def test_linux_keypad_navigation_mapping(self):
        """Linux keypad 4/6 events must behave like left/right navigation."""
        self.assertEqual(navigation_step_for_key("right"), 1)
        self.assertEqual(navigation_step_for_key("6"), 1)
        self.assertEqual(navigation_step_for_key("kp_6"), 1)
        self.assertEqual(navigation_step_for_key("left"), -1)
        self.assertEqual(navigation_step_for_key("4"), -1)
        self.assertEqual(navigation_step_for_key("kp_4"), -1)
        self.assertIsNone(navigation_step_for_key("d"))

    def test_qt_auto_repeat_detection(self):
        """Native Qt repeat events must be distinguishable from first presses."""
        repeated = SimpleNamespace(
            guiEvent=SimpleNamespace(isAutoRepeat=lambda: True)
        )
        initial = SimpleNamespace(
            guiEvent=SimpleNamespace(isAutoRepeat=lambda: False)
        )
        fallback = SimpleNamespace(guiEvent=None)
        self.assertTrue(is_auto_repeat_key_event(repeated))
        self.assertFalse(is_auto_repeat_key_event(initial))
        self.assertFalse(is_auto_repeat_key_event(fallback))

    def test_mock_prediction_and_click_record_protocol(self):
        """Mock output must append one canonical active click polygon."""
        result = self._result()
        bbox = [0, 0, 100, 80]
        prediction = run_mock_prompt_inference([(25, 30)], [], bbox)
        self.assertEqual(prediction["bbox_tile"], [20.0, 16.0, 80.0, 64.0])

        polygon = add_click_polygon(result, prediction, (25, 30))
        self.assertEqual(polygon.id, "tile_0001_poly_0004")
        self.assertEqual(next_polygon_id(result), "tile_0001_poly_0005")
        self.assertEqual(polygon.source, "click")
        self.assertEqual(polygon.status, "active")
        self.assertEqual(polygon.points_global, [])
        self.assertEqual(
            polygon.click_history,
            [{"type": "positive", "point": [25.0, 30.0], "mode": "add"}],
        )
        self.assertEqual(polygon.metadata["created_by"], "review_tiles_add_mode")
        self.assertTrue(polygon.metadata["mock_prompt"])

    def test_invalid_prediction_does_not_modify_polygons(self):
        """Validation failure must occur before the result list is mutated."""
        result = self._result()
        original_ids = [polygon.id for polygon in result.polygons]
        invalid_prediction = {
            "points_tile": [[1, 1], [2, 2]],
            "score": 0.0,
            "bbox_tile": [0, 0, 10, 10],
        }
        with self.assertRaises(ValueError):
            add_click_polygon(result, invalid_prediction, (5, 5))
        self.assertEqual([polygon.id for polygon in result.polygons], original_ids)

    def test_added_polygon_survives_edited_save_and_reload(self):
        """Click additions must persist through the canonical edited-result path."""
        result = self._result()
        prediction = run_mock_prompt_inference([(50, 50)], [], [0, 0, 100, 100])
        added = add_click_polygon(result, prediction, (50, 50))

        with tempfile.TemporaryDirectory() as directory:
            destination = save_review_result(result, Path(directory))
            loaded = TileResult.load(destination)

        restored = next(item for item in loaded.polygons if item.id == added.id)
        self.assertEqual(restored.source, "click")
        self.assertEqual(restored.status, "active")
        self.assertEqual(restored.points_tile, added.points_tile)
        self.assertEqual(restored.click_history, added.click_history)


if __name__ == "__main__":
    unittest.main()
