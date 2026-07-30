"""GUI-free tests for the stage-6 Refine session foundation."""

import unittest
from pathlib import Path

from workflow.prompt_infer_adapter import PromptSessionCache
from workflow.refine_session import RefineSession, RefineSessionState
from workflow.tile_manager import PolygonRecord


class RefineSessionTest(unittest.TestCase):
    """Verify session lifecycle, preview updates, and cache behavior."""

    @staticmethod
    def _polygon(tile_id="tile_0001", polygon_id="tile_0001_poly_0001"):
        """Create one canonical polygon that already belongs to a tile result."""
        return PolygonRecord(
            id=polygon_id,
            tile_id=tile_id,
            status="active",
            source="auto",
            score=0.9,
            points_tile=[[10, 10], [40, 10], [40, 40], [10, 40]],
            bbox_tile=[10, 10, 40, 40],
            points_global=[],
            click_history=[],
            metadata={"raw_index": 0},
        )

    @staticmethod
    def _prediction(points, score):
        """Create one valid adapter-like prediction from accumulated clicks."""
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x1 = min(x_values) - 2
        y1 = min(y_values) - 2
        x2 = max(x_values) + 8
        y2 = max(y_values) + 8
        return {
            "points_tile": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "score": score,
            "bbox_tile": [x1, y1, x2, y2],
            "metadata": {"model_score": score + 0.05},
        }

    def test_start_add_accept_updates_preview_history_and_clicks(self):
        """A running session must keep preview edits in memory until accepted."""
        polygon = self._polygon()
        session = RefineSession.start(polygon)
        callback_calls = []

        def callback(positive_points, negative_points):
            callback_calls.append((list(positive_points), list(negative_points)))
            return self._prediction(positive_points, 0.7 + 0.1 * len(positive_points))

        first = session.add_positive_point((12, 14), callback)
        second = session.add_positive_point((18, 22), callback)
        third = session.add_negative_point((16, 20), callback)
        prompt_points, prompt_labels = session.get_prompt_points_and_labels()
        accepted = session.accept()

        self.assertEqual(session.state, RefineSessionState.ACCEPTED)
        self.assertEqual(
            callback_calls,
            [
                ([(12, 14)], []),
                ([(12, 14), (18, 22)], []),
                ([(12, 14), (18, 22)], [(16, 20)]),
            ],
        )
        self.assertEqual(session.positive_points, [(12, 14), (18, 22)])
        self.assertEqual(session.negative_points, [(16, 20)])
        self.assertEqual(prompt_points, [[12, 14], [18, 22], [16, 20]])
        self.assertEqual(prompt_labels, [1, 1, 0])
        self.assertTrue(session.is_modified)
        self.assertEqual(len(session.history), 3)
        self.assertEqual(first.source, "refine")
        self.assertEqual(second.source, "refine")
        self.assertEqual(third.source, "refine")
        self.assertEqual(accepted.points_tile, third.points_tile)
        self.assertEqual(accepted.click_history[-3]["point"], [12.0, 14.0])
        self.assertEqual(accepted.click_history[-3]["label"], 1)
        self.assertEqual(accepted.click_history[-2]["point"], [18.0, 22.0])
        self.assertEqual(accepted.click_history[-2]["label"], 1)
        self.assertEqual(accepted.click_history[-1]["point"], [16.0, 20.0])
        self.assertEqual(accepted.click_history[-1]["label"], 0)
        self.assertEqual(accepted.metadata["raw_index"], 0)
        self.assertIn("refine_session_id", accepted.metadata)
        self.assertEqual(polygon.source, "auto")
        self.assertEqual(polygon.click_history, [])

    def test_cancel_restores_original_polygon_without_committing_preview(self):
        """Cancel must keep the source polygon untouched and end the session."""
        polygon = self._polygon()
        session = RefineSession.start(polygon)
        session.add_positive_point(
            (15, 16),
            lambda positive_points, negative_points: self._prediction(positive_points, 0.8),
        )

        cancelled = session.cancel()

        self.assertEqual(session.state, RefineSessionState.CANCELLED)
        self.assertEqual(cancelled.points_tile, polygon.points_tile)
        self.assertEqual(cancelled.source, polygon.source)
        self.assertEqual(session.current_polygon.points_tile, polygon.points_tile)

    def test_invalid_prediction_does_not_mutate_running_session(self):
        """Session state should stay atomic when Prompt output is malformed."""
        session = RefineSession.start(self._polygon())

        with self.assertRaises(ValueError):
            session.add_positive_point(
                (20, 20),
                lambda positive_points, negative_points: {"score": 0.1},
            )

        self.assertEqual(session.state, RefineSessionState.RUNNING)
        self.assertEqual(session.positive_points, [])
        self.assertEqual(session.history, [])
        self.assertFalse(session.is_modified)

    def test_prompt_session_cache_reuses_image_and_embedding_keys(self):
        """Cache should avoid repeated image loads and duplicate embedding prep."""
        cache = PromptSessionCache()
        image_path = Path("tile_0001.png")
        loaded_paths = []
        prepared = []

        def loader(path):
            loaded_paths.append(path)
            return {"path": str(path)}

        predictor = object()
        first_image = cache.get_image(image_path, loader)
        second_image = cache.get_image(image_path, loader)
        prepared_first = cache.ensure_embedding(
            predictor,
            ("tile_0001", (0, 0, 32, 32)),
            lambda: prepared.append("prepared"),
        )
        prepared_second = cache.ensure_embedding(
            predictor,
            ("tile_0001", (0, 0, 32, 32)),
            lambda: prepared.append("prepared-again"),
        )
        cache.clear()

        self.assertEqual(first_image, second_image)
        self.assertEqual(len(loaded_paths), 1)
        self.assertTrue(prepared_first)
        self.assertFalse(prepared_second)
        self.assertEqual(prepared, ["prepared"])
        self.assertIsNone(cache.image_path)
        self.assertIsNone(cache.embedding_key)


if __name__ == "__main__":
    unittest.main()
