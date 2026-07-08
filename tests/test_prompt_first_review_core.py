"""GUI-free tests for Prompt-first session creation, updates, and accept/cancel."""

import tempfile
import unittest
from pathlib import Path

from workflow.prompt_review_session import PromptReviewSession, PromptReviewSessionState
from workflow.tile_manager import TileResult
from workflow.tile_review import next_source_polygon_id, save_review_result


class PromptFirstReviewCoreTest(unittest.TestCase):
    """Verify Prompt-first session behavior without opening Matplotlib."""

    @staticmethod
    def _prediction(positive_points, negative_points):
        """Return one deterministic Prompt adapter payload for test clicks."""
        seed_x = positive_points[0][0]
        seed_y = positive_points[0][1]
        expand = len(positive_points) + len(negative_points)
        return {
            "points_tile": [
                [seed_x - 3, seed_y - 3],
                [seed_x + 10 + expand, seed_y - 3],
                [seed_x + 10 + expand, seed_y + 10 + expand],
                [seed_x - 3, seed_y + 10 + expand],
            ],
            "score": 0.5 + 0.1 * expand,
            "bbox_tile": [seed_x - 5, seed_y - 5, seed_x + 20, seed_y + 20],
            "metadata": {"mock_prompt": True},
        }

    def test_start_positive_negative_and_accept(self):
        """Prompt-first sessions should accumulate clicks and save only the final polygon."""
        session = PromptReviewSession.start(
            tile_id="tile_0014",
            bbox=[10, 10, 40, 40],
            bbox_source="auto_contains",
            selected_auto_polygon_id="tile_0014_poly_0003",
            selected_auto_score=0.91,
            selected_auto_distance=0.0,
        )
        self.assertEqual(session.state, PromptReviewSessionState.RUNNING)

        first = session.add_positive_point(
            (20, 22),
            lambda positive_points, negative_points: self._prediction(
                positive_points,
                negative_points,
            ),
        )
        second = session.add_positive_point(
            (24, 25),
            lambda positive_points, negative_points: self._prediction(
                positive_points,
                negative_points,
            ),
        )
        third = session.add_negative_point(
            (18, 19),
            lambda positive_points, negative_points: self._prediction(
                positive_points,
                negative_points,
            ),
        )
        prompt_points, prompt_labels = session.get_prompt_points_and_labels()

        result = TileResult(
            tile_id="tile_0014",
            offset=[0, 0],
            image_path=Path("tiles/tile_0014.png"),
            polygons=[],
        )
        accepted = session.accept(
            polygon_id=next_source_polygon_id(result, "prompt"),
            source_name="prompt",
        )
        result.polygons.append(accepted)

        self.assertEqual(first.source, "prompt")
        self.assertEqual(second.source, "prompt")
        self.assertEqual(third.source, "prompt")
        self.assertEqual(session.state, PromptReviewSessionState.ACCEPTED)
        self.assertEqual(prompt_points, [[20, 22], [24, 25], [18, 19]])
        self.assertEqual(prompt_labels, [1, 1, 0])
        self.assertEqual(accepted.id, "tile_0014_prompt_0001")
        self.assertEqual(accepted.source, "prompt")
        self.assertEqual(accepted.metadata["created_by"], "prompt_first_review")
        self.assertEqual(accepted.metadata["bbox_source"], "auto_contains")
        self.assertEqual(
            accepted.metadata["selected_auto_polygon_id"],
            "tile_0014_poly_0003",
        )
        self.assertEqual(
            accepted.click_history,
            [
                {
                    "type": "positive",
                    "point": [20.0, 22.0],
                    "label": 1,
                    "mode": "prompt_first",
                    "session_id": session.session_id,
                },
                {
                    "type": "positive",
                    "point": [24.0, 25.0],
                    "label": 1,
                    "mode": "prompt_first",
                    "session_id": session.session_id,
                },
                {
                    "type": "negative",
                    "point": [18.0, 19.0],
                    "label": 0,
                    "mode": "prompt_first",
                    "session_id": session.session_id,
                },
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            saved_path = save_review_result(result, Path(directory))
            restored = TileResult.load(saved_path)

        self.assertEqual(restored.polygons[0].source, "prompt")
        self.assertEqual(restored.polygons[0].points_tile, accepted.points_tile)

    def test_cancel_session_does_not_produce_a_polygon(self):
        """Cancelling a running Prompt-first session should keep TileResult untouched."""
        session = PromptReviewSession.start(
            tile_id="tile_0014",
            bbox=[10, 10, 40, 40],
            bbox_source="fixed_fallback",
        )
        session.add_positive_point(
            (20, 22),
            lambda positive_points, negative_points: self._prediction(
                positive_points,
                negative_points,
            ),
        )
        session.cancel()

        result = TileResult(
            tile_id="tile_0014",
            offset=[0, 0],
            image_path=Path("tiles/tile_0014.png"),
            polygons=[],
        )
        self.assertEqual(session.state, PromptReviewSessionState.CANCELLED)
        self.assertEqual(result.polygons, [])


if __name__ == "__main__":
    unittest.main()
