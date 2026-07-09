"""Core tests for ignoring outside negative points during Prompt review sessions."""

import unittest

import numpy as np

from workflow.prompt_review_session import PromptReviewSession
from workflow.refine_session import RefineSession
from workflow.tile_manager import PolygonRecord


class NegativePointValidationTest(unittest.TestCase):
    """Verify negative-point validity rules without opening any GUI."""

    @staticmethod
    def _polygon() -> PolygonRecord:
        """Create one simple square polygon in tile coordinates."""
        return PolygonRecord(
            id="tile_0001_poly_0001",
            tile_id="tile_0001",
            status="active",
            source="auto",
            score=0.9,
            points_tile=[[10, 10], [40, 10], [40, 40], [10, 40]],
            bbox_tile=[10, 10, 40, 40],
            points_global=[],
            click_history=[],
            metadata={},
        )

    @staticmethod
    def _prediction(mask=None, polygon_points=None):
        """Return one adapter-like prediction payload with optional mask override."""
        return {
            "points_tile": polygon_points or [[10, 10], [40, 10], [40, 40], [10, 40]],
            "score": 0.8,
            "bbox_tile": [10, 10, 40, 40],
            "mask": mask,
            "metadata": {"mock_prompt": True},
        }

    def test_prompt_negative_point_inside_polygon_is_allowed(self):
        """A negative point inside the preview polygon should be accepted."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
        )
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(),
        )

        self.assertTrue(session.can_accept_negative_point((25, 25)))

    def test_prompt_negative_point_outside_polygon_is_rejected(self):
        """A negative point outside the preview polygon should be ignored by default."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
        )
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(),
        )
        before_polygon = [point[:] for point in session.current_polygon.points_tile]
        before_negative_count = len(session.negative_points)

        self.assertFalse(session.can_accept_negative_point((50, 50)))
        self.assertEqual(len(session.negative_points), before_negative_count)
        self.assertEqual(session.current_polygon.points_tile, before_polygon)

    def test_prompt_negative_point_on_polygon_boundary_is_allowed(self):
        """A negative point on the polygon boundary should count as valid."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
        )
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(),
        )

        self.assertTrue(session.can_accept_negative_point((10, 20)))

    def test_prompt_negative_point_without_current_polygon_is_rejected(self):
        """A session with no preview polygon should ignore right-click negatives."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
        )

        self.assertFalse(session.can_accept_negative_point((12, 12)))

    def test_refine_session_uses_current_mask_before_polygon(self):
        """When a current mask exists, mask membership should override polygon geometry."""
        session = RefineSession.start(self._polygon())
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        mask[0, 55, 55] = 1
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(
                mask=mask,
                polygon_points=[[10, 10], [40, 10], [40, 40], [10, 40]],
            ),
        )

        self.assertTrue(session.can_accept_negative_point((55, 55)))
        self.assertFalse(session.can_accept_negative_point((45, 45)))

    def test_allow_outside_negative_points_flag_overrides_default_validation(self):
        """The escape hatch should allow outside negative points when explicitly enabled."""
        session = RefineSession.start(self._polygon())

        self.assertTrue(
            session.can_accept_negative_point(
                (60, 60),
                allow_outside_negative_points=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
