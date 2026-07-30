"""Core tests for preview-polygon-first negative-point validation."""

import unittest

import numpy as np

from workflow.prompt_review_session import PromptReviewSession
from workflow.refine_session import RefineSession
from workflow.tile_manager import PolygonRecord


def _polygon_record(points_tile, polygon_id="tile_0001_poly_0001") -> PolygonRecord:
    """Create one canonical polygon record with explicit tile-space vertices."""
    x_values = [float(point[0]) for point in points_tile]
    y_values = [float(point[1]) for point in points_tile]
    return PolygonRecord(
        id=polygon_id,
        tile_id="tile_0001",
        status="active",
        source="auto",
        score=0.9,
        points_tile=[[float(x), float(y)] for x, y in points_tile],
        bbox_tile=[
            float(min(x_values)),
            float(min(y_values)),
            float(max(x_values)),
            float(max(y_values)),
        ],
        points_global=[],
        click_history=[],
        metadata={},
    )


class NegativePointValidationTest(unittest.TestCase):
    """Verify negative-point validity against the visible preview polygon."""

    @staticmethod
    def _prediction(points_tile, mask=None, mask_coordinate_space="crop"):
        """Return one adapter-like prediction payload for session updates."""
        x_values = [float(point[0]) for point in points_tile]
        y_values = [float(point[1]) for point in points_tile]
        return {
            "points_tile": [[float(x), float(y)] for x, y in points_tile],
            "score": 0.8,
            "bbox_tile": [
                float(min(x_values)),
                float(min(y_values)),
                float(max(x_values)),
                float(max(y_values)),
            ],
            "mask": mask,
            "metadata": {
                "mock_prompt": True,
                "mask_coordinate_space": mask_coordinate_space,
            },
        }

    def _prompt_session_with_polygon(self, points_tile):
        """Create one prompt session whose current preview polygon is already available."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
        )
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(points_tile),
        )
        return session

    def test_rectangle_inside_point_is_accepted(self):
        """A point clearly inside the cyan preview rectangle should be accepted."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertTrue(session.can_accept_negative_point((25, 25)))

    def test_rectangle_outside_point_is_ignored(self):
        """A point far outside the preview rectangle should be ignored."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertFalse(session.can_accept_negative_point((60, 60)))

    def test_rectangle_boundary_point_is_accepted(self):
        """A point exactly on the preview boundary should be accepted."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertTrue(session.can_accept_negative_point((10, 20)))

    def test_point_three_pixels_outside_boundary_is_accepted_by_tolerance(self):
        """A point just outside the boundary should be accepted within the default tolerance."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertTrue(session.can_accept_negative_point((43, 20), tolerance=5.0))

    def test_point_ten_pixels_outside_boundary_is_ignored(self):
        """A point far outside the boundary should still be ignored with tolerance enabled."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertFalse(session.can_accept_negative_point((50, 20), tolerance=5.0))

    def test_concave_polygon_true_interior_point_is_accepted(self):
        """A point inside a concave polygon body should be accepted."""
        concave = [[10, 10], [40, 10], [40, 20], [20, 20], [20, 40], [10, 40]]
        session = self._prompt_session_with_polygon(concave)
        self.assertTrue(session.can_accept_negative_point((15, 30), tolerance=5.0))

    def test_concave_polygon_notch_point_is_ignored(self):
        """A point in the concave notch outside the polygon should be ignored."""
        concave = [[10, 10], [40, 10], [40, 20], [20, 20], [20, 40], [10, 40]]
        session = self._prompt_session_with_polygon(concave)
        self.assertFalse(session.can_accept_negative_point((30, 30), tolerance=5.0))

    def test_crop_mask_does_not_override_polygon(self):
        """A crop-space mask must not override the visible tile-space preview polygon."""
        polygon = _polygon_record([[10, 10], [40, 10], [40, 40], [10, 40]])
        session = RefineSession.start(polygon)
        crop_mask = np.zeros((1, 16, 16), dtype=np.uint8)
        crop_mask[0, 5, 5] = 1
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points: self._prediction(
                [[10, 10], [40, 10], [40, 40], [10, 40]],
                mask=crop_mask,
                mask_coordinate_space="crop",
            ),
        )
        self.assertTrue(session.can_accept_negative_point((25, 25), tolerance=5.0))
        self.assertFalse(session.last_negative_point_debug["used_mask"])

    def test_allow_outside_negative_points_true_accepts_outside_click(self):
        """The explicit override should allow outside negative points."""
        session = RefineSession.start(_polygon_record([[10, 10], [40, 10], [40, 40], [10, 40]]))
        self.assertTrue(
            session.can_accept_negative_point(
                (60, 60),
                allow_outside_negative_points=True,
            )
        )

    def test_float_and_int_points_are_both_supported(self):
        """Float and integer click coordinates should be normalized consistently."""
        session = self._prompt_session_with_polygon([[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertTrue(session.can_accept_negative_point((25.2, 25.4), tolerance=5.0))
        self.assertTrue(session.can_accept_negative_point((25, 25), tolerance=5.0))


if __name__ == "__main__":
    unittest.main()
