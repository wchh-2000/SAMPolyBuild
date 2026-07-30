"""Tests for refined polygon quality gates and primary-component filtering."""

import unittest

import numpy as np

from tools.point_refine_utils import keep_primary_component
from workflow.prompt_review_session import PromptReviewSession
from workflow.refine_polygon_validation import (
    RefinePolygonValidationConfig,
    validate_refined_polygon,
)
from workflow.refine_session import RefineSession
from workflow.tile_manager import PolygonRecord


def _polygon(tile_id="tile_0001", polygon_id="tile_0001_poly_0001", points=None):
    """Create one canonical polygon record for refine-session tests."""
    points = points or [[10, 10], [40, 10], [40, 40], [10, 40]]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return PolygonRecord(
        id=polygon_id,
        tile_id=tile_id,
        status="active",
        source="auto",
        score=0.9,
        points_tile=[[float(x), float(y)] for x, y in points],
        bbox_tile=[min(xs), min(ys), max(xs), max(ys)],
        points_global=[],
        click_history=[],
        metadata={},
    )


class RefinePolygonValidationTest(unittest.TestCase):
    """Verify invalid refined polygons are rejected before they overwrite preview state."""

    def test_keep_primary_component_prefers_positive_component(self):
        """Multiple mask fragments should collapse to the positive-point component only."""
        mask = np.zeros((1, 30, 30), dtype=np.uint8)
        mask[0, 2:8, 2:8] = 1
        mask[0, 15:28, 15:28] = 1
        refined, applied, _ = keep_primary_component(mask, positive_points=[(4, 4)])
        self.assertTrue(applied)
        self.assertEqual(int(refined[0, 4, 4]), 1)
        self.assertEqual(int(refined[0, 20, 20]), 0)

    def test_self_intersecting_polygon_is_rejected(self):
        """Bow-tie polygons should fail self-intersection validation."""
        config = RefinePolygonValidationConfig()
        validation = validate_refined_polygon(
            [[10, 10], [40, 40], [40, 10], [10, 40]],
            [[10, 10], [40, 10], [40, 40], [10, 40]],
            [10, 10, 40, 40],
            config,
        )
        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["reason"], "self_intersection")

    def test_too_many_points_polygon_is_rejected(self):
        """Over-detailed polygons should fail the max point-count threshold."""
        config = RefinePolygonValidationConfig(max_polygon_points=10)
        circle_like = [
            [20 + np.cos(index / 20.0 * np.pi * 2) * 10, 20 + np.sin(index / 20.0 * np.pi * 2) * 10]
            for index in range(25)
        ]
        validation = validate_refined_polygon(
            circle_like,
            [[10, 10], [40, 10], [40, 40], [10, 40]],
            [0, 0, 50, 50],
            config,
        )
        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["reason"], "too_many_points")

    def test_area_change_polygon_is_rejected(self):
        """Area jumps outside the configured ratio window should be rejected."""
        config = RefinePolygonValidationConfig(max_area_change_ratio=1.5, min_area_change_ratio=0.3)
        validation = validate_refined_polygon(
            [[0, 0], [60, 0], [60, 60], [0, 60]],
            [[10, 10], [40, 10], [40, 40], [10, 40]],
            [0, 0, 60, 60],
            config,
        )
        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["reason"], "area_grew_too_much")

    def test_invalid_refine_does_not_overwrite_current_polygon(self):
        """Rejected refine output must keep the previous current_polygon unchanged."""
        original = _polygon()
        session = RefineSession.start(
            original,
            validation_config=RefinePolygonValidationConfig(
                max_polygon_points=80,
                min_polygon_area=50.0,
                max_area_change_ratio=1.5,
                min_area_change_ratio=0.3,
                reject_invalid_refine=True,
            ),
        )
        before_points = [point[:] for point in session.current_polygon.points_tile]
        before_history = len(session.history)

        def invalid_prediction(positive_points, negative_points, **kwargs):
            return {
                "points_tile": [[0, 0], [100, 0], [0, 100], [100, 100]],
                "score": 0.1,
                "bbox_tile": [0, 0, 100, 100],
                "metadata": {},
            }

        result = session.add_negative_point((20, 20), invalid_prediction)
        self.assertEqual(result.points_tile, before_points)
        self.assertEqual(session.current_polygon.points_tile, before_points)
        self.assertEqual(len(session.history), before_history)
        self.assertFalse(session.last_refine_validation["accepted"])

    def test_prompt_session_invalid_refine_does_not_create_new_version(self):
        """Prompt-first sessions should keep the previous preview when refine output is invalid."""
        session = PromptReviewSession.start(
            tile_id="tile_0001",
            bbox=[0, 0, 64, 64],
            bbox_source="fixed_fallback",
            validation_config=RefinePolygonValidationConfig(reject_invalid_refine=True),
        )
        session.add_positive_point(
            (20, 20),
            lambda positive_points, negative_points, **kwargs: {
                "points_tile": [[10, 10], [40, 10], [40, 40], [10, 40]],
                "score": 0.7,
                "bbox_tile": [10, 10, 40, 40],
                "metadata": {},
            },
        )
        first_history = len(session.history)
        first_points = [point[:] for point in session.current_polygon.points_tile]
        session.add_negative_point(
            (18, 18),
            lambda positive_points, negative_points, **kwargs: {
                "points_tile": [[0, 0], [90, 0], [90, 90], [0, 90]],
                "score": 0.2,
                "bbox_tile": [0, 0, 90, 90],
                "metadata": {},
            },
        )
        self.assertEqual(session.current_polygon.points_tile, first_points)
        self.assertEqual(len(session.history), first_history)
        self.assertFalse(session.last_refine_validation["accepted"])


if __name__ == "__main__":
    unittest.main()
