"""GUI-free tests for tile-local Auto bbox selection in Prompt-first review."""

import unittest

from workflow.auto_bbox_provider import select_bbox_for_click
from workflow.tile_manager import PolygonRecord


class AutoBBoxProviderTest(unittest.TestCase):
    """Verify Prompt-first bbox selection rules against canonical polygons."""

    @staticmethod
    def _polygon(
        polygon_id: str,
        *,
        score: float = 0.9,
        status: str = "active",
        bbox_tile=None,
        points_tile=None,
    ):
        """Create one canonical Auto polygon for bbox selection tests."""
        return PolygonRecord(
            id=polygon_id,
            tile_id="tile_0001",
            status=status,
            source="auto",
            score=score,
            points_tile=points_tile or [[10, 10], [40, 10], [40, 40], [10, 40]],
            bbox_tile=bbox_tile or [10, 10, 40, 40],
            points_global=[],
            click_history=[],
            metadata={},
        )

    def test_selects_containing_bbox(self):
        """A click inside one bbox should return the containing Auto candidate."""
        selected = select_bbox_for_click(
            polygons=[self._polygon("tile_0001_poly_0001")],
            click=(20, 20),
            nearest_threshold=80,
            tile_width=100,
            tile_height=100,
        )
        self.assertEqual(selected["source"], "auto_contains")
        self.assertEqual(selected["polygon_id"], "tile_0001_poly_0001")
        self.assertEqual(selected["bbox"], [10.0, 10.0, 40.0, 40.0])

    def test_multiple_contains_prefer_score_then_smaller_area(self):
        """Containing ties should prefer higher score and then smaller area."""
        candidates = [
            self._polygon("tile_0001_poly_0001", score=0.8, bbox_tile=[0, 0, 80, 80]),
            self._polygon("tile_0001_poly_0002", score=0.95, bbox_tile=[0, 0, 90, 90]),
            self._polygon("tile_0001_poly_0003", score=0.95, bbox_tile=[5, 5, 50, 50]),
        ]
        selected = select_bbox_for_click(
            polygons=candidates,
            click=(20, 20),
            nearest_threshold=80,
            tile_width=100,
            tile_height=100,
        )
        self.assertEqual(selected["polygon_id"], "tile_0001_poly_0003")
        self.assertEqual(selected["source"], "auto_contains")

    def test_selects_nearest_bbox_when_click_is_outside(self):
        """The nearest active bbox within threshold should be selected."""
        selected = select_bbox_for_click(
            polygons=[
                self._polygon("tile_0001_poly_0001", bbox_tile=[10, 10, 20, 20]),
                self._polygon("tile_0001_poly_0002", bbox_tile=[60, 60, 80, 80]),
            ],
            click=(25, 18),
            nearest_threshold=10,
            tile_width=100,
            tile_height=100,
        )
        self.assertEqual(selected["source"], "auto_nearest")
        self.assertEqual(selected["polygon_id"], "tile_0001_poly_0001")

    def test_returns_none_when_nearest_bbox_is_too_far(self):
        """No candidate should be returned beyond the nearest-threshold."""
        selected = select_bbox_for_click(
            polygons=[self._polygon("tile_0001_poly_0001", bbox_tile=[10, 10, 20, 20])],
            click=(90, 90),
            nearest_threshold=5,
            tile_width=100,
            tile_height=100,
        )
        self.assertIsNone(selected)

    def test_derives_bbox_from_points_when_bbox_tile_is_missing(self):
        """Prompt-first mode must derive one bbox from points_tile if needed."""
        polygon = self._polygon(
            "tile_0001_poly_0001",
            bbox_tile=[],
            points_tile=[[3, 4], [15, 4], [15, 12], [3, 12]],
        )
        selected = select_bbox_for_click(
            polygons=[polygon],
            click=(10, 10),
            nearest_threshold=10,
            tile_width=100,
            tile_height=100,
        )
        self.assertEqual(selected["bbox"], [3.0, 4.0, 15.0, 12.0])

    def test_deleted_polygons_do_not_participate(self):
        """Deleted Auto polygons must never act as Prompt bbox providers."""
        selected = select_bbox_for_click(
            polygons=[
                self._polygon("tile_0001_poly_0001", status="deleted"),
                self._polygon("tile_0001_poly_0002", bbox_tile=[50, 50, 70, 70]),
            ],
            click=(15, 15),
            nearest_threshold=100,
            tile_width=100,
            tile_height=100,
        )
        self.assertEqual(selected["polygon_id"], "tile_0001_poly_0002")
        self.assertEqual(selected["source"], "auto_nearest")

    def test_bbox_is_clipped_to_tile_bounds(self):
        """Provider output must stay inside tile bounds after clipping."""
        selected = select_bbox_for_click(
            polygons=[self._polygon("tile_0001_poly_0001", bbox_tile=[-5, -5, 120, 110])],
            click=(10, 10),
            nearest_threshold=80,
            tile_width=100,
            tile_height=90,
        )
        self.assertEqual(selected["bbox"], [0.0, 0.0, 100.0, 90.0])


if __name__ == "__main__":
    unittest.main()
