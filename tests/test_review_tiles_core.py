"""GUI-free tests for tile review loading, selection, status, and persistence."""

import tempfile
import unittest
from pathlib import Path

from workflow.manifest import TileManifestEntry
from workflow.tile_manager import PolygonRecord, TileResult
from workflow.tile_review import (
    first_review_index,
    load_review_result,
    save_review_result,
    select_polygon_at_point,
    set_polygon_status,
)


class ReviewTilesCoreTest(unittest.TestCase):
    """Verify review behavior without creating a Matplotlib window."""

    @staticmethod
    def _tile(tile_id="tile_0001", x_offset=0):
        """Create one canonical manifest tile for isolated tests."""
        return TileManifestEntry(
            tile_id=tile_id,
            image_path=Path("tiles") / f"{tile_id}.png",
            x_offset=x_offset,
            y_offset=0,
            width=100,
            height=100,
            is_edge_tile=False,
        )

    @staticmethod
    def _result(tile, status="active", score=0.91):
        """Create one valid square polygon while retaining all protocol fields."""
        polygon = PolygonRecord(
            id=f"{tile.tile_id}_poly_0001",
            tile_id=tile.tile_id,
            status=status,
            source="auto",
            score=score,
            points_tile=[[10, 10], [40, 10], [40, 40], [10, 40]],
            bbox_tile=[10, 10, 40, 40],
            points_global=[],
            click_history=[],
            metadata={"raw_index": 0},
        )
        return TileResult(
            tile_id=tile.tile_id,
            offset=[tile.x_offset, tile.y_offset],
            image_path=tile.image_path,
            polygons=[polygon],
        )

    def test_auto_load_status_changes_and_edited_save(self):
        """Auto results must support soft delete, restore, and lossless edited save."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auto_dir = root / "auto_results"
            edited_dir = root / "edited_results"
            auto_dir.mkdir()
            tile = self._tile()
            original = self._result(tile)
            original.save(auto_dir / f"{tile.tile_id}.json")

            loaded, from_edited = load_review_result(tile, auto_dir, edited_dir)
            self.assertFalse(from_edited)
            deleted = set_polygon_status(loaded, loaded.polygons[0].id, "deleted")
            self.assertEqual(deleted.status, "deleted")
            restored = set_polygon_status(loaded, loaded.polygons[0].id, "active")
            self.assertEqual(restored.status, "active")
            set_polygon_status(loaded, loaded.polygons[0].id, "deleted")
            save_review_result(loaded, edited_dir)

            saved = TileResult.load(edited_dir / f"{tile.tile_id}.json")
            self.assertEqual(saved.polygons[0].status, "deleted")
            self.assertEqual(saved.polygons[0].id, original.polygons[0].id)
            self.assertEqual(saved.polygons[0].score, original.polygons[0].score)
            self.assertEqual(saved.polygons[0].source, original.polygons[0].source)
            self.assertEqual(saved.polygons[0].points_tile, original.polygons[0].points_tile)
            self.assertEqual(saved.polygons[0].metadata, original.polygons[0].metadata)

    def test_edited_result_has_priority_over_auto(self):
        """An existing edited JSON must be loaded instead of its Auto source."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auto_dir = root / "auto_results"
            edited_dir = root / "edited_results"
            auto_dir.mkdir()
            edited_dir.mkdir()
            tile = self._tile()
            self._result(tile, status="active", score=0.9).save(
                auto_dir / f"{tile.tile_id}.json"
            )
            self._result(tile, status="deleted", score=0.9).save(
                edited_dir / f"{tile.tile_id}.json"
            )

            loaded, from_edited = load_review_result(tile, auto_dir, edited_dir)
            self.assertTrue(from_edited)
            self.assertEqual(loaded.polygons[0].status, "deleted")

    def test_polygon_selection_inside_near_and_too_far(self):
        """Containment must win, then nearest boundary within the configured threshold."""
        tile = self._tile()
        result = self._result(tile)
        self.assertEqual(
            select_polygon_at_point(result, (20, 20), max_distance=5).id,
            result.polygons[0].id,
        )
        self.assertEqual(
            select_polygon_at_point(result, (43, 20), max_distance=5).id,
            result.polygons[0].id,
        )
        self.assertIsNone(select_polygon_at_point(result, (80, 80), max_distance=5))

    def test_empty_result_and_resume_index(self):
        """Missing Auto results must remain saveable and resume at the first unreviewed tile."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auto_dir = root / "auto_results"
            edited_dir = root / "edited_results"
            tiles = [self._tile("tile_0001", 0), self._tile("tile_0002", 100)]
            empty, from_edited = load_review_result(tiles[0], auto_dir, edited_dir)
            self.assertFalse(from_edited)
            self.assertEqual(empty.polygons, [])
            save_review_result(empty, edited_dir)
            self.assertEqual(first_review_index(tiles, edited_dir), 1)
            self.assertEqual(first_review_index(tiles, edited_dir, "tile_0001"), 0)
            self.assertEqual(TileResult.load(edited_dir / "tile_0001.json").polygons, [])


if __name__ == "__main__":
    unittest.main()
