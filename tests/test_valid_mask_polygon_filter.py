"""Model-free tests for filtering Auto polygons against tile valid pixels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.filter_tile_results_by_valid_mask import (
    filter_result,
    polygon_validity,
    resolve_manifest_tile_path,
)
from workflow.tile_manager import PolygonRecord, TileResult


class ValidMaskPolygonFilterTest(unittest.TestCase):
    @staticmethod
    def _result():
        return TileResult(
            tile_id="tile_0001",
            offset=[0, 0],
            polygons=[
                PolygonRecord(
                    id="tile_0001_poly_0001",
                    tile_id="tile_0001",
                    status="active",
                    source="auto",
                    score=0.9,
                    points_tile=[[0, 0], [10, 0], [10, 10], [0, 10]],
                    bbox_tile=[0, 0, 10, 10],
                )
            ],
        )

    def test_polygon_validity_counts_pixel_center_overlap(self):
        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:, :5] = True
        validity = polygon_validity(
            [[0, 0], [10, 0], [10, 10], [0, 10]], valid_mask
        )
        self.assertEqual(validity["polygon_pixels"], 100)
        self.assertEqual(validity["valid_pixels"], 50)
        self.assertEqual(validity["valid_ratio"], 0.5)

    def test_manifest_tile_path_resolution_is_local_and_traversal_safe(self):
        """The CLI must resolve canonical tile paths without importing another CLI."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "output"
            manifest_path = workspace / "manifests" / "manifest.json"
            tile_path = workspace / "tiles" / "tile_0001.tif"
            manifest_path.parent.mkdir(parents=True)
            tile_path.parent.mkdir()
            manifest_path.touch()
            tile_path.touch()
            self.assertEqual(
                resolve_manifest_tile_path(
                    manifest_path, Path("tiles") / "tile_0001.tif"
                ),
                tile_path.resolve(),
            )
            with self.assertRaises(ValueError):
                resolve_manifest_tile_path(manifest_path, Path("..") / "outside.tif")

    def test_low_valid_overlap_is_rejected_without_mutating_input(self):
        result = self._result()
        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:, :5] = True
        filtered, rejected = filter_result(result, valid_mask, 0.9, 20)
        self.assertEqual(len(result.polygons), 1)
        self.assertNotIn("valid_mask_filter", result.polygons[0].metadata)
        self.assertEqual(filtered.polygons, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["valid_ratio"], 0.5)
        self.assertIn("valid_ratio_below_threshold", rejected[0]["reasons"])

    def test_polygon_inside_valid_imagery_is_preserved_with_audit(self):
        result = self._result()
        filtered, rejected = filter_result(
            result, np.ones((10, 10), dtype=bool), 0.9, 20
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(filtered.polygons), 1)
        audit = filtered.polygons[0].metadata["valid_mask_filter"]
        self.assertEqual(audit["valid_pixels"], 100)
        self.assertEqual(audit["valid_ratio"], 1.0)

    def test_tiny_polygon_is_rejected_by_minimum_valid_pixels(self):
        result = self._result()
        result.polygons[0] = PolygonRecord(
            id="tile_0001_poly_0001",
            tile_id="tile_0001",
            status="active",
            source="auto",
            score=0.9,
            points_tile=[[0, 0], [2, 0], [2, 2], [0, 2]],
        )
        filtered, rejected = filter_result(
            result, np.ones((10, 10), dtype=bool), 0.9, 20
        )
        self.assertEqual(filtered.polygons, [])
        self.assertIn("valid_pixels_below_threshold", rejected[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
