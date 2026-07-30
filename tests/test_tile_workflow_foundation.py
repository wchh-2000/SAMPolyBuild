"""Standard-library tests for tile workflow data contracts and persistence."""

import tempfile
import unittest
from pathlib import Path

from workflow.coordinate_transform import global_to_tile, tile_to_global
from workflow.manifest import TileManifest, TileManifestEntry
from workflow.tile_manager import PolygonRecord, TileManager, TileResult


class TileWorkflowFoundationTest(unittest.TestCase):
    """Verify round trips without importing image or model dependencies."""

    def test_manifest_round_trip(self):
        """A canonical manifest must survive save and load unchanged."""
        manifest = TileManifest(
            image_name="source.tif",
            image_width=2048,
            image_height=1536,
            tile_size=1024,
            overlap=256,
            tiles=[
                TileManifestEntry(
                    tile_id="tile_0001",
                    image_path=Path("tiles/tile_0001.png"),
                    x_offset=0,
                    y_offset=0,
                    width=1024,
                    height=1024,
                    is_edge_tile=False,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.save(path)
            loaded = TileManifest.load(path)
        self.assertEqual(loaded.to_dict(), manifest.to_dict())

    def test_tile_result_round_trip_and_polygon_closure(self):
        """TileResult persistence must retain lifecycle and reserved fields."""
        polygon = PolygonRecord(
            id="auto_0001",
            tile_id="tile_0001",
            status="active",
            source="auto",
            score=0.9,
            points_tile=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        result = TileResult(
            tile_id="tile_0001",
            offset=[100, 200],
            polygons=[polygon],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tile_0001.json"
            result.save(path)
            loaded = TileResult.load(path)
        self.assertEqual(loaded.to_dict(), result.to_dict())
        self.assertEqual(loaded.polygons[0].points_tile[0], loaded.polygons[0].points_tile[-1])

    def test_coordinate_translation_is_reversible(self):
        """Translation to global coordinates and back must preserve a point."""
        global_point = tile_to_global([12, 34], [100, 200])
        self.assertEqual(global_point, [112.0, 234.0])
        self.assertEqual(global_to_tile(global_point, [100, 200]), [12.0, 34.0])

    def test_tile_manager_paths_and_empty_result(self):
        """The manager must create only the documented workflow directories."""
        with tempfile.TemporaryDirectory() as directory:
            manager = TileManager(Path(directory) / "data")
            paths = manager.ensure_directories()
            self.assertTrue(all(path.is_dir() for path in paths.values()))
            self.assertEqual(
                manager.tile_result_path("tile_0001", "edited").name,
                "tile_0001.json",
            )
        empty = TileResult.create_empty("tile_0001", [0, 0])
        self.assertEqual(empty.polygons, [])

    def test_invalid_geometry_and_manifest_bounds_are_rejected(self):
        """Invalid polygons and out-of-bounds tiles must fail with clear errors."""
        with self.assertRaisesRegex(ValueError, "distinct vertices"):
            PolygonRecord(
                id="invalid",
                tile_id="tile_0001",
                status="active",
                source="auto",
                score=None,
                points_tile=[[0, 0], [0, 0], [1, 1]],
            )
        with self.assertRaisesRegex(ValueError, "exceed source-image bounds"):
            TileManifest(
                image_name="source.tif",
                image_width=100,
                image_height=100,
                tile_size=64,
                overlap=16,
                tiles=[
                    TileManifestEntry(
                        tile_id="outside",
                        image_path=Path("tiles/outside.png"),
                        x_offset=80,
                        y_offset=0,
                        width=64,
                        height=64,
                        is_edge_tile=True,
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
