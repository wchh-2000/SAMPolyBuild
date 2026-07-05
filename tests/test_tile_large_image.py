"""Integration tests for large-image tiling and canonical manifest output."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.tile_large_image import run
from workflow.manifest import TileManifest

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
    from_origin = None


@unittest.skipUnless(rasterio is not None, "rasterio is required for tiling integration tests")
class TileLargeImageTest(unittest.TestCase):
    """Exercise the real window reader and PNG tile writer on a small synthetic raster."""

    def test_tiling_order_offsets_edges_and_relative_paths(self):
        """A 2500x1800 image must produce the expected row-major 3x3 grid."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "large_image.tif"
            output_dir = root / "output"
            image = np.ones((3, 1800, 2500), dtype=np.uint8)
            with rasterio.open(
                input_path,
                mode="w",
                driver="GTiff",
                width=2500,
                height=1800,
                count=3,
                dtype="uint8",
                transform=from_origin(100, 200, 1, 1),
            ) as dataset:
                dataset.write(image)

            args = Namespace(
                input=str(input_path),
                output_dir=str(output_dir),
                tile_size=1024,
                overlap=256,
                prefix="large_image",
                bands=(1, 2, 3),
                tile_format="png",
                skip_empty=False,
                empty_threshold=0.98,
                min_valid_ratio=0.0,
                overwrite=False,
            )
            run(args)

            manifest_path = output_dir / "manifests" / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((output_dir / "tiles_manifest.json").is_file())
            manifest = TileManifest.load(manifest_path)

            self.assertEqual(len(manifest.tiles), 9)
            self.assertEqual(
                [tile.tile_id for tile in manifest.tiles],
                [f"tile_{index:04d}" for index in range(1, 10)],
            )
            self.assertEqual(
                [(tile.x_offset, tile.y_offset) for tile in manifest.tiles],
                [
                    (0, 0), (768, 0), (1536, 0),
                    (0, 768), (768, 768), (1536, 768),
                    (0, 1536), (768, 1536), (1536, 1536),
                ],
            )
            self.assertEqual((manifest.tiles[-1].width, manifest.tiles[-1].height), (964, 264))
            for index, tile in enumerate(manifest.tiles):
                touches_right = tile.x_offset + tile.width == manifest.image_width
                touches_bottom = tile.y_offset + tile.height == manifest.image_height
                self.assertEqual(tile.is_edge_tile, touches_right or touches_bottom)
                self.assertFalse(tile.image_path.is_absolute())
                self.assertEqual(tile.image_path, Path(f"tiles/tile_{index + 1:04d}.png"))
                self.assertTrue((output_dir / tile.image_path).is_file())


if __name__ == "__main__":
    unittest.main()
