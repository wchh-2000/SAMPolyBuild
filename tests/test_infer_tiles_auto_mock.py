"""Model-free tests for canonical per-tile Auto inference orchestration."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from infer_tiles_auto import run
from workflow.manifest import TileManifest, TileManifestEntry
from workflow.tile_manager import TileResult


class InferTilesAutoMockTest(unittest.TestCase):
    """Validate output contracts, empty results, skip, overwrite, and errors."""

    def _workspace(self, root: Path, tile_count: int = 3):
        """Create a canonical manifest and placeholder tile/config/checkpoint files."""
        workspace = root / "workspace"
        tiles_dir = workspace / "tiles"
        manifests_dir = workspace / "manifests"
        tiles_dir.mkdir(parents=True)
        entries = []
        for index in range(1, tile_count + 1):
            tile_id = f"tile_{index:04d}"
            image_path = Path("tiles") / f"{tile_id}.png"
            (workspace / image_path).touch()
            entries.append(
                TileManifestEntry(
                    tile_id=tile_id,
                    image_path=image_path,
                    x_offset=(index - 1) * 64,
                    y_offset=0,
                    width=64,
                    height=64,
                    is_edge_tile=index == tile_count,
                )
            )
        manifest_path = manifests_dir / "manifest.json"
        TileManifest(
            image_name="source.png",
            image_width=tile_count * 64,
            image_height=64,
            tile_size=64,
            overlap=0,
            tiles=entries,
        ).save(manifest_path)
        config_path = root / "config.py"
        checkpoint_path = root / "model.pth"
        config_path.touch()
        checkpoint_path.touch()
        return manifest_path, config_path, checkpoint_path

    @staticmethod
    def _args(manifest_path, config_path, checkpoint_path, output_dir, overwrite=False):
        """Build CLI-equivalent arguments for the core run function."""
        return Namespace(
            manifest=str(manifest_path),
            output_dir=str(output_dir),
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            device="cpu",
            score_threshold=0.5,
            batch_size=2,
            num_workers=0,
            overwrite=overwrite,
        )

    @staticmethod
    def _raw_polygon(tile_id: str, score: float = 0.91):
        """Return one record shaped exactly like existing infer_auto output."""
        return {
            "image_id": tile_id,
            "category_id": 1,
            "segmentation": [[1, 1, 20, 1, 20, 20, 1, 20, 1, 1]],
            "bbox": [1, 1, 19, 19],
            "area": 361,
            "score": score,
            "score_cls": 0.95,
        }

    def test_outputs_empty_result_skip_and_overwrite(self):
        """Every tile must have a result file and overwrite must rerun inference."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, config_path, checkpoint_path = self._workspace(root)
            output_dir = root / "auto_results"
            calls = []

            def fake_infer(tile_paths, **kwargs):
                calls.append([path.stem for path in tile_paths])
                return [self._raw_polygon("tile_0001")]

            args = self._args(
                manifest_path, config_path, checkpoint_path, output_dir
            )
            summary = run(args, infer_batch=fake_infer)
            self.assertEqual(summary["success_tiles"], 3)
            self.assertEqual(summary["total_polygons"], 1)
            self.assertEqual(len(calls), 1)

            first = TileResult.load(output_dir / "tile_0001.json")
            second = TileResult.load(output_dir / "tile_0002.json")
            self.assertEqual(first.polygons[0].id, "tile_0001_poly_0001")
            self.assertEqual(first.polygons[0].status, "active")
            self.assertEqual(first.polygons[0].source, "auto")
            self.assertTrue(first.polygons[0].points_tile)
            self.assertEqual(first.polygons[0].bbox_tile, [1.0, 1.0, 20.0, 20.0])
            self.assertEqual(first.polygons[0].metadata["raw_index"], 0)
            self.assertEqual(second.polygons, [])

            def must_not_run(tile_paths, **kwargs):
                raise AssertionError("Existing results should have been skipped")

            skipped = run(args, infer_batch=must_not_run)
            self.assertEqual(skipped["skipped_tiles"], 3)

            args.overwrite = True
            overwritten = run(args, infer_batch=fake_infer)
            self.assertEqual(overwritten["success_tiles"], 3)
            self.assertEqual(len(calls), 2)

    def test_batch_failure_falls_back_and_writes_error_json(self):
        """One failing fallback tile must not prevent other TileResults being saved."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, config_path, checkpoint_path = self._workspace(root, tile_count=2)
            output_dir = root / "auto_results"

            def partially_failing_infer(tile_paths, **kwargs):
                if len(tile_paths) > 1:
                    raise RuntimeError("synthetic batch failure")
                if tile_paths[0].stem == "tile_0002":
                    raise RuntimeError("synthetic tile failure")
                return [self._raw_polygon(tile_paths[0].stem)]

            summary = run(
                self._args(manifest_path, config_path, checkpoint_path, output_dir),
                infer_batch=partially_failing_infer,
            )
            self.assertEqual(summary["success_tiles"], 1)
            self.assertEqual(summary["failed_tiles"], 1)
            failed = TileResult.load(output_dir / "tile_0002.json")
            self.assertEqual(failed.polygons, [])
            self.assertIn("synthetic tile failure", failed.error)


if __name__ == "__main__":
    unittest.main()
