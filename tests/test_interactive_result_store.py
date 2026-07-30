import json
import os
import tempfile
import unittest

from utils.interactive_result_store import InteractiveResultStore


class InteractiveResultStoreTest(unittest.TestCase):
    def test_load_hit_edit_delete_and_save(self):
        initial = [
            {
                "image_id": "large_image",
                "bbox": [100, 200, 20, 20],
                "segmentation": [[100, 200, 120, 200, 120, 220, 100, 220]],
                "score": 0.9,
                "tile_id": "tile_1",
            },
            {
                "image_id": "large_image",
                "bbox": [1000, 1000, 20, 20],
                "segmentation": [[1000, 1000, 1020, 1000, 1020, 1020, 1000, 1020]],
                "score": 0.8,
                "tile_id": "tile_2",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            initial_path = os.path.join(directory, "initial.json")
            output_path = os.path.join(directory, "final.json")
            with open(initial_path, "w", encoding="utf-8") as stream:
                json.dump(initial, stream)

            store = InteractiveResultStore(
                "tile.png", tile_offset=(100, 200), output_path=output_path
            )
            loaded, warnings = store.load_initial_results(
                initial_path, image_id="large_image", viewport_size=(256, 256)
            )

            self.assertEqual(loaded, 2)
            self.assertEqual(warnings, [])
            self.assertEqual(len(store.visible_instances((256, 256))), 1)
            self.assertEqual(store.hit_test_local((10, 10)), "auto_0000")

            updated = store.update_from_local(
                "auto_0000",
                [0, 0, 30, 30],
                [[0, 0], [30, 0], [30, 30], [0, 30]],
                [[10, 10], [20, 20]],
                [1, 0],
                0.95,
            )
            self.assertEqual(updated["source"], "edited")
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["bbox"], [100.0, 200.0, 30.0, 30.0])

            prompt_id = store.create_prompt(
                [40, 40, 60, 60],
                [[40, 40], [60, 40], [60, 60], [40, 60]],
                [[50, 50]],
                [1],
            )
            self.assertEqual(store.instances[prompt_id]["source"], "prompt")
            store.mark_deleted(prompt_id)
            self.assertEqual(store.instances[prompt_id]["status"], "deleted")

            store.save()
            with open(output_path, "r", encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["coordinate_space"], "global")
            self.assertEqual(len(saved["instances"]), 3)
            self.assertEqual(saved["instances"][prompt_id]["status"], "deleted")


if __name__ == "__main__":
    unittest.main()
