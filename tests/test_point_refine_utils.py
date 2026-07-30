"""GUI-free tests for point-guided mask refinement helpers."""

import unittest

import numpy as np

from tools.point_refine_utils import (
    refine_mask_by_connected_components,
    refine_mask_by_distance,
)


class PointRefineUtilsTest(unittest.TestCase):
    """Verify connected-component and distance point refinement behavior."""

    def test_connected_component_removes_negative_component(self):
        """Negative-hit components should be removed while positive ones are kept."""
        mask = np.zeros((1, 20, 30), dtype=np.uint8)
        mask[0, 2:8, 2:8] = 1
        mask[0, 10:18, 18:26] = 1
        refined, applied, _ = refine_mask_by_connected_components(
            mask,
            positive_points=[(4, 4)],
            negative_points=[(20, 12)],
        )
        self.assertTrue(applied)
        self.assertEqual(int(refined[0, 4, 4]), 1)
        self.assertEqual(int(refined[0, 12, 20]), 0)

    def test_distance_mode_removes_negative_side_but_keeps_positive(self):
        """Distance refinement should keep positive neighborhoods inside one component."""
        mask = np.ones((1, 20, 20), dtype=np.uint8)
        refined, applied, _ = refine_mask_by_distance(
            mask,
            positive_points=[(4, 10)],
            negative_points=[(16, 10)],
            negative_margin=0.0,
        )
        self.assertTrue(applied)
        self.assertEqual(int(refined[0, 10, 4]), 1)
        self.assertEqual(int(refined[0, 10, 16]), 0)


if __name__ == "__main__":
    unittest.main()
