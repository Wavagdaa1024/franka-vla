# -*- coding: utf-8 -*-
import unittest
import numpy as np

from obstacle_avoidance.perception.envelope_extractor import (
    fit_plane_ransac,
    fit_3d_obb,
    extract_obstacle_envelopes
)


class TestEnvelopeExtraction(unittest.TestCase):
    def test_ransac_table_fit(self):
        # Synthetic flat table at Z = 0.0 + small noise
        rng = np.random.default_rng(42)
        n_table = 2000
        x = rng.uniform(0.3, 0.7, n_table)
        y = rng.uniform(-0.3, 0.3, n_table)
        z = rng.normal(0.0, 0.002, n_table)
        table_pts = np.column_stack((x, y, z))

        # Add a block (obstacle) on top of the table at X=0.5, Y=0.0, Z=0.02..0.15
        n_block = 300
        bx = rng.uniform(0.48, 0.52, n_block)
        by = rng.uniform(-0.02, 0.02, n_block)
        bz = rng.uniform(0.03, 0.12, n_block)
        block_pts = np.column_stack((bx, by, bz))

        pts_all = np.vstack((table_pts, block_pts))

        plane, inliers, above = fit_plane_ransac(pts_all, distance_threshold=0.008)
        self.assertIsNotNone(plane)
        # Normal should be approximately [0, 0, 1]
        self.assertGreater(plane[2], 0.95)
        # Table inliers should capture majority of table
        self.assertGreater(np.sum(inliers), 1800)
        # Points above table should include block points
        self.assertGreater(np.sum(above), 250)

    def test_obb_fitting(self):
        # Synthetic box cluster with margin
        rng = np.random.default_rng(42)
        pts = rng.uniform([-0.05, -0.05, 0.0], [0.05, 0.05, 0.10], size=(200, 3))
        obb = fit_3d_obb(pts, margin=0.03)

        # Center should be near [0, 0, 0.05]
        np.testing.assert_allclose(obb.center, [0.0, 0.0, 0.05], atol=0.02)
        # Extents should cover original size (0.1, 0.1, 0.1) + 2*margin (0.06)
        self.assertGreaterEqual(obb.extents[0], 0.14)
        self.assertGreaterEqual(obb.extents[1], 0.14)
        self.assertGreaterEqual(obb.extents[2], 0.14)


if __name__ == "__main__":
    unittest.main()
