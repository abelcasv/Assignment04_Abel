"""
tests/test_partitioning.py — Unit Tests for 2D Block Partitioning and Assembly

Covers:
- Standard divisible grid: n=500, b=250 -> 2x2 blocks of 250x250.
- Irregular edge blocks: n=4, b=3 -> 2x2 blocks with shapes (3,3), (3,1), (1,3), (1,1).
- Non-divisible larger grid: n=250, b=100 -> 3x3 blocks with edge dimensions 50.
- Re-assembly roundtrip identity: assemble(partition(M)) == M.
- C-order memory contiguity validation.
- Shape validation and error handling on non-2D, non-square, or non-float64 inputs.
"""

import unittest
import numpy as np

from src.dask_matrix.partitioning import (
    compute_grid_dimensions,
    get_block_slices,
    partition_matrix_2d,
    assemble_matrix_2d,
)


class TestPartitioning(unittest.TestCase):

    def test_compute_grid_dimensions(self):
        self.assertEqual(compute_grid_dimensions(250, 250), 1)
        self.assertEqual(compute_grid_dimensions(500, 250), 2)
        self.assertEqual(compute_grid_dimensions(750, 250), 3)
        self.assertEqual(compute_grid_dimensions(1000, 250), 4)
        self.assertEqual(compute_grid_dimensions(4, 3), 2)
        self.assertEqual(compute_grid_dimensions(250, 100), 3)
        with self.assertRaises(ValueError):
            compute_grid_dimensions(0, 250)
        with self.assertRaises(ValueError):
            compute_grid_dimensions(250, 0)

    def test_partition_and_assemble_divisible(self):
        n, b = 500, 250
        rng = np.random.default_rng(12345)
        original = rng.standard_normal((n, n)).astype(np.float64)

        blocks = partition_matrix_2d(original, b)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(blocks[0][0].shape, (250, 250))
        self.assertEqual(blocks[0][1].shape, (250, 250))
        self.assertEqual(blocks[1][0].shape, (250, 250))
        self.assertEqual(blocks[1][1].shape, (250, 250))

        # Check C-contiguity
        for row in blocks:
            for blk in row:
                self.assertTrue(blk.flags.c_contiguous)

        # Assemble and test exact equality
        assembled = assemble_matrix_2d(blocks, n, b)
        self.assertTrue(assembled.flags.c_contiguous)
        np.testing.assert_array_equal(original, assembled)

    def test_irregular_edge_blocks_n4_b3(self):
        """Mandatory edge-block fixture from protocol: n=4, b=3."""
        n, b = 4, 3
        data = np.arange(16, dtype=np.float64).reshape((4, 4))
        blocks = partition_matrix_2d(data, b)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        # Block (0, 0): rows 0..3, cols 0..3 -> shape (3, 3)
        self.assertEqual(blocks[0][0].shape, (3, 3))
        # Block (0, 1): rows 0..3, cols 3..4 -> shape (3, 1)
        self.assertEqual(blocks[0][1].shape, (3, 1))
        # Block (1, 0): rows 3..4, cols 0..3 -> shape (1, 3)
        self.assertEqual(blocks[1][0].shape, (1, 3))
        # Block (1, 1): rows 3..4, cols 3..4 -> shape (1, 1)
        self.assertEqual(blocks[1][1].shape, (1, 1))

        # Assemble and check exact roundtrip
        assembled = assemble_matrix_2d(blocks, n, b)
        np.testing.assert_array_equal(data, assembled)

    def test_invalid_matrix_rejection(self):
        # Non-2D
        with self.assertRaises(ValueError):
            partition_matrix_2d(np.ones(10, dtype=np.float64), 5)
        # Non-square
        with self.assertRaises(ValueError):
            partition_matrix_2d(np.ones((4, 5), dtype=np.float64), 2)
        # Non-float64
        with self.assertRaises(ValueError):
            partition_matrix_2d(np.ones((4, 4), dtype=np.int32), 2)


if __name__ == "__main__":
    unittest.main()
