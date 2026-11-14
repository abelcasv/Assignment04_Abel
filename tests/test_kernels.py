"""
tests/test_kernels.py — Unit Tests for Elementary Matrix Kernels

Covers:
- block_multiply: basic matrix multiplication, shape validation, C-order output.
- block_add: element-wise addition, shape mismatch rejection.
- compute_matrix_checksum: summation verification, NaN/Inf rejection.
"""

import unittest
import numpy as np

from src.dask_matrix.kernels import (
    block_multiply,
    block_add,
    compute_matrix_checksum,
)


class TestKernels(unittest.TestCase):

    def test_block_multiply_basic(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float64)

        result = block_multiply(a, b)
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_block_multiply_dimension_mismatch(self):
        a = np.ones((2, 3), dtype=np.float64)
        b = np.ones((4, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            block_multiply(a, b)

    def test_block_add_basic(self):
        p1 = np.ones((3, 3), dtype=np.float64) * 2.0
        p2 = np.ones((3, 3), dtype=np.float64) * 3.0
        expected = np.ones((3, 3), dtype=np.float64) * 5.0

        result = block_add(p1, p2)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, expected)

    def test_block_add_shape_mismatch(self):
        p1 = np.ones((3, 2), dtype=np.float64)
        p2 = np.ones((3, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            block_add(p1, p2)

    def test_compute_matrix_checksum(self):
        mat = np.arange(16, dtype=np.float64).reshape((4, 4))
        expected_sum = float(np.sum(np.arange(16)))
        self.assertEqual(compute_matrix_checksum(mat), expected_sum)

    def test_compute_matrix_checksum_nan_inf(self):
        nan_mat = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            compute_matrix_checksum(nan_mat)

        inf_mat = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            compute_matrix_checksum(inf_mat)


if __name__ == "__main__":
    unittest.main()
