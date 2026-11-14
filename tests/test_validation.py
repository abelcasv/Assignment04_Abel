"""
tests/test_validation.py — Unit Tests for Numerical Correctness Validation

Tests:
- Exact match passing with 0 diff.
- Tolerance compliance with atol=1e-9 and rtol=1e-12.
- Detection of tolerance violations.
- Detection and rejection of NaN/Inf values.
- Dimension mismatch error reporting.
- Verification against committed analytical 4x4 fixtures.
"""

import os
import unittest
import numpy as np

from src.dask_matrix.validation import MatrixValidator


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_DIR = os.path.join(REPO_ROOT, "data", "validation")


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.validator = MatrixValidator(atol=1e-9, rtol=1e-12)

    def test_exact_match(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        res = self.validator.validate_matrices(mat, mat)
        self.assertTrue(res.passed)
        self.assertEqual(res.max_abs_diff, 0.0)
        self.assertEqual(res.max_rel_diff, 0.0)
        self.assertTrue(res.all_finite)

    def test_tolerance_within_bound(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Add tiny perturbation 1e-10 (below atol=1e-9)
        pert = ref + 1e-10
        res = self.validator.validate_matrices(pert, ref)
        self.assertTrue(res.passed)
        self.assertAlmostEqual(res.max_abs_diff, 1e-10)

    def test_tolerance_exceeded(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Perturbation 1e-8 (exceeds atol=1e-9)
        pert = ref + 1e-8
        res = self.validator.validate_matrices(pert, ref)
        self.assertFalse(res.passed)
        self.assertIn("exceeded tolerance", res.error_message)

    def test_nan_rejection(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        nan_mat = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
        res = self.validator.validate_matrices(nan_mat, ref)
        self.assertFalse(res.passed)
        self.assertFalse(res.all_finite)
        self.assertIn("Non-finite", res.error_message)

    def test_shape_mismatch(self):
        m1 = np.ones((2, 2), dtype=np.float64)
        m2 = np.ones((3, 3), dtype=np.float64)
        res = self.validator.validate_matrices(m1, m2)
        self.assertFalse(res.passed)
        self.assertIn("Shape mismatch", res.error_message)

    def test_analytical_4x4_fixtures(self):
        """Validates that known_A @ known_B == expected_C using committed fixtures."""
        file_a = os.path.join(VAL_DIR, "known_A_4.csv")
        file_b = os.path.join(VAL_DIR, "known_B_4.csv")
        file_c = os.path.join(VAL_DIR, "expected_C_4.csv")

        if os.path.isfile(file_a) and os.path.isfile(file_b) and os.path.isfile(file_c):
            mat_a = np.loadtxt(file_a, delimiter=",", dtype=np.float64)
            mat_b = np.loadtxt(file_b, delimiter=",", dtype=np.float64)
            expected_c = np.loadtxt(file_c, delimiter=",", dtype=np.float64)

            computed_c = np.matmul(mat_a, mat_b)
            res = self.validator.validate_matrices(computed_c, expected_c)
            self.assertTrue(res.passed)
            self.assertEqual(res.max_abs_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
