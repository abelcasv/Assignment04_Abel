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

    def test_inf_rejection(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        inf_mat = np.array([[1.0, np.inf], [3.0, -np.inf]], dtype=np.float64)
        res = self.validator.validate_matrices(inf_mat, ref)
        self.assertFalse(res.passed)
        self.assertFalse(res.all_finite)
        self.assertIn("Non-finite", res.error_message)

    def test_rtol_relative_scaling(self):
        ref = np.array([[1.0e6, 2.0e6], [3.0e6, 4.0e6]], dtype=np.float64)
        # 5e-7 is > atol (1e-9) and <= atol + rtol * 1e6 (= 1.001e-6), so it must pass
        within_rtol = ref.copy()
        within_rtol[0, 0] += 5e-7
        self.assertTrue(self.validator.validate_matrices(within_rtol, ref).passed)
        # 5e-6 exceeds atol + rtol * 1e6, so it must fail
        beyond_rtol = ref.copy()
        beyond_rtol[0, 0] += 5e-6
        self.assertFalse(self.validator.validate_matrices(beyond_rtol, ref).passed)

    def test_identity_and_zero_fixtures(self):
        mat_a = np.loadtxt(os.path.join(VAL_DIR, "known_A_4.csv"), delimiter=",", dtype=np.float64)
        mat_id = np.loadtxt(os.path.join(VAL_DIR, "identity_4.csv"), delimiter=",", dtype=np.float64)
        mat_zero = np.loadtxt(os.path.join(VAL_DIR, "zero_4.csv"), delimiter=",", dtype=np.float64)
        self.assertTrue(self.validator.validate_identity_fixture(mat_a, np.matmul(mat_a, mat_id)).passed)
        self.assertTrue(self.validator.validate_zero_fixture(np.matmul(mat_a, mat_zero)).passed)

    def test_empty_and_zero_dim_rejection(self):
        empty = np.zeros((0, 0), dtype=np.float64)
        res_empty = self.validator.validate_matrices(empty, empty)
        self.assertFalse(res_empty.passed)
        self.assertIn("Empty", res_empty.error_message)

        scalar = np.array(1.0, dtype=np.float64)
        res_scalar = self.validator.validate_matrices(scalar, scalar)
        self.assertFalse(res_scalar.passed)
        self.assertIn("Empty or 0-D", res_scalar.error_message)

    def test_analytical_4x4_fixtures(self):
        """Validates that known_A @ known_B == expected_C using committed fixtures."""
        file_a = os.path.join(VAL_DIR, "known_A_4.csv")
        file_b = os.path.join(VAL_DIR, "known_B_4.csv")
        file_c = os.path.join(VAL_DIR, "expected_C_4.csv")

        self.assertTrue(os.path.isfile(file_a), f"Missing fixture: {file_a}")
        self.assertTrue(os.path.isfile(file_b), f"Missing fixture: {file_b}")
        self.assertTrue(os.path.isfile(file_c), f"Missing fixture: {file_c}")

        mat_a = np.loadtxt(file_a, delimiter=",", dtype=np.float64)
        mat_b = np.loadtxt(file_b, delimiter=",", dtype=np.float64)
        expected_c = np.loadtxt(file_c, delimiter=",", dtype=np.float64)

        computed_c = np.matmul(mat_a, mat_b)
        res = self.validator.validate_matrices(computed_c, expected_c)
        self.assertTrue(res.passed)
        self.assertEqual(res.max_abs_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
