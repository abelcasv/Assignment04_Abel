"""
tests/test_permission_gate.py — Verification of Gate G1 / DEC-01 Permission Enforcement

Ensures that:
- Single worker (p=1) is strictly and exclusively authorized without requiring approval files.
- Process counts below 1 (p <= 0) raise ValueError.
- Non-integral types, booleans, floats, and strings raise TypeError.
- Multiple local workers (p > 1) raise PermissionError when no written instructor
  permission is located in docs/decisions/.
- Under no circumstances is p > 1 silently converted or fallen back to p = 1.
- ClusterManager validates constructor parameters strictly before cluster creation.
- ClusterManager rejects invalid/unauthorized values prior to any cluster initialization.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.dask_matrix.cluster_manager import (
    ClusterManager,
    verify_multiworker_permission,
)


class TestPermissionGate(unittest.TestCase):

    def test_single_worker_permitted(self):
        """p=1 must be authorized prior to instructor permission."""
        self.assertTrue(verify_multiworker_permission(1))
        # Support numpy integer scalar for p=1
        self.assertTrue(verify_multiworker_permission(np.int64(1)))

    def test_non_positive_values_rejected_with_value_error(self):
        """p <= 0 must raise ValueError."""
        non_positive_counts = [0, -1, -10]
        for p in non_positive_counts:
            with self.assertRaises(ValueError) as ctx:
                verify_multiworker_permission(p)
            self.assertIn("positive integer", str(ctx.exception))

        with self.assertRaises(ValueError):
            verify_multiworker_permission(np.int64(0))
        with self.assertRaises(ValueError):
            verify_multiworker_permission(np.int64(-1))

    def test_invalid_types_rejected_with_type_error(self):
        """Non-integral types and booleans must raise TypeError."""
        invalid_types = [
            1.5,
            0.5,
            "1",
            "2",
            True,
            False,
            None,
            "",
        ]
        for val in invalid_types:
            with self.assertRaises(TypeError) as ctx:
                verify_multiworker_permission(val)  # type: ignore
            self.assertIn("must be an integer", str(ctx.exception))

    def test_multiworker_blocked_without_permission(self):
        """p > 1 must raise PermissionError if DEC-01 is unresolved."""
        blocked_counts = [2, 4, 8, 10, 10**9]
        for p in blocked_counts:
            with self.assertRaises(PermissionError) as ctx:
                verify_multiworker_permission(p)
            err_msg = str(ctx.exception)
            self.assertIn("DEC-01", err_msg)
            self.assertIn("BLOCKED", err_msg)
            self.assertIn(f"p={p}", err_msg)
            self.assertIn("Do NOT silently substitute p=1", err_msg)

        # Support numpy integer scalar for p=2
        with self.assertRaises(PermissionError):
            verify_multiworker_permission(np.int64(2))


class TestClusterManagerValidation(unittest.TestCase):
    """Verifies that ClusterManager validates parameters at construction time."""

    def test_constructor_accepts_valid_parameters(self):
        """p=1 and threads=1 are accepted without error."""
        cm = ClusterManager(worker_count=1, threads_per_worker=1)
        self.assertEqual(cm.worker_count, 1)
        self.assertEqual(cm.threads_per_worker, 1)

        # Accepts numpy integer scalars
        cm_np = ClusterManager(worker_count=np.int64(1), threads_per_worker=np.int64(1))
        self.assertEqual(cm_np.worker_count, 1)
        self.assertEqual(cm_np.threads_per_worker, 1)

    def test_constructor_rejects_non_positive_worker_count(self):
        """worker_count < 1 raises ValueError."""
        for val in [0, -1, -5]:
            with self.assertRaises(ValueError) as ctx:
                ClusterManager(worker_count=val)
            self.assertIn("positive integer", str(ctx.exception))

    def test_constructor_rejects_invalid_worker_count_types(self):
        """Non-integral and boolean worker_count raises TypeError."""
        for val in [True, False, 1.5, 0.5, "1", "2", None, ""]:
            with self.assertRaises(TypeError) as ctx:
                ClusterManager(worker_count=val)  # type: ignore
            self.assertIn("must be an integer", str(ctx.exception))

    def test_constructor_rejects_invalid_threads_per_worker(self):
        """threads_per_worker must be positive integer and reject booleans/floats."""
        for val in [0, -1]:
            with self.assertRaises(ValueError):
                ClusterManager(threads_per_worker=val)
        for val in [True, False, 1.5, "1", None]:
            with self.assertRaises(TypeError):
                ClusterManager(threads_per_worker=val)  # type: ignore


class TestClusterManagerSideEffectOrdering(unittest.TestCase):
    """Verifies that invalid or unauthorized configurations never create clusters."""

    def test_invalid_and_blocked_configurations_never_initialize_cluster(self):
        """Cluster initialization must never be reached for p=0, p=-1, p=2, or invalid types."""
        # 1. Non-positive p=0 rejected at constructor before start
        with self.assertRaises(ValueError):
            ClusterManager(worker_count=0)

        # 2. Negative p=-1 rejected at constructor before start
        with self.assertRaises(ValueError):
            ClusterManager(worker_count=-1)

        # 3. Invalid type rejected at constructor before start
        with self.assertRaises(TypeError):
            ClusterManager(worker_count="2")  # type: ignore

        # 4. p=2 accepted at constructor, but start() raises PermissionError before LocalCluster init
        cm_blocked = ClusterManager(worker_count=2)
        with patch("distributed.LocalCluster", create=True) as mock_lc:
            with self.assertRaises(PermissionError) as ctx:
                cm_blocked.start()
            self.assertIn("DEC-01", str(ctx.exception))
            # Verify LocalCluster was NEVER instantiated
            mock_lc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
