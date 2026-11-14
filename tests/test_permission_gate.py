"""
tests/test_permission_gate.py — Verification of Gate G1 / DEC-01 Permission Enforcement

Ensures that:
- Single worker (p=1) is immediately permitted without requiring approval files.
- Multiple local workers (p > 1) raise PermissionError when no written instructor
  permission is located in docs/decisions/.
- Under no circumstances is p > 1 silently converted or fallen back to p = 1.
"""

import unittest
from src.dask_matrix.cluster_manager import verify_multiworker_permission


class TestPermissionGate(unittest.TestCase):

    def test_single_worker_permitted(self):
        """p=1 must always be permitted prior to instructor permission."""
        self.assertTrue(verify_multiworker_permission(1))
        self.assertTrue(verify_multiworker_permission(0))

    def test_multiworker_blocked_without_permission(self):
        """p > 1 must raise PermissionError if DEC-01 is unresolved."""
        blocked_counts = [2, 4, 8, 10]
        for p in blocked_counts:
            with self.assertRaises(PermissionError) as ctx:
                verify_multiworker_permission(p)
            err_msg = str(ctx.exception)
            self.assertIn("DEC-01", err_msg)
            self.assertIn("BLOCKED", err_msg)
            self.assertIn(f"p={p}", err_msg)
            self.assertIn("Do NOT silently substitute p=1", err_msg)


if __name__ == "__main__":
    unittest.main()
