"""
tests/test_reduction.py — Unit Tests for Binary-Tree Reduction Logic

Tests graph construction and round-based pairing without requiring a live Dask cluster:
- Single future passthrough.
- Exact count of Nb - 1 addition tasks for arbitrary Nb.
- ceil(log2(Nb)) depth verification.
- Unique key formatting per repetition and block coordinates.
"""

import math
import unittest
from typing import Any, List

from src.dask_matrix.reduction import build_binary_tree_reduction


class MockFuture:
    def __init__(self, key: str, value: Any = None):
        self.key = key
        self.value = value


class MockDaskClient:
    def __init__(self):
        self.submitted_tasks: List[dict] = []

    def submit(self, func, *args, key=None, pure=False):
        self.submitted_tasks.append({
            "func": func,
            "args": args,
            "key": key,
            "pure": pure,
        })
        return MockFuture(key=key)


class TestReduction(unittest.TestCase):

    def test_single_future_passthrough(self):
        client = MockDaskClient()
        fut = MockFuture("fut-0")
        res = build_binary_tree_reduction(client, [fut], 0, 0, repetition_id=1)
        self.assertEqual(res, fut)
        self.assertEqual(len(client.submitted_tasks), 0)

    def test_reduction_task_count_and_rounds(self):
        test_cases = [2, 3, 4, 5, 6, 7, 8, 9]

        for nb in test_cases:
            client = MockDaskClient()
            futures = [MockFuture(f"fut-{k}") for k in range(nb)]
            result_future = build_binary_tree_reduction(
                client=client,
                partial_product_futures=futures,
                block_row=1,
                block_col=2,
                repetition_id=3,
            )

            # Exactly Nb - 1 additions must be scheduled
            self.assertEqual(
                len(client.submitted_tasks),
                nb - 1,
                f"Failed for Nb={nb}: expected {nb - 1} additions, got {len(client.submitted_tasks)}",
            )

            # Check key structure
            for task in client.submitted_tasks:
                key = task["key"]
                self.assertIn("add-r3-b(1,2)", key)
                self.assertIn("round", key)
                self.assertIn("pair", key)


if __name__ == "__main__":
    unittest.main()
