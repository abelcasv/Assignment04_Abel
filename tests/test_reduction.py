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

            # Check key structure and tree depth = ceil(log2(Nb))
            rounds_seen = set()
            depth_by_key = {f"fut-{k}": 0 for k in range(nb)}
            for task in client.submitted_tasks:
                key = task["key"]
                self.assertIn("add-r3-b(1,2)", key)
                self.assertIn("round", key)
                self.assertIn("pair", key)
                round_part = key.split("-round")[1].split("-pair")[0]
                rounds_seen.add(int(round_part))
                left, right = task["args"]
                depth_by_key[key] = max(depth_by_key[left.key], depth_by_key[right.key]) + 1

            expected_depth = math.ceil(math.log2(nb))
            self.assertEqual(len(rounds_seen), expected_depth)
            self.assertEqual(max(rounds_seen) + 1, expected_depth)
            self.assertEqual(depth_by_key[result_future.key], expected_depth)

    def test_empty_futures_raises_value_error(self):
        client = MockDaskClient()
        with self.assertRaises(ValueError):
            build_binary_tree_reduction(client, [], 0, 0)

    def test_key_namespace_isolation(self):
        client = MockDaskClient()
        futures = [MockFuture("f0"), MockFuture("f1")]
        res = build_binary_tree_reduction(
            client=client,
            partial_product_futures=futures,
            block_row=0,
            block_col=1,
            repetition_id=2,
            key_namespace="n500-c4",
        )
        self.assertEqual(res.key, "add-n500-c4-r2-b(0,1)-round0-pair0")


if __name__ == "__main__":
    unittest.main()
