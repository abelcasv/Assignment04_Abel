"""
tests/test_worker_count_validation.py — Verification of Worker Count & Single-Host Topology Validation

Ensures that:
- Single worker (p=1) and valid multi-worker counts (p > 1) are accepted as positive integers.
- Process counts below 1 (p <= 0) raise ValueError.
- Non-integral types, booleans, floats, and strings raise TypeError.
- ClusterManager validates constructor parameters strictly before cluster creation.
- ClusterManager rejects invalid values prior to any cluster initialization.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.dask_matrix.cluster_manager import (
    ClusterManager,
    ClusterTopology,
    validate_cluster_topology,
    validate_injected_cluster_manager,
    validate_worker_count,
)


class TestWorkerCountValidation(unittest.TestCase):

    def test_single_worker_permitted(self):
        """p=1 must be accepted as a valid positive integer worker count."""
        self.assertTrue(validate_worker_count(1))
        # Support numpy integer scalar for p=1
        self.assertTrue(validate_worker_count(np.int64(1)))

    def test_non_positive_values_rejected_with_value_error(self):
        """p <= 0 must raise ValueError."""
        non_positive_counts = [0, -1, -10]
        for p in non_positive_counts:
            with self.assertRaises(ValueError) as ctx:
                validate_worker_count(p)
            self.assertIn("positive integer", str(ctx.exception))

        with self.assertRaises(ValueError):
            validate_worker_count(np.int64(0))
        with self.assertRaises(ValueError):
            validate_worker_count(np.int64(-1))

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
                validate_worker_count(val)  # type: ignore
            self.assertIn("must be an integer", str(ctx.exception))

    def test_multiworker_positive_integers_permitted(self):
        """Positive integer worker counts p > 1 for local strong scaling are permitted."""
        for p in [2, 4, 8, 10]:
            self.assertTrue(validate_worker_count(p))
        self.assertTrue(validate_worker_count(np.int64(2)))


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
    """Verifies that invalid configurations never create clusters."""

    def test_invalid_configurations_never_initialize_cluster(self):
        """Cluster initialization must never be reached for p=0, p=-1, or invalid types."""
        # 1. Non-positive p=0 rejected at constructor before start
        with self.assertRaises(ValueError):
            ClusterManager(worker_count=0)

        # 2. Negative p=-1 rejected at constructor before start
        with self.assertRaises(ValueError):
            ClusterManager(worker_count=-1)

        # 3. Invalid type rejected at constructor before start
        with self.assertRaises(TypeError):
            ClusterManager(worker_count="2")  # type: ignore

        # 4. Mutated invalid worker_count rejected in start() before LocalCluster init
        cm_mutated = ClusterManager(worker_count=1)
        cm_mutated.worker_count = 0
        with patch("distributed.LocalCluster", create=True) as mock_lc:
            with self.assertRaises(ValueError) as ctx:
                cm_mutated.start()
            self.assertIn("positive integer", str(ctx.exception))
            mock_lc.assert_not_called()


class TestInjectedManagerAndTopologyValidation(unittest.TestCase):
    """Verifies fail-closed contracts for injected managers and ClusterTopology descriptors."""

    def test_validate_injected_cluster_manager_rejects_unconfigured_mock_fallback(self):
        """Bare MagicMock or missing worker_count must fail even if start.return_value looks valid."""
        bare_mock = MagicMock()
        bare_mock.start.return_value = ClusterTopology(
            worker_count_requested=1,
            worker_count_observed=1,
            threads_per_worker=1,
            driver_pid=100,
            scheduler_pid=100,
        )
        with self.assertRaises(TypeError):
            validate_injected_cluster_manager(bare_mock, expected_worker_count=1)

        class MissingWorkerCountDouble:
            class _Start:
                return_value = ClusterTopology(
                    worker_count_requested=1,
                    worker_count_observed=1,
                    threads_per_worker=1,
                    driver_pid=100,
                    scheduler_pid=100,
                )
            start = _Start()

        with self.assertRaises(AttributeError):
            validate_injected_cluster_manager(MissingWorkerCountDouble(), expected_worker_count=1)

    def test_validate_cluster_topology_fail_closed(self):
        """validate_cluster_topology rejects None, missing attrs, wrong types, and count mismatches."""
        with self.assertRaises(ValueError):
            validate_cluster_topology(None, expected_worker_count=1)

        class MissingObserved:
            worker_count_requested = 1

        with self.assertRaises(ValueError):
            validate_cluster_topology(MissingObserved(), expected_worker_count=1)

        with self.assertRaises(TypeError):
            validate_cluster_topology(MagicMock(), expected_worker_count=1)

        bad_req = ClusterTopology(
            worker_count_requested=2,
            worker_count_observed=1,
            threads_per_worker=1,
            driver_pid=1,
            scheduler_pid=1,
        )
        with self.assertRaises(ValueError):
            validate_cluster_topology(bad_req, expected_worker_count=1)

        bad_threads = ClusterTopology(
            worker_count_requested=1,
            worker_count_observed=1,
            threads_per_worker=2,
            driver_pid=1,
            scheduler_pid=1,
        )
        with self.assertRaises(ValueError):
            validate_cluster_topology(bad_threads, expected_worker_count=1, expected_threads_per_worker=1)


class TestClusterManagerStateAndTimeoutGuards(unittest.TestCase):
    """Verifies repeat-start rejection, close topology reset, and timeout_s/startup guards."""

    @patch("src.dask_matrix.cluster_manager.validate_worker_count")
    @patch("distributed.LocalCluster")
    @patch("distributed.Client")
    def test_repeat_start_raises_and_close_resets_topology_for_restart(
        self, mock_client_cls, mock_cluster_cls, mock_perm
    ):
        """Calling start() while active raises RuntimeError; close() resets topology so start() works again."""
        mock_cluster = MagicMock()
        mock_cluster.workers = {"worker_0": MagicMock(pid=os.getpid())}
        mock_cluster.scheduler.pid = os.getpid()
        mock_cluster_cls.return_value = mock_cluster
        mock_client_cls.return_value = MagicMock()

        cm = ClusterManager(worker_count=1, threads_per_worker=1)
        topo = cm.start(timeout_s=10.0)
        self.assertIsNotNone(cm.topology)
        self.assertEqual(topo.worker_count_observed, 1)

        with self.assertRaises(RuntimeError):
            cm.start(timeout_s=10.0)
        mock_cluster_cls.assert_called_once()

        cm.close(timeout_s=1.0)
        self.assertIsNone(cm.cluster)
        self.assertIsNone(cm.client)
        self.assertIsNone(cm.topology)

        cm.start(timeout_s=10.0)
        self.assertEqual(mock_cluster_cls.call_count, 2)

    def test_invalid_timeout_s_rejected_without_close_or_cluster(self):
        """Non-positive, NaN, and non-real timeout_s fail before try block without calling close()."""
        cm = ClusterManager(worker_count=1)
        with patch.object(cm, "close") as mock_close, patch("distributed.LocalCluster", create=True) as mock_lc:
            for bad_timeout in (0.0, -5.0, float("nan"), float("inf")):
                with self.assertRaises(TimeoutError):
                    cm.start(timeout_s=bad_timeout)
            for bad_type in (True, False, "10", None):
                with self.assertRaises(TypeError):
                    cm.start(timeout_s=bad_type)  # type: ignore
            mock_close.assert_not_called()
            mock_lc.assert_not_called()

    @patch("src.dask_matrix.cluster_manager.validate_worker_count")
    @patch("distributed.LocalCluster")
    @patch("distributed.Client")
    def test_small_timeout_startup_failure_uses_positive_close_floor_and_bad_workers_tears_down(
        self, mock_client_cls, mock_cluster_cls, mock_perm
    ):
        """Startup failures enforce >=1.0s close floor and reject observed worker count mismatch."""
        mock_cluster = MagicMock()
        mock_cluster.workers = {}
        mock_cluster.scheduler.pid = os.getpid()
        mock_cluster_cls.return_value = mock_cluster
        mock_client_cls.return_value = MagicMock()

        cm = ClusterManager(worker_count=1)
        with patch.object(cm, "close", wraps=cm.close) as spy_close:
            with self.assertRaises(ValueError) as ctx:
                cm.start(timeout_s=0.2)
            self.assertIn("failed worker count verification", str(ctx.exception))
            spy_close.assert_called_once()
            self.assertGreaterEqual(spy_close.call_args.kwargs["timeout_s"], 1.0)
        self.assertIsNone(cm.topology)
        self.assertIsNone(cm.cluster)
        self.assertIsNone(cm.client)


if __name__ == "__main__":
    unittest.main()
