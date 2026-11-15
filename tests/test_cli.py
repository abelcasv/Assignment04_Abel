"""
tests/test_cli.py — Unit Tests for CLI Parameter Validation and DEC-01 Enforcement

Verifies:
- validate_configuration enforces positive integers and DEC-01 permission.
- CLI --check-config rejects p=0, p=-2, and unauthorized p > 1 without launching clusters.
- CLI --check-config accepts p=1.
- Injected cluster managers in CampaignRunner and PilotRunner strictly enforce fail-closed contract:
  * Constructor validation (worker_count=1, 2, 0, -1, 1.5, True, False, None, missing, np.int64(1))
  * Run-time validation prior to manifest reading, matrix loading, and cluster startup
  * Post-startup verification of observed worker count
"""

from io import StringIO
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch
import numpy as np

from src.cli import main as cli_main, validate_configuration
from src.dask_matrix.campaign_runner import CampaignRunner
from src.dask_matrix.cluster_manager import (
    ClusterManager,
    ClusterTopology,
    validate_injected_cluster_manager,
)
from src.dask_matrix.pilot_runner import PilotRunner


class TestValidateConfiguration(unittest.TestCase):
    """Unit tests for validate_configuration function."""

    def test_valid_configuration_accepted(self):
        """Valid n, b, p=1 succeeds."""
        with patch("sys.stdout", new_callable=StringIO):
            self.assertTrue(validate_configuration(250, 250, 1))

    def test_non_positive_worker_count_raises_value_error(self):
        """p <= 0 raises ValueError."""
        for p in [0, -1, -5]:
            with self.assertRaises(ValueError) as ctx:
                validate_configuration(250, 250, p)
            self.assertIn("positive integer", str(ctx.exception))

    def test_invalid_types_raise_type_error(self):
        """Non-integral types and booleans raise TypeError."""
        for p in [1.5, 0.5, "1", "2", True, False, None, ""]:
            with self.assertRaises(TypeError):
                validate_configuration(250, 250, p)  # type: ignore

    def test_multiworker_blocked_without_permission(self):
        """p > 1 raises PermissionError when DEC-01 permission is absent."""
        for p in [2, 4, 8]:
            with self.assertRaises(PermissionError) as ctx:
                validate_configuration(250, 250, p)
            self.assertIn("DEC-01", str(ctx.exception))


class TestCliCheckConfig(unittest.TestCase):
    """Unit tests for CLI --check-config argument execution."""

    def test_check_config_valid_single_worker_succeeds(self):
        """--check-config with default p=1 or -p 1 returns exit code 0."""
        with patch("sys.stdout", new_callable=StringIO):
            ret = cli_main(["--check-config", "--n", "250", "-b", "250", "-p", "1"])
            self.assertEqual(ret, 0)

    def test_check_config_zero_workers_fails(self):
        """--check-config -p 0 returns exit code 1."""
        with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO) as err:
            ret = cli_main(["--check-config", "--n", "250", "-b", "250", "-p", "0"])
            self.assertEqual(ret, 1)
            self.assertIn("positive integer", err.getvalue())

    def test_check_config_negative_workers_fails(self):
        """--check-config -p -2 returns exit code 1."""
        with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO) as err:
            ret = cli_main(["--check-config", "--n", "250", "-b", "250", "-p", "-2"])
            self.assertEqual(ret, 1)
            self.assertIn("positive integer", err.getvalue())

    def test_check_config_multiworker_fails_dec01(self):
        """--check-config -p 2 returns exit code 1 due to DEC-01 permission error."""
        with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO) as err:
            ret = cli_main(["--check-config", "--n", "250", "-b", "250", "-p", "2"])
            self.assertEqual(ret, 1)
            self.assertIn("DEC-01", err.getvalue())


class TestRunnerClusterManagerInjection(unittest.TestCase):
    """
    Exhaustive contract tests for injected cluster managers in CampaignRunner and PilotRunner.
    Covers all 10 edge cases and fail-closed validation semantics.
    """

    def _create_duck_double(self, worker_count_val=None, has_attr=True):
        """Helper to create a duck-typed test double with explicit worker_count."""
        class DuckClusterManager:
            pass
        obj = DuckClusterManager()
        if has_attr:
            obj.worker_count = worker_count_val
        return obj

    # --- CampaignRunner Constructor Validation ---

    def test_campaign_runner_accepts_single_worker_manager(self):
        """CampaignRunner accepts injected manager with worker_count=1."""
        cm = ClusterManager(worker_count=1)
        runner = CampaignRunner(cluster_manager=cm)
        self.assertIs(runner.cluster_manager, cm)

    def test_campaign_runner_accepts_numpy_int64(self):
        """CampaignRunner accepts injected manager with worker_count=np.int64(1)."""
        cm = self._create_duck_double(worker_count_val=np.int64(1))
        runner = CampaignRunner(cluster_manager=cm)
        self.assertIs(runner.cluster_manager, cm)

    def test_campaign_runner_rejects_worker_count_two(self):
        """CampaignRunner rejects injected manager with worker_count=2 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=2)
        with self.assertRaises(ValueError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("requires worker_count=1", str(ctx.exception))

    def test_campaign_runner_rejects_worker_count_zero(self):
        """CampaignRunner rejects injected manager with worker_count=0 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=0)
        with self.assertRaises(ValueError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("positive integer", str(ctx.exception))

    def test_campaign_runner_rejects_worker_count_negative(self):
        """CampaignRunner rejects injected manager with worker_count=-1 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=-1)
        with self.assertRaises(ValueError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("positive integer", str(ctx.exception))

    def test_campaign_runner_rejects_worker_count_float(self):
        """CampaignRunner rejects injected manager with worker_count=1.5 (TypeError)."""
        cm = self._create_duck_double(worker_count_val=1.5)
        with self.assertRaises(TypeError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_campaign_runner_rejects_worker_count_bool(self):
        """CampaignRunner rejects injected manager with worker_count=True and False (TypeError)."""
        for b in [True, False]:
            cm = self._create_duck_double(worker_count_val=b)
            with self.assertRaises(TypeError) as ctx:
                CampaignRunner(cluster_manager=cm)
            self.assertIn("must be an integer", str(ctx.exception))

    def test_campaign_runner_rejects_worker_count_none(self):
        """CampaignRunner rejects injected manager with worker_count=None (TypeError)."""
        cm = self._create_duck_double(worker_count_val=None)
        with self.assertRaises(TypeError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_campaign_runner_rejects_missing_worker_count(self):
        """CampaignRunner rejects injected manager missing worker_count (AttributeError)."""
        cm = self._create_duck_double(has_attr=False)
        with self.assertRaises(AttributeError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("worker_count", str(ctx.exception))

    def test_campaign_runner_rejects_unconfigured_mock(self):
        """CampaignRunner rejects generic MagicMock without worker_count or topology (AttributeError)."""
        cm = MagicMock(spec=[])
        with self.assertRaises(AttributeError) as ctx:
            CampaignRunner(cluster_manager=cm)
        self.assertIn("worker_count", str(ctx.exception))

    # --- PilotRunner Constructor Validation ---

    def test_pilot_runner_accepts_single_worker_manager(self):
        """PilotRunner accepts injected manager with worker_count=1."""
        cm = ClusterManager(worker_count=1)
        runner = PilotRunner(cluster_manager=cm)
        self.assertIs(runner.cluster_manager, cm)

    def test_pilot_runner_accepts_numpy_int64(self):
        """PilotRunner accepts injected manager with worker_count=np.int64(1)."""
        cm = self._create_duck_double(worker_count_val=np.int64(1))
        runner = PilotRunner(cluster_manager=cm)
        self.assertIs(runner.cluster_manager, cm)

    def test_pilot_runner_rejects_worker_count_two(self):
        """PilotRunner rejects injected manager with worker_count=2 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=2)
        with self.assertRaises(ValueError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("requires worker_count=1", str(ctx.exception))

    def test_pilot_runner_rejects_worker_count_zero(self):
        """PilotRunner rejects injected manager with worker_count=0 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=0)
        with self.assertRaises(ValueError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("positive integer", str(ctx.exception))

    def test_pilot_runner_rejects_worker_count_negative(self):
        """PilotRunner rejects injected manager with worker_count=-1 (ValueError)."""
        cm = self._create_duck_double(worker_count_val=-1)
        with self.assertRaises(ValueError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("positive integer", str(ctx.exception))

    def test_pilot_runner_rejects_worker_count_float(self):
        """PilotRunner rejects injected manager with worker_count=1.5 (TypeError)."""
        cm = self._create_duck_double(worker_count_val=1.5)
        with self.assertRaises(TypeError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_pilot_runner_rejects_worker_count_bool(self):
        """PilotRunner rejects injected manager with worker_count=True and False (TypeError)."""
        for b in [True, False]:
            cm = self._create_duck_double(worker_count_val=b)
            with self.assertRaises(TypeError) as ctx:
                PilotRunner(cluster_manager=cm)
            self.assertIn("must be an integer", str(ctx.exception))

    def test_pilot_runner_rejects_worker_count_none(self):
        """PilotRunner rejects injected manager with worker_count=None (TypeError)."""
        cm = self._create_duck_double(worker_count_val=None)
        with self.assertRaises(TypeError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_pilot_runner_rejects_missing_worker_count(self):
        """PilotRunner rejects injected manager missing worker_count (AttributeError)."""
        cm = self._create_duck_double(has_attr=False)
        with self.assertRaises(AttributeError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("worker_count", str(ctx.exception))

    def test_pilot_runner_rejects_unconfigured_mock(self):
        """PilotRunner rejects generic MagicMock without worker_count or topology (AttributeError)."""
        cm = MagicMock(spec=[])
        with self.assertRaises(AttributeError) as ctx:
            PilotRunner(cluster_manager=cm)
        self.assertIn("worker_count", str(ctx.exception))


class TestRunnerSideEffectOrdering(unittest.TestCase):
    """
    Verifies that runtime manager validation fails closed BEFORE manifest reading,
    matrix CSV reads, cluster startup, or resource initialization.
    """

    def test_campaign_runner_runtime_validation_aborts_before_io_and_startup(self):
        """CampaignRunner.run() rejects mutated/invalid manager before manifest/matrix IO or cluster start."""
        mock_cm = MagicMock()
        mock_cm.worker_count = 1  # Passes constructor
        runner = CampaignRunner(cluster_manager=mock_cm)

        # Mutate to invalid worker_count after constructor
        mock_cm.worker_count = 2

        with patch("src.dask_matrix.campaign_runner.load_sha256_manifest") as mock_load_manifest, \
             patch.object(CampaignRunner, "_verify_and_load_matrix") as mock_verify_load, \
             patch.object(mock_cm, "start") as mock_cm_start:

            with self.assertRaises(ValueError) as ctx:
                runner.run()

            self.assertIn("requires worker_count=1", str(ctx.exception))
            mock_load_manifest.assert_not_called()
            mock_verify_load.assert_not_called()
            mock_cm_start.assert_not_called()

    def test_pilot_runner_runtime_validation_aborts_before_io_and_startup(self):
        """PilotRunner.run() rejects mutated/invalid manager before manifest/matrix IO or cluster start."""
        mock_cm = MagicMock()
        mock_cm.worker_count = 1  # Passes constructor
        runner = PilotRunner(cluster_manager=mock_cm)

        # Mutate to invalid worker_count after constructor
        mock_cm.worker_count = 2

        with patch("src.dask_matrix.pilot_runner.load_sha256_manifest") as mock_load_manifest, \
             patch.object(PilotRunner, "_verify_and_load_matrix") as mock_verify_load, \
             patch.object(mock_cm, "start") as mock_cm_start:

            with self.assertRaises(ValueError) as ctx:
                runner.run()

            self.assertIn("requires worker_count=1", str(ctx.exception))
            mock_load_manifest.assert_not_called()
            mock_verify_load.assert_not_called()
            mock_cm_start.assert_not_called()

    def test_campaign_runner_post_startup_observed_worker_count_verified(self):
        """CampaignRunner.run() rejects cluster startup if observed workers != 1."""
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_csv = os.path.join(tmp_dir, "raw.csv")
            meta_dir = os.path.join(tmp_dir, "meta")
            os.makedirs(meta_dir, exist_ok=True)

            mock_cm = MagicMock()
            mock_cm.worker_count = 1
            bad_topology = ClusterTopology(
                worker_count_requested=1,
                worker_count_observed=2,  # Multi-worker anomaly observed
                threads_per_worker=1,
                driver_pid=100,
                scheduler_pid=100,
                nanny_pids=[101, 102],
                worker_pids=[201, 202],
                all_pids=[100, 101, 102, 201, 202],
                cluster_startup_s=0.2,
            )
            mock_cm.start.return_value = bad_topology

            runner = CampaignRunner(
                results_csv=res_csv,
                metadata_dir=meta_dir,
                cluster_manager=mock_cm,
            )

            with patch("src.dask_matrix.campaign_runner.load_sha256_manifest", return_value={}), \
                 patch.object(CampaignRunner, "_verify_and_load_matrix", return_value=(np.ones((250, 250)), "digest")), \
                 patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul") as mock_engine, \
                 patch("src.dask_matrix.campaign_runner.ResourceWatchdog") as mock_wd:

                summary = runner.run()

                self.assertEqual(summary["completed_samples"], 0)
                self.assertEqual(summary["failed_samples"], 36)
                mock_engine.assert_not_called()
                mock_wd.assert_not_called()

                with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
                self.assertIn("failed worker count verification", sidecar["failure_reason"])

    def test_pilot_runner_post_startup_observed_worker_count_verified(self):
        """PilotRunner.run() rejects cluster startup if observed workers != 1."""
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_csv = os.path.join(tmp_dir, "raw.csv")
            meta_dir = os.path.join(tmp_dir, "meta")
            os.makedirs(meta_dir, exist_ok=True)

            mock_cm = MagicMock()
            mock_cm.worker_count = 1
            bad_topology = ClusterTopology(
                worker_count_requested=1,
                worker_count_observed=2,  # Multi-worker anomaly observed
                threads_per_worker=1,
                driver_pid=100,
                scheduler_pid=100,
                nanny_pids=[101, 102],
                worker_pids=[201, 202],
                all_pids=[100, 101, 102, 201, 202],
                cluster_startup_s=0.2,
            )
            mock_cm.start.return_value = bad_topology

            runner = PilotRunner(
                results_csv=res_csv,
                metadata_dir=meta_dir,
                cluster_manager=mock_cm,
            )

            with patch("src.dask_matrix.pilot_runner.load_sha256_manifest", return_value={}), \
                 patch.object(PilotRunner, "_verify_and_load_matrix", return_value=(np.ones((250, 250)), "digest")), \
                 patch("src.dask_matrix.pilot_runner.DistributedBlockMatmul") as mock_engine, \
                 patch("src.dask_matrix.pilot_runner.ResourceWatchdog") as mock_wd:

                summary = runner.run()

                self.assertEqual(summary["completed_samples"], 0)
                self.assertEqual(summary["failed_samples"], 6)
                mock_engine.assert_not_called()
                mock_wd.assert_not_called()

                with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
                self.assertIn("failed worker count verification", sidecar["failure_reason"])


if __name__ == "__main__":
    unittest.main()
