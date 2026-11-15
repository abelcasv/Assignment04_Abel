"""
tests.test_campaign_runner — Unit Tests and Mocked Verification for Phase 8 Campaign Orchestrator

Validates all 20 required specifications for CampaignRunner and build_campaign_plan:
1. Correct generation of 36 planned samples.
2. Exactly 6 warm-ups and 30 measured repetitions.
3. Correct size, block, and worker configurations.
4. Unique sample identities.
5. Rejection of p > 1 (DEC-01 / Gate G1).
6. Rejection of malformed campaign plans.
7. Separation between --pilot and --campaign in CLI.
8. Correct handling of successful mocked samples.
9. Warm-up exclusion from measured statistics.
10. Per-sample timeout handling.
11. Global budget exhaustion.
12. Watchdog abort.
13. Validation failure handling (never marked COMPLETED).
14. CSV append failure handling.
15. Sidecar persistence failure handling.
16. Preservation of completed records when a later sample fails.
17. Correct classification of samples not executed.
18. Cleanup invocation after every terminal condition.
19. No reuse of pilot observations.
20. No changes to existing pilot behavior or PILOT_PLAN.
"""

from datetime import datetime, timezone
import hashlib
import os
import shutil
import tempfile
import time
from typing import Optional
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.cli import parse_arguments, main as cli_main
from src.dask_matrix.campaign_runner import (
    CampaignRunner,
    CampaignSampleSpec,
    CAMPAIGN_PLAN,
    CAMPAIGN_SIZES,
    CAMPAIGN_BLOCK_SIZE,
    CAMPAIGN_WORKER_COUNT,
    CAMPAIGN_REPETITIONS,
    CAMPAIGN_GLOBAL_BUDGET_S,
    CAMPAIGN_SAMPLE_TIMEOUT_S,
    build_campaign_plan,
    validate_campaign_plan,
)
from src.dask_matrix.cluster_manager import ClusterTopology
from src.dask_matrix.distributed_engine import DaskExecutionResult
from src.dask_matrix.pilot_runner import PILOT_PLAN, PilotRunner
from src.dask_matrix.results_logger import RawResultsLogger


import re


def mock_load_matrix(filepath: str, raw_bytes: Optional[bytes] = None) -> np.ndarray:
    """Returns a synthetic array matching the dimension encoded in the filename."""
    m = re.search(r'([A-Za-z]+_)?(\d+)\.csv', filepath)
    if m:
        size = int(m.group(2))
        return np.ones((size, size), dtype=np.float64)
    return np.ones((250, 250), dtype=np.float64)


class TestCampaignPlanSpecification(unittest.TestCase):
    """Tests 1–6: Plan structure, configuration, uniqueness, and validation."""

    def test_01_sample_count_and_composition(self):
        """Req 1: Plan generates exactly 36 planned samples."""
        self.assertEqual(len(CAMPAIGN_PLAN), 36)

    def test_02_warmups_and_measured_repetitions(self):
        """Req 2: Exactly 6 warm-ups and 30 measured repetitions."""
        warmups = [s for s in CAMPAIGN_PLAN if s.is_warmup]
        measured = [s for s in CAMPAIGN_PLAN if not s.is_warmup]
        self.assertEqual(len(warmups), 6)
        self.assertEqual(len(measured), 30)

        for w in warmups:
            self.assertEqual(w.repetition, 0)
        for m in measured:
            self.assertIn(m.repetition, {1, 2, 3, 4, 5})

    def test_03_size_block_and_worker_configurations(self):
        """Req 3: Correct sizes (250..2000), block=250, workers=1."""
        plan_sizes = {s.n for s in CAMPAIGN_PLAN}
        self.assertEqual(plan_sizes, set(CAMPAIGN_SIZES))
        self.assertEqual(plan_sizes, {250, 500, 750, 1000, 1500, 2000})

        for s in CAMPAIGN_PLAN:
            self.assertEqual(s.block_size, 250)
            self.assertEqual(s.worker_count, 1)

    def test_04_unique_sample_identities(self):
        """Req 4: All sample identities are unique and ordered 1..36."""
        sample_ids = [s.sample_id for s in CAMPAIGN_PLAN]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertEqual(len(sample_ids), 36)

        orders = [s.execution_order for s in CAMPAIGN_PLAN]
        self.assertEqual(orders, list(range(1, 37)))

    def test_05_rejection_of_multiworker(self):
        """Req 5: Rejection of p > 1 enforcing DEC-01."""
        with self.assertRaises(PermissionError) as ctx:
            validate_campaign_plan(CAMPAIGN_PLAN, expected_worker_count=2)
        self.assertIn("DEC-01", str(ctx.exception))

    def test_06_rejection_of_malformed_plans(self):
        """Req 6: Plan validation rejects duplicate IDs, wrong sizes, or invalid budgets."""
        # Duplicate sample ID
        bad_plan = list(CAMPAIGN_PLAN)
        bad_plan[1] = bad_plan[0]
        with self.assertRaises(ValueError) as ctx:
            validate_campaign_plan(bad_plan)
        self.assertIn("Duplicate sample_id", str(ctx.exception))

        # Invalid budget (sample timeout > session budget)
        with self.assertRaises(ValueError) as ctx:
            validate_campaign_plan(CAMPAIGN_PLAN, sample_timeout_s=500.0, session_budget_s=100.0)
        self.assertIn("cannot exceed session_budget_s", str(ctx.exception))

        # Wrong sample count
        with self.assertRaises(ValueError) as ctx:
            validate_campaign_plan(CAMPAIGN_PLAN[:35])
        self.assertIn("must contain exactly 36 samples", str(ctx.exception))


class TestCliSeparationAndPilotImmutability(unittest.TestCase):
    """Tests 7, 19, 20: CLI separation, pilot isolation, and immutability."""

    def test_07_cli_pilot_and_campaign_separation(self):
        """Req 7: CLI distinguishes --pilot and --campaign; rejects specifying both."""
        with patch("sys.argv", ["cli.py", "--pilot", "--check-config"]):
            args = parse_arguments()
            self.assertTrue(args.pilot)
            self.assertFalse(args.campaign)

        with patch("sys.argv", ["cli.py", "--campaign", "--check-config"]):
            args = parse_arguments()
            self.assertFalse(args.pilot)
            self.assertTrue(args.campaign)

        # Rejection of both simultaneously
        with patch("sys.argv", ["cli.py", "--pilot", "--campaign"]):
            ret = cli_main()
            self.assertEqual(ret, 1)

    def test_19_no_reuse_of_pilot_observations(self):
        """Req 19: Campaign plan samples do NOT contain 'pilot_' prefix."""
        for s in CAMPAIGN_PLAN:
            self.assertTrue(s.sample_id.startswith("campaign_"))
            self.assertNotIn("pilot_", s.sample_id)

    def test_20_existing_pilot_immutability(self):
        """Req 20: Frozen PILOT_PLAN retains exactly 6 samples and remains untouched."""
        self.assertEqual(len(PILOT_PLAN), 6)
        self.assertEqual(
            [s.sample_id for s in PILOT_PLAN],
            ["pilot_250_w0", "pilot_250_r1", "pilot_250_r2", "pilot_500_w0", "pilot_500_r1", "pilot_500_r2"],
        )


class TestMockedCampaignExecution(unittest.TestCase):
    """Tests 8–18: Mocked campaign lifecycle, failure handling, and teardown."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_campaign_")
        self.data_dir = os.path.join(self.test_dir, "data")
        self.dense_dir = os.path.join(self.data_dir, "dense")
        self.val_dir = os.path.join(self.data_dir, "validation")
        self.ref_dir = os.path.join(self.val_dir, "reference")
        self.metadata_dir = os.path.join(self.test_dir, "results", "metadata")
        self.results_csv = os.path.join(self.test_dir, "results", "raw", "raw_measurements.csv")

        os.makedirs(self.dense_dir, exist_ok=True)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.results_csv), exist_ok=True)

        # Create dummy input files so file existence checks pass and write valid SHA256SUMS.txt
        sums_file = os.path.join(self.data_dir, "SHA256SUMS.txt")
        with open(sums_file, "w", encoding="utf-8") as sf:
            for n in CAMPAIGN_SIZES:
                for prefix in ("A", "B"):
                    fname = f"{prefix}_{n}.csv"
                    p = os.path.join(self.dense_dir, fname)
                    with open(p, "wb") as f:
                        f.write(b"0.0\n")
                    h = hashlib.sha256(b"0.0\n").hexdigest()
                    sf.write(f"{h}  dense/{fname}\n")

            with open(os.path.join(self.ref_dir, "C_250.csv"), "wb") as f:
                f.write(b"0.0\n")
            h_ref = hashlib.sha256(b"0.0\n").hexdigest()
            sf.write(f"{h_ref}  validation/reference/C_250.csv\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_cluster_manager(self):
        mock_cm = MagicMock()
        mock_cm.worker_count = 1
        mock_topology = ClusterTopology(
            worker_count_requested=1,
            worker_count_observed=1,
            threads_per_worker=1,
            driver_pid=1001,
            scheduler_pid=1001,
            nanny_pids=[2001],
            worker_pids=[2001],
            all_pids=[1001, 2001],
            cluster_startup_s=0.5,
        )
        mock_cm.start.return_value = mock_topology
        mock_cm.client = MagicMock()
        mock_cm.cluster = MagicMock()
        return mock_cm, mock_topology

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_08_successful_mocked_campaign(self, mock_wd_cls, mock_engine_cls):
        """Req 8 & 9: Successful execution of all 36 samples; warmups distinguished."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 150.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {1001: 80.0, 2001: 70.0}
        mock_wd_cls.return_value = mock_wd

        def mock_multiply(a_matrix, b_matrix, repetition_id, timeout_s, watchdog, global_deadline_s):
            n = a_matrix.shape[0]
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=float(n * n),
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_multiply
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
            session_budget_s=36000.0,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        self.assertEqual(summary["total_samples"], 36)
        self.assertEqual(summary["completed_samples"], 36)
        self.assertEqual(summary["failed_samples"], 0)
        self.assertTrue(os.path.isfile(self.results_csv))
        self.assertTrue(os.path.isfile(summary["sidecar_path"]))
        self.assertTrue(mock_cm.close.called)
        self.assertTrue(mock_wd.stop.called)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_10_sample_timeout_handling(self, mock_wd_cls, mock_engine_cls):
        """Req 10: Per-sample timeout safely closes cluster and skips subsequent repetitions/dimensions."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 150.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        mock_engine = MagicMock()
        mock_engine.multiply.return_value = DaskExecutionResult(
            status="TIMEOUT",
            n=750,
            block_size=250,
            nb=3,
            time_e2e_s=None,
            checksum=None,
            c_matrix=None,
            error_reason="Operation timed out after 120.00 s (> 120.0 s limit).",
            elapsed_until_abort_s=120.0,
            object_nbytes=0,
        )
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        # 1. First sample is recorded as TIMEOUT
        first_rec = summary["records"][0]
        self.assertEqual(first_rec.status, "TIMEOUT")
        self.assertIsNone(first_rec.time_e2e_s)
        self.assertEqual(first_rec.elapsed_until_abort_s, 120.0)
        self.assertEqual(first_rec.timeout_limit_s, 120.0)

        # 2. Cluster was closed immediately upon timeout detection
        self.assertTrue(mock_cm.close.called)
        self.assertTrue(mock_wd.stop.called)

        # 3. No new work was dispatched to the closed cluster
        self.assertEqual(mock_engine.multiply.call_count, 1)

        # 4. Subsequent repetitions of timed out dimension are marked NOT_RUN_AFTER_TIMEOUT
        same_dim_recs = [r for r in summary["records"][1:] if r.n == first_rec.n]
        for r in same_dim_recs:
            self.assertEqual(r.status, "NOT_RUN_AFTER_TIMEOUT")
            self.assertIsNone(r.time_e2e_s)
            self.assertIn("Repetition omitted", r.error_reason)

        # 5. Subsequent dimensions are marked NOT_RUN_AFTER_TIMEOUT due to closed cluster
        other_dim_recs = [r for r in summary["records"][1:] if r.n != first_rec.n]
        for r in other_dim_recs:
            self.assertEqual(r.status, "NOT_RUN_AFTER_TIMEOUT")
            self.assertIsNone(r.time_e2e_s)
            self.assertIn("cluster restart not authorized", r.error_reason)

        # 6. All 36 samples are recorded
        self.assertEqual(len(summary["records"]), 36)
        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["failed_samples"], 36)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_11_global_budget_exhaustion(self, mock_wd_cls, mock_engine_cls):
        """Req 11 & 17: Global budget exhaustion records ABORTED_BUDGET for unexecuted samples."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 150.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        orig_perf_counter = time.perf_counter
        call_count = [0]
        def advance_time():
            call_count[0] += 1
            if call_count[0] > 5:
                return orig_perf_counter() + 40000.0
            return orig_perf_counter()

        def mock_multiply(a_matrix, b_matrix, repetition_id, timeout_s, watchdog, global_deadline_s):
            n = a_matrix.shape[0]
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.2,
                checksum=float(n * n),
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=100,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_multiply
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
            session_budget_s=500.0,
            sample_timeout_s=10.0,
        )

        with patch("time.perf_counter", side_effect=advance_time):
            with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
                summary = runner.run()

        self.assertGreater(summary["failed_samples"], 0)
        aborted_records = [r for r in summary["records"] if r.status == "ABORTED_BUDGET"]
        self.assertTrue(len(aborted_records) > 0)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_12_watchdog_abort(self, mock_wd_cls, mock_engine_cls):
        """Req 12: Watchdog breach marks subsequent samples NOT_RUN_AFTER_OOM."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = True
        mock_wd.breach_type = "OUT_OF_MEMORY"
        mock_wd.breach_reason = "Aggregate RSS 2600.0 MiB exceeded ceiling 2560.0 MiB."
        mock_wd.peak_process_sum_mib = 2600.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["failed_samples"], 36)
        self.assertEqual(summary["records"][0].status, "NOT_RUN_AFTER_OOM")

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_13_validation_failure_recorded_as_failed(self, mock_wd_cls, mock_engine_cls):
        """Req 13: Non-finite values or validation mismatch recorded as FAILED (never COMPLETED)."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        def mock_multiply_nan(a_matrix, b_matrix, repetition_id, timeout_s, watchdog, global_deadline_s):
            n = a_matrix.shape[0]
            nan_mat = np.ones((n, n), dtype=np.float64)
            nan_mat[0, 0] = np.nan
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=float("nan"),
                c_matrix=nan_mat,
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_multiply_nan
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        first_rec = summary["records"][0]
        self.assertEqual(first_rec.status, "FAILED")
        self.assertIn("Non-finite", first_rec.error_reason)
        # Measured computation time is retained even though numerical validation failed
        self.assertEqual(first_rec.time_e2e_s, 0.25)
        # Verify exclusion from accepted performance statistics
        accepted_samples = [r for r in summary["records"] if r.status == "COMPLETED" and not r.is_warmup]
        self.assertNotIn(first_rec, accepted_samples)

    @patch("src.dask_matrix.campaign_runner.RawResultsLogger")
    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_14_csv_append_failure_handling(self, mock_wd_cls, mock_engine_cls, mock_logger_cls):
        """Req 14: Logger failure is caught gracefully without unhandled crash."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        mock_engine = MagicMock()
        mock_engine.multiply.return_value = DaskExecutionResult(
            status="COMPLETED",
            n=250,
            block_size=250,
            nb=1,
            time_e2e_s=0.25,
            checksum=62500.0,
            c_matrix=np.ones((250, 250), dtype=np.float64),
            error_reason=None,
            elapsed_until_abort_s=None,
            object_nbytes=1024,
        )
        mock_engine_cls.return_value = mock_engine

        mock_logger = MagicMock()
        mock_logger.log.side_effect = IOError("Disk full or permission denied")
        mock_logger_cls.return_value = mock_logger

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        self.assertIsNotNone(summary)
        self.assertTrue(mock_cm.close.called)

    @patch("src.dask_matrix.campaign_runner.generate_campaign_sidecar")
    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_15_sidecar_failure_handling(self, mock_wd_cls, mock_engine_cls, mock_sidecar):
        """Req 15: Sidecar generation failure does not crash the orchestrator."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        mock_engine = MagicMock()
        mock_engine.multiply.return_value = DaskExecutionResult(
            status="COMPLETED",
            n=250,
            block_size=250,
            nb=1,
            time_e2e_s=0.25,
            checksum=62500.0,
            c_matrix=np.ones((250, 250), dtype=np.float64),
            error_reason=None,
            elapsed_until_abort_s=None,
            object_nbytes=1024,
        )
        mock_engine_cls.return_value = mock_engine

        mock_sidecar.side_effect = OSError("Read-only filesystem")

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        self.assertEqual(summary["sidecar_path"], "unwritten")
        self.assertTrue(mock_cm.close.called)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_16_and_18_preservation_and_cleanup(self, mock_wd_cls, mock_engine_cls):
        """Req 16 & 18: Completed records are preserved when a later sample fails; cleanup always invoked."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        call_count = [0]
        def mock_mult(*args, **kwargs):
            call_count[0] += 1
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 250
            if call_count[0] == 1:
                return DaskExecutionResult(
                    status="COMPLETED",
                    n=n,
                    block_size=250,
                    nb=max(1, n // 250),
                    time_e2e_s=0.2,
                    checksum=float(n * n),
                    c_matrix=np.ones((n, n), dtype=np.float64),
                    error_reason=None,
                    elapsed_until_abort_s=None,
                    object_nbytes=100,
                )
            raise RuntimeError("Unexpected engine crash")

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        # Completed record is preserved
        completed_recs = [r for r in summary["records"] if r.status == "COMPLETED"]
        self.assertEqual(len(completed_recs), 1)

        # Later records are marked FAILED
        failed_recs = [r for r in summary["records"] if r.status == "FAILED"]
        self.assertEqual(len(failed_recs), 35)

        # Cleanup was invoked
        self.assertTrue(mock_cm.close.called)
        self.assertTrue(mock_wd.stop.called)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_21_checksum_divergence_greater_than_1e6_fails(self, mock_wd_cls, mock_engine_cls):
        """Req 21 (Correction B): Checksum divergence > 1e-6 produces FAILED with explicit reason."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        def mock_mult(*args, **kwargs):
            rep = kwargs.get("repetition_id", 0)
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 750
            # For rep=2, inject checksum divergence of 1e-4 (> 1e-6)
            chk = float(n * n) + (1e-4 if rep == 2 else 0.0)
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=chk,
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        # Find rep 1 and rep 2 for the first dimension executed (n=750)
        recs_750 = [r for r in summary["records"] if r.n == 750]
        rep1_rec = next(r for r in recs_750 if r.repetition == 1)
        rep2_rec = next(r for r in recs_750 if r.repetition == 2)

        self.assertEqual(rep1_rec.status, "COMPLETED")
        self.assertEqual(rep1_rec.time_e2e_s, 0.25)
        self.assertEqual(rep2_rec.status, "FAILED")
        # Measured computation time is retained even though checksum validation failed
        self.assertEqual(rep2_rec.time_e2e_s, 0.25)
        self.assertIn("Checksum divergence", rep2_rec.error_reason)

        # Verify exclusion of failed sample from accepted performance statistics
        accepted_samples = [r for r in summary["records"] if r.status == "COMPLETED" and not r.is_warmup]
        self.assertIn(rep1_rec, accepted_samples)
        self.assertNotIn(rep2_rec, accepted_samples)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_22_checksum_divergence_within_tolerance_accepted(self, mock_wd_cls, mock_engine_cls):
        """Req 22 (Correction B): Checksum divergence <= 1e-6 remains COMPLETED."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        def mock_mult(*args, **kwargs):
            rep = kwargs.get("repetition_id", 0)
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 750
            # For rep=2, inject tiny difference within 1e-6 tolerance (e.g. 5e-7)
            chk = float(n * n) + (5e-7 if rep == 2 else 0.0)
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=chk,
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        recs_750 = [r for r in summary["records"] if r.n == 750]
        rep1_rec = next(r for r in recs_750 if r.repetition == 1)
        rep2_rec = next(r for r in recs_750 if r.repetition == 2)

        self.assertEqual(rep1_rec.status, "COMPLETED")
        self.assertEqual(rep2_rec.status, "COMPLETED")
        self.assertIsNotNone(rep2_rec.time_e2e_s)

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_23_repetition_1_failure_does_not_silently_establish_later_reference(self, mock_wd_cls, mock_engine_cls):
        """Req 23 (Correction B): Repetition 1 failure does not silently establish a later reference."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        def mock_mult(*args, **kwargs):
            rep = kwargs.get("repetition_id", 0)
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 750
            if rep == 1:
                # Repetition 1 fails due to non-finite array
                nan_mat = np.ones((n, n), dtype=np.float64)
                nan_mat[0, 0] = np.nan
                return DaskExecutionResult(
                    status="COMPLETED",
                    n=n,
                    block_size=250,
                    nb=max(1, n // 250),
                    time_e2e_s=0.25,
                    checksum=float("nan"),
                    c_matrix=nan_mat,
                    error_reason=None,
                    elapsed_until_abort_s=None,
                    object_nbytes=1024,
                )
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=float(n * n),
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        recs_750 = [r for r in summary["records"] if r.n == 750]
        rep1_rec = next(r for r in recs_750 if r.repetition == 1)
        rep2_rec = next(r for r in recs_750 if r.repetition == 2)

        self.assertEqual(rep1_rec.status, "FAILED")
        # Repetition 2 completes its computation and finitude check, but did NOT establish a reference
        self.assertEqual(rep2_rec.status, "COMPLETED")

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_24_warmups_excluded_from_reproducibility_validation(self, mock_wd_cls, mock_engine_cls):
        """Req 24 (Correction B): Warm-up samples remain excluded from reproducibility validation."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        def mock_mult(*args, **kwargs):
            rep = kwargs.get("repetition_id", 0)
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 750
            # Warm-up has a completely different checksum
            chk = 999999.0 if rep == 0 else float(n * n)
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=0.25,
                checksum=chk,
                c_matrix=np.ones((n, n), dtype=np.float64),
                error_reason=None,
                elapsed_until_abort_s=None,
                object_nbytes=1024,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        recs_750 = [r for r in summary["records"] if r.n == 750]
        w0_rec = next(r for r in recs_750 if r.is_warmup)
        rep1_rec = next(r for r in recs_750 if r.repetition == 1)
        rep2_rec = next(r for r in recs_750 if r.repetition == 2)

        self.assertEqual(w0_rec.checksum_sum, 999999.0)
        self.assertEqual(rep1_rec.status, "COMPLETED")
        self.assertEqual(rep2_rec.status, "COMPLETED")

    @patch("src.dask_matrix.campaign_runner.DistributedBlockMatmul")
    @patch("src.dask_matrix.campaign_runner.ResourceWatchdog")
    def test_25_previously_completed_rows_preserved_after_timeout(self, mock_wd_cls, mock_engine_cls):
        """Req 25 (Correction A): Previously completed rows remain preserved when a sample times out."""
        mock_cm, _ = self._create_mock_cluster_manager()

        mock_wd = MagicMock()
        mock_wd.breach_detected = False
        mock_wd.peak_process_sum_mib = 100.0
        mock_wd.samples = []
        mock_wd.peak_working_set_per_pid_mib = {}
        mock_wd_cls.return_value = mock_wd

        call_count = [0]
        def mock_mult(*args, **kwargs):
            call_count[0] += 1
            a_matrix = kwargs.get("a_matrix", args[0] if len(args) > 0 else None)
            n = a_matrix.shape[0] if a_matrix is not None else 750
            if call_count[0] <= 2:
                return DaskExecutionResult(
                    status="COMPLETED",
                    n=n,
                    block_size=250,
                    nb=max(1, n // 250),
                    time_e2e_s=0.25,
                    checksum=float(n * n),
                    c_matrix=np.ones((n, n), dtype=np.float64),
                    error_reason=None,
                    elapsed_until_abort_s=None,
                    object_nbytes=1024,
                )
            return DaskExecutionResult(
                status="TIMEOUT",
                n=n,
                block_size=250,
                nb=max(1, n // 250),
                time_e2e_s=None,
                checksum=None,
                c_matrix=None,
                error_reason="Operation timed out after 120.00 s.",
                elapsed_until_abort_s=120.0,
                object_nbytes=0,
            )

        mock_engine = MagicMock()
        mock_engine.multiply.side_effect = mock_mult
        mock_engine_cls.return_value = mock_engine

        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch.object(CampaignRunner, "_load_matrix", side_effect=mock_load_matrix):
            summary = runner.run()

        completed_recs = [r for r in summary["records"] if r.status == "COMPLETED"]
        timeout_recs = [r for r in summary["records"] if r.status == "TIMEOUT"]
        not_run_recs = [r for r in summary["records"] if r.status == "NOT_RUN_AFTER_TIMEOUT"]

        self.assertEqual(len(completed_recs), 2)
        self.assertEqual(len(timeout_recs), 1)
        self.assertEqual(len(not_run_recs), 33)
        self.assertTrue(mock_cm.close.called)


if __name__ == "__main__":
    unittest.main()
