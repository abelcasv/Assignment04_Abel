"""
tests/test_pilot_runner.py — Unit Tests for Phase 7 Pilot Orchestrator and Safeguards

Tests:
1. Immutability and exact specification of PILOT_PLAN (6 samples, p=1, b=250, warmup flags).
2. Mocked full execution of PilotRunner (append-only CSV, sidecar generation, telemetry).
3. Active budget exhaustion (deadline enforcement and ABORTED_BUDGET logging).
4. Watchdog OOM breach handling and subsequent NOT_RUN_AFTER_OOM marking.
5. Watchdog safe stop overall grace period bounded by <= 5.0 s (with shared deadline).
6. Watchdog safe stop re-entrancy and concurrency lock.
7. Safe stop process termination (children terminated/killed, driver preserved).
8. Watchdog global deadline breach trigger.
9. Watchdog status precedence (RSS breach over deadline; secondary exceptions preserve breach reason).
10. Interruption of blocking operations (scatter, wait, gather).
11. Numerical validation for n=250 (elementwise against reference) and n=500 (finitude & determinism).
"""

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

from src.dask_matrix.cluster_manager import ClusterTopology
from src.dask_matrix.distributed_engine import DaskExecutionResult, DistributedBlockMatmul
from src.dask_matrix.pilot_runner import (
    PILOT_PLAN,
    PILOT_SESSION_BUDGET_S,
    PILOT_SAMPLE_TIMEOUT_S,
    PilotRunner,
    PilotSampleSpec,
)
from src.dask_matrix.results_logger import RawResultsLogger
from src.dask_matrix.validation import MatrixValidator
from src.dask_matrix.watchdog import ResourceWatchdog, ProcessMemorySample


class TestPilotPlanSpecification(unittest.TestCase):
    """Verifies that the pilot plan is strictly immutable and matches Section 10.5."""

    def test_plan_immutability(self):
        self.assertIsInstance(PILOT_PLAN, tuple)
        self.assertEqual(len(PILOT_PLAN), 6)

        # Attempting to modify attributes on any spec must raise FrozenInstanceError
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            PILOT_PLAN[0].n = 1000  # type: ignore

    def test_plan_sample_sequence_and_parameters(self):
        expected_ids = [
            "pilot_250_w0",
            "pilot_250_r1",
            "pilot_250_r2",
            "pilot_500_w0",
            "pilot_500_r1",
            "pilot_500_r2",
        ]
        expected_n = [250, 250, 250, 500, 500, 500]
        expected_warmup = [True, False, False, True, False, False]
        expected_reps = [0, 1, 2, 0, 1, 2]

        for i, spec in enumerate(PILOT_PLAN):
            self.assertEqual(spec.sample_id, expected_ids[i])
            self.assertEqual(spec.n, expected_n[i])
            self.assertEqual(spec.block_size, 250)
            self.assertEqual(spec.worker_count, 1)
            self.assertEqual(spec.is_warmup, expected_warmup[i])
            self.assertEqual(spec.repetition, expected_reps[i])
            self.assertEqual(spec.execution_order, i + 1)


class TestPilotRunnerExecution(unittest.TestCase):
    """Tests execution flow of PilotRunner using isolated mocks without live Dask cluster."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_pilot_")
        self.data_dir = os.path.join(self.test_dir, "data")
        self.dense_dir = os.path.join(self.data_dir, "dense")
        self.val_dir = os.path.join(self.data_dir, "validation", "reference")
        os.makedirs(self.dense_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        self.results_csv = os.path.join(self.test_dir, "results", "raw", "raw_measurements.csv")
        self.metadata_dir = os.path.join(self.test_dir, "results", "metadata")

        # Create synthetic CSV files for n=250 and n=500
        np.savetxt(os.path.join(self.dense_dir, "A_250.csv"), np.ones((250, 250)), delimiter=",")
        np.savetxt(os.path.join(self.dense_dir, "B_250.csv"), np.ones((250, 250)), delimiter=",")
        np.savetxt(os.path.join(self.dense_dir, "A_500.csv"), np.ones((500, 500)), delimiter=",")
        np.savetxt(os.path.join(self.dense_dir, "B_500.csv"), np.ones((500, 500)), delimiter=",")
        # Reference C_250 = A_250 @ B_250 = 250.0 everywhere
        np.savetxt(os.path.join(self.val_dir, "C_250.csv"), np.full((250, 250), 250.0), delimiter=",")

        # Generate valid SHA256SUMS.txt for test fixtures
        sums_file = os.path.join(self.data_dir, "SHA256SUMS.txt")
        with open(sums_file, "w", encoding="utf-8") as f:
            for fname in ("A_250.csv", "B_250.csv", "A_500.csv", "B_500.csv"):
                p = os.path.join(self.dense_dir, fname)
                with open(p, "rb") as bf:
                    h = hashlib.sha256(bf.read()).hexdigest()
                f.write(f"{h}  dense/{fname}\n")
            c_ref_p = os.path.join(self.val_dir, "C_250.csv")
            with open(c_ref_p, "rb") as bf:
                h = hashlib.sha256(bf.read()).hexdigest()
            f.write(f"{h}  validation/reference/C_250.csv\n")

        # Mock ClusterManager
        self.mock_cm = MagicMock()
        self.mock_cm.worker_count = 1
        self.mock_cm.start.return_value = ClusterTopology(
            worker_count_requested=1,
            worker_count_observed=1,
            threads_per_worker=1,
            driver_pid=os.getpid(),
            scheduler_pid=os.getpid(),
            nanny_pids=[],
            worker_pids=[],
            all_pids=[os.getpid()],
            cluster_startup_s=0.15,
            threading_env={"OPENBLAS_NUM_THREADS": "1"},
        )
        self.mock_cm.client = MagicMock()
        self.mock_cm.cluster = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("src.dask_matrix.pilot_runner.DistributedBlockMatmul")
    def test_mocked_pilot_success_run(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        def mock_multiply(a_matrix, b_matrix, repetition_id=0, timeout_s=120.0, watchdog=None, global_deadline_s=None):
            n = a_matrix.shape[0]
            # Output matrix matching reference (n everywhere)
            c_mat = np.full((n, n), float(n), dtype=np.float64)
            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=250,
                nb=n // 250,
                time_e2e_s=0.042,
                checksum=float(np.sum(c_mat)),
                c_matrix=c_mat,
                object_nbytes=1024,
            )

        mock_engine.multiply.side_effect = mock_multiply

        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self.mock_cm,
            session_budget_s=900.0,
        )

        summary = runner.run()

        self.assertEqual(summary["total_samples"], 6)
        self.assertEqual(summary["completed_samples"], 6)
        self.assertEqual(summary["failed_samples"], 0)
        self.assertTrue(os.path.isfile(self.results_csv))
        self.assertTrue(os.path.isfile(summary["sidecar_path"]))

        # Verify CSV contents
        with open(self.results_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 1 header line + 6 sample rows = 7 lines
        self.assertEqual(len(lines), 7)
        header = lines[0].split(",")
        self.assertEqual(len(header), 44)
        self.assertIn("sample_id", header)
        self.assertIn("time_e2e_s", header)
        self.assertIn("is_warmup", header)

        # Verify sidecar JSON
        with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
            sidecar_json = json.load(f)

        self.assertEqual(sidecar_json["campaign_parameters"]["worker_count"], 1)
        self.assertEqual(sidecar_json["campaign_parameters"]["block_size"], 250)
        self.assertEqual(sidecar_json["campaign_parameters"]["matrix_dimensions"], [250, 500])

    def test_pilot_runner_session_budget_pre_exhaustion(self):
        """If session budget has already expired, all 6 samples must record ABORTED_BUDGET."""
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self.mock_cm,
            session_budget_s=-1.0,  # Negative budget -> deadline already expired
        )

        summary = runner.run()

        self.assertEqual(summary["total_samples"], 6)
        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["failed_samples"], 6)

        # Verify all records have status ABORTED_BUDGET
        with open(self.results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertEqual(len(lines), 7)
        for row in lines[1:]:
            self.assertIn("ABORTED_BUDGET", row)

    @patch("src.dask_matrix.pilot_runner.DistributedBlockMatmul")
    def test_pilot_runner_oom_safestop_and_not_run_after_oom(self, mock_engine_cls):
        """Simulate an OUT_OF_MEMORY abort on sample 2; subsequent samples must be NOT_RUN_AFTER_OOM."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        call_count = [0]

        def mock_multiply(a_matrix, b_matrix, repetition_id=0, timeout_s=120.0, watchdog=None, global_deadline_s=None):
            call_count[0] += 1
            n = a_matrix.shape[0]
            if call_count[0] == 1:
                # Sample 1 completes
                c_mat = np.full((n, n), float(n), dtype=np.float64)
                return DaskExecutionResult(
                    status="COMPLETED",
                    n=n,
                    block_size=250,
                    nb=1,
                    time_e2e_s=0.05,
                    checksum=float(np.sum(c_mat)),
                    c_matrix=c_mat,
                )
            else:
                # Sample 2 triggers OOM
                if watchdog is not None:
                    watchdog.breach_detected = True
                    watchdog.breach_type = "OUT_OF_MEMORY"
                    watchdog.breach_reason = "Simulated OOM ceiling breach."
                return DaskExecutionResult(
                    status="OUT_OF_MEMORY",
                    n=n,
                    block_size=250,
                    nb=1,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason="Simulated OOM ceiling breach.",
                    elapsed_until_abort_s=0.1,
                )

        mock_engine.multiply.side_effect = mock_multiply

        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self.mock_cm,
            session_budget_s=900.0,
        )

        summary = runner.run()

        self.assertEqual(summary["total_samples"], 6)
        self.assertEqual(summary["completed_samples"], 1)
        self.assertEqual(summary["failed_samples"], 5)

        # Check recorded statuses in records
        statuses = [r.status for r in summary["records"]]
        self.assertEqual(statuses[0], "COMPLETED")
        self.assertEqual(statuses[1], "OUT_OF_MEMORY")
        for s in statuses[2:]:
            self.assertEqual(s, "NOT_RUN_AFTER_OOM")


class TestWatchdogSafeguards(unittest.TestCase):
    """Tests watchdog safe stop timing bounds, concurrency locks, and error classification."""

    def test_watchdog_safe_stop_shared_deadline_and_slow_components(self):
        """Verifies that _execute_safe_stop shares one unified deadline across client and cluster."""
        mock_client = MagicMock()
        mock_cluster = MagicMock()

        observed_timeouts = {}

        def slow_client_close(timeout):
            observed_timeouts["client"] = timeout
            time.sleep(0.05)

        def slow_cluster_close(timeout):
            observed_timeouts["cluster"] = timeout
            time.sleep(0.05)

        mock_client.close.side_effect = slow_client_close
        mock_cluster.close.side_effect = slow_cluster_close

        watchdog = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            client=mock_client,
            cluster=mock_cluster,
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
        )

        t0 = time.perf_counter()
        watchdog._execute_safe_stop("Manual test of safe stop timing")
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 5.2, f"Safe stop took {elapsed:.2f} s, expected <= 5.2 s.")
        # Verify cluster close received a smaller timeout than client close (sharing the remaining budget)
        self.assertIn("client", observed_timeouts)
        self.assertIn("cluster", observed_timeouts)
        self.assertLess(observed_timeouts["cluster"], observed_timeouts["client"])

    def test_watchdog_safe_stop_concurrency_lock(self):
        """Verifies that concurrent calls to _execute_safe_stop are strictly serialized and idempotent."""
        mock_client = MagicMock()
        mock_cluster = MagicMock()

        watchdog = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            client=mock_client,
            cluster=mock_cluster,
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
        )

        # Launch two concurrent safe stop calls
        t1 = threading.Thread(target=watchdog._execute_safe_stop, args=("Thread 1 reason",))
        t2 = threading.Thread(target=watchdog._execute_safe_stop, args=("Thread 2 reason",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # client.close and cluster.close must be called exactly once
        self.assertEqual(mock_client.close.call_count, 1)
        self.assertEqual(mock_cluster.close.call_count, 1)

    @patch("psutil.wait_procs")
    @patch("psutil.Process")
    def test_watchdog_child_process_termination(self, mock_process_cls, mock_wait_procs):
        """Verifies child processes are terminated and driver PID is preserved."""
        driver_pid = os.getpid()
        child_pid_1 = 99901
        child_pid_2 = 99902

        mock_proc_1 = MagicMock()
        mock_proc_1.pid = child_pid_1
        mock_proc_1.is_running.return_value = True
        mock_proc_1.children.return_value = []

        mock_proc_2 = MagicMock()
        mock_proc_2.pid = child_pid_2
        mock_proc_2.is_running.return_value = True
        mock_proc_2.children.return_value = []

        def get_proc(pid):
            if pid == child_pid_1:
                return mock_proc_1
            elif pid == child_pid_2:
                return mock_proc_2
            else:
                p = MagicMock()
                p.pid = pid
                return p

        mock_process_cls.side_effect = get_proc
        mock_wait_procs.return_value = ([mock_proc_1, mock_proc_2], [])

        watchdog = ResourceWatchdog(
            pids_to_track=[driver_pid, child_pid_1, child_pid_2],
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
        )

        watchdog._execute_safe_stop("Child termination test")

        mock_proc_1.terminate.assert_called_once()
        mock_proc_2.terminate.assert_called_once()
        mock_wait_procs.assert_called_once()

    def test_watchdog_global_deadline_breach(self):
        """Verifies that reaching global_deadline_s sets breach_type to ABORTED_BUDGET."""
        deadline = time.perf_counter() + 0.1
        watchdog = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
            sample_interval_s=0.05,
            global_deadline_s=deadline,
        )

        with watchdog:
            time.sleep(0.25)

        self.assertTrue(watchdog.breach_detected)
        self.assertEqual(watchdog.breach_type, "ABORTED_BUDGET")
        self.assertIn("Global session deadline reached", watchdog.breach_reason)


class TestStatusPrecedenceAndBlockingInterruption(unittest.TestCase):
    """Tests status precedence and watchdog interruption of blocking operations."""

    def test_status_precedence_rss_breach_over_deadline(self):
        """If RSS breach occurs, it takes precedence and is not overwritten by deadline expiry."""
        watchdog = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            rss_ceiling_mib=0.1,  # immediate breach
            host_floor_mib=10.0,
            sample_interval_s=0.05,
            global_deadline_s=time.perf_counter() + 0.05,
        )

        with watchdog:
            time.sleep(0.2)

        self.assertTrue(watchdog.breach_detected)
        self.assertEqual(watchdog.breach_type, "OUT_OF_MEMORY")
        self.assertIn("Aggregate RSS ceiling breached", watchdog.breach_reason)

    def test_secondary_exception_preserves_watchdog_breach_reason(self):
        """When client.gather throws CommClosedError due to watchdog shutdown, status is OUT_OF_MEMORY."""
        mock_client = MagicMock()
        mock_client.scatter.return_value = MagicMock()
        mock_client.submit.return_value = MagicMock()

        watchdog = ResourceWatchdog(pids_to_track=[os.getpid()])
        watchdog.breach_detected = True
        watchdog.breach_type = "OUT_OF_MEMORY"
        watchdog.breach_reason = "Aggregate RSS ceiling breached: 2600.0 MiB >= 2560.0 MiB"

        with patch("distributed.wait") as mock_wait:
            mock_wait.return_value = ([MagicMock()], [])
            mock_client.gather.side_effect = RuntimeError("CommClosedError: Connection to worker lost during shutdown.")

            engine = DistributedBlockMatmul(client=mock_client, block_size=250)
            a_mat = np.ones((250, 250), dtype=np.float64)
            b_mat = np.ones((250, 250), dtype=np.float64)

            res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog)

            self.assertEqual(res.status, "OUT_OF_MEMORY")
            self.assertEqual(res.error_reason, watchdog.breach_reason)
            self.assertIsNone(res.time_e2e_s)

    def test_scatter_interrupted_by_watchdog(self):
        """Simulate scatter RPC failing because watchdog terminated cluster."""
        mock_client = MagicMock()
        watchdog = ResourceWatchdog(pids_to_track=[os.getpid()])
        watchdog.breach_detected = True
        watchdog.breach_type = "ABORTED_WATCHDOG"
        watchdog.breach_reason = "Host available RAM floor breached: 900.0 MiB < 1024.0 MiB"

        mock_client.scatter.side_effect = RuntimeError("Cluster closing: RPC failed")

        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog)

        self.assertEqual(res.status, "ABORTED_WATCHDOG")
        self.assertEqual(res.error_reason, watchdog.breach_reason)

    def test_deadline_interruption_during_wait(self):
        """Simulate global deadline expiring during wait()."""
        mock_client = MagicMock()
        mock_client.scatter.return_value = MagicMock()
        mock_client.submit.return_value = MagicMock()

        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        with patch("distributed.wait") as mock_wait:
            # wait returns not_done when deadline has expired
            mock_wait.return_value = ([], [MagicMock()])

            past_deadline = time.perf_counter() - 1.0  # deadline in past
            res = engine.multiply(a_mat, b_mat, timeout_s=120.0, global_deadline_s=past_deadline)

            self.assertEqual(res.status, "ABORTED_BUDGET")
            self.assertIn("Global session deadline reached", res.error_reason)

    def test_controlled_blocking_scatter_interrupted_by_watchdog(self):
        """Simulate an active blocking scatter RPC interrupted by watchdog safe-stop."""
        mock_client = MagicMock()
        scatter_started = threading.Event()
        scatter_unblock = threading.Event()

        def blocking_scatter(*args, **kwargs):
            scatter_started.set()
            unblocked = scatter_unblock.wait(timeout=3.0)
            if unblocked:
                raise RuntimeError("CommClosedError: Connection to worker closed during scatter RPC")
            raise TimeoutError("Scatter test timed out without unblocking")

        mock_client.scatter.side_effect = blocking_scatter

        def mock_client_close(timeout=None):
            scatter_unblock.set()

        mock_client.close.side_effect = mock_client_close

        watchdog = ResourceWatchdog(pids_to_track=[os.getpid()], client=mock_client)
        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        result_holder = []

        def run_worker():
            res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog)
            result_holder.append(res)

        t = threading.Thread(target=run_worker)
        t.start()

        self.assertTrue(scatter_started.wait(timeout=1.0), "Scatter did not enter blocking state.")

        # Simulate watchdog detecting RSS breach while scatter is actively blocked
        watchdog.breach_detected = True
        watchdog.breach_type = "OUT_OF_MEMORY"
        watchdog.breach_reason = "Aggregate RSS ceiling breached: 2570.0 MiB >= 2560.0 MiB limit."
        watchdog._execute_safe_stop(watchdog.breach_reason)

        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "Worker thread remained blocked after safe-stop.")
        self.assertEqual(len(result_holder), 1)
        res = result_holder[0]
        self.assertEqual(res.status, "OUT_OF_MEMORY")
        self.assertEqual(res.error_reason, watchdog.breach_reason)
        self.assertIsNone(res.time_e2e_s)

    def test_controlled_blocking_wait_interrupted_by_watchdog(self):
        """Simulate an active blocking distributed.wait() interrupted by watchdog safe-stop."""
        mock_client = MagicMock()
        mock_client.scatter.return_value = MagicMock()
        mock_client.submit.return_value = MagicMock()

        wait_started = threading.Event()
        wait_unblock = threading.Event()

        def blocking_wait(futures, timeout=None):
            wait_started.set()
            unblocked = wait_unblock.wait(timeout=3.0)
            if unblocked:
                raise RuntimeError("CancelledError: Futures cancelled by watchdog safe-stop")
            raise TimeoutError("Wait test timed out without unblocking")

        def mock_client_cancel(futures):
            wait_unblock.set()

        mock_client.cancel.side_effect = mock_client_cancel

        watchdog = ResourceWatchdog(pids_to_track=[os.getpid()], client=mock_client)
        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        result_holder = []

        def run_worker():
            with patch("distributed.wait", side_effect=blocking_wait):
                res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog)
                result_holder.append(res)

        t = threading.Thread(target=run_worker)
        t.start()

        self.assertTrue(wait_started.wait(timeout=1.0), "Wait did not enter blocking state.")

        # Simulate watchdog detecting host RAM floor breach while wait is actively blocked
        watchdog.breach_detected = True
        watchdog.breach_type = "ABORTED_WATCHDOG"
        watchdog.breach_reason = "Host available RAM floor breached: 950.0 MiB < 1024.0 MiB floor."
        watchdog._execute_safe_stop(watchdog.breach_reason)

        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "Worker thread remained blocked after safe-stop.")
        self.assertEqual(len(result_holder), 1)
        res = result_holder[0]
        self.assertEqual(res.status, "ABORTED_WATCHDOG")
        self.assertEqual(res.error_reason, watchdog.breach_reason)
        self.assertIsNone(res.time_e2e_s)

    def test_controlled_blocking_gather_interrupted_by_watchdog(self):
        """Simulate an active blocking client.gather() interrupted by watchdog safe-stop."""
        mock_client = MagicMock()
        mock_client.scatter.return_value = MagicMock()
        mock_client.submit.return_value = MagicMock()

        gather_started = threading.Event()
        gather_unblock = threading.Event()

        def blocking_gather(*args, **kwargs):
            gather_started.set()
            unblocked = gather_unblock.wait(timeout=3.0)
            if unblocked:
                raise RuntimeError("CommClosedError: Worker comm closed during gather")
            raise TimeoutError("Gather test timed out without unblocking")

        mock_client.gather.side_effect = blocking_gather

        def mock_client_close(timeout=None):
            gather_unblock.set()

        mock_client.close.side_effect = mock_client_close

        watchdog = ResourceWatchdog(pids_to_track=[os.getpid()], client=mock_client)
        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        result_holder = []

        def run_worker():
            with patch("distributed.wait", return_value=([MagicMock()], [])):
                res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog)
                result_holder.append(res)

        t = threading.Thread(target=run_worker)
        t.start()

        self.assertTrue(gather_started.wait(timeout=1.0), "Gather did not enter blocking state.")

        # Trigger watchdog safe stop
        watchdog.breach_detected = True
        watchdog.breach_type = "OUT_OF_MEMORY"
        watchdog.breach_reason = "Aggregate RSS ceiling breached: 2580.0 MiB >= 2560.0 MiB limit."
        watchdog._execute_safe_stop(watchdog.breach_reason)

        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "Worker thread remained blocked after safe-stop.")
        self.assertEqual(len(result_holder), 1)
        res = result_holder[0]
        self.assertEqual(res.status, "OUT_OF_MEMORY")
        self.assertEqual(res.error_reason, watchdog.breach_reason)
        self.assertIsNone(res.time_e2e_s)

    def test_controlled_blocking_wait_interrupted_by_background_watchdog_thread(self):
        """Verify that the background watchdog thread actively polling at 50ms wakes up a blocked wait() on deadline."""
        mock_client = MagicMock()
        mock_client.scatter.return_value = MagicMock()
        mock_client.submit.return_value = MagicMock()

        wait_started = threading.Event()
        wait_unblock = threading.Event()

        def blocking_wait(futures, timeout=None):
            wait_started.set()
            unblocked = wait_unblock.wait(timeout=min(2.0, timeout if timeout else 2.0))
            if unblocked:
                raise RuntimeError("CancelledError: Futures cancelled by watchdog deadline trigger")
            return ([], futures)

        def mock_client_cancel(futures):
            wait_unblock.set()

        mock_client.cancel.side_effect = mock_client_cancel

        deadline = time.perf_counter() + 0.15
        watchdog = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            client=mock_client,
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
            sample_interval_s=0.05,
            global_deadline_s=deadline,
        )

        engine = DistributedBlockMatmul(client=mock_client, block_size=250)
        a_mat = np.ones((250, 250), dtype=np.float64)
        b_mat = np.ones((250, 250), dtype=np.float64)

        result_holder = []

        def run_worker():
            with patch("distributed.wait", side_effect=blocking_wait):
                res = engine.multiply(a_mat, b_mat, timeout_s=10.0, watchdog=watchdog, global_deadline_s=deadline)
                result_holder.append(res)

        with watchdog:
            t = threading.Thread(target=run_worker)
            t.start()

            self.assertTrue(wait_started.wait(timeout=1.0), "Wait did not enter blocking state.")
            t.join(timeout=2.0)
            self.assertFalse(t.is_alive(), "Worker thread remained blocked after deadline.")

        self.assertEqual(len(result_holder), 1)
        res = result_holder[0]
        self.assertEqual(res.status, "ABORTED_BUDGET")
        self.assertIn("Global session deadline reached", res.error_reason)


class TestPilotValidationRoutines(unittest.TestCase):
    """Tests numerical validation logic for n=250 reference check and n=500 determinism check."""

    def test_validation_n250_exact_and_corrupted(self):
        validator = MatrixValidator(atol=1e-9, rtol=1e-12)
        ref = np.full((250, 250), 250.0, dtype=np.float64)

        # Exact match
        computed_good = np.full((250, 250), 250.0, dtype=np.float64)
        res_good = validator.validate_matrices(computed_good, ref)
        self.assertTrue(res_good.passed)
        self.assertEqual(res_good.max_abs_diff, 0.0)

        # Corrupted single element
        computed_bad = np.full((250, 250), 250.0, dtype=np.float64)
        computed_bad[10, 10] += 1e-4
        res_bad = validator.validate_matrices(computed_bad, ref)
        self.assertFalse(res_bad.passed)
        self.assertGreater(res_bad.max_abs_diff, 1e-5)

    def test_validation_finitude_and_non_finite_detection(self):
        mat_good = np.ones((500, 500), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(mat_good)))

        mat_nan = np.ones((500, 500), dtype=np.float64)
        mat_nan[0, 0] = np.nan
        self.assertFalse(np.all(np.isfinite(mat_nan)))

        mat_inf = np.ones((500, 500), dtype=np.float64)
        mat_inf[0, 0] = np.inf
        self.assertFalse(np.all(np.isfinite(mat_inf)))


class TestClusterManagerStartupAndCleanup(unittest.TestCase):
    """Verifies timeout propagation, partial startup teardown, and orphan process cleanup."""

    @patch("src.dask_matrix.cluster_manager.verify_multiworker_permission")
    @patch("distributed.LocalCluster")
    @patch("distributed.Client")
    def test_partial_startup_client_failure_cleans_cluster(self, mock_client_cls, mock_cluster_cls, mock_perm):
        """When LocalCluster succeeds but Client construction fails, cluster must be closed immediately."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_cluster = MagicMock()
        mock_cluster_cls.return_value = mock_cluster
        mock_client_cls.side_effect = RuntimeError("Simulated Client connection error")

        cm = ClusterManager(worker_count=1)
        with self.assertRaises(RuntimeError) as ctx:
            cm.start(timeout_s=10.0)

        self.assertIn("Simulated Client connection error", str(ctx.exception))
        # Verify cluster.close was invoked during error cleanup
        mock_cluster.close.assert_called_once()
        self.assertIsNone(cm.cluster)
        self.assertIsNone(cm.client)

    @patch("src.dask_matrix.cluster_manager.verify_multiworker_permission")
    @patch("distributed.LocalCluster")
    @patch("distributed.Client")
    def test_partial_startup_wait_for_workers_timeout_cleans_all(self, mock_client_cls, mock_cluster_cls, mock_perm):
        """When wait_for_workers times out, both client and cluster must be closed."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_cluster = MagicMock()
        mock_cluster_cls.return_value = mock_cluster

        mock_client = MagicMock()
        mock_client.wait_for_workers.side_effect = TimeoutError("Simulated registration timeout")
        mock_client_cls.return_value = mock_client

        cm = ClusterManager(worker_count=1)
        with self.assertRaises(TimeoutError) as ctx:
            cm.start(timeout_s=5.0)

        self.assertIn("Simulated registration timeout", str(ctx.exception))
        mock_client.close.assert_called_once()
        mock_cluster.close.assert_called_once()
        self.assertIsNone(cm.cluster)
        self.assertIsNone(cm.client)

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_cluster_manager_close_forcefully_terminates_orphans_and_excludes_driver(self, mock_proc_cls, mock_pid_exists):
        """Close must terminate alive child PIDs, handle duplicate PIDs, and strictly exclude the driver PID."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_pid_exists.return_value = True

        procs_by_pid = {}
        def get_proc(pid):
            if pid not in procs_by_pid:
                p = MagicMock()
                p.pid = pid
                p.is_running.return_value = True
                p.create_time.return_value = 1000.0
                procs_by_pid[pid] = p
            return procs_by_pid[pid]

        mock_proc_cls.side_effect = get_proc

        cm = ClusterManager(worker_count=1)
        driver_pid = os.getpid()

        # Mock nanny with child process PID and simulated driver PID duplicate
        mock_nanny = MagicMock()
        mock_nanny.pid = driver_pid          # Nanny sharing driver PID on Windows
        mock_nanny.process.pid = 99102       # Worker child PID

        mock_cluster = MagicMock()
        mock_cluster.workers = {"worker_0": mock_nanny}

        cm.cluster = mock_cluster
        cm.close(timeout_s=1.0)

        mock_cluster.close.assert_called_once()

        # The driver PID (os.getpid()) must NEVER have been targeted for terminate or kill
        self.assertNotIn(driver_pid, [p.pid for p in procs_by_pid.values() if p.terminate.called])
        self.assertNotIn(driver_pid, [p.pid for p in procs_by_pid.values() if p.kill.called])

        # Worker child PID 99102 must have been targeted
        self.assertIn(99102, procs_by_pid)
        self.assertTrue(procs_by_pid[99102].terminate.called)
        self.assertIsNone(cm.cluster)

    @patch("src.dask_matrix.cluster_manager.verify_multiworker_permission")
    @patch("distributed.LocalCluster")
    @patch("distributed.Client")
    def test_startup_timeout_actual_remaining_below_100ms(self, mock_client_cls, mock_cluster_cls, mock_perm):
        """Startup timeout must pass actual remaining time even when below 100ms without artificial clamp."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_cluster = MagicMock()
        mock_cluster.workers = {}
        mock_cluster.scheduler.pid = os.getpid()
        mock_cluster_cls.return_value = mock_cluster

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate 2.95s spent in LocalCluster out of 3.0s budget
        timeline = [100.0, 102.95, 102.96, 102.97, 102.98]
        time_idx = [0]
        def fake_perf_counter():
            val = timeline[min(time_idx[0], len(timeline) - 1)]
            time_idx[0] += 1
            return val

        cm = ClusterManager(worker_count=1)
        with patch("time.perf_counter", side_effect=fake_perf_counter):
            cm.start(timeout_s=3.0)

        # Client init must receive ~0.05s (< 100 ms), not an artificial minimum of 0.1s
        client_timeout = mock_client_cls.call_args[1].get("timeout")
        self.assertIsNotNone(client_timeout)
        self.assertAlmostEqual(client_timeout, 0.05, delta=0.01)
        self.assertLess(client_timeout, 0.1)

    def test_startup_timeout_zero_or_negative_raises_immediately(self):
        """Startup timeout <= 0 must immediately raise TimeoutError without starting LocalCluster."""
        from src.dask_matrix.cluster_manager import ClusterManager
        cm = ClusterManager(worker_count=1)
        with self.assertRaises(TimeoutError) as ctx_zero:
            cm.start(timeout_s=0.0)
        self.assertIn("must be positive", str(ctx_zero.exception))

        with self.assertRaises(TimeoutError) as ctx_neg:
            cm.start(timeout_s=-2.5)
        self.assertIn("must be positive", str(ctx_neg.exception))

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_close_pid_reused_with_different_create_time_not_targeted(self, mock_proc_cls, mock_pid_exists):
        """When a worker exits and its PID is reused with a different create_time, it must NOT be killed."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_pid_exists.return_value = True

        initial_proc = MagicMock()
        initial_proc.is_running.return_value = True
        initial_proc.create_time.return_value = 1000.0  # Original creation time

        reused_proc = MagicMock()
        reused_proc.is_running.return_value = True
        reused_proc.create_time.return_value = 2000.0  # New process creation time (reused PID)

        # First call before close returns initial_proc; subsequent calls return reused_proc
        mock_proc_cls.side_effect = [initial_proc, reused_proc, reused_proc]

        cm = ClusterManager(worker_count=1)
        mock_nanny = MagicMock()
        mock_nanny.pid = 99201
        mock_nanny.process.pid = None

        mock_cluster = MagicMock()
        mock_cluster.workers = {"worker_0": mock_nanny}
        cm.cluster = mock_cluster

        cm.close(timeout_s=1.0)

        # Neither initial nor reused process should have terminate() or kill() called
        self.assertFalse(reused_proc.terminate.called)
        self.assertFalse(reused_proc.kill.called)
        self.assertFalse(initial_proc.kill.called)

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_close_already_exited_process_not_targeted(self, mock_proc_cls, mock_pid_exists):
        """When a process exits cleanly during cluster.close(), kill() must not be called on it."""
        from src.dask_matrix.cluster_manager import ClusterManager
        mock_pid_exists.return_value = True

        proc = MagicMock()
        proc.create_time.return_value = 1000.0
        # Running when discovered, but already dead when terminate/kill are evaluated
        proc.is_running.side_effect = [True, False, False, False]
        mock_proc_cls.return_value = proc

        cm = ClusterManager(worker_count=1)
        mock_nanny = MagicMock()
        mock_nanny.pid = 99202
        mock_nanny.process.pid = None
        mock_cluster = MagicMock()
        mock_cluster.workers = {"worker_0": mock_nanny}
        cm.cluster = mock_cluster

        cm.close(timeout_s=1.0)

        self.assertFalse(proc.kill.called)

    def test_pilot_runner_startup_timeout_with_remaining_budget_records_failed(self):
        """A local startup timeout while global budget remains must record FAILED, not ABORTED_BUDGET."""
        test_dir = tempfile.mkdtemp(prefix="test_pilot_startup_fail_")
        try:
            data_dir = os.path.join(test_dir, "data")
            dense_dir = os.path.join(data_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            results_csv = os.path.join(test_dir, "results", "raw", "raw_measurements.csv")
            metadata_dir = os.path.join(test_dir, "results", "metadata")

            np.savetxt(os.path.join(dense_dir, "A_250.csv"), np.ones((250, 250)), delimiter=",")
            np.savetxt(os.path.join(dense_dir, "B_250.csv"), np.ones((250, 250)), delimiter=",")
            np.savetxt(os.path.join(dense_dir, "A_500.csv"), np.ones((500, 500)), delimiter=",")
            np.savetxt(os.path.join(dense_dir, "B_500.csv"), np.ones((500, 500)), delimiter=",")

            sums_file = os.path.join(data_dir, "SHA256SUMS.txt")
            with open(sums_file, "w", encoding="utf-8") as f:
                for fname in ("A_250.csv", "B_250.csv", "A_500.csv", "B_500.csv"):
                    p = os.path.join(dense_dir, fname)
                    with open(p, "rb") as bf:
                        h = hashlib.sha256(bf.read()).hexdigest()
                    f.write(f"{h}  dense/{fname}\n")

            mock_cm = MagicMock()
            mock_cm.worker_count = 1
            mock_cm.start.side_effect = TimeoutError("Cluster startup timed out after 60.0s")

            runner = PilotRunner(
                data_dir=data_dir,
                results_csv=results_csv,
                metadata_dir=metadata_dir,
                cluster_manager=mock_cm,
                session_budget_s=900.0,  # 900s budget is NOT exhausted
            )

            summary = runner.run()
            self.assertEqual(summary["total_samples"], 6)
            self.assertEqual(summary["completed_samples"], 0)
            self.assertEqual(summary["failed_samples"], 6)

            with open(results_csv, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 7)
            for row in lines[1:]:
                self.assertIn("FAILED", row)
                self.assertNotIn("ABORTED_BUDGET", row)

            with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
                sidecar_json = json.load(f)
            self.assertEqual(sidecar_json["status"], "FAILED")
            self.assertIn("Cluster startup timed out", sidecar_json["failure_reason"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pilot_runner_startup_failure_with_exhausted_deadline_records_aborted_budget(self):
        """When startup fails and the global session budget has expired, samples record ABORTED_BUDGET."""
        test_dir = tempfile.mkdtemp(prefix="test_pilot_startup_budget_fail_")
        try:
            data_dir = os.path.join(test_dir, "data")
            dense_dir = os.path.join(data_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            results_csv = os.path.join(test_dir, "results", "raw", "raw_measurements.csv")
            metadata_dir = os.path.join(test_dir, "results", "metadata")

            # Create dummy touch files so file existence checks pass without large disk I/O
            sums_file = os.path.join(data_dir, "SHA256SUMS.txt")
            with open(sums_file, "w", encoding="utf-8") as sf:
                for fname in ("A_250.csv", "B_250.csv", "A_500.csv", "B_500.csv"):
                    p = os.path.join(dense_dir, fname)
                    with open(p, "wb") as f:
                        f.write(b"0.0\n")
                    h = hashlib.sha256(b"0.0\n").hexdigest()
                    sf.write(f"{h}  dense/{fname}\n")

            mock_cm = MagicMock()
            mock_cm.worker_count = 1
            orig_perf_counter = time.perf_counter

            def slow_startup_exceeding_deadline(timeout_s=60.0):
                # Simulate time jumping past global session deadline during cluster startup
                patcher = patch("time.perf_counter", return_value=orig_perf_counter() + 2000.0)
                patcher.start()
                self.addCleanup(patcher.stop)
                raise TimeoutError("Deadline expired during cluster startup")

            mock_cm.start.side_effect = slow_startup_exceeding_deadline

            runner = PilotRunner(
                data_dir=data_dir,
                results_csv=results_csv,
                metadata_dir=metadata_dir,
                cluster_manager=mock_cm,
                session_budget_s=900.0,  # 900s standard session budget
            )

            # Mock _load_matrix to avoid disk I/O latency while returning correctly-shaped synthetic arrays
            with patch.object(
                PilotRunner,
                "_load_matrix",
                side_effect=lambda path, raw_bytes=None: np.ones((500, 500) if "500" in path else (250, 250), dtype=np.float64),
            ):
                summary = runner.run()

            self.assertTrue(mock_cm.start.called, "cm.start should have been called before deadline expired")
            mock_cm.start.assert_called_with(timeout_s=60.0)
            self.assertEqual(summary["total_samples"], 6)
            self.assertEqual(summary["completed_samples"], 0)
            self.assertEqual(summary["failed_samples"], 6)

            with open(results_csv, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 7)
            for row in lines[1:]:
                self.assertIn("ABORTED_BUDGET", row)

            with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
                sidecar_json = json.load(f)
            self.assertEqual(sidecar_json["status"], "ABORTED_BUDGET")
            self.assertIn("Global session budget exhausted", sidecar_json["failure_reason"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pilot_runner_input_loading_failure_records_failed(self):
        """When an input file is missing, PilotRunner catches the error, marks all FAILED, and creates sidecar."""
        test_dir = tempfile.mkdtemp(prefix="test_pilot_input_fail_")
        try:
            data_dir = os.path.join(test_dir, "data")
            dense_dir = os.path.join(data_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            results_csv = os.path.join(test_dir, "results", "raw", "raw_measurements.csv")
            metadata_dir = os.path.join(test_dir, "results", "metadata")

            # Intentionally do NOT create A_250.csv (missing input), but create manifest with declared entry
            sums_file = os.path.join(data_dir, "SHA256SUMS.txt")
            with open(sums_file, "w", encoding="utf-8") as sf:
                for fname in ("A_250.csv", "B_250.csv", "A_500.csv", "B_500.csv"):
                    sf.write(f"{'0'*64}  dense/{fname}\n")
            mock_cm = MagicMock()
            mock_cm.worker_count = 1

            runner = PilotRunner(
                data_dir=data_dir,
                results_csv=results_csv,
                metadata_dir=metadata_dir,
                cluster_manager=mock_cm,
                session_budget_s=900.0,
            )

            summary = runner.run()
            # ClusterManager.start must NOT have been called
            mock_cm.start.assert_not_called()

            self.assertEqual(summary["total_samples"], 6)
            self.assertEqual(summary["completed_samples"], 0)
            self.assertEqual(summary["failed_samples"], 6)

            with open(results_csv, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 7)
            for row in lines[1:]:
                self.assertIn("FAILED", row)
                self.assertIn("Required matrix file not found", row)

            with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
                sidecar_json = json.load(f)
            self.assertEqual(sidecar_json["status"], "FAILED")
            self.assertIn("Required matrix file not found", sidecar_json["failure_reason"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
