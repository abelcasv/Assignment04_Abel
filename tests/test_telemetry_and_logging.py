"""
tests/test_telemetry_and_logging.py — Unit Tests for Telemetry, Sidecar, Logger, Watchdog & Engine

Covers previously untested paths and remediated defects:
- RawResultsLogger: %.17g precision for checksum_sum >= 1e8, LF line endings, multi-row reopen.
- TelemetryCollector: real Dask metrics schema & startstops task-stream schema (+ legacy fallbacks).
- sidecar: repo-root git SHA invocation, configurable thresholds, LF line endings.
- cluster_manager: nanny_pids deduplication when Nanny.pid equals worker subprocess pid.
- watchdog: per-sample window isolation, sub-interval boundary sampling, host RAM floor
  ABORTED_WATCHDOG trigger, and FutureState -> Future cancellation.
- distributed_engine: multi-block (nb > 1) execution, cross-dimension unique task keys, timeout.
"""

import csv
import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.dask_matrix.cluster_manager import _extract_worker_hierarchy_pids
from src.dask_matrix.distributed_engine import DistributedBlockMatmul
from src.dask_matrix.results_logger import MeasurementRecord, RawResultsLogger
from src.dask_matrix.sidecar import generate_campaign_sidecar, get_git_commit_sha
from src.dask_matrix.telemetry import TelemetryCollector
from src.dask_matrix.watchdog import ResourceWatchdog


def _make_record(sample_id: str, n: int, checksum: float) -> MeasurementRecord:
    return MeasurementRecord(
        run_id="run_test",
        sample_id=sample_id,
        timestamp_utc="2026-11-22T00:00:00+00:00",
        method="dask_block_2d",
        framework="dask_distributed",
        framework_version="2026.8.0",
        source_revision="abc123",
        compiler_version=None,
        build_flags=None,
        n=n,
        dtype="float64",
        input_A_sha256="a" * 64,
        input_B_sha256="b" * 64,
        block_size=250,
        algorithm_variant="binary_tree_reduction",
        worker_count=1,
        process_count=2,
        physical_machine_count=1,
        worker_threads=1,
        openmp_threads=None,
        blas_threads=1,
        repetition=1,
        is_warmup=False,
        execution_order=1,
        time_kernel_s=None,
        time_e2e_s=0.123456,
        timing_boundary_id="TB_DASK_E2E",
        cluster_startup_s=1.0,
        input_load_s=0.1,
        client_to_worker_payload_bytes=None,
        worker_to_worker_payload_bytes=None,
        worker_to_client_payload_bytes=None,
        compressed_transfer_bytes=None,
        transfer_metric_source=None,
        object_nbytes=1024,
        rss_driver_mib=100.0,
        rss_process_sum_peak_sampled_mib=200.0,
        peak_working_set_mib=120.0,
        dask_managed_memory_mib=None,
        checksum_sum=checksum,
        status="COMPLETED",
        error_reason=None,
        timeout_limit_s=120.0,
        elapsed_until_abort_s=None,
    )


class TestResultsLoggerPrecisionAndReopen(unittest.TestCase):

    def test_checksum_full_float64_precision_lf_and_append_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "raw.csv")
            large_chk_a = 1999493827.1234567
            large_chk_b = 1999493827.1239998

            logger1 = RawResultsLogger(csv_path)
            logger1.log(_make_record("s1", 2000, large_chk_a))

            # Reopen existing CSV and append another row (must NOT rewrite header)
            logger2 = RawResultsLogger(csv_path)
            logger2.log(_make_record("s2", 2000, large_chk_b))

            with open(csv_path, "rb") as bf:
                raw_bytes = bf.read()
            self.assertNotIn(b"\r", raw_bytes)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            # Full precision preserves sub-1e-3 differences >= 1e8 that :.6e destroyed
            self.assertNotEqual(rows[0]["checksum_sum"], rows[1]["checksum_sum"])
            self.assertAlmostEqual(float(rows[0]["checksum_sum"]), large_chk_a, places=6)
            self.assertAlmostEqual(float(rows[1]["checksum_sum"]), large_chk_b, places=6)


class TestTelemetryCollectorSchemas(unittest.TestCase):

    def test_worker_memory_stats_native_metrics_and_legacy_fallback(self):
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            "workers": {
                "tcp://127.0.0.1:5000": {
                    "nthreads": 1,
                    "metrics": {
                        "managed_bytes": 4096,
                        "memory": 104857600,
                        "spilled_bytes": {"memory": 512, "disk": 1024},
                    },
                },
                "tcp://127.0.0.1:5001": {
                    "nthreads": 2,
                    "memory": {"managed": 2048, "process": 52428800, "spilled": 256},
                },
            }
        }
        collector = TelemetryCollector(mock_client)
        stats = collector.get_worker_memory_stats()
        self.assertEqual(stats["tcp://127.0.0.1:5000"]["managed_bytes"], 4096)
        self.assertEqual(stats["tcp://127.0.0.1:5000"]["process_bytes"], 104857600)
        self.assertEqual(stats["tcp://127.0.0.1:5000"]["spilled_bytes"], 1536)
        self.assertEqual(stats["tcp://127.0.0.1:5001"]["managed_bytes"], 2048)
        self.assertEqual(stats["tcp://127.0.0.1:5001"]["spilled_bytes"], 256)

    def test_parse_task_stream_native_startstops_and_legacy(self):
        mock_client = MagicMock()
        collector = TelemetryCollector(mock_client)
        stream = [
            {
                "key": "matmul-n500-c1-r1-(0,0,0)",
                "startstops": [
                    {"action": "transfer", "start": 10.0, "stop": 10.2},
                    {"action": "deserialize", "start": 10.2, "stop": 10.3},
                    {"action": "compute", "start": 10.3, "stop": 11.0},
                ],
            },
            {
                "key": "add-n500-c1-r1-b(0,0)-round0-pair0",
                "startstops": [
                    {"action": "compute", "start": 11.0, "stop": 11.25},
                ],
            },
        ]
        diag = collector.parse_task_stream(stream)
        self.assertEqual(diag.total_tasks, 2)
        self.assertEqual(diag.matmul_tasks_count, 1)
        self.assertEqual(diag.reduction_tasks_count, 1)
        self.assertAlmostEqual(diag.mean_matmul_duration_s, 0.7)
        self.assertAlmostEqual(diag.mean_reduction_duration_s, 0.25)
        self.assertAlmostEqual(diag.total_compute_s, 0.95)
        self.assertAlmostEqual(diag.total_transfer_s, 0.2)
        self.assertAlmostEqual(diag.total_deserialization_s, 0.1)

    def test_parse_task_stream_tuple_startstops_real_dask_format(self):
        mock_client = MagicMock()
        collector = TelemetryCollector(mock_client)
        stream = [
            {
                "key": "matmul-n32-c1-r1-(0,0,0)",
                "startstops": (
                    {"action": "transfer", "start": 20.0, "stop": 20.05},
                    {"action": "deserialize", "start": 20.05, "stop": 20.08},
                    {"action": "compute", "start": 20.08, "stop": 20.58},
                ),
            },
            {
                "key": "add-n32-c1-r1-b(0,0)-round0-pair0",
                "startstops": (
                    {"action": "compute", "start": 20.6, "stop": 20.75},
                ),
            },
        ]
        diag = collector.parse_task_stream(stream)
        self.assertEqual(diag.total_tasks, 2)
        self.assertEqual(diag.matmul_tasks_count, 1)
        self.assertEqual(diag.reduction_tasks_count, 1)
        self.assertAlmostEqual(diag.mean_matmul_duration_s, 0.5)
        self.assertAlmostEqual(diag.mean_reduction_duration_s, 0.15)
        self.assertAlmostEqual(diag.total_compute_s, 0.65)
        self.assertAlmostEqual(diag.total_transfer_s, 0.05)
        self.assertAlmostEqual(diag.total_deserialization_s, 0.03)

    def test_worker_transfer_stats_native_metrics(self):
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            "workers": {
                "tcp://127.0.0.1:5000": {
                    "metrics": {
                        "transfer": {
                            "incoming_bytes": 1000000,
                            "outgoing_bytes": 500000,
                            "incoming_count_total": 4,
                            "outgoing_count_total": 2,
                        }
                    }
                }
            }
        }
        collector = TelemetryCollector(mock_client)
        tstats = collector.get_worker_transfer_stats()
        self.assertEqual(tstats["tcp://127.0.0.1:5000"]["incoming_bytes"], 1000000)
        self.assertEqual(tstats["tcp://127.0.0.1:5000"]["outgoing_bytes"], 500000)
        self.assertEqual(tstats["tcp://127.0.0.1:5000"]["incoming_count_total"], 4)
        self.assertEqual(tstats["tcp://127.0.0.1:5000"]["outgoing_count_total"], 2)


class TestSidecarAndHierarchy(unittest.TestCase):

    def test_sidecar_uses_custom_thresholds_lf_and_repo_root_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                sha = get_git_commit_sha()
            finally:
                os.chdir(old_cwd)
            self.assertRegex(sha, r"^[0-9a-f]{40}$")

            path = generate_campaign_sidecar(
                run_id="custom_thresh_test",
                campaign_name="Test",
                worker_count=1,
                block_size=250,
                n_values=[250],
                repetitions=1,
                output_dir=tmp,
                timeout_per_sample_s=45.0,
                aggregate_rss_ceiling_mib=1500.0,
                host_available_ram_floor_mib=512.0,
            )
            with open(path, "rb") as bf:
                self.assertNotIn(b"\r", bf.read())
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["campaign_parameters"]["timeout_per_sample_s"], 45.0)
            self.assertEqual(data["campaign_parameters"]["aggregate_rss_ceiling_mib"], 1500.0)
            self.assertEqual(data["campaign_parameters"]["host_available_ram_floor_mib"], 512.0)

    def test_worker_hierarchy_deduplicates_inprocess_nanny_pid(self):
        nanny_same = MagicMock()
        nanny_same.pid = 21036
        nanny_same.process.pid = 21036

        nanny_distinct = MagicMock()
        nanny_distinct.pid = 10001
        nanny_distinct.process.pid = 21037

        nannies, workers = _extract_worker_hierarchy_pids({"w0": nanny_same, "w1": nanny_distinct})
        self.assertEqual(workers, [21036, 21037])
        self.assertEqual(nannies, [10001])


class TestWatchdogAndEngineEnhancements(unittest.TestCase):

    def test_per_sample_memory_window_isolation(self):
        wd = ResourceWatchdog(pids_to_track=[os.getpid()], rss_ceiling_mib=100000.0, host_floor_mib=1.0)
        from src.dask_matrix.watchdog import ProcessMemorySample

        # Pre-existing huge sample from an earlier n=2000 iteration
        wd.samples.append(
            ProcessMemorySample(
                timestamp_epoch_s=1.0,
                elapsed_s=1.0,
                rss_driver_mib=9000.0,
                rss_process_sum_mib=9999.0,
                host_available_ram_mib=4000.0,
                per_pid_rss_mib={os.getpid(): 9999.0},
            )
        )
        wd.peak_process_sum_mib = 9999.0

        # New short n=250 sample where no background sample arrived inside interval
        idx = wd.mark_sample_start()
        peak_rss, driver_rss, peak_ws = wd.get_window_metrics(idx)
        self.assertIsNotNone(peak_rss)
        self.assertLess(peak_rss, 9999.0)
        self.assertIsNotNone(driver_rss)
        self.assertLess(driver_rss, 9000.0)
        self.assertIsNotNone(peak_ws)
        self.assertLess(peak_ws, 9999.0)

    def test_host_floor_breach_sets_aborted_watchdog(self):
        wd = ResourceWatchdog(
            pids_to_track=[os.getpid()],
            rss_ceiling_mib=100000.0,
            host_floor_mib=10_000_000.0,  # Guaranteed below threshold
            sample_interval_s=0.02,
        )
        with wd:
            time.sleep(0.1)
        self.assertTrue(wd.breach_detected)
        self.assertEqual(wd.breach_type, "ABORTED_WATCHDOG")

    def test_safe_stop_wraps_dask_futurestate_as_future(self):
        from distributed import Future
        from distributed.client import FutureState

        mock_client = MagicMock()
        st = FutureState("k-123")
        mock_client.futures = {"k-123": st}
        wd = ResourceWatchdog(pids_to_track=[os.getpid()], client=mock_client)
        with patch.object(Future, "_bind_late", return_value=None):
            wd._execute_safe_stop("test breach")
        mock_client.cancel.assert_called_once()
        cancelled_arg = mock_client.cancel.call_args[0][0]
        self.assertEqual(len(cancelled_arg), 1)
        self.assertIsInstance(cancelled_arg[0], Future)
        self.assertEqual(cancelled_arg[0].key, "k-123")
        self.assertTrue(mock_client.cancel.call_args[1].get("force"))

    def test_engine_multiblock_unique_keys_and_timeout(self):
        mock_client = MagicMock()
        submitted_keys = []

        def fake_scatter(arr, broadcast=False, hash=False):
            return arr

        def fake_submit(fn, *args, key=None, pure=False):
            submitted_keys.append(key)
            val = fn(*[getattr(a, "value", a) for a in args])
            fut = MagicMock()
            fut.key = key
            fut.value = val
            return fut

        mock_client.scatter.side_effect = fake_scatter
        mock_client.submit.side_effect = fake_submit
        mock_client.gather.side_effect = lambda fut: fut.value
        mock_client.nbytes.return_value = {"w0": 256}
        mock_client.scheduler_info.return_value = {
            "workers": {
                "tcp://127.0.0.1:5000": {
                    "nthreads": 1,
                    "metrics": {
                        "managed_bytes": 1048576,
                        "memory": 52428800,
                        "spilled_bytes": {"memory": 0, "disk": 0},
                    },
                }
            }
        }

        engine = DistributedBlockMatmul(client=mock_client, block_size=2)
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
        b = np.eye(4, dtype=np.float64)

        with patch("distributed.wait", side_effect=lambda futs, timeout=None: (futs, [])):
            res1 = engine.multiply(a, b, repetition_id=1)
            res2 = engine.multiply(a[:2, :2], b[:2, :2], repetition_id=1)

        self.assertEqual(res1.status, "COMPLETED")
        self.assertEqual(res1.nb, 2)
        self.assertTrue(np.allclose(res1.c_matrix, a))
        self.assertEqual(res1.object_nbytes, 256)
        self.assertIn("wait_s", res1.phase_breakdown_s)
        self.assertAlmostEqual(res1.dask_managed_memory_mib, 1.0)
        self.assertEqual(res2.status, "COMPLETED")
        # Unique namespaces across invocations even with identical repetition_id
        self.assertEqual(len(submitted_keys), len(set(submitted_keys)))
        self.assertTrue(any("matmul-n4-c1-r1-" in k for k in submitted_keys))
        self.assertTrue(any("add-n4-c1-r1-" in k for k in submitted_keys))
        self.assertTrue(any("matmul-n2-c2-r1-" in k for k in submitted_keys))

        # Timeout path cancels pending futures
        with patch("distributed.wait", side_effect=lambda futs, timeout=None: ([], futs)):
            res_to = engine.multiply(a, b, repetition_id=1, timeout_s=5.0)
        self.assertEqual(res_to.status, "TIMEOUT")
        self.assertIsNone(res_to.time_e2e_s)
        mock_client.cancel.assert_called()


if __name__ == "__main__":
    unittest.main()
