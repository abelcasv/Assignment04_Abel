"""
tests/test_results_logger.py — Unit Tests for Raw Results CSV Logger

Covers:
- Header initialization matching SPEC.md / protocol schema.
- Nullable fields serialized as empty strings (NULL), never 0.0 or 0.
- Append-only row recording without data loss.
"""

import os
import tempfile
import unittest

from src.dask_matrix.results_logger import MeasurementRecord, RawResultsLogger


class TestResultsLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "test_raw.csv")
        self.logger = RawResultsLogger(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_header_creation(self):
        self.assertTrue(os.path.isfile(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        self.assertIn("run_id", header)
        self.assertIn("time_e2e_s", header)
        self.assertIn("status", header)
        self.assertIn("rss_process_sum_peak_sampled_mib", header)

    def test_log_row_with_nullable_fields(self):
        record = MeasurementRecord(
            run_id="run-123",
            sample_id="run-123_s1",
            timestamp_utc="2026-11-14T01:00:00Z",
            method="dask_2d_block",
            framework="dask_distributed",
            framework_version="2024.1.0",
            source_revision="9ebf1f6",
            compiler_version=None,  # Nullable
            build_flags=None,       # Nullable
            n=250,
            dtype="float64",
            input_A_sha256="hashA",
            input_B_sha256="hashB",
            block_size=250,
            algorithm_variant="binary_tree_reduction",
            worker_count=1,
            process_count=3,
            physical_machine_count=1,
            worker_threads=1,
            openmp_threads=None,    # Nullable
            blas_threads=1,
            repetition=1,
            is_warmup=False,
            execution_order=1,
            time_kernel_s=None,     # Nullable for Dask
            time_e2e_s=0.123456,
            timing_boundary_id="TB_DASK_E2E",
            cluster_startup_s=1.25,
            input_load_s=0.05,
            client_to_worker_payload_bytes=None,
            worker_to_worker_payload_bytes=None,
            worker_to_client_payload_bytes=None,
            compressed_transfer_bytes=None,
            transfer_metric_source=None,
            object_nbytes=1000000,
            rss_driver_mib=150.0,
            rss_process_sum_peak_sampled_mib=450.0,
            peak_working_set_mib=200.0,
            dask_managed_memory_mib=None,
            checksum_sum=1234.5678,
            status="COMPLETED",
            error_reason=None,
            timeout_limit_s=120.0,
            elapsed_until_abort_s=None,
        )

        self.logger.log(record)

        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # Header + 1 row
        row = lines[1].strip().split(",")
        self.assertEqual(row[0], "run-123")
        self.assertEqual(row[7], "")     # compiler_version must be empty
        self.assertEqual(row[8], "")     # build_flags must be empty
        self.assertEqual(row[24], "")    # time_kernel_s must be empty


if __name__ == "__main__":
    unittest.main()
