"""
tests/test_input_integrity.py — Regression tests for input matrix SHA-256 verification.
Enforces fail-closed integrity across PilotRunner, CampaignRunner, and input_verifier.
Strictly isolated using temporary fixtures and mocks; zero Dask clusters or real benchmarks.
"""

import hashlib
import io
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.dask_matrix.campaign_runner import CampaignRunner, CampaignSampleSpec
from src.dask_matrix.cluster_manager import ClusterTopology
from src.dask_matrix.input_verifier import (
    InputIntegrityError,
    load_sha256_manifest,
    compute_file_sha256,
    compute_bytes_sha256,
    read_and_verify_matrix_bytes,
    verify_and_load_matrix,
)
from src.dask_matrix.pilot_runner import PilotRunner


class TestInputVerifierUnit(unittest.TestCase):
    """Unit tests for low-level input verification utilities."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_input_verifier_")
        self.data_dir = os.path.join(self.test_dir, "data")
        self.dense_dir = os.path.join(self.data_dir, "dense")
        os.makedirs(self.dense_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.data_dir, "SHA256SUMS.txt")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_matrix_file(self, rel_name: str, content: bytes) -> Tuple[str, str]:
        path = os.path.join(self.dense_dir, rel_name)
        with open(path, "wb") as f:
            f.write(content)
        digest = hashlib.sha256(content).hexdigest().lower()
        return path, digest

    def test_load_manifest_success(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("# Comment line\n\n")
            f.write("a" * 64 + "  dense/A_250.csv\n")
            f.write("B" * 64 + " dense\\B_250.csv\n")

        manifest = load_sha256_manifest(self.manifest_path)
        self.assertEqual(len(manifest), 2)
        self.assertEqual(manifest["dense/A_250.csv"], "a" * 64)
        self.assertEqual(manifest["dense/B_250.csv"], "B" * 64)

    def test_load_manifest_missing_file_raises(self):
        missing_path = os.path.join(self.test_dir, "nonexistent_manifest.txt")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_sha256_manifest(missing_path)
        self.assertIn("manifest file not found", str(ctx.exception).lower())

    def test_load_manifest_malformed_syntax_raises(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("only_one_token_here\n")
        with self.assertRaises(InputIntegrityError) as ctx:
            load_sha256_manifest(self.manifest_path)
        self.assertIn("Malformed manifest line", str(ctx.exception))

    def test_load_manifest_duplicate_entry_raises(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("a" * 64 + "  dense/A_250.csv\n")
            f.write("b" * 64 + "  dense\\A_250.csv\n")

        with self.assertRaises(InputIntegrityError) as ctx:
            load_sha256_manifest(self.manifest_path)
        self.assertEqual(ctx.exception.reason, "Duplicate entry in SHA-256 manifest")
        self.assertIn("Duplicate manifest entry", str(ctx.exception))

    def test_load_sha256_checksums_compatibility_wrapper(self):
        """Confirms load_sha256_checksums can be called without arguments using DEFAULT_SUMS_FILE in both runners."""
        from src.dask_matrix.campaign_runner import (
            load_sha256_checksums as campaign_load_checksums,
            DEFAULT_SUMS_FILE as CAMPAIGN_DEFAULT_SUMS_FILE,
        )
        from src.dask_matrix.pilot_runner import (
            load_sha256_checksums as pilot_load_checksums,
            DEFAULT_SUMS_FILE as PILOT_DEFAULT_SUMS_FILE,
        )

        with patch("src.dask_matrix.campaign_runner.load_sha256_manifest", return_value={"test": "abc"}) as mock_c:
            res = campaign_load_checksums()
            self.assertEqual(res, {"test": "abc"})
            mock_c.assert_called_once_with(CAMPAIGN_DEFAULT_SUMS_FILE)

        with patch("src.dask_matrix.pilot_runner.load_sha256_manifest", return_value={"test": "xyz"}) as mock_p:
            res = pilot_load_checksums()
            self.assertEqual(res, {"test": "xyz"})
            mock_p.assert_called_once_with(PILOT_DEFAULT_SUMS_FILE)

        custom_path = os.path.join(self.test_dir, "custom_sums.txt")
        with patch("src.dask_matrix.campaign_runner.load_sha256_manifest", return_value={"k": "v"}) as mock_c_custom:
            res = campaign_load_checksums(custom_path)
            self.assertEqual(res, {"k": "v"})
            mock_c_custom.assert_called_once_with(custom_path)

    def test_verify_and_load_matrix_success(self):
        path, digest = self._create_matrix_file("A_250.csv", b"1.0,2.0\n3.0,4.0\n")
        manifest = {"dense/A_250.csv": digest}

        mat, verified_sha = verify_and_load_matrix(path, "dense/A_250.csv", manifest)
        self.assertEqual(verified_sha, digest)
        self.assertEqual(mat.shape, (2, 2))
        np.testing.assert_array_equal(mat, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_stale_manifest_hash_rejected(self):
        """Modified file with stale manifest hash: must be rejected."""
        path, orig_digest = self._create_matrix_file("A_250.csv", b"1.0,2.0\n3.0,4.0\n")
        manifest = {"dense/A_250.csv": orig_digest}

        # Modify file on disk without updating manifest
        with open(path, "wb") as f:
            f.write(b"9.0,9.0\n9.0,9.0\n")

        with self.assertRaises(InputIntegrityError) as ctx:
            verify_and_load_matrix(path, "dense/A_250.csv", manifest)

        err = ctx.exception
        self.assertEqual(err.rel_path, "dense/A_250.csv")
        self.assertEqual(err.reason, "Checksum mismatch")
        self.assertEqual(err.expected_digest, orig_digest)
        self.assertNotEqual(err.actual_digest, orig_digest)
        self.assertIn("Actual file SHA-256 does not match", str(err))

    def test_missing_manifest_entry_rejected(self):
        path, _ = self._create_matrix_file("A_250.csv", b"1.0,2.0\n")
        manifest = {}  # Empty manifest

        with self.assertRaises(InputIntegrityError) as ctx:
            verify_and_load_matrix(path, "dense/A_250.csv", manifest)

        err = ctx.exception
        self.assertEqual(err.rel_path, "dense/A_250.csv")
        self.assertEqual(err.reason, "Missing entry in SHA-256 manifest")
        self.assertIn("Missing entry in SHA-256 manifest", str(err))

    def test_malformed_digest_rejected(self):
        path, _ = self._create_matrix_file("A_250.csv", b"1.0,2.0\n")
        manifest = {"dense/A_250.csv": "not_a_valid_64_hex_hash"}

        with self.assertRaises(InputIntegrityError) as ctx:
            verify_and_load_matrix(path, "dense/A_250.csv", manifest)

        err = ctx.exception
        self.assertEqual(err.reason, "Malformed SHA-256 digest in manifest")
        self.assertIn("Malformed SHA-256 digest", str(err))

    def test_uppercase_vs_lowercase_handling(self):
        path, digest = self._create_matrix_file("A_250.csv", b"1.0,2.0\n3.0,4.0\n")
        # Manifest has uppercase digest
        manifest = {"dense/A_250.csv": digest.upper()}

        mat, verified_sha = verify_and_load_matrix(path, "dense/A_250.csv", manifest)
        self.assertEqual(verified_sha, digest.lower())
        self.assertEqual(mat.shape, (2, 2))

    def test_missing_matrix_file_rejected(self):
        path = os.path.join(self.dense_dir, "missing.csv")
        manifest = {"dense/missing.csv": "a" * 64}

        with self.assertRaises(FileNotFoundError) as ctx:
            verify_and_load_matrix(path, "dense/missing.csv", manifest)
        self.assertIn("Required matrix file not found", str(ctx.exception))


class TestRunnerInputIntegrityEnforcement(unittest.TestCase):
    """Runner-level tests ensuring CampaignRunner and PilotRunner fail closed before cluster startup."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_runner_integrity_")
        self.data_dir = os.path.join(self.test_dir, "data")
        self.dense_dir = os.path.join(self.data_dir, "dense")
        self.val_dir = os.path.join(self.data_dir, "validation", "reference")
        os.makedirs(self.dense_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        self.results_csv = os.path.join(self.test_dir, "results", "raw", "raw_measurements.csv")
        self.metadata_dir = os.path.join(self.test_dir, "results", "metadata")
        self.manifest_path = os.path.join(self.data_dir, "SHA256SUMS.txt")

        # Create valid dummy files for n=250 and n=500
        self.hashes = {}
        for fname in ("A_250.csv", "B_250.csv", "A_500.csv", "B_500.csv"):
            p = os.path.join(self.dense_dir, fname)
            content = f"1.0,2.0\n3.0,4.0\n".encode("utf-8")
            with open(p, "wb") as f:
                f.write(content)
            self.hashes[f"dense/{fname}"] = hashlib.sha256(content).hexdigest().lower()

        # Write valid manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for k, v in self.hashes.items():
                f.write(f"{v}  {k}\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_cluster_manager(self):
        mock_cm = MagicMock()
        mock_cm.start.return_value = ClusterTopology(
            worker_count_requested=1,
            worker_count_observed=1,
            threads_per_worker=1,
            driver_pid=os.getpid(),
            scheduler_pid=os.getpid(),
            nanny_pids=[],
            worker_pids=[],
            all_pids=[os.getpid()],
            cluster_startup_s=0.1,
        )
        mock_cm.client = MagicMock()
        mock_cm.cluster = MagicMock()
        return mock_cm

    def test_pilot_runner_stale_manifest_hash_rejected_before_cluster_startup(self):
        """PilotRunner: Modified file with stale manifest hash must abort before ClusterManager.start()."""
        # Tamper with A_250.csv without updating manifest
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        with open(a_path, "wb") as f:
            f.write(b"CORRUPTED_DATA_HERE\n")

        mock_cm = self._create_mock_cluster_manager()
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        summary = runner.run()
        # Invariant: cluster was NEVER started
        mock_cm.start.assert_not_called()

        self.assertEqual(summary["total_samples"], 6)
        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["failed_samples"], 6)

        # Inspect sidecar failure reason
        with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("checksum mismatch", sidecar["failure_reason"].lower())
        self.assertIn("dense/A_250.csv", sidecar["failure_reason"])

    def test_campaign_runner_stale_manifest_hash_rejected_before_cluster_startup(self):
        """CampaignRunner: Modified file with stale manifest hash must abort before ClusterManager.start()."""
        # Write dummy files for remaining campaign dimensions
        from src.dask_matrix.campaign_runner import CAMPAIGN_SIZES
        with open(self.manifest_path, "a", encoding="utf-8") as mf:
            for n in CAMPAIGN_SIZES:
                for prefix in ("A", "B"):
                    fname = f"{prefix}_{n}.csv"
                    rel = f"dense/{fname}"
                    if rel not in self.hashes:
                        p = os.path.join(self.dense_dir, fname)
                        with open(p, "wb") as f:
                            f.write(b"0.0\n")
                        h = hashlib.sha256(b"0.0\n").hexdigest().lower()
                        mf.write(f"{h}  {rel}\n")

        # Corrupt B_500.csv
        b_path = os.path.join(self.dense_dir, "B_500.csv")
        with open(b_path, "wb") as f:
            f.write(b"TAMPERED_CONTENT\n")

        mock_cm = self._create_mock_cluster_manager()
        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        summary = runner.run()
        # Invariant: cluster was NEVER started
        mock_cm.start.assert_not_called()

        self.assertEqual(summary["total_samples"], 36)
        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["failed_samples"], 36)

        with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("checksum mismatch", sidecar["failure_reason"].lower())
        self.assertIn("dense/B_500.csv", sidecar["failure_reason"])

    def test_pilot_runner_missing_manifest_rejected_before_cluster_startup(self):
        """PilotRunner: Missing SHA256SUMS.txt must abort before ClusterManager.start()."""
        os.remove(self.manifest_path)
        mock_cm = self._create_mock_cluster_manager()
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        summary = runner.run()
        mock_cm.start.assert_not_called()
        self.assertEqual(summary["failed_samples"], 6)

        with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("manifest file not found", sidecar["failure_reason"].lower())

    def test_campaign_runner_missing_manifest_rejected_before_cluster_startup(self):
        """CampaignRunner: Missing SHA256SUMS.txt must abort before ClusterManager.start()."""
        os.remove(self.manifest_path)
        mock_cm = self._create_mock_cluster_manager()
        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        summary = runner.run()
        mock_cm.start.assert_not_called()
        self.assertEqual(summary["failed_samples"], 36)

        with open(summary["sidecar_path"], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("manifest file not found", sidecar["failure_reason"].lower())

    def test_verified_digest_recorded_in_measurement_records(self):
        """Verified actual digests must be what gets recorded in CSV measurement records."""
        # Custom mini-plan for PilotRunner to execute sample 0 successfully
        mock_cm = self._create_mock_cluster_manager()
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=mock_cm,
        )

        with patch("src.dask_matrix.pilot_runner.DistributedBlockMatmul") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.execute.return_value = MagicMock(
                status="COMPLETED",
                time_e2e_s=0.05,
                checksum=123.456,
                c_matrix=np.ones((250, 250)),
                object_nbytes=1000,
                elapsed_until_abort_s=0.05,
            )
            mock_engine_cls.return_value = mock_engine

            with patch.object(runner, "_load_matrix", return_value=np.ones((250, 250))):
                summary = runner.run()

        # Check recorded CSV
        with open(self.results_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            rows = [line.strip().split(",") for line in f if line.strip()]

        a_idx = header.index("input_A_sha256")
        b_idx = header.index("input_B_sha256")

        # Invariant: the recorded digests match the verified digests
        for row in rows:
            if row[header.index("n")] == "250":
                self.assertEqual(row[a_idx], self.hashes["dense/A_250.csv"])
                self.assertEqual(row[b_idx], self.hashes["dense/B_250.csv"])

    def test_campaign_runner_verify_and_load_matrix_passes_exact_bytes_and_digest(self):
        """CampaignRunner._verify_and_load_matrix passes exact verified raw_bytes to _load_matrix."""
        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self._create_mock_cluster_manager(),
        )
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        manifest = {"dense/A_250.csv": self.hashes["dense/A_250.csv"]}

        with open(a_path, "rb") as f:
            expected_bytes = f.read()
        expected_digest = hashlib.sha256(expected_bytes).hexdigest().lower()

        captured = {}
        original_load = runner._load_matrix

        def spy_load(path, raw_bytes=None):
            captured["path"] = path
            captured["raw_bytes"] = raw_bytes
            return original_load(path, raw_bytes=raw_bytes)

        with patch.object(runner, "_load_matrix", side_effect=spy_load):
            mat, verified_sha = runner._verify_and_load_matrix(a_path, "dense/A_250.csv", manifest)

        self.assertEqual(verified_sha, expected_digest)
        self.assertEqual(captured["path"], a_path)
        self.assertEqual(captured["raw_bytes"], expected_bytes)
        np.testing.assert_array_equal(mat, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_pilot_runner_verify_and_load_matrix_passes_exact_bytes_and_digest(self):
        """PilotRunner._verify_and_load_matrix passes exact verified raw_bytes to _load_matrix."""
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self._create_mock_cluster_manager(),
        )
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        manifest = {"dense/A_250.csv": self.hashes["dense/A_250.csv"]}

        with open(a_path, "rb") as f:
            expected_bytes = f.read()
        expected_digest = hashlib.sha256(expected_bytes).hexdigest().lower()

        captured = {}
        original_load = runner._load_matrix

        def spy_load(path, raw_bytes=None):
            captured["path"] = path
            captured["raw_bytes"] = raw_bytes
            return original_load(path, raw_bytes=raw_bytes)

        with patch.object(runner, "_load_matrix", side_effect=spy_load):
            mat, verified_sha = runner._verify_and_load_matrix(a_path, "dense/A_250.csv", manifest)

        self.assertEqual(verified_sha, expected_digest)
        self.assertEqual(captured["path"], a_path)
        self.assertEqual(captured["raw_bytes"], expected_bytes)
        np.testing.assert_array_equal(mat, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_campaign_runner_type_error_propagates_without_disk_fallback(self):
        """CampaignRunner propagates TypeError raised during matrix loading without retrying disk read."""
        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self._create_mock_cluster_manager(),
        )
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        manifest = {"dense/A_250.csv": self.hashes["dense/A_250.csv"]}

        call_args_list = []

        def failing_load(path, raw_bytes=None):
            call_args_list.append((path, raw_bytes))
            raise TypeError("simulated parsing/loading TypeError")

        with patch.object(runner, "_load_matrix", side_effect=failing_load):
            with self.assertRaises(TypeError) as ctx:
                runner._verify_and_load_matrix(a_path, "dense/A_250.csv", manifest)
            self.assertIn("simulated parsing/loading TypeError", str(ctx.exception))

        # Must be called exactly once with raw_bytes provided; no secondary fallback without raw_bytes
        self.assertEqual(len(call_args_list), 1)
        self.assertIsNotNone(call_args_list[0][1])

    def test_pilot_runner_type_error_propagates_without_disk_fallback(self):
        """PilotRunner propagates TypeError raised during matrix loading without retrying disk read."""
        runner = PilotRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self._create_mock_cluster_manager(),
        )
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        manifest = {"dense/A_250.csv": self.hashes["dense/A_250.csv"]}

        call_args_list = []

        def failing_load(path, raw_bytes=None):
            call_args_list.append((path, raw_bytes))
            raise TypeError("simulated parsing/loading TypeError")

        with patch.object(runner, "_load_matrix", side_effect=failing_load):
            with self.assertRaises(TypeError) as ctx:
                runner._verify_and_load_matrix(a_path, "dense/A_250.csv", manifest)
            self.assertIn("simulated parsing/loading TypeError", str(ctx.exception))

        # Must be called exactly once with raw_bytes provided; no secondary fallback without raw_bytes
        self.assertEqual(len(call_args_list), 1)
        self.assertIsNotNone(call_args_list[0][1])

    def test_parsing_type_error_in_parse_matrix_bytes_propagates_without_second_disk_read(self):
        """Unpatched runner propagates TypeError from parse_matrix_bytes with strictly one file open."""
        import builtins
        runner = CampaignRunner(
            data_dir=self.data_dir,
            results_csv=self.results_csv,
            metadata_dir=self.metadata_dir,
            cluster_manager=self._create_mock_cluster_manager(),
        )
        a_path = os.path.join(self.dense_dir, "A_250.csv")
        manifest = {"dense/A_250.csv": self.hashes["dense/A_250.csv"]}

        open_calls = []
        original_open = builtins.open

        def tracked_open(path, *args, **kwargs):
            if str(path) == str(a_path):
                open_calls.append(path)
            return original_open(path, *args, **kwargs)

        with patch("src.dask_matrix.campaign_runner.parse_matrix_bytes", side_effect=TypeError("parser TypeError")):
            with patch("builtins.open", side_effect=tracked_open):
                with self.assertRaises(TypeError) as ctx:
                    runner._verify_and_load_matrix(a_path, "dense/A_250.csv", manifest)
                self.assertIn("parser TypeError", str(ctx.exception))

        # Invariant: open was called exactly once to read the verified byte snapshot; never reopened
        self.assertEqual(len(open_calls), 1)


if __name__ == "__main__":
    unittest.main()
