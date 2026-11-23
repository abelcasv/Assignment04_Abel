"""
tests/test_scripts.py — Unit Tests for Reproducibility & Analysis Scripts

Covers:
- scripts/verify_dataset.py: full-file CR detection beyond byte 4096, missing/malformed
  manifest & missing file guards, CLI manifest argument support.
- scripts/prepare_a3_runtime.py: synthesized fallback fixtures match A3_FIXTURE_HASHES
  exactly even when external_repos/ is absent.
- scripts/analyze_campaign.py: fails before writing derived files when a dimension has
  zero COMPLETED measured records.
"""

import csv
import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_campaign
from scripts.prepare_a3_runtime import A3_FIXTURE_HASHES, install_a3_fixtures, sha256_of_file
from scripts.verify_dataset import main as verify_dataset_main, verify_matrix_contents


class TestVerifyDatasetScript(unittest.TestCase):

    def test_detects_crlf_beyond_4096_bytes_and_supports_cli_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            dense_dir = os.path.join(data_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)

            # First row exceeds 4096 bytes (~5500 bytes), ending with CRLF (\r\n)
            long_row = ",".join(["0.5" * 1] * 250)  # 250 cols -> use 20-char floats -> ~5500B
            long_row_20 = ",".join([f"0.{i:018d}" for i in range(250)])
            rows = [long_row_20 for _ in range(250)]
            payload = ("\r\n".join(rows) + "\r\n").encode("utf-8")

            csv_path = os.path.join(dense_dir, "A_250.csv")
            with open(csv_path, "wb") as f:
                f.write(payload)
            digest = hashlib.sha256(payload).hexdigest()

            manifest_path = os.path.join(data_dir, "SHA256SUMS.txt")
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as mf:
                mf.write(f"{digest}  dense/A_250.csv\n")

            passed, failed, errors = verify_matrix_contents(manifest_path)
            self.assertEqual(passed, 0)
            self.assertEqual(failed, 1)
            self.assertTrue(any("Windows CR" in e for e in errors))

            # CLI arg invocation returns exit code 1 when CR is detected past byte 4096
            self.assertEqual(verify_dataset_main([manifest_path]), 1)

    def test_verify_matrix_contents_guards_missing_manifest_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_manifest = os.path.join(tmp, "missing.txt")
            p, f_cnt, errs = verify_matrix_contents(missing_manifest)
            self.assertEqual(p, 0)
            self.assertEqual(f_cnt, 1)
            self.assertIn("Manifest not found", errs[0])

            manifest_path = os.path.join(tmp, "SHA256SUMS.txt")
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as mf:
                mf.write("malformed_line_without_two_fields\n")
                mf.write(f"{'a'*64}  dense/A_250.csv\n")
            p2, f_cnt2, errs2 = verify_matrix_contents(manifest_path)
            self.assertEqual(p2, 0)
            self.assertEqual(f_cnt2, 2)
            self.assertTrue(any("malformed entry" in e for e in errs2))
            self.assertTrue(any("Missing file" in e for e in errs2))


class TestPrepareA3RuntimeSynthesis(unittest.TestCase):

    def test_synthesized_fixtures_match_certified_sha256_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_val_dir = os.path.join(tmp, "val")
            absent_external_dir = os.path.join(tmp, "does_not_exist")
            ok = install_a3_fixtures(target_val_dir, absent_external_dir)
            self.assertTrue(ok)
            for fname, expected_hash in A3_FIXTURE_HASHES.items():
                fpath = os.path.join(target_val_dir, fname)
                self.assertTrue(os.path.isfile(fpath))
                self.assertEqual(sha256_of_file(fpath).lower(), expected_hash.lower())


class TestAnalyzeCampaignEmptyDimensionGuard(unittest.TestCase):

    def test_raises_before_writing_when_dimension_has_zero_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_csv = os.path.join(tmp, "raw_measurements.csv")
            sidecar_json = os.path.join(tmp, "sidecar.json")
            out_dir = os.path.join(tmp, "analysis")
            a1_csv = os.path.join(tmp, "a1.csv")
            a3_csv = os.path.join(tmp, "a3.csv")

            for p in (sidecar_json, a1_csv, a3_csv):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("{}")

            # 36 rows for TARGET_RUN_ID, but all n=2000 rows are TIMEOUT (0 COMPLETED)
            header = ["run_id", "n", "is_warmup", "status", "time_e2e_s", "checksum_sum"]
            with open(raw_csv, "w", encoding="utf-8", newline="\n") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(header)
                for n in analyze_campaign.EXPECTED_SIZES:
                    for rep in range(6):
                        is_w = "true" if rep == 0 else "false"
                        st = "TIMEOUT" if n == 2000 else "COMPLETED"
                        w.writerow([analyze_campaign.TARGET_RUN_ID, n, is_w, st, "0.5", "10.0"])

            with patch.multiple(
                analyze_campaign,
                RAW_CSV_PATH=raw_csv,
                SIDECAR_PATH=sidecar_json,
                OUTPUT_DIR=out_dir,
                A1_SUMMARY_PATH=a1_csv,
                A3_SUMMARY_PATH=a3_csv,
                STRONG_SCALING_MANIFEST_PATH="",
            ):
                with self.assertRaises(ValueError) as ctx:
                    analyze_campaign.main()
            self.assertIn("n=2000", str(ctx.exception))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "campaign_measurements.csv")))


class TestAnalyzeStrongScalingHelper(unittest.TestCase):

    def test_computes_speedup_for_complete_cells_and_omits_partial_timeout(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "analysis")
            os.makedirs(out_dir, exist_ok=True)
            manifest_path = os.path.join(tmp, "strong_scaling_manifest.json")
            manifest = {
                "schema_version": "1.0",
                "physical_machine_count": 1,
                "block_size": 250,
                "sizes": [1000],
                "repetitions_per_dimension": 2,
                "worker_counts_planned": [1, 2, 10],
                "runs": [
                    {"worker_count": 1, "run_id": "ss_p1", "total_samples": 3},
                    {"worker_count": 2, "run_id": "ss_p2", "total_samples": 3},
                    {"worker_count": 10, "run_id": "ss_p10", "total_samples": 3},
                ],
            }
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f)

            header = [
                "run_id", "n", "worker_count", "process_count", "is_warmup", "status",
                "time_e2e_s", "checksum_sum", "rss_driver_mib",
                "rss_process_sum_peak_sampled_mib", "dask_managed_memory_mib",
            ]
            col = {name: i for i, name in enumerate(header)}
            data_rows = [
                # p=1: warmup + 2 completed (median = 10.0)
                ["ss_p1", "1000", "1", "2", "true", "COMPLETED", "15.0", "100.0", "200.0", "300.0", "50.0"],
                ["ss_p1", "1000", "1", "2", "false", "COMPLETED", "9.0", "100.0", "200.0", "300.0", "50.0"],
                ["ss_p1", "1000", "1", "2", "false", "COMPLETED", "11.0", "100.0", "200.0", "310.0", "50.0"],
                # p=2: warmup + 2 completed (median = 5.0 -> S=2.0, E=1.0)
                ["ss_p2", "1000", "2", "3", "true", "COMPLETED", "8.0", "100.0", "210.0", "400.0", "60.0"],
                ["ss_p2", "1000", "2", "3", "false", "COMPLETED", "4.0", "100.0", "210.0", "410.0", "60.0"],
                ["ss_p2", "1000", "2", "3", "false", "COMPLETED", "6.0", "100.0", "210.0", "420.0", "60.0"],
                # p=10: warmup + 1 completed + 1 TIMEOUT -> PARTIAL_TIMEOUT, S/E omitted
                ["ss_p10", "1000", "10", "11", "true", "COMPLETED", "30.0", "100.0", "250.0", "1200.0", "70.0"],
                ["ss_p10", "1000", "10", "11", "false", "COMPLETED", "40.0", "100.0", "250.0", "1250.0", "70.0"],
                ["ss_p10", "1000", "10", "11", "false", "TIMEOUT", "", "", "250.0", "1260.0", ""],
            ]

            info = analyze_campaign._analyze_strong_scaling(
                header=header,
                data_rows=data_rows,
                col=col,
                manifest_path=manifest_path,
                output_dir=out_dir,
            )
            self.assertEqual(info["workers_completed_all_sizes"], [1, 2])
            self.assertTrue(os.path.isfile(info["plot_png"]))

            with open(info["summary_csv"], "r", encoding="utf-8") as f:
                summary_rows = {int(r["worker_count"]): r for r in csv.DictReader(f)}

            self.assertEqual(summary_rows[1]["status_summary"], "COMPLETED")
            self.assertAlmostEqual(float(summary_rows[1]["speedup_vs_p1"]), 1.0, places=6)
            self.assertEqual(summary_rows[2]["status_summary"], "COMPLETED")
            self.assertAlmostEqual(float(summary_rows[2]["speedup_vs_p1"]), 2.0, places=6)
            self.assertAlmostEqual(float(summary_rows[2]["efficiency_vs_p1"]), 1.0, places=6)
            self.assertEqual(summary_rows[10]["status_summary"], "PARTIAL_TIMEOUT")
            self.assertEqual(summary_rows[10]["speedup_vs_p1"], "")
            self.assertEqual(summary_rows[10]["efficiency_vs_p1"], "")
            self.assertAlmostEqual(float(summary_rows[10]["median_time_e2e_s"]), 40.0, places=6)


if __name__ == "__main__":
    unittest.main()
