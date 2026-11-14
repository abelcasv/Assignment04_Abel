"""
tests/test_verify_provenance.py — Unit tests for provenance integrity verifier.
Strictly isolated using temporary directories (unittest); zero experimental execution.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.verify_provenance import (
    compute_file_sha256,
    extract_artifact_entries,
    resolve_artifact_path,
    verify_provenance_manifest,
)


class TestVerifyProvenance(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _write_lf_text(self, filepath: Path, content: str) -> str:
        """Write text with explicit LF line endings and return its SHA-256."""
        raw = content.encode("utf-8")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(raw)
        return hashlib.sha256(raw).hexdigest().upper()

    def _write_binary(self, filepath: Path, content: bytes) -> str:
        """Write binary bytes and return its SHA-256."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(content)
        return hashlib.sha256(content).hexdigest().upper()

    def test_compute_file_sha256(self):
        sample = self.tmp_path / "sample.txt"
        expected_hash = self._write_lf_text(sample, "hello world\n")
        self.assertEqual(compute_file_sha256(sample), expected_hash)

    def test_resolve_artifact_path_valid(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()
        child = repo_root / "results" / "data.csv"
        child.parent.mkdir(parents=True)
        child.touch()

        resolved, err = resolve_artifact_path("results/data.csv", repo_root)
        self.assertIsNone(err)
        self.assertEqual(resolved, child.resolve())

        # Windows-style backslashes
        resolved_win, err_win = resolve_artifact_path(r"results\data.csv", repo_root)
        self.assertIsNone(err_win)
        self.assertEqual(resolved_win, child.resolve())

    def test_resolve_artifact_path_traversal_rejection(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        # Relative path escaping repo root
        resolved, err = resolve_artifact_path("../outside.txt", repo_root)
        self.assertIsNone(resolved)
        self.assertIn("Path traversal escapes repository root", err)

        # Absolute path escaping repo root
        outside = (self.tmp_path / "outside.txt").resolve()
        resolved_abs, err_abs = resolve_artifact_path(str(outside), repo_root)
        self.assertIsNone(resolved_abs)
        self.assertIn("Absolute path escapes repository root", err_abs)

    def test_verify_manifest_success_lf_text_and_binary(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        script_path = repo_root / "scripts" / "analyze.py"
        csv_path = repo_root / "results" / "raw" / "raw.csv"
        sidecar_path = repo_root / "results" / "meta" / "sidecar.json"
        plot_path = repo_root / "results" / "analysis" / "plot.png"

        script_hash = self._write_lf_text(script_path, "import sys\nprint('ok')\n")
        csv_hash = self._write_lf_text(csv_path, "dim,time\n250,0.123\n500,0.456\n")
        sidecar_hash = self._write_lf_text(sidecar_path, '{"status": "SUCCESS"}\n')
        plot_hash = self._write_binary(plot_path, b"\x89PNG\r\n\x1a\n\x00\x00\x00fake_image_data")

        manifest = {
            "analysis_schema_version": "1.0",
            "source_script": "scripts/analyze.py",
            "source_script_sha256": script_hash,
            "raw_measurements_csv": {
                "path": "results/raw/raw.csv",
                "sha256": csv_hash
            },
            "campaign_sidecar_json": {
                "path": "results/meta/sidecar.json",
                "sha256": sidecar_hash
            },
            "derived_artifacts": {
                "plot_png": {
                    "path": "results/analysis/plot.png",
                    "sha256": plot_hash
                }
            }
        }

        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertTrue(success)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(records), 4)
        for r in records:
            self.assertEqual(r["status"], "PASS")

    def test_verify_manifest_hash_mismatch(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        csv_path = repo_root / "results" / "raw.csv"
        self._write_lf_text(csv_path, "dim,time\n250,0.123\n")

        manifest = {
            "raw_measurements_csv": {
                "path": "results/raw.csv",
                "sha256": "0" * 64  # Wrong hash
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertFalse(success)
        self.assertEqual(len(errors), 1)
        self.assertIn("Hash mismatch", errors[0])
        self.assertEqual(records[0]["status"], "HASH_MISMATCH")

    def test_verify_manifest_missing_mandatory_file(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        manifest = {
            "raw_measurements_csv": {
                "path": "results/missing.csv",
                "sha256": "A" * 64
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertFalse(success)
        self.assertEqual(len(errors), 1)
        self.assertIn("Missing mandatory artifact", errors[0])
        self.assertEqual(records[0]["status"], "MISSING_FILE")

    def test_verify_manifest_missing_optional_file(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        manifest = {
            "raw_measurements_csv": {
                "path": "results/optional.csv",
                "sha256": "A" * 64,
                "optional": True
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertTrue(success)
        self.assertEqual(len(errors), 0)
        self.assertEqual(records[0]["status"], "SKIPPED_OPTIONAL_MISSING")

    def test_verify_manifest_invalid_hash_format(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        file_path = repo_root / "file.txt"
        self._write_lf_text(file_path, "content\n")

        manifest = {
            "raw_measurements_csv": {
                "path": "file.txt",
                "sha256": "NOT_A_VALID_HEX_DIGEST"
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertFalse(success)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid SHA-256 format", errors[0])
        self.assertEqual(records[0]["status"], "INVALID_HASH_FORMAT")

    def test_verify_manifest_path_traversal_detection(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        manifest = {
            "raw_measurements_csv": {
                "path": "../../../etc/passwd",
                "sha256": "B" * 64
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        success, records, errors = verify_provenance_manifest(manifest_file, repo_root)
        self.assertFalse(success)
        self.assertEqual(len(errors), 1)
        self.assertIn("Path security violation", errors[0])
        self.assertEqual(records[0]["status"], "PATH_TRAVERSAL")

    def test_cli_execution_success(self):
        repo_root = self.tmp_path / "repo"
        repo_root.mkdir()

        sample = repo_root / "data.csv"
        h = self._write_lf_text(sample, "a,b\n1,2\n")

        manifest = {
            "raw_measurements_csv": {
                "path": "data.csv",
                "sha256": h
            }
        }
        manifest_file = repo_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        res = subprocess.run(
            [sys.executable, "scripts/verify_provenance.py", "--manifest", str(manifest_file), "--repo-root", str(repo_root)],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("[SUCCESS] All provenance artifacts verified", res.stdout)


if __name__ == "__main__":
    unittest.main()
