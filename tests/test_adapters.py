"""
tests/test_adapters.py — Unit Tests for Non-Invasive A1 and A3 Baseline Adapters

Validates:
- Missing binary / runtime directory returns NOT_RUN fail-closed.
- Exact argv[1] == runtime_dir and cwd == runtime_dir passed to subprocess.run.
- CSV extraction from c_results.csv (A1) and parallel_results.csv (A3) matching
  exact (method, n, threads, repetition).
- Fallback multi-line stdout parsing matching real benchmark.c and parallel_benchmark.c outputs.
- Fail-closed status="FAILED" when process exits 0 without parseable timing.
- Process non-zero exit codes and TimeoutExpired mapped accurately.
"""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.adapters.a1_adapter import A1SequentialAdapter
from src.adapters.a3_adapter import A3OpenMPAdapter


class TestA1SequentialAdapter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin_path = os.path.join(self.tmp.name, "a1_benchmark.exe")
        self.runtime_dir = os.path.join(self.tmp.name, "a1_runtime")
        with open(self.bin_path, "wb") as f:
            f.write(b"fake_binary")
        os.makedirs(self.runtime_dir, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_binary_or_runtime_dir_returns_not_run(self):
        missing_bin = A1SequentialAdapter(binary_path=self.bin_path + ".missing", runtime_dir=self.runtime_dir)
        res = missing_bin.run_dimension(250)
        self.assertEqual(res.status, "NOT_RUN")
        self.assertIsNone(res.time_kernel_s)

        missing_dir = A1SequentialAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir + "_missing")
        res2 = missing_dir.run_dimension(250)
        self.assertEqual(res2.status, "NOT_RUN")
        self.assertIsNone(res2.time_kernel_s)

    def test_parses_c_results_csv_and_passes_runtime_dir_arg(self):
        results_dir = os.path.join(self.runtime_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "c_results.csv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("language,n,repetition,time_s,memory_mb\n")
            f.write("c,250,1,0.091200,5.830000\n")
            f.write("c,250,3,0.088800,5.830000\n")
            f.write("c,500,1,0.772500,12.100000\n")

        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        adapter = A1SequentialAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a1_adapter.subprocess.run", return_value=mock_proc) as mock_run:
            res = adapter.run_dimension(250, repetition=3, timeout_s=45.0)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd, [self.bin_path, self.runtime_dir])
        self.assertEqual(mock_run.call_args[1]["cwd"], self.runtime_dir)
        self.assertEqual(res.status, "COMPLETED")
        self.assertAlmostEqual(res.time_kernel_s, 0.0888)

    def test_fallback_multiline_stdout_parser(self):
        stdout = (
            "All correctness checks passed.\n\n"
            "Benchmarking n=100...\n"
            "  Rep 1/5: 0.0050 s | 4.10 MB (RSS)\n"
            "Benchmarking n=250...\n"
            "  Rep 1/5: 0.0912 s | 5.83 MB (RSS)\n"
            "  Rep 2/5: 0.0899 s | 5.83 MB (RSS)\n"
        )
        mock_proc = MagicMock(returncode=0, stdout=stdout, stderr="")
        adapter = A1SequentialAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a1_adapter.subprocess.run", return_value=mock_proc):
            res = adapter.run_dimension(250, repetition=2)

        self.assertEqual(res.status, "COMPLETED")
        self.assertAlmostEqual(res.time_kernel_s, 0.0899)

    def test_unparseable_output_fails_closed(self):
        mock_proc = MagicMock(returncode=0, stdout="Benchmarking n=250...\n", stderr="")
        adapter = A1SequentialAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a1_adapter.subprocess.run", return_value=mock_proc):
            res = adapter.run_dimension(250, repetition=1)

        self.assertEqual(res.status, "FAILED")
        self.assertIsNone(res.time_kernel_s)
        self.assertIn("Failed to parse A1 kernel timing", res.error_reason)

    def test_nonzero_exit_and_timeout_handling(self):
        adapter = A1SequentialAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a1_adapter.subprocess.run", return_value=MagicMock(returncode=2, stdout="", stderr="err")):
            res_err = adapter.run_dimension(250)
        self.assertEqual(res_err.status, "FAILED")
        self.assertIn("non-zero code 2", res_err.error_reason)

        with patch("src.adapters.a1_adapter.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="a1", timeout=5.0)):
            res_to = adapter.run_dimension(250, timeout_s=5.0)
        self.assertEqual(res_to.status, "TIMEOUT")
        self.assertIsNone(res_to.time_kernel_s)


class TestA3OpenMPAdapter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin_path = os.path.join(self.tmp.name, "a3_parallel_benchmark.exe")
        self.runtime_dir = os.path.join(self.tmp.name, "a3_runtime")
        with open(self.bin_path, "wb") as f:
            f.write(b"fake_binary")
        os.makedirs(self.runtime_dir, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_parallel_results_csv_and_sets_omp_env(self):
        results_dir = os.path.join(self.runtime_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "parallel_results.csv"), "w", encoding="utf-8", newline="\n") as f:
            f.write(
                "method,algorithm,parallel_model,vectorisation,n,threads,actual_threads,"
                "oversubscribed,numeric_type,repetition,time_s,peak_memory_mib,structural_memory_mib\n"
            )
            f.write("omp_ikj,ikj,openmp,autovec_O3,2000,1,1,0,double,2,16.374453,97.36,91.55\n")
            f.write("omp_ikj,ikj,openmp,autovec_O3,2000,10,10,0,double,2,1.544338,97.36,91.55\n")
            f.write("seq_tiled_b32,tiled_b32,sequential,autovec_O3,2000,1,1,0,double,1,4.720163,97.36,91.55\n")

        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        adapter = A3OpenMPAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a3_adapter.subprocess.run", return_value=mock_proc) as mock_run:
            res_omp10 = adapter.run_benchmark(n=2000, method_name="omp_ikj", threads=10, repetition=2)
            res_tiled = adapter.run_benchmark(n=2000, method_name="seq_tiled_b32", threads=1, repetition=1)

        cmd = mock_run.call_args_list[0][0][0]
        env = mock_run.call_args_list[0][1]["env"]
        self.assertEqual(cmd, [self.bin_path, self.runtime_dir])
        self.assertEqual(env["OMP_NUM_THREADS"], "10")
        self.assertEqual(res_omp10.status, "COMPLETED")
        self.assertAlmostEqual(res_omp10.time_kernel_s, 1.544338)
        self.assertEqual(res_tiled.status, "COMPLETED")
        self.assertAlmostEqual(res_tiled.time_kernel_s, 4.720163)

    def test_fallback_multiline_stdout_parser(self):
        stdout = (
            "Benchmarking Matrix Dimension: n = 250\n"
            "  seq_ikj            (seq, 1 th)  | Warm-up... Done. Repetitions: 0.0061s 0.0060s 0.0059s\n"
            "  omp_ikj            (omp,  1 th) | Warm-up... Done (actual: 1 th). Repetitions: 0.0062s 0.0060s\n"
            "  omp_ikj            (omp, 10 th) | Warm-up... Done (actual: 10 th). Repetitions: 0.0020s 0.0019s\n"
        )
        mock_proc = MagicMock(returncode=0, stdout=stdout, stderr="")
        adapter = A3OpenMPAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a3_adapter.subprocess.run", return_value=mock_proc):
            res = adapter.run_benchmark(n=250, method_name="omp_ikj", threads=10, repetition=2)

        self.assertEqual(res.status, "COMPLETED")
        self.assertAlmostEqual(res.time_kernel_s, 0.0019)

    def test_unparseable_output_fails_closed(self):
        mock_proc = MagicMock(returncode=0, stdout="Benchmarking Matrix Dimension: n = 250\n", stderr="")
        adapter = A3OpenMPAdapter(binary_path=self.bin_path, runtime_dir=self.runtime_dir)
        with patch("src.adapters.a3_adapter.subprocess.run", return_value=mock_proc):
            res = adapter.run_benchmark(n=250, method_name="omp_ikj", threads=1, repetition=1)

        self.assertEqual(res.status, "FAILED")
        self.assertIsNone(res.time_kernel_s)


if __name__ == "__main__":
    unittest.main()
