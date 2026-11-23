"""
src.adapters.a3_adapter — Non-Invasive Execution Adapter for Assignment 3

Provides the runner interface for the frozen Assignment 3 OpenMP baseline:
- Never modifies source code in baselines/a3_openmp/parallel_benchmark.c or its Makefile.
- Prepares the expected runtime layout in build/a3_runtime/ via scripts/prepare_a3_runtime.py.
- Invokes build/bin/a3_parallel_benchmark.exe with build/a3_runtime as base_dir and cwd.
- Sets OMP_NUM_THREADS explicitly while recognizing parallel_benchmark.c uses per-region
  `num_threads(threads)` clauses and selects the exact `(n, method_name, threads, repetition)`
  measurement from results/parallel_results.csv (with stdout fallback).
- Fails closed (`status="FAILED"`) when the requested measurement cannot be parsed.
- Enforces timeout of 120 s per sample.
"""

import csv
from dataclasses import dataclass
import os
import re
import subprocess
import time
from typing import Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A3_BINARY_PATH = os.path.join(REPO_ROOT, "build", "bin", "a3_parallel_benchmark.exe")
A3_RUNTIME_DIR = os.path.join(REPO_ROOT, "build", "a3_runtime")


@dataclass
class A3ExecutionResult:
    """Outcome of an Assignment 3 execution invocation."""
    status: str
    n: int
    method_name: str
    threads: int
    time_kernel_s: Optional[float]
    stdout: str
    stderr: str
    error_reason: Optional[str] = None
    elapsed_until_abort_s: Optional[float] = None


class A3OpenMPAdapter:
    """Adapter for executing and parsing Assignment 3 OpenMP baseline runs."""

    def __init__(
        self,
        binary_path: str = A3_BINARY_PATH,
        runtime_dir: str = A3_RUNTIME_DIR,
    ):
        self.binary_path = binary_path
        self.runtime_dir = runtime_dir

    def is_compiled(self) -> bool:
        """Checks if the A3 binary has been compiled to build/bin/."""
        return os.path.isfile(self.binary_path)

    def run_benchmark(
        self,
        n: int,
        method_name: str = "omp_ikj",
        threads: int = 1,
        repetition: int = 1,
        is_warmup: bool = False,
        timeout_s: float = 120.0,
    ) -> A3ExecutionResult:
        """
        Executes the A3 OpenMP binary and extracts the target measurement.

        Args:
            n: Matrix dimension (250, 500, 750, 1000, 1500, or 2000).
            method_name: "seq_ikj", "seq_tiled_b32", "omp_ikj", or "omp_tiled_b32".
            threads: Target thread count (1 for sequential methods; 1, 2, 4, 8, ... for OpenMP).
            repetition: Repetition index (1..5).
            is_warmup: Whether this run is a warm-up.
            timeout_s: Maximum execution duration before timing out (default: 120 s).

        Returns:
            A3ExecutionResult.
        """
        if not self.is_compiled():
            return A3ExecutionResult(
                status="NOT_RUN",
                n=n,
                method_name=method_name,
                threads=threads,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Binary not compiled at {self.binary_path}. Compilation belongs to Phase 5.",
            )

        if not os.path.isdir(self.runtime_dir):
            return A3ExecutionResult(
                status="NOT_RUN",
                n=n,
                method_name=method_name,
                threads=threads,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Runtime directory not found at {self.runtime_dir}. Run scripts/prepare_a3_runtime.py.",
            )

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)

        cmd = [self.binary_path, self.runtime_dir]
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.runtime_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            elapsed = time.perf_counter() - t0

            if proc.returncode != 0:
                return A3ExecutionResult(
                    status="FAILED",
                    n=n,
                    method_name=method_name,
                    threads=threads,
                    time_kernel_s=None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error_reason=f"Process exited with non-zero code {proc.returncode}.",
                    elapsed_until_abort_s=elapsed,
                )

            kernel_time = self._parse_timing_from_csv(n, method_name, threads, repetition)
            if kernel_time is None:
                kernel_time = self._parse_timing(
                    proc.stdout, n, method_name, threads=threads, repetition=repetition
                )

            if kernel_time is None:
                return A3ExecutionResult(
                    status="FAILED",
                    n=n,
                    method_name=method_name,
                    threads=threads,
                    time_kernel_s=None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error_reason=(
                        f"Failed to parse A3 timing for method={method_name}, n={n}, "
                        f"threads={threads}, repetition={repetition} from parallel_results.csv or stdout."
                    ),
                    elapsed_until_abort_s=elapsed,
                )

            return A3ExecutionResult(
                status="COMPLETED",
                n=n,
                method_name=method_name,
                threads=threads,
                time_kernel_s=kernel_time,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            return A3ExecutionResult(
                status="TIMEOUT",
                n=n,
                method_name=method_name,
                threads=threads,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Execution timed out after {elapsed:.2f} s (> {timeout_s:.1f} s limit).",
                elapsed_until_abort_s=elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return A3ExecutionResult(
                status="FAILED",
                n=n,
                method_name=method_name,
                threads=threads,
                time_kernel_s=None,
                stdout="",
                stderr=str(e),
                error_reason=str(e),
                elapsed_until_abort_s=elapsed,
            )

    def _parse_timing_from_csv(
        self, n: int, method_name: str, threads: int = 1, repetition: int = 1
    ) -> Optional[float]:
        """Extracts kernel time from <runtime_dir>/results/parallel_results.csv when present."""
        csv_path = os.path.join(self.runtime_dir, "results", "parallel_results.csv")
        if not os.path.isfile(csv_path):
            return None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if (
                        row.get("method", "").strip() == method_name
                        and int(row["n"]) == n
                        and int(row["threads"]) == threads
                        and int(row["repetition"]) == repetition
                    ):
                        return float(row["time_s"])
        except Exception:
            return None
        return None

    def _parse_timing(
        self,
        stdout: str,
        n: int,
        method_name: str,
        threads: int = 1,
        repetition: int = 1,
    ) -> Optional[float]:
        """
        Extracts kernel execution time from A3 parallel_benchmark.c console output.

        Matches multi-line blocks:
            Benchmarking Matrix Dimension: n = 250
              seq_ikj            (seq, 1 th)  | Warm-up... Done. Repetitions: 0.0060s ...
              omp_ikj            (omp,  1 th) | Warm-up... Done (actual: 1 th). Repetitions: 0.0060s ...
        """
        if repetition < 1:
            return None
        current_n: Optional[int] = None
        row_pattern = re.compile(
            rf"^\s*{re.escape(method_name)}\s+\((?:omp|seq),\s*{threads}\s*th\b.*Repetitions:\s*(.+)$"
        )
        for line in stdout.splitlines():
            dim_match = re.match(
                r"^\s*Benchmarking\s+Matrix\s+Dimension:\s*n\s*=\s*(\d+)\b", line
            )
            if dim_match:
                current_n = int(dim_match.group(1))
                continue
            if current_n == n:
                match = row_pattern.match(line)
                if match:
                    rep_times = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*s\b", match.group(1))
                    if repetition <= len(rep_times):
                        try:
                            return float(rep_times[repetition - 1])
                        except ValueError:
                            return None
        return None
