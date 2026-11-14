"""
src.adapters.a3_adapter — Non-Invasive Execution Adapter for Assignment 3

Provides the runner interface for the frozen Assignment 3 OpenMP baseline:
- Never modifies source code in baselines/a3_openmp/parallel_benchmark.c or its Makefile.
- Prepares the expected runtime layout in build/a3_runtime/ via scripts/prepare_a3_runtime.py.
- Invokes build/bin/a3_parallel_benchmark.exe with build/a3_runtime as cwd.
- Sets OMP_NUM_THREADS explicitly to control thread allocation.
- Captures stdout, parses kernel execution time, and records observations to raw_measurements.csv.
- Enforces timeout of 120 s per sample.
"""

from dataclasses import dataclass
import os
import re
import subprocess
import time
from typing import Dict, List, Optional

from src.dask_matrix.results_logger import MeasurementRecord, RawResultsLogger


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
        Executes the A3 OpenMP binary.

        Args:
            n: Matrix dimension (250, 500, 750, 1000, 1500, or 2000).
            method_name: "seq_ikj", "omp_ikj", or "omp_tiled_b32".
            threads: Number of OpenMP threads.
            repetition: Repetition index.
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

        # Set environment with specified thread count
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)

        cmd = [self.binary_path]
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

            kernel_time = self._parse_timing(proc.stdout, n, method_name)

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

    def _parse_timing(self, stdout: str, n: int, method_name: str) -> Optional[float]:
        """Extracts kernel execution time for a specific algorithm and dimension."""
        # A3 output typically prints CSV or formatted table lines:
        # e.g., "omp_ikj, 1000, 4, 0.123456"
        for line in stdout.splitlines():
            if method_name in line and str(n) in line:
                match = re.search(r"([0-9]+\.[0-9]+)", line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
        return None
