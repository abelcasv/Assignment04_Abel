"""
src.adapters.a1_adapter — Non-Invasive Execution Adapter for Assignment 1

Provides the runner interface for the frozen Assignment 1 sequential baseline:
- Never modifies the source code in baselines/a1_sequential/benchmark.c or its Makefile.
- Prepares the expected runtime layout in build/a1_runtime/ via scripts/prepare_a1_runtime.py.
- Invokes build/bin/a1_benchmark.exe with build/a1_runtime as argv[1] and cwd.
- Captures stdout and CSV outputs, parses exact execution time, and fails closed on missing timing.
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
A1_BINARY_PATH = os.path.join(REPO_ROOT, "build", "bin", "a1_benchmark.exe")
A1_RUNTIME_DIR = os.path.join(REPO_ROOT, "build", "a1_runtime")


@dataclass
class A1ExecutionResult:
    """Outcome of an Assignment 1 execution invocation."""
    status: str
    n: int
    time_kernel_s: Optional[float]
    stdout: str
    stderr: str
    error_reason: Optional[str] = None
    elapsed_until_abort_s: Optional[float] = None


class A1SequentialAdapter:
    """Adapter for executing and parsing Assignment 1 sequential baseline runs."""

    def __init__(
        self,
        binary_path: str = A1_BINARY_PATH,
        runtime_dir: str = A1_RUNTIME_DIR,
    ):
        self.binary_path = binary_path
        self.runtime_dir = runtime_dir

    def is_compiled(self) -> bool:
        """Checks if the A1 binary has been compiled to build/bin/."""
        return os.path.isfile(self.binary_path)

    def run_dimension(
        self,
        n: int,
        repetition: int = 1,
        is_warmup: bool = False,
        timeout_s: float = 120.0,
    ) -> A1ExecutionResult:
        """
        Executes the A1 sequential binary.

        Args:
            n: Matrix dimension (100, 250, 500, 750, or 1000).
            repetition: Repetition index (1..5).
            is_warmup: Whether this run is a warm-up.
            timeout_s: Maximum execution duration before timing out (default: 120 s).

        Returns:
            A1ExecutionResult.
        """
        if not self.is_compiled():
            return A1ExecutionResult(
                status="NOT_RUN",
                n=n,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Binary not compiled at {self.binary_path}. Build the C executable first.",
            )

        if not os.path.isdir(self.runtime_dir):
            return A1ExecutionResult(
                status="NOT_RUN",
                n=n,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Runtime directory not found at {self.runtime_dir}. Run scripts/prepare_a1_runtime.py.",
            )

        cmd = [self.binary_path, self.runtime_dir]
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.runtime_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            elapsed = time.perf_counter() - t0

            if proc.returncode != 0:
                return A1ExecutionResult(
                    status="FAILED",
                    n=n,
                    time_kernel_s=None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error_reason=f"Process exited with non-zero code {proc.returncode}.",
                    elapsed_until_abort_s=elapsed,
                )

            kernel_time = self._parse_timing_from_csv(n, repetition)
            if kernel_time is None:
                kernel_time = self._parse_timing(proc.stdout, n, repetition=repetition)

            if kernel_time is None:
                return A1ExecutionResult(
                    status="FAILED",
                    n=n,
                    time_kernel_s=None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error_reason=(
                        f"Failed to parse A1 kernel timing for n={n}, repetition={repetition} "
                        "from c_results.csv or stdout."
                    ),
                    elapsed_until_abort_s=elapsed,
                )

            return A1ExecutionResult(
                status="COMPLETED",
                n=n,
                time_kernel_s=kernel_time,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            return A1ExecutionResult(
                status="TIMEOUT",
                n=n,
                time_kernel_s=None,
                stdout="",
                stderr="",
                error_reason=f"Execution timed out after {elapsed:.2f} s (> {timeout_s:.1f} s limit).",
                elapsed_until_abort_s=elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return A1ExecutionResult(
                status="FAILED",
                n=n,
                time_kernel_s=None,
                stdout="",
                stderr=str(e),
                error_reason=str(e),
                elapsed_until_abort_s=elapsed,
            )

    def _parse_timing_from_csv(self, n: int, repetition: int = 1) -> Optional[float]:
        """Extracts kernel time from <runtime_dir>/results/c_results.csv when present."""
        csv_path = os.path.join(self.runtime_dir, "results", "c_results.csv")
        if not os.path.isfile(csv_path):
            return None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if (
                        row.get("language", "").strip().lower() == "c"
                        and int(row["n"]) == n
                        and int(row["repetition"]) == repetition
                    ):
                        return float(row["time_s"])
        except Exception:
            return None
        return None

    def _parse_timing(self, stdout: str, n: int, repetition: int = 1) -> Optional[float]:
        """
        Extracts kernel execution time from A1 benchmark console output.

        Matches benchmark.c multi-line sections:
            Benchmarking n=250...
              Rep 1/5: 0.0912 s | 5.83 MB (RSS)
        """
        current_n: Optional[int] = None
        for line in stdout.splitlines():
            dim_match = re.match(r"^\s*Benchmarking\s+n\s*=\s*(\d+)\b", line)
            if dim_match:
                current_n = int(dim_match.group(1))
                continue
            if current_n == n:
                rep_match = re.search(
                    rf"Rep\s+{repetition}/\d+\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s\b",
                    line,
                )
                if rep_match:
                    try:
                        return float(rep_match.group(1))
                    except ValueError:
                        return None
        return None
