"""
src.dask_matrix.results_logger — Append-Only Raw Measurements Logger

Implements the strict raw data recording contract defined in SPEC.md:
- Schema: 44 fields capturing identification, stack, dimensions, parallelism,
  repetition, timings, telemetry, memory, and failure statuses.
- Append-only integrity: writes to results/raw/raw_measurements.csv without overwriting.
- Nullable semantics: absent values are empty strings (NULL), never 0.0 or 0.
- Explicit status tracking: COMPLETED, TIMEOUT, OUT_OF_MEMORY, FAILED, ABORTED_WATCHDOG.
"""

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import os
import threading
from typing import Any, Dict, List, Optional


DEFAULT_RESULTS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "raw", "raw_measurements.csv")
)


@dataclass
class MeasurementRecord:
    """Represents a single observation row in results/raw/raw_measurements.csv."""
    # Identification
    run_id: str
    sample_id: str
    timestamp_utc: str
    # Method & Software Stack
    method: str                          # e.g., "dask_2d_block", "a1_sequential", "a3_omp_ikj"
    framework: str                       # e.g., "dask_distributed", "c_openmp", "c_native"
    framework_version: str
    source_revision: str                 # Git commit SHA
    compiler_version: Optional[str]      # e.g., "gcc 14.2.0"
    build_flags: Optional[str]           # e.g., "-O3 -mavx2"
    # Problem Dimensions & Inputs
    n: int
    dtype: str                           # "float64"
    input_A_sha256: str
    input_B_sha256: str
    block_size: Optional[int]            # 250 for Dask
    algorithm_variant: str               # "binary_tree_reduction", "ikj_loop"
    # Parallelism & Topology
    worker_count: int                    # p (worker processes)
    process_count: int                   # total OS processes in tree (1 + 2p)
    physical_machine_count: int          # strictly 1 for local experiments
    worker_threads: int                  # 1
    openmp_threads: Optional[int]        # for A3 OpenMP
    blas_threads: int                    # strictly 1
    # Repetition & Sequence
    repetition: int                      # 0 for warmup, 1..5 for measured
    is_warmup: bool
    execution_order: int
    # Timings
    time_kernel_s: Optional[float]       # for A1/A3 harnesses; None for Dask
    time_e2e_s: Optional[float]          # for Dask end-to-end; None for A1/A3
    timing_boundary_id: str              # "TB_DASK_E2E", "TB_A1_KERNEL", "TB_A3_KERNEL"
    cluster_startup_s: Optional[float]   # measured outside primary timer
    input_load_s: Optional[float]        # measured outside primary timer
    # Communication Telemetry (Dask-specific)
    client_to_worker_payload_bytes: Optional[int]
    worker_to_worker_payload_bytes: Optional[int]
    worker_to_client_payload_bytes: Optional[int]
    compressed_transfer_bytes: Optional[int]
    transfer_metric_source: Optional[str]
    object_nbytes: Optional[int]
    # Memory Telemetry
    rss_driver_mib: Optional[float]
    rss_process_sum_peak_sampled_mib: Optional[float]
    peak_working_set_mib: Optional[float]
    dask_managed_memory_mib: Optional[float]
    # Validation & Output Consumption
    checksum_sum: Optional[float]
    status: str                          # "COMPLETED", "TIMEOUT", "OUT_OF_MEMORY", "FAILED"
    error_reason: Optional[str]
    timeout_limit_s: Optional[float]
    elapsed_until_abort_s: Optional[float]


class RawResultsLogger:
    """Thread-safe append-only CSV writer for experimental observations."""

    def __init__(self, filepath: str = DEFAULT_RESULTS_FILE):
        self.filepath = os.path.abspath(filepath)
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Initializes the CSV file with the standard header if it does not exist."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.isfile(self.filepath) or os.path.getsize(self.filepath) == 0:
            header = [f.name for f in fields(MeasurementRecord)]
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(header)

    def log(self, record: MeasurementRecord) -> None:
        """
        Appends a single measurement observation to the CSV.

        Args:
            record: MeasurementRecord instance.
        """
        data = asdict(record)
        field_names = [f.name for f in fields(MeasurementRecord)]
        row = []

        for name in field_names:
            val = data.get(name)
            if val is None:
                row.append("")  # Empty string for NULL
            elif isinstance(val, bool):
                row.append("true" if val else "false")
            elif isinstance(val, float):
                if val == 0.0:
                    row.append("0.0")
                elif name == "checksum_sum":
                    # Preserve full IEEE 754 float64 precision across all matrix dimensions
                    row.append(f"{val:.17g}")
                elif abs(val) < 1e-4 or abs(val) >= 1e8:
                    row.append(f"{val:.6e}")
                else:
                    row.append(f"{val:.6f}")
            else:
                row.append(str(val))

        with self._lock:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(row)
