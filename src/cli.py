"""
src.cli — Command-Line Interface for Assignment 4 Execution and Verification

Provides the unified execution entry point for distributed matrix multiplication,
correctness checks, and benchmark campaigns:
- Validates parameters (n, b, p, repetitions, timeouts, memory ceilings).
- Enforces DEC-01 (Gate G1): rejects p > 1 immediately if permission evidence is absent.
- Provides a --check-config / --dry-run flag for static validation without launching clusters.
- Integrates ResourceWatchdog, DistributedBlockMatmul, and RawResultsLogger.
"""

import argparse
import os
import sys
import uuid
from typing import List, Optional
import numpy as np

from src.dask_matrix.cluster_manager import (
    ClusterManager,
    verify_multiworker_permission,
)
from src.dask_matrix.distributed_engine import DistributedBlockMatmul
from src.dask_matrix.partitioning import compute_grid_dimensions
from src.dask_matrix.results_logger import MeasurementRecord, RawResultsLogger
from src.dask_matrix.sidecar import generate_campaign_sidecar
from src.dask_matrix.validation import MatrixValidator
from src.dask_matrix.watchdog import ResourceWatchdog


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DENSE_DIR = os.path.join(DATA_DIR, "dense")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 4: Distributed 2D Block Matrix Multiplication (Dask Distributed)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=250,
        help="Matrix dimension n (e.g. 100, 250, 500, 750, 1000, 1500, 2000).",
    )
    parser.add_argument(
        "--block-size",
        "-b",
        type=int,
        default=250,
        help="Reference block size b (default: 250).",
    )
    parser.add_argument(
        "--workers",
        "-p",
        type=int,
        default=1,
        help="Number of worker processes p (default: 1). p > 1 requires DEC-01 permission.",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=5,
        help="Number of timed repetitions (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-sample timeout limit in seconds (default: 120.0 s).",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Static configuration check only. Validates arguments and exits without running calculations.",
    )
    parser.add_argument(
        "--validate-fixtures",
        action="store_true",
        help="Run correctness verification on analytical fixtures (4x4) and n=250 reference.",
    )
    return parser.parse_args()


def load_matrix_csv(filepath: str) -> np.ndarray:
    """Loads a dense matrix from CSV into a C-order float64 array."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Input matrix not found at '{filepath}'. Run 'python scripts/generate_dataset.py' "
            f"if working from a fresh repository clone (see data/README.md)."
        )
    # Fast CSV parsing via numpy
    mat = np.loadtxt(filepath, delimiter=",", dtype=np.float64)
    return np.ascontiguousarray(mat, dtype=np.float64)


def validate_configuration(n: int, b: int, p: int) -> bool:
    """
    Validates input parameters against protocol constraints and Gate G1.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If n <= 0 or b <= 0.
        PermissionError: If p > 1 and DEC-01 permission is missing.
    """
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")
    if b <= 0:
        raise ValueError(f"Block size b must be positive, got {b}.")

    nb = compute_grid_dimensions(n, b)
    print(f"[CONFIG CHECK] Problem dimension: {n} x {n}")
    print(f"[CONFIG CHECK] Block size: {b} x {b}")
    print(f"[CONFIG CHECK] Grid dimension: {nb} x {nb} blocks ({nb**2} blocks total)")
    print(f"[CONFIG CHECK] Multiplication tasks: {nb**3}, Reduction additions: {nb**2 * (nb - 1)}")

    # Strict check of Gate G1 / DEC-01 permission
    verify_multiworker_permission(p)
    print(f"[CONFIG CHECK] Worker count: p={p} (Permission verified).")

    return True


def main() -> int:
    args = parse_arguments()

    try:
        validate_configuration(args.n, args.block_size, args.workers)
    except Exception as e:
        sys.stderr.write(f"\n[CONFIGURATION ERROR] {e}\n")
        return 1

    if args.check_config:
        print("\n[SUCCESS] Configuration check passed. Exiting (--check-config).")
        return 0

    print("\n[NOTE] Execution of matrix calculations or worker clusters belongs to Phase 5–8.")
    print("Use --check-config for static verification in Phase 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
