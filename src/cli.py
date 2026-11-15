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
import numbers
import os
import sys
import uuid
from typing import List, Optional

# Ensure repository root is in sys.path when executed directly
REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT_DIR not in sys.path:
    sys.path.insert(0, REPO_ROOT_DIR)

import numpy as np

from src.dask_matrix.campaign_runner import (
    CampaignRunner,
    CAMPAIGN_PLAN,
    validate_campaign_plan,
)
from src.dask_matrix.cluster_manager import (
    ClusterManager,
    verify_multiworker_permission,
)
from src.dask_matrix.distributed_engine import DistributedBlockMatmul
from src.dask_matrix.partitioning import compute_grid_dimensions
from src.dask_matrix.pilot_runner import PilotRunner, PILOT_PLAN
from src.dask_matrix.results_logger import MeasurementRecord, RawResultsLogger
from src.dask_matrix.sidecar import generate_campaign_sidecar
from src.dask_matrix.validation import MatrixValidator
from src.dask_matrix.watchdog import ResourceWatchdog


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DENSE_DIR = os.path.join(DATA_DIR, "dense")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
        help="Number of worker processes p (default: 1). Must be a positive integer (p >= 1). Only p=1 is currently authorized; p > 1 requires written instructor permission under DEC-01.",
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
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Execute the bounded Phase 7 Pilot experiment (fixed immutable matrix: n in {250, 500}, b=250, p=1, 6 runs).",
    )
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="Execute the full Phase 8 single-worker campaign (n in {250, 500, 750, 1000, 1500, 2000}, b=250, p=1, 36 runs).",
    )
    return parser.parse_args(argv)


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
        TypeError: If n, b, or p are boolean or not integral numbers.
        ValueError: If n <= 0, b <= 0, or p < 1.
        PermissionError: If p > 1 and DEC-01 permission is missing.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"Matrix dimension n must be an integer, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if isinstance(b, bool) or not isinstance(b, numbers.Integral):
        raise TypeError(f"Block size b must be an integer, got {type(b).__name__}.")
    if b <= 0:
        raise ValueError(f"Block size b must be positive, got {b}.")

    # Central validation of Gate G1 / DEC-01 permission and process count
    verify_multiworker_permission(p)

    nb = compute_grid_dimensions(n, b)
    print(f"[CONFIG CHECK] Problem dimension: {n} x {n}")
    print(f"[CONFIG CHECK] Block size: {b} x {b}")
    print(f"[CONFIG CHECK] Grid dimension: {nb} x {nb} blocks ({nb**2} blocks total)")
    print(f"[CONFIG CHECK] Multiplication tasks: {nb**3}, Reduction additions: {nb**2 * (nb - 1)}")
    print(f"[CONFIG CHECK] Worker count: p={p} (Permission verified).")

    return True


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)

    if args.pilot and args.campaign:
        sys.stderr.write("[ERROR] Cannot specify both --pilot and --campaign simultaneously.\n")
        return 1

    if args.campaign:
        print("[CAMPAIGN MODE] Full Phase 8 Single-Worker Campaign selected.")
        print("[CAMPAIGN MODE] Enforcing fixed campaign matrix: n in {250, 500, 750, 1000, 1500, 2000}, b=250, p=1, 36 runs.")

        if args.workers != 1:
            sys.stderr.write(f"[CAMPAIGN ERROR] Single-worker campaign strictly requires p=1. Got --workers={args.workers}.\n")
            return 1
        if args.block_size != 250:
            print(f"[CAMPAIGN WARNING] Argument --block-size={args.block_size} ignored. Campaign strictly uses b=250.")

        if args.check_config:
            print("[CAMPAIGN CONFIG CHECK] Validating static campaign requirements...")
            validate_campaign_plan(CAMPAIGN_PLAN)
            for n in (250, 500, 750, 1000, 1500, 2000):
                validate_configuration(n, 250, 1)
            print(f"[CAMPAIGN CONFIG CHECK] Plan contains {len(CAMPAIGN_PLAN)} planned executions.")
            print("[SUCCESS] Campaign configuration check passed. Exiting (--check-config).")
            return 0

        # Execute Campaign
        runner = CampaignRunner()
        summary = runner.run()
        if summary.get("failed_samples", 0) > 0 and summary.get("completed_samples", 0) == 0:
            return 1
        return 0

    if args.pilot:
        print("[PILOT MODE] Bounded Phase 7 Pilot selected.")
        print("[PILOT MODE] Enforcing fixed pilot matrix: n in {250, 500}, b=250, p=1, 6 runs.")
        # Guard against external dimension/worker args attempting to alter pilot
        if args.n not in (250, 500):
            print(f"[PILOT WARNING] Argument --n={args.n} ignored. Pilot strictly evaluates n in {{250, 500}}.")
        if args.workers != 1:
            sys.stderr.write(f"[PILOT ERROR] Pilot strictly requires p=1. Got --workers={args.workers}.\n")
            return 1
        if args.block_size != 250:
            print(f"[PILOT WARNING] Argument --block-size={args.block_size} ignored. Pilot strictly uses b=250.")

        if args.check_config:
            print("[PILOT CONFIG CHECK] Validating static pilot requirements...")
            validate_configuration(250, 250, 1)
            validate_configuration(500, 250, 1)
            print(f"[PILOT CONFIG CHECK] Plan contains {len(PILOT_PLAN)} immutable executions.")
            print("[SUCCESS] Pilot configuration check passed. Exiting (--check-config).")
            return 0

        # Execute Pilot
        runner = PilotRunner()
        summary = runner.run()
        if summary.get("failed_samples", 0) > 0 and summary.get("completed_samples", 0) == 0:
            return 1
        return 0

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
    print("Use --pilot to execute the bounded Phase 7 Pilot.")
    print("Use --campaign to execute the full Phase 8 Single-Worker Campaign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
