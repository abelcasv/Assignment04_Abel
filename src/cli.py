"""
src.cli — Command-Line Interface for Assignment 4 Execution and Verification

Provides the unified execution entry point for distributed matrix multiplication,
correctness checks, and benchmark campaigns:
- Validates parameters (n, b, p, repetitions, timeouts, memory ceilings).
- Enforces positive integral worker counts (p >= 1).
- Provides a --check-config flag for static validation without launching clusters.
- Integrates ResourceWatchdog, DistributedBlockMatmul, and RawResultsLogger.
"""

import argparse
import math
import numbers
import os
import sys
from typing import List, Optional

# Ensure repository root is in sys.path when executed directly
REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT_DIR not in sys.path:
    sys.path.insert(0, REPO_ROOT_DIR)

from src.dask_matrix.campaign_runner import (
    CampaignRunner,
    CAMPAIGN_PLAN,
    validate_campaign_plan,
)
from src.dask_matrix.cluster_manager import (
    verify_multiworker_permission,
)
from src.dask_matrix.partitioning import compute_grid_dimensions
from src.dask_matrix.pilot_runner import PilotRunner, PILOT_PLAN


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
        help="Number of worker processes p (default: 1). Must be a positive integer (p >= 1).",
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
        help="Execute the bounded single-worker pilot experiment (fixed matrix: n in {250, 500}, b=250, p=1, 6 runs).",
    )
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="Execute the full single-worker campaign (n in {250, 500, 750, 1000, 1500, 2000}, b=250, p=1, 36 runs).",
    )
    return parser.parse_args(argv)


def validate_configuration(
    n: int,
    b: int,
    p: int,
    repetitions: int = 5,
    timeout: float = 120.0,
) -> bool:
    """
    Validates input parameters against specification constraints.

    Returns:
        True if configuration is valid.

    Raises:
        TypeError: If n, b, p, repetitions are booleans/non-integral, or timeout is non-real.
        ValueError: If n <= 0, b <= 0, p < 1, repetitions < 1, or timeout <= 0 / non-finite.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"Matrix dimension n must be an integer, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if isinstance(b, bool) or not isinstance(b, numbers.Integral):
        raise TypeError(f"Block size b must be an integer, got {type(b).__name__}.")
    if b <= 0:
        raise ValueError(f"Block size b must be positive, got {b}.")

    if isinstance(repetitions, bool) or not isinstance(repetitions, numbers.Integral):
        raise TypeError(f"Repetitions must be an integer, got {type(repetitions).__name__}.")
    if repetitions < 1:
        raise ValueError(f"Repetitions must be at least 1, got {repetitions}.")

    if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
        raise TypeError(f"Timeout must be a finite positive number, got {type(timeout).__name__}.")
    timeout_f = float(timeout)
    if not math.isfinite(timeout_f) or timeout_f <= 0.0:
        raise ValueError(f"Timeout must be finite and positive, got {timeout}.")

    # Validate positive integer worker count (p >= 1)
    verify_multiworker_permission(p)

    nb = compute_grid_dimensions(n, b)
    print(f"[CONFIG CHECK] Problem dimension: {n} x {n}")
    print(f"[CONFIG CHECK] Block size: {b} x {b}")
    print(f"[CONFIG CHECK] Grid dimension: {nb} x {nb} blocks ({nb**2} blocks total)")
    print(f"[CONFIG CHECK] Multiplication tasks: {nb**3}, Reduction additions: {nb**2 * (nb - 1)}")
    print(f"[CONFIG CHECK] Repetitions: {repetitions}, Sample timeout: {timeout_f:.1f} s")
    print(f"[CONFIG CHECK] Worker count: p={p} (Configuration verified).")

    return True


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)

    if args.pilot and args.campaign:
        sys.stderr.write("[ERROR] Cannot specify both --pilot and --campaign simultaneously.\n")
        return 1

    if args.validate_fixtures:
        if args.pilot or args.campaign:
            sys.stderr.write("[ERROR] Cannot combine --validate-fixtures with --pilot or --campaign.\n")
            return 1
        from scripts.verify_correctness import run_correctness_checks
        return 0 if run_correctness_checks() else 1

    if args.campaign:
        try:
            verify_multiworker_permission(args.workers)
        except Exception as e:
            sys.stderr.write(f"[CAMPAIGN ERROR] {e}\n")
            return 1

        target_sizes = (250, 500, 750, 1000, 1500, 2000) if args.workers == 1 else (1000, 1500, 2000)
        if args.workers == 1:
            print("[CAMPAIGN MODE] Full Single-Worker Campaign selected.")
            print("[CAMPAIGN MODE] Enforcing fixed campaign matrix: n in {250, 500, 750, 1000, 1500, 2000}, b=250, p=1, 36 runs.")
        else:
            print(f"[CAMPAIGN MODE] Strong-Scaling Multi-Worker Campaign selected (p={args.workers}, 1 physical host).")
            print(f"[CAMPAIGN MODE] Enforcing fixed strong-scaling matrix: n in {{1000, 1500, 2000}}, b=250, p={args.workers}, 18 runs.")

        if args.block_size != 250:
            print(f"[CAMPAIGN WARNING] Argument --block-size={args.block_size} ignored. Campaign strictly uses b=250.")

        if args.check_config:
            print("[CAMPAIGN CONFIG CHECK] Validating static campaign requirements...")
            runner_preview = CampaignRunner(worker_count=args.workers, sizes=target_sizes)
            validate_campaign_plan(
                runner_preview.plan,
                expected_sizes=target_sizes,
                expected_block_size=250,
                expected_worker_count=args.workers,
            )
            for n in target_sizes:
                validate_configuration(n, 250, args.workers, repetitions=args.repetitions, timeout=args.timeout)
            print(f"[CAMPAIGN CONFIG CHECK] Plan contains {len(runner_preview.plan)} planned executions.")
            print("[SUCCESS] Campaign configuration check passed. Exiting (--check-config).")
            return 0

        # Execute Campaign
        runner = CampaignRunner(worker_count=args.workers, sizes=target_sizes) if args.workers != 1 else CampaignRunner()
        summary = runner.run()
        if summary.get("failed_samples", 0) > 0:
            return 1
        return 0

    if args.pilot:
        print("[PILOT MODE] Bounded Pilot selected.")
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
            validate_configuration(250, 250, 1, repetitions=args.repetitions, timeout=args.timeout)
            validate_configuration(500, 250, 1, repetitions=args.repetitions, timeout=args.timeout)
            print(f"[PILOT CONFIG CHECK] Plan contains {len(PILOT_PLAN)} immutable executions.")
            print("[SUCCESS] Pilot configuration check passed. Exiting (--check-config).")
            return 0

        # Execute Pilot
        runner = PilotRunner()
        summary = runner.run()
        if summary.get("failed_samples", 0) > 0:
            return 1
        return 0

    try:
        validate_configuration(
            args.n,
            args.block_size,
            args.workers,
            repetitions=args.repetitions,
            timeout=args.timeout,
        )
    except Exception as e:
        sys.stderr.write(f"\n[CONFIGURATION ERROR] {e}\n")
        return 1

    if args.check_config:
        print("\n[SUCCESS] Configuration check passed. Exiting (--check-config).")
        return 0

    print("\n[NOTE] Select an execution or verification mode flag:")
    print("Use --check-config for static parameter verification.")
    print("Use --validate-fixtures to run numerical correctness verification.")
    print("Use --pilot to execute the bounded single-worker pilot.")
    print("Use --campaign to execute the full single-worker campaign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
