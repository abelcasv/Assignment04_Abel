"""
src.dask_matrix.pilot_runner — Bounded Pilot Execution Orchestrator

Implements the bounded Pilot experimental specification defined in SPEC.md.

Fixed, immutable execution plan:
1. pilot_250_w0: n=250, b=250, p=1, rep=0, warmup=True  (Latencia base, 1 mult, 0 red)
2. pilot_250_r1: n=250, b=250, p=1, rep=1, warmup=False (Medida 1)
3. pilot_250_r2: n=250, b=250, p=1, rep=2, warmup=False (Medida 2)
4. pilot_500_w0: n=500, b=250, p=1, rep=0, warmup=True  (Particionado 2D, 8 mult, 4 red)
5. pilot_500_r1: n=500, b=250, p=1, rep=1, warmup=False (Medida 1)
6. pilot_500_r2: n=500, b=250, p=1, rep=2, warmup=False (Medida 2)

Invariable parameters:
- Method: dask_block_2d exclusively
- Matrix sizes: n in {250, 500}
- Block size: b = 250
- Worker count: p = 1 exclusively
- Threads per worker: 1
- OpenBLAS num_threads: 1
- Repetitions: 1 warmup + 2 measured repetitions per size (6 executions total)
- Sample timeout: 120.0 s
- Global session budget: 900.0 s (15 minutes)
- Aggregate RSS ceiling: 2560.0 MiB (2.5 GiB)
- Host available RAM floor: 1024.0 MiB (1.0 GiB)
- Watchdog sampling interval: 250 ms
- Dask worker memory_limit: None
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import numbers
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from src.dask_matrix.cluster_manager import (
    ClusterManager,
    ClusterTopology,
    validate_cluster_topology,
    validate_injected_cluster_manager,
    verify_multiworker_permission,
)
from src.dask_matrix.distributed_engine import (
    DaskExecutionResult,
    DistributedBlockMatmul,
)
from src.dask_matrix.results_logger import (
    MeasurementRecord,
    RawResultsLogger,
    DEFAULT_RESULTS_FILE,
)
from src.dask_matrix.input_verifier import (
    InputIntegrityError,
    load_sha256_manifest,
    compute_file_sha256,
    compute_bytes_sha256,
    read_and_verify_matrix_bytes,
    parse_matrix_bytes,
    verify_and_load_matrix,
)
from src.dask_matrix.sidecar import (
    generate_campaign_sidecar,
    get_git_commit_sha,
    METADATA_DIR,
)
from src.dask_matrix.validation import MatrixValidator
from src.dask_matrix.watchdog import (
    ResourceWatchdog,
    AGGREGATE_RSS_CEILING_MIB,
    HOST_AVAILABLE_RAM_FLOOR_MIB,
    WATCHDOG_SAMPLE_INTERVAL_S,
    begin_sample_memory_window,
    extract_sample_memory_metrics,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_DENSE_DIR = os.path.join(DEFAULT_DATA_DIR, "dense")
DEFAULT_VAL_DIR = os.path.join(DEFAULT_DATA_DIR, "validation")
DEFAULT_SUMS_FILE = os.path.join(DEFAULT_DATA_DIR, "SHA256SUMS.txt")

# Formal Pilot Operational Constants
PILOT_SESSION_BUDGET_S = 900.0        # 15 minutes absolute maximum session budget
PILOT_SAMPLE_TIMEOUT_S = 120.0        # 120 seconds per end-to-end sample
PILOT_METHOD = "dask_block_2d"
PILOT_FRAMEWORK = "dask_distributed"
PILOT_TIMING_BOUNDARY_ID = "TB_DASK_E2E"
PILOT_ALGORITHM_VARIANT = "binary_tree_reduction"
PILOT_DTYPE = "float64"
PILOT_BLOCK_SIZE = 250
PILOT_WORKER_COUNT = 1
PILOT_WORKER_THREADS = 1
PILOT_BLAS_THREADS = 1
PILOT_PHYSICAL_MACHINE_COUNT = 1


@dataclass(frozen=True)
class PilotSampleSpec:
    """Immutable specification for a single pilot sample execution."""
    sample_id: str
    n: int
    block_size: int
    worker_count: int
    repetition: int
    is_warmup: bool
    execution_order: int


# Fixed, immutable pilot execution matrix (exactly 6 samples)
PILOT_PLAN: Tuple[PilotSampleSpec, ...] = (
    PilotSampleSpec(
        sample_id="pilot_250_w0",
        n=250,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=0,
        is_warmup=True,
        execution_order=1,
    ),
    PilotSampleSpec(
        sample_id="pilot_250_r1",
        n=250,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=1,
        is_warmup=False,
        execution_order=2,
    ),
    PilotSampleSpec(
        sample_id="pilot_250_r2",
        n=250,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=2,
        is_warmup=False,
        execution_order=3,
    ),
    PilotSampleSpec(
        sample_id="pilot_500_w0",
        n=500,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=0,
        is_warmup=True,
        execution_order=4,
    ),
    PilotSampleSpec(
        sample_id="pilot_500_r1",
        n=500,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=1,
        is_warmup=False,
        execution_order=5,
    ),
    PilotSampleSpec(
        sample_id="pilot_500_r2",
        n=500,
        block_size=PILOT_BLOCK_SIZE,
        worker_count=PILOT_WORKER_COUNT,
        repetition=2,
        is_warmup=False,
        execution_order=6,
    ),
)


def load_sha256_checksums(
    sums_path: str = DEFAULT_SUMS_FILE,
) -> Dict[str, str]:
    """Loads recorded SHA-256 checksums from manifest with default path backward compatibility."""
    return load_sha256_manifest(sums_path)


class PilotRunner:
    """
    Dedicated orchestrator executing exclusively the bounded Pilot matrix.
    Enforces the 15-minute global budget, 120s sample timeout, and resource watchdog limits.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        results_csv: str = DEFAULT_RESULTS_FILE,
        metadata_dir: str = METADATA_DIR,
        cluster_manager: Optional[ClusterManager] = None,
        session_budget_s: float = PILOT_SESSION_BUDGET_S,
        sample_timeout_s: float = PILOT_SAMPLE_TIMEOUT_S,
        rss_ceiling_mib: float = AGGREGATE_RSS_CEILING_MIB,
        host_floor_mib: float = HOST_AVAILABLE_RAM_FLOOR_MIB,
        watchdog_interval_s: float = WATCHDOG_SAMPLE_INTERVAL_S,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.results_csv = os.path.abspath(results_csv)
        self.metadata_dir = os.path.abspath(metadata_dir)
        self.cluster_manager = cluster_manager
        self.session_budget_s = session_budget_s
        self.sample_timeout_s = sample_timeout_s
        self.rss_ceiling_mib = rss_ceiling_mib
        self.host_floor_mib = host_floor_mib
        self.watchdog_interval_s = watchdog_interval_s

        self.dense_dir = os.path.join(self.data_dir, "dense")
        self.val_dir = os.path.join(self.data_dir, "validation")
        self.sums_file = os.path.join(self.data_dir, "SHA256SUMS.txt")

        validate_injected_cluster_manager(self.cluster_manager, expected_worker_count=PILOT_WORKER_COUNT)

    def _get_framework_version(self) -> str:
        """Retrieves distributed version string."""
        try:
            import distributed
            return distributed.__version__
        except Exception:
            return "unknown"

    def _load_matrix(self, filepath: str, raw_bytes: Optional[bytes] = None) -> np.ndarray:
        """Loads a dense matrix CSV into a C-contiguous float64 array."""
        if raw_bytes is not None:
            return parse_matrix_bytes(raw_bytes)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Required matrix file not found: {filepath}")
        mat = np.loadtxt(filepath, delimiter=",", dtype=np.float64)
        return np.ascontiguousarray(mat, dtype=np.float64)

    def _verify_and_load_matrix(
        self, filepath: str, rel_path: str, manifest: Dict[str, str]
    ) -> Tuple[np.ndarray, str]:
        """
        Verifies input matrix against manifest and parses directly from verified byte snapshot.
        Eliminates TOCTOU race conditions and propagates any parsing/loading exceptions directly
        without retrying from disk.
        """
        raw_bytes, verified_sha = read_and_verify_matrix_bytes(filepath, rel_path, manifest)
        mat = self._load_matrix(filepath, raw_bytes=raw_bytes)
        return mat, verified_sha

    def run(self) -> Dict[str, Any]:
        """
        Executes the bounded pilot matrix within the strict operational envelope.

        Returns:
            Summary dictionary of execution metrics, sample statuses, and sidecar path.
        """
        session_start_mono = time.perf_counter()
        session_start_utc = datetime.now(timezone.utc).isoformat()
        deadline_mono = session_start_mono + self.session_budget_s
        run_id = f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"

        print("================================================================================")
        print("ASSIGNMENT 4 — BOUNDED PILOT EXECUTION")
        print("================================================================================")
        print(f"Run ID:                 {run_id}")
        print(f"Session Start (UTC):    {session_start_utc}")
        print(f"Global Session Budget:  {self.session_budget_s:.1f} s (15 minutes)")
        print(f"Per-Sample Timeout:     {self.sample_timeout_s:.1f} s")
        print(f"Aggregate RSS Ceiling:  {self.rss_ceiling_mib:.1f} MiB (Watchdog)")
        print(f"Host RAM Floor:         {self.host_floor_mib:.1f} MiB (Watchdog)")
        print(f"Watchdog Interval:      {self.watchdog_interval_s * 1000.0:.0f} ms")
        print(f"Worker Configuration:   p={PILOT_WORKER_COUNT}, threads={PILOT_WORKER_THREADS}, OpenBLAS={PILOT_BLAS_THREADS}")
        print(f"Matrix Dimension Set:   n in {{250, 500}}, block_size={PILOT_BLOCK_SIZE}")
        print(f"Sample Plan:            {len(PILOT_PLAN)} runs ({[s.sample_id for s in PILOT_PLAN]})")
        print("================================================================================\n")

        # 1. Validate worker configuration
        verify_multiworker_permission(PILOT_WORKER_COUNT)

        validate_injected_cluster_manager(self.cluster_manager, expected_worker_count=PILOT_WORKER_COUNT)

        cm = self.cluster_manager
        if cm is None:
            cm = ClusterManager(
                worker_count=PILOT_WORKER_COUNT,
                threads_per_worker=PILOT_WORKER_THREADS,
                worker_memory_limit=None,
            )

        topology: Optional[ClusterTopology] = None
        watchdog: Optional[ResourceWatchdog] = None
        engine: Optional[DistributedBlockMatmul] = None
        logger = RawResultsLogger(self.results_csv)

        source_rev = get_git_commit_sha()
        dist_ver = self._get_framework_version()

        execution_records: List[MeasurementRecord] = []
        c_500_r1_checksum: Optional[float] = None
        early_abort_status: Optional[str] = None
        early_abort_reason: Optional[str] = None
        timed_out_n: Optional[int] = None
        timed_out_sample_id: Optional[str] = None
        inputs_cache: Dict[int, Tuple[np.ndarray, np.ndarray, str, str, float]] = {}
        sidecar_path: str = "unwritten"

        try:
            # 2. Check input files and hashes within session budget
            recorded_sums = load_sha256_manifest(self.sums_file)

            # 3. Pre-load input matrices (synchronous local disk reads; outside sample timer, inside session budget)
            for n in (250, 500):
                t_load_start = time.perf_counter()
                a_path = os.path.join(self.dense_dir, f"A_{n}.csv")
                b_path = os.path.join(self.dense_dir, f"B_{n}.csv")

                rel_a = f"dense/A_{n}.csv"
                rel_b = f"dense/B_{n}.csv"
                a_mat, sha_a = self._verify_and_load_matrix(a_path, rel_a, recorded_sums)
                b_mat, sha_b = self._verify_and_load_matrix(b_path, rel_b, recorded_sums)
                load_duration = time.perf_counter() - t_load_start

                inputs_cache[n] = (a_mat, b_mat, sha_a, sha_b, load_duration)
                print(f"[INPUTS LOADED] n={n}: A={a_mat.shape}, B={b_mat.shape} (load time: {load_duration:.3f} s)")

            # Load reference for n=250 validation (verified against SHA256SUMS.txt)
            c_250_ref_path = os.path.join(self.val_dir, "reference", "C_250.csv")
            c_250_ref: Optional[np.ndarray] = None
            if os.path.isfile(c_250_ref_path):
                c_250_ref, _ = self._verify_and_load_matrix(
                    c_250_ref_path, "validation/reference/C_250.csv", recorded_sums
                )
                print(f"[REFERENCE LOADED] Loaded C_250.csv for analytical validation: shape={c_250_ref.shape}")

            # 4. Check deadline before cluster startup
            rem_startup = deadline_mono - time.perf_counter()
            if rem_startup <= 0:
                early_abort_status = "ABORTED_BUDGET"
                early_abort_reason = "Global session budget reached before cluster startup."
            else:
                startup_timeout = min(60.0, rem_startup)
                print(f"\n[CLUSTER] Starting LocalCluster (p=1, processes=True, threads=1, timeout={startup_timeout:.1f}s)...")
                started_topology = cm.start(timeout_s=startup_timeout)
                validate_cluster_topology(
                    started_topology,
                    expected_worker_count=PILOT_WORKER_COUNT,
                    expected_threads_per_worker=PILOT_WORKER_THREADS,
                )
                topology = started_topology
                print(f"[CLUSTER] Started successfully in {topology.cluster_startup_s:.2f} s.")
                print(f"[CLUSTER] Observed PIDs: Driver={topology.driver_pid}, Scheduler={topology.scheduler_pid}, "
                      f"Nannies={topology.nanny_pids}, Workers={topology.worker_pids} (Total unique: {len(topology.all_pids)})")

                # Start ResourceWatchdog on tracked PIDs
                watchdog = ResourceWatchdog(
                    pids_to_track=topology.all_pids,
                    client=cm.client,
                    cluster=cm.cluster,
                    rss_ceiling_mib=self.rss_ceiling_mib,
                    host_floor_mib=self.host_floor_mib,
                    sample_interval_s=self.watchdog_interval_s,
                    global_deadline_s=deadline_mono,
                )
                watchdog.start()
                engine = DistributedBlockMatmul(client=cm.client, block_size=PILOT_BLOCK_SIZE)

            # 5. Execute Fixed Pilot Matrix Plan
            print("\n[ORCHESTRATOR] Beginning 6-sample pilot execution sequence...")

            for spec in PILOT_PLAN:
                now_mono = time.perf_counter()
                rem_session = deadline_mono - now_mono
                sample_ts = datetime.now(timezone.utc).isoformat()
                a_mat, b_mat, sha_a, sha_b, load_dur = inputs_cache[spec.n]

                # Check 1: Early abort condition from previous step or deadline exhaustion
                if early_abort_status is not None:
                    status_to_record = early_abort_status
                    if early_abort_status == "NOT_RUN_AFTER_TIMEOUT":
                        if timed_out_n is not None and spec.n == timed_out_n:
                            abort_desc = f"Repetition omitted following timeout in sample {timed_out_sample_id} (n={spec.n}) per protocol §10.1."
                        else:
                            abort_desc = f"Skipped: pilot halted and cluster closed following timeout in sample {timed_out_sample_id} (n={timed_out_n}); cluster restart not authorized."
                    else:
                        abort_desc = early_abort_reason or "Aborted prior to sample execution."
                    rec = MeasurementRecord(
                        run_id=run_id,
                        sample_id=spec.sample_id,
                        timestamp_utc=sample_ts,
                        method=PILOT_METHOD,
                        framework=PILOT_FRAMEWORK,
                        framework_version=dist_ver,
                        source_revision=source_rev,
                        compiler_version=None,
                        build_flags=None,
                        n=spec.n,
                        dtype=PILOT_DTYPE,
                        input_A_sha256=sha_a,
                        input_B_sha256=sha_b,
                        block_size=spec.block_size,
                        algorithm_variant=PILOT_ALGORITHM_VARIANT,
                        worker_count=spec.worker_count,
                        process_count=len(topology.all_pids) if topology else 1,
                        physical_machine_count=PILOT_PHYSICAL_MACHINE_COUNT,
                        worker_threads=PILOT_WORKER_THREADS,
                        openmp_threads=None,
                        blas_threads=PILOT_BLAS_THREADS,
                        repetition=spec.repetition,
                        is_warmup=spec.is_warmup,
                        execution_order=spec.execution_order,
                        time_kernel_s=None,
                        time_e2e_s=None,
                        timing_boundary_id=PILOT_TIMING_BOUNDARY_ID,
                        cluster_startup_s=topology.cluster_startup_s if topology else None,
                        input_load_s=load_dur,
                        client_to_worker_payload_bytes=None,
                        worker_to_worker_payload_bytes=None,
                        worker_to_client_payload_bytes=None,
                        compressed_transfer_bytes=None,
                        transfer_metric_source=None,
                        object_nbytes=None,
                        rss_driver_mib=None,
                        rss_process_sum_peak_sampled_mib=None,
                        peak_working_set_mib=None,
                        dask_managed_memory_mib=None,
                        checksum_sum=None,
                        status=status_to_record,
                        error_reason=abort_desc,
                        timeout_limit_s=self.sample_timeout_s,
                        elapsed_until_abort_s=None,
                    )
                    logger.log(rec)
                    execution_records.append(rec)
                    print(f"[{spec.sample_id}] Skipped -> {status_to_record} ({abort_desc})")
                    continue

                # Check 2: Global Session Deadline Check
                if rem_session <= 0.0:
                    early_abort_status = "ABORTED_BUDGET"
                    early_abort_reason = f"Global session budget of {self.session_budget_s:.1f} s exhausted."
                    rec = MeasurementRecord(
                        run_id=run_id,
                        sample_id=spec.sample_id,
                        timestamp_utc=sample_ts,
                        method=PILOT_METHOD,
                        framework=PILOT_FRAMEWORK,
                        framework_version=dist_ver,
                        source_revision=source_rev,
                        compiler_version=None,
                        build_flags=None,
                        n=spec.n,
                        dtype=PILOT_DTYPE,
                        input_A_sha256=sha_a,
                        input_B_sha256=sha_b,
                        block_size=spec.block_size,
                        algorithm_variant=PILOT_ALGORITHM_VARIANT,
                        worker_count=spec.worker_count,
                        process_count=len(topology.all_pids) if topology else 1,
                        physical_machine_count=PILOT_PHYSICAL_MACHINE_COUNT,
                        worker_threads=PILOT_WORKER_THREADS,
                        openmp_threads=None,
                        blas_threads=PILOT_BLAS_THREADS,
                        repetition=spec.repetition,
                        is_warmup=spec.is_warmup,
                        execution_order=spec.execution_order,
                        time_kernel_s=None,
                        time_e2e_s=None,
                        timing_boundary_id=PILOT_TIMING_BOUNDARY_ID,
                        cluster_startup_s=topology.cluster_startup_s if topology else None,
                        input_load_s=load_dur,
                        client_to_worker_payload_bytes=None,
                        worker_to_worker_payload_bytes=None,
                        worker_to_client_payload_bytes=None,
                        compressed_transfer_bytes=None,
                        transfer_metric_source=None,
                        object_nbytes=None,
                        rss_driver_mib=None,
                        rss_process_sum_peak_sampled_mib=None,
                        peak_working_set_mib=None,
                        dask_managed_memory_mib=None,
                        checksum_sum=None,
                        status="ABORTED_BUDGET",
                        error_reason=early_abort_reason,
                        timeout_limit_s=self.sample_timeout_s,
                        elapsed_until_abort_s=0.0,
                    )
                    logger.log(rec)
                    execution_records.append(rec)
                    print(f"[{spec.sample_id}] ABORTED_BUDGET: {early_abort_reason}")
                    continue

                # Check 3: Watchdog Pre-Execution Check
                if watchdog.breach_detected:
                    if watchdog.breach_type == "OUT_OF_MEMORY":
                        early_abort_status = "NOT_RUN_AFTER_OOM"
                    elif watchdog.breach_type in ("ABORTED_WATCHDOG", "ABORTED_BUDGET"):
                        early_abort_status = watchdog.breach_type
                    else:
                        early_abort_status = "FAILED"
                    early_abort_reason = watchdog.breach_reason or "Watchdog breach prior to execution."
                    rec = MeasurementRecord(
                        run_id=run_id,
                        sample_id=spec.sample_id,
                        timestamp_utc=sample_ts,
                        method=PILOT_METHOD,
                        framework=PILOT_FRAMEWORK,
                        framework_version=dist_ver,
                        source_revision=source_rev,
                        compiler_version=None,
                        build_flags=None,
                        n=spec.n,
                        dtype=PILOT_DTYPE,
                        input_A_sha256=sha_a,
                        input_B_sha256=sha_b,
                        block_size=spec.block_size,
                        algorithm_variant=PILOT_ALGORITHM_VARIANT,
                        worker_count=spec.worker_count,
                        process_count=len(topology.all_pids) if topology else 1,
                        physical_machine_count=PILOT_PHYSICAL_MACHINE_COUNT,
                        worker_threads=PILOT_WORKER_THREADS,
                        openmp_threads=None,
                        blas_threads=PILOT_BLAS_THREADS,
                        repetition=spec.repetition,
                        is_warmup=spec.is_warmup,
                        execution_order=spec.execution_order,
                        time_kernel_s=None,
                        time_e2e_s=None,
                        timing_boundary_id=PILOT_TIMING_BOUNDARY_ID,
                        cluster_startup_s=topology.cluster_startup_s if topology else None,
                        input_load_s=load_dur,
                        client_to_worker_payload_bytes=None,
                        worker_to_worker_payload_bytes=None,
                        worker_to_client_payload_bytes=None,
                        compressed_transfer_bytes=None,
                        transfer_metric_source=None,
                        object_nbytes=None,
                        rss_driver_mib=None,
                        rss_process_sum_peak_sampled_mib=watchdog.peak_process_sum_mib,
                        peak_working_set_mib=None,
                        dask_managed_memory_mib=None,
                        checksum_sum=None,
                        status=early_abort_status,
                        error_reason=early_abort_reason,
                        timeout_limit_s=self.sample_timeout_s,
                        elapsed_until_abort_s=None,
                    )
                    logger.log(rec)
                    execution_records.append(rec)
                    print(f"[{spec.sample_id}] Skipped -> {early_abort_status}: {early_abort_reason}")
                    continue

                # 6. Execute Multiplication Sample
                cur_timeout = min(self.sample_timeout_s, max(0.1, rem_session))
                print(f"[{spec.sample_id}] Running (n={spec.n}, rep={spec.repetition}, warmup={spec.is_warmup}, timeout={cur_timeout:.1f}s)...")
                sample_start_idx = begin_sample_memory_window(watchdog)

                res: DaskExecutionResult = engine.multiply(
                    a_matrix=a_mat,
                    b_matrix=b_mat,
                    repetition_id=spec.repetition,
                    timeout_s=cur_timeout,
                    watchdog=watchdog,
                    global_deadline_s=deadline_mono,
                )

                # 7. Post-execution Numerical Correctness Validation (outside primary timer)
                final_status = res.status
                final_error = res.error_reason

                if res.status == "COMPLETED":
                    if res.c_matrix is None:
                        final_status = "FAILED"
                        final_error = "Validation failure: Completed execution returned None c_matrix."
                        print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                    # Invariant 1: Finitude and Shape Check (all n)
                    elif res.c_matrix.shape != (spec.n, spec.n) or res.c_matrix.dtype != np.float64:
                        final_status = "FAILED"
                        final_error = f"Validation failure: Invalid shape/dtype ({res.c_matrix.shape}, {res.c_matrix.dtype})"
                        print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                    elif not np.all(np.isfinite(res.c_matrix)):
                        final_status = "FAILED"
                        final_error = "Validation failure: Non-finite values (NaN or Inf) detected."
                        print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                    else:
                        # Invariant 2: Ground-truth reference validation for n=250
                        if spec.n == 250 and c_250_ref is not None:
                            validator = MatrixValidator()
                            v_res = validator.validate_matrices(res.c_matrix, c_250_ref)
                            if not v_res.passed:
                                final_status = "FAILED"
                                final_error = f"Validation failure: {v_res.error_message}"
                                print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                            else:
                                print(f"[VALIDATION PASS] {spec.sample_id} verified against C_250.csv (|max_diff|={v_res.max_abs_diff:.2e})")

                        # Invariant 3: Finitude and cross-repetition determinism for n=500 (partial control per protocol §9.3/§10.5)
                        elif spec.n == 500:
                            print(f"[VALIDATION CONTROL] {spec.sample_id} finitude verified. Running cross-repetition determinism check (partial control per protocol §9.3).")
                            if not spec.is_warmup:
                                if res.checksum is None:
                                    final_status = "FAILED"
                                    final_error = "Validation failure: Completed execution returned None checksum."
                                    print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                                elif spec.repetition == 1:
                                    if final_status == "COMPLETED":
                                        c_500_r1_checksum = res.checksum
                                elif spec.repetition == 2:
                                    if c_500_r1_checksum is not None:
                                        chk_diff = abs(res.checksum - c_500_r1_checksum)
                                        if chk_diff > 1e-6:
                                            final_status = "FAILED"
                                            final_error = f"Validation failure: Checksum divergence {chk_diff:.2e} > 1e-6 against rep=1 reference ({c_500_r1_checksum:.4f})."
                                            print(f"[VALIDATION FAIL] {spec.sample_id}: {final_error}")
                                        else:
                                            print(f"[VALIDATION PASS] n=500 cross-repetition determinism confirmed: r1 and r2 checksums match (diff={chk_diff:.2e})")

                # Telemetry extraction (strictly per-sample window on real watchdog)
                peak_rss, driver_rss, peak_ws = extract_sample_memory_metrics(
                    watchdog, sample_start_idx
                )

                # Build record
                rec = MeasurementRecord(
                    run_id=run_id,
                    sample_id=spec.sample_id,
                    timestamp_utc=sample_ts,
                    method=PILOT_METHOD,
                    framework=PILOT_FRAMEWORK,
                    framework_version=dist_ver,
                    source_revision=source_rev,
                    compiler_version=None,
                    build_flags=None,
                    n=spec.n,
                    dtype=PILOT_DTYPE,
                    input_A_sha256=sha_a,
                    input_B_sha256=sha_b,
                    block_size=spec.block_size,
                    algorithm_variant=PILOT_ALGORITHM_VARIANT,
                    worker_count=spec.worker_count,
                    process_count=len(topology.all_pids) if topology else 1,
                    physical_machine_count=PILOT_PHYSICAL_MACHINE_COUNT,
                    worker_threads=PILOT_WORKER_THREADS,
                    openmp_threads=None,
                    blas_threads=PILOT_BLAS_THREADS,
                    repetition=spec.repetition,
                    is_warmup=spec.is_warmup,
                    execution_order=spec.execution_order,
                    time_kernel_s=None,
                    time_e2e_s=res.time_e2e_s,
                    timing_boundary_id=PILOT_TIMING_BOUNDARY_ID,
                    cluster_startup_s=topology.cluster_startup_s if topology else None,
                    input_load_s=load_dur,
                    client_to_worker_payload_bytes=None,
                    worker_to_worker_payload_bytes=None,
                    worker_to_client_payload_bytes=None,
                    compressed_transfer_bytes=None,
                    transfer_metric_source=None,
                    object_nbytes=res.object_nbytes,
                    rss_driver_mib=driver_rss,
                    rss_process_sum_peak_sampled_mib=peak_rss,
                    peak_working_set_mib=peak_ws,
                    dask_managed_memory_mib=None,
                    checksum_sum=res.checksum,
                    status=final_status,
                    error_reason=final_error,
                    timeout_limit_s=self.sample_timeout_s,
                    elapsed_until_abort_s=res.elapsed_until_abort_s,
                )
                logger.log(rec)
                execution_records.append(rec)

                time_str = f"{res.time_e2e_s:.4f} s" if res.time_e2e_s is not None else "NULL"
                chk_str = f"{res.checksum:.4f}" if res.checksum is not None else "NULL"
                peak_str = f"{peak_rss:.1f} MiB" if peak_rss is not None else "NULL"
                print(f"[{spec.sample_id}] Status: {final_status} | Time: {time_str} | Checksum: {chk_str} | Peak RSS: {peak_str}")

                # Handle abort states
                if final_status == "OUT_OF_MEMORY":
                    early_abort_status = "NOT_RUN_AFTER_OOM"
                    early_abort_reason = f"OOM in sample {spec.sample_id}: {final_error}"
                elif final_status in ("ABORTED_WATCHDOG", "ABORTED_BUDGET"):
                    early_abort_status = final_status
                    early_abort_reason = final_error
                elif final_status == "TIMEOUT":
                    print(f"[{spec.sample_id}] Sample timed out (> {self.sample_timeout_s:.1f} s limit).")
                    print(f"[{spec.sample_id}] Enforcing protocol §10.1: safely closing cluster and omitting remaining repetitions.")
                    timed_out_n = spec.n
                    timed_out_sample_id = spec.sample_id
                    early_abort_status = "NOT_RUN_AFTER_TIMEOUT"
                    early_abort_reason = (
                        f"Cluster safely closed following timeout in sample {spec.sample_id} (n={spec.n}). "
                        f"Subsequent repetitions of n={spec.n} omitted per protocol §10.1; "
                        f"cluster restart not authorized."
                    )
                    if watchdog is not None:
                        try:
                            watchdog.stop()
                        except Exception as we:
                            print(f"[CLEANUP WARNING] Watchdog shutdown error: {we}")
                        watchdog = None

                    if cm is not None:
                        try:
                            cm.close(timeout_s=5.0)
                        except Exception as ce:
                            print(f"[CLEANUP WARNING] Cluster teardown error: {ce}")
                        cm = None

        except Exception as e:
            print(f"\n[ORCHESTRATOR ERROR] Unexpected exception during pilot: {e}")
            if early_abort_status is None:
                if time.perf_counter() >= deadline_mono:
                    early_abort_status = "ABORTED_BUDGET"
                    early_abort_reason = f"Global session budget exhausted: {e}"
                else:
                    early_abort_status = "FAILED"
                    early_abort_reason = f"Startup or execution failure: {e}"

            # Record any unrecorded specs as FAILED/ABORTED
            recorded_ids = {r.sample_id for r in execution_records}
            for spec in PILOT_PLAN:
                if spec.sample_id not in recorded_ids:
                    a_mat, b_mat, sha_a, sha_b, load_dur = inputs_cache.get(spec.n, (None, None, "", "", None))
                    rec = MeasurementRecord(
                        run_id=run_id,
                        sample_id=spec.sample_id,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        method=PILOT_METHOD,
                        framework=PILOT_FRAMEWORK,
                        framework_version=dist_ver,
                        source_revision=source_rev,
                        compiler_version=None,
                        build_flags=None,
                        n=spec.n,
                        dtype=PILOT_DTYPE,
                        input_A_sha256=sha_a,
                        input_B_sha256=sha_b,
                        block_size=spec.block_size,
                        algorithm_variant=PILOT_ALGORITHM_VARIANT,
                        worker_count=spec.worker_count,
                        process_count=len(topology.all_pids) if topology else 1,
                        physical_machine_count=PILOT_PHYSICAL_MACHINE_COUNT,
                        worker_threads=PILOT_WORKER_THREADS,
                        openmp_threads=None,
                        blas_threads=PILOT_BLAS_THREADS,
                        repetition=spec.repetition,
                        is_warmup=spec.is_warmup,
                        execution_order=spec.execution_order,
                        time_kernel_s=None,
                        time_e2e_s=None,
                        timing_boundary_id=PILOT_TIMING_BOUNDARY_ID,
                        cluster_startup_s=topology.cluster_startup_s if topology else None,
                        input_load_s=load_dur,
                        client_to_worker_payload_bytes=None,
                        worker_to_worker_payload_bytes=None,
                        worker_to_client_payload_bytes=None,
                        compressed_transfer_bytes=None,
                        transfer_metric_source=None,
                        object_nbytes=None,
                        rss_driver_mib=None,
                        rss_process_sum_peak_sampled_mib=None,
                        peak_working_set_mib=None,
                        dask_managed_memory_mib=None,
                        checksum_sum=None,
                        status=early_abort_status,
                        error_reason=early_abort_reason,
                        timeout_limit_s=self.sample_timeout_s,
                        elapsed_until_abort_s=None,
                    )
                    try:
                        logger.log(rec)
                    except Exception:
                        pass
                    execution_records.append(rec)

        finally:
            # 8. Clean Shutdown
            print("\n[CLEANUP] Stopping ResourceWatchdog and tearing down LocalCluster...")
            if watchdog is not None:
                try:
                    watchdog.stop()
                except Exception as we:
                    print(f"[CLEANUP WARNING] Watchdog shutdown error: {we}")

            if cm is not None:
                try:
                    cm.close(timeout_s=5.0)
                except Exception as ce:
                    print(f"[CLEANUP WARNING] Cluster teardown error: {ce}")

            # 9. Generate Campaign Sidecar
            sidecar_topology = {}
            if topology is not None:
                sidecar_topology = {
                    "driver_pid": topology.driver_pid,
                    "scheduler_pid": topology.scheduler_pid,
                    "nanny_pids": topology.nanny_pids,
                    "worker_pids": topology.worker_pids,
                    "all_pids": topology.all_pids,
                    "observed_worker_count": topology.worker_count_observed,
                    "threads_per_worker": topology.threads_per_worker,
                    "cluster_startup_s": topology.cluster_startup_s,
                }
            elif cm is not None:
                sidecar_topology = {
                    "startup_status": early_abort_status or "FAILED",
                    "driver_pid": os.getpid(),
                    "all_pids": [os.getpid()],
                }

            completed_count = sum(1 for r in execution_records if r.status == "COMPLETED")
            failed_count = sum(1 for r in execution_records if r.status != "COMPLETED")
            sidecar_status = early_abort_status if early_abort_status is not None else (
                "COMPLETED" if failed_count == 0 else ("FAILED" if completed_count == 0 else "PARTIAL")
            )
            sidecar_failure_reason = early_abort_reason
            if sidecar_failure_reason is None and failed_count > 0:
                sidecar_failure_reason = next(
                    (r.error_reason for r in execution_records if r.status != "COMPLETED" and r.error_reason),
                    f"{failed_count} of {len(execution_records)} samples did not complete.",
                )

            try:
                sidecar_path = generate_campaign_sidecar(
                    run_id=run_id,
                    campaign_name="Bounded Pilot Campaign",
                    worker_count=PILOT_WORKER_COUNT,
                    block_size=PILOT_BLOCK_SIZE,
                    n_values=[250, 500],
                    repetitions=2,
                    cluster_topology=sidecar_topology,
                    output_dir=self.metadata_dir,
                    status=sidecar_status,
                    failure_reason=sidecar_failure_reason,
                    timeout_per_sample_s=self.sample_timeout_s,
                    aggregate_rss_ceiling_mib=self.rss_ceiling_mib,
                    host_available_ram_floor_mib=self.host_floor_mib,
                )
                print(f"[SIDECAR] Generated metadata companion: {sidecar_path}")
            except Exception as se:
                print(f"[SIDECAR ERROR] Failed to generate sidecar: {se}")

        session_duration = time.perf_counter() - session_start_mono
        completed_count = sum(1 for r in execution_records if r.status == "COMPLETED")
        failed_count = sum(1 for r in execution_records if r.status != "COMPLETED")

        print("\n================================================================================")
        print("PILOT EXECUTION COMPLETE")
        print("================================================================================")
        print(f"Total Session Duration: {session_duration:.2f} s / {self.session_budget_s:.1f} s")
        print(f"Samples Attempted:      {len(execution_records)} / {len(PILOT_PLAN)}")
        print(f"Samples Completed:      {completed_count}")
        print(f"Samples Incomplete:     {failed_count}")
        print(f"Results CSV:            {self.results_csv}")
        print(f"Sidecar JSON:           {sidecar_path}")
        print("================================================================================\n")

        return {
            "run_id": run_id,
            "session_duration_s": session_duration,
            "session_budget_s": self.session_budget_s,
            "total_samples": len(PILOT_PLAN),
            "completed_samples": completed_count,
            "failed_samples": failed_count,
            "sidecar_path": sidecar_path,
            "results_csv": self.results_csv,
            "records": execution_records,
        }
