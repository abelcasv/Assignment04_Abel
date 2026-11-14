"""
src.dask_matrix.pilot_runner — Phase 7 Bounded Pilot Execution Orchestrator

Implements the formal and binding Pilot experimental specification defined in:
- docs/experimental_protocol.md Section 10.5
- SPEC.md and CLAUDE.md

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
- Worker count: p = 1 exclusively (DEC-01 / Gate G1)
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
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from src.dask_matrix.cluster_manager import (
    ClusterManager,
    ClusterTopology,
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


def load_sha256_checksums(sums_path: str = DEFAULT_SUMS_FILE) -> Dict[str, str]:
    """Loads recorded SHA-256 checksums from data/SHA256SUMS.txt."""
    sums: Dict[str, str] = {}
    if os.path.isfile(sums_path):
        with open(sums_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    h, rel = parts[0], parts[1].replace("\\", "/").strip()
                    sums[rel] = h
    return sums


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PilotRunner:
    """
    Dedicated orchestrator executing exclusively the bounded Phase 7 Pilot matrix.
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

    def _get_framework_version(self) -> str:
        """Retrieves distributed version string."""
        try:
            import distributed
            return distributed.__version__
        except Exception:
            return "unknown"

    def _load_matrix(self, filepath: str) -> np.ndarray:
        """Loads a dense matrix CSV into a C-contiguous float64 array."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Required matrix file not found: {filepath}")
        mat = np.loadtxt(filepath, delimiter=",", dtype=np.float64)
        return np.ascontiguousarray(mat, dtype=np.float64)

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
        print("ASSIGNMENT 4 — PHASE 7 BOUNDED PILOT EXECUTION")
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

        # 1. Enforce Gate G1
        verify_multiworker_permission(PILOT_WORKER_COUNT)

        cm = self.cluster_manager
        owns_cluster = False
        if cm is None:
            cm = ClusterManager(
                worker_count=PILOT_WORKER_COUNT,
                threads_per_worker=PILOT_WORKER_THREADS,
                worker_memory_limit=None,
            )
            owns_cluster = True

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
        inputs_cache: Dict[int, Tuple[np.ndarray, np.ndarray, str, str, float]] = {}
        sidecar_path: str = "unwritten"

        try:
            # 2. Check input files and hashes within session budget
            recorded_sums = load_sha256_checksums(self.sums_file)

            # 3. Pre-load input matrices (synchronous local disk reads; outside sample timer, inside session budget)
            for n in (250, 500):
                t_load_start = time.perf_counter()
                a_path = os.path.join(self.dense_dir, f"A_{n}.csv")
                b_path = os.path.join(self.dense_dir, f"B_{n}.csv")

                a_mat = self._load_matrix(a_path)
                b_mat = self._load_matrix(b_path)
                load_duration = time.perf_counter() - t_load_start

                # Resolve input SHA-256
                rel_a = f"dense/A_{n}.csv"
                rel_b = f"dense/B_{n}.csv"
                sha_a = recorded_sums.get(rel_a) or compute_file_sha256(a_path)
                sha_b = recorded_sums.get(rel_b) or compute_file_sha256(b_path)

                inputs_cache[n] = (a_mat, b_mat, sha_a, sha_b, load_duration)
                print(f"[INPUTS LOADED] n={n}: A={a_mat.shape}, B={b_mat.shape} (load time: {load_duration:.3f} s)")

            # Load reference for n=250 validation
            c_250_ref_path = os.path.join(self.val_dir, "reference", "C_250.csv")
            c_250_ref: Optional[np.ndarray] = None
            if os.path.isfile(c_250_ref_path):
                c_250_ref = self._load_matrix(c_250_ref_path)
                print(f"[REFERENCE LOADED] Loaded C_250.csv for analytical validation: shape={c_250_ref.shape}")

            # 4. Check deadline before cluster startup
            rem_startup = deadline_mono - time.perf_counter()
            if rem_startup <= 0:
                early_abort_status = "ABORTED_BUDGET"
                early_abort_reason = "Global session budget reached before cluster startup."
            else:
                startup_timeout = min(60.0, rem_startup)
                print(f"\n[CLUSTER] Starting LocalCluster (p=1, processes=True, threads=1, timeout={startup_timeout:.1f}s)...")
                topology = cm.start(timeout_s=startup_timeout)
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
                        process_count=len(topology.all_pids) if topology else 2,
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
                        error_reason=early_abort_reason,
                        timeout_limit_s=self.sample_timeout_s,
                        elapsed_until_abort_s=None,
                    )
                    logger.log(rec)
                    execution_records.append(rec)
                    print(f"[{spec.sample_id}] Skipped -> {status_to_record} ({early_abort_reason})")
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
                        process_count=len(topology.all_pids) if topology else 2,
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
                    early_abort_status = "NOT_RUN_AFTER_OOM" if watchdog.breach_type == "OUT_OF_MEMORY" else "FAILED"
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
                        process_count=len(topology.all_pids) if topology else 2,
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

                res: DaskExecutionResult = engine.multiply(
                    a_matrix=a_mat,
                    b_matrix=b_mat,
                    repetition_id=spec.repetition,
                    timeout_s=cur_timeout,
                    watchdog=watchdog,
                    global_deadline_s=deadline_mono,
                )

                # 7. Post-execution Numerical Correctness Validation (outside primary timer)
                validation_passed = True
                if res.status == "COMPLETED" and res.c_matrix is not None:
                    # Invariant 1: Finitude and Shape Check (all n)
                    if res.c_matrix.shape != (spec.n, spec.n) or res.c_matrix.dtype != np.float64:
                        print(f"[VALIDATION FAIL] Invalid shape/dtype for {spec.sample_id}: shape={res.c_matrix.shape}, dtype={res.c_matrix.dtype}")
                        validation_passed = False
                    elif not np.all(np.isfinite(res.c_matrix)):
                        print(f"[VALIDATION FAIL] Non-finite values (NaN or Inf) detected in {spec.sample_id}")
                        validation_passed = False

                    # Invariant 2: Ground-truth reference validation for n=250
                    if spec.n == 250 and c_250_ref is not None and validation_passed:
                        validator = MatrixValidator()
                        v_res = validator.validate_matrices(res.c_matrix, c_250_ref)
                        if not v_res.passed:
                            print(f"[VALIDATION FAIL] Elementwise check failed for {spec.sample_id}: {v_res.error_message}")
                            validation_passed = False
                        else:
                            print(f"[VALIDATION PASS] {spec.sample_id} verified against C_250.csv (|max_diff|={v_res.max_abs_diff:.2e})")

                    # Invariant 3: Finitude and cross-repetition determinism for n=500 (partial control per protocol §9.3/§10.5)
                    elif spec.n == 500 and validation_passed:
                        print(f"[VALIDATION CONTROL] {spec.sample_id} finitude verified. Running cross-repetition determinism check (partial control per protocol §9.3).")
                        if spec.repetition == 1 and not spec.is_warmup:
                            c_500_r1_checksum = res.checksum
                        elif spec.repetition == 2 and not spec.is_warmup:
                            if c_500_r1_checksum is not None:
                                chk_diff = abs(res.checksum - c_500_r1_checksum)
                                if chk_diff > 1e-6:
                                    print(f"[VALIDATION WARNING] n=500 checksum mismatch between r1 ({c_500_r1_checksum}) and r2 ({res.checksum}): diff={chk_diff}")
                                else:
                                    print(f"[VALIDATION PASS] n=500 cross-repetition determinism confirmed: r1 and r2 checksums match (diff={chk_diff:.2e})")

                # Telemetry extraction
                peak_rss = watchdog.peak_process_sum_mib
                driver_rss = watchdog.samples[-1].rss_driver_mib if watchdog.samples else None
                peak_ws = max(watchdog.peak_working_set_per_pid_mib.values()) if watchdog.peak_working_set_per_pid_mib else None

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
                    process_count=len(topology.all_pids) if topology else 2,
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
                    status=res.status,
                    error_reason=res.error_reason,
                    timeout_limit_s=self.sample_timeout_s,
                    elapsed_until_abort_s=res.elapsed_until_abort_s,
                )
                logger.log(rec)
                execution_records.append(rec)

                time_str = f"{res.time_e2e_s:.4f} s" if res.time_e2e_s is not None else "NULL"
                chk_str = f"{res.checksum:.4f}" if res.checksum is not None else "NULL"
                print(f"[{spec.sample_id}] Status: {res.status} | Time: {time_str} | Checksum: {chk_str} | Peak RSS: {peak_rss:.1f} MiB")

                # Handle abort states
                if res.status == "OUT_OF_MEMORY":
                    early_abort_status = "NOT_RUN_AFTER_OOM"
                    early_abort_reason = f"OOM in sample {spec.sample_id}: {res.error_reason}"
                elif res.status in ("ABORTED_WATCHDOG", "ABORTED_BUDGET"):
                    early_abort_status = res.status
                    early_abort_reason = res.error_reason
                elif res.status == "TIMEOUT":
                    # Mark remaining repetitions of this n as NOT_RUN_TIMEOUT
                    print(f"[{spec.sample_id}] Sample timed out. Protocol stops repetitions for this dimension.")

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
                    a_mat, b_mat, sha_a, sha_b, load_dur = inputs_cache.get(spec.n, (None, None, "", "", 0.0))
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
                        process_count=len(topology.all_pids) if topology else 2,
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
                    logger.log(rec)
                    execution_records.append(rec)

        finally:
            # 8. Clean Shutdown
            print("\n[CLEANUP] Stopping ResourceWatchdog and tearing down LocalCluster...")
            if watchdog is not None:
                try:
                    watchdog.stop()
                except Exception as we:
                    print(f"[CLEANUP WARNING] Watchdog shutdown error: {we}")

            if owns_cluster and cm is not None:
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
                "COMPLETED" if failed_count == 0 else "PARTIAL"
            )

            try:
                sidecar_path = generate_campaign_sidecar(
                    run_id=run_id,
                    campaign_name="Phase 7 Bounded Pilot Campaign",
                    worker_count=PILOT_WORKER_COUNT,
                    block_size=PILOT_BLOCK_SIZE,
                    n_values=[250, 500],
                    repetitions=2,
                    cluster_topology=sidecar_topology,
                    output_dir=self.metadata_dir,
                    status=sidecar_status,
                    failure_reason=early_abort_reason,
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
