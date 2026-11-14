"""
src.dask_matrix.cluster_manager — LocalCluster Lifecycle and Gate G1 Enforcement

Manages Dask Distributed LocalCluster and Client instances:
- Enforces DEC-01 / Gate G1: local multi-worker execution (p > 1) is strictly BLOCKED
  unless explicit written instructor permission evidence is present in docs/decisions/.
- Enforces strict single-threading per worker (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc.).
- Measures cluster startup time separately from multiplication time.
- Inspects and records the nominal 1 + 2p process hierarchy (Driver/Client, Nannies, Workers).
"""

from dataclasses import dataclass, field
import os
import sys
import time
from typing import Any, Dict, List, Optional
import psutil


DECISIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "decisions")
)
PERMISSION_FILE_CANDIDATES = [
    "DEC-01-instructor-permission.md",
    "instructor_permission.md",
    "DEC-01_permission.md",
]

THREADING_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def apply_single_thread_env() -> Dict[str, str]:
    """Sets internal BLAS and OpenMP threading limits to 1."""
    applied = {}
    for var, val in THREADING_ENV_VARS.items():
        os.environ[var] = val
        applied[var] = val
    return applied


def verify_multiworker_permission(p: int) -> bool:
    """
    Verifies that local multi-worker execution (p > 1) is permitted by checking
    for written instructor permission evidence in docs/decisions/.

    Args:
        p: Number of workers requested.

    Returns:
        True if p == 1 or valid permission evidence exists.

    Raises:
        PermissionError: If p > 1 and no permission file is present.
    """
    if p <= 1:
        return True

    if not os.path.isdir(DECISIONS_DIR):
        raise PermissionError(
            f"[DEC-01 / Gate G1 BLOCKED] Execution with p={p} workers requires written "
            f"instructor permission, but directory '{DECISIONS_DIR}' does not exist. "
            f"Single-worker (p=1) is the only permitted configuration. Falling back "
            f"silently to p=1 is strictly prohibited. Do NOT silently substitute p=1."
        )

    found = False
    for candidate in PERMISSION_FILE_CANDIDATES:
        path = os.path.join(DECISIONS_DIR, candidate)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            found = True
            break

    if not found:
        raise PermissionError(
            f"[DEC-01 / Gate G1 BLOCKED] Local multi-worker execution with p={p} workers "
            f"is strictly BLOCKED because no written instructor permission was found in "
            f"'{DECISIONS_DIR}'. Candidates checked: {PERMISSION_FILE_CANDIDATES}. "
            f"Under course rules (PDF page 7, A4-11), local multi-worker execution requires "
            f"explicit permission. Do NOT silently substitute p=1."
        )

    return True


@dataclass
class ClusterTopology:
    """Metadata detailing the active cluster process hierarchy."""
    worker_count_requested: int
    worker_count_observed: int
    threads_per_worker: int
    driver_pid: int
    scheduler_pid: int
    nanny_pids: List[int] = field(default_factory=list)
    worker_pids: List[int] = field(default_factory=list)
    all_pids: List[int] = field(default_factory=list)
    cluster_startup_s: float = 0.0
    threading_env: Dict[str, str] = field(default_factory=dict)


class ClusterManager:
    """Manages the creation, metadata capture, and shutdown of a LocalCluster."""

    def __init__(
        self,
        worker_count: int = 1,
        threads_per_worker: int = 1,
        worker_memory_limit: Optional[str] = None,
    ):
        self.worker_count = worker_count
        self.threads_per_worker = threads_per_worker
        self.worker_memory_limit = worker_memory_limit
        self.cluster: Optional[Any] = None
        self.client: Optional[Any] = None
        self.topology: Optional[ClusterTopology] = None

    def start(self, timeout_s: float = 60.0) -> ClusterTopology:
        """
        Starts the LocalCluster and Client, enforcing thread limits and DEC-01.

        Args:
            timeout_s: Maximum duration in seconds to wait for workers to register (default: 60.0).

        Returns:
            ClusterTopology descriptor.

        Raises:
            PermissionError: If worker_count > 1 and DEC-01 permission is missing.
            ImportError: If dask.distributed is not installed.
        """
        verify_multiworker_permission(self.worker_count)
        applied_env = apply_single_thread_env()

        try:
            from distributed import LocalCluster, Client
        except ImportError as e:
            raise ImportError(
                "dask.distributed is required to start a LocalCluster. "
                "Ensure dask and distributed are installed in your Python environment."
            ) from e

        t0 = time.perf_counter()

        try:
            # Immediate check for non-positive startup timeout budget
            if timeout_s <= 0.0:
                raise TimeoutError(
                    f"Cluster startup timeout_s must be positive, got {timeout_s:.4f}s."
                )

            # In LocalCluster with processes=True, workers run under Nanny processes.
            # The scheduler runs in the driver process or its own thread/loop.
            self.cluster = LocalCluster(
                n_workers=self.worker_count,
                threads_per_worker=self.threads_per_worker,
                processes=True,
                memory_limit=self.worker_memory_limit,
                silence_logs=True,
            )

            # Check remaining budget after LocalCluster construction
            elapsed_lc = time.perf_counter() - t0
            rem_timeout = timeout_s - elapsed_lc
            if rem_timeout <= 0.0:
                raise TimeoutError(
                    f"Cluster startup exceeded timeout_s={timeout_s:.1f}s during LocalCluster initialization."
                )

            # Pass actual remaining timeout to Client connection (no artificial minimum)
            self.client = Client(self.cluster, timeout=rem_timeout)

            # Check remaining budget after Client construction
            elapsed_cl = time.perf_counter() - t0
            rem_timeout = timeout_s - elapsed_cl
            if rem_timeout <= 0.0:
                raise TimeoutError(
                    f"Cluster startup exceeded timeout_s={timeout_s:.1f}s during Client initialization."
                )

            # Wait for worker registration within actual remaining budget (no artificial minimum)
            self.client.wait_for_workers(n_workers=self.worker_count, timeout=rem_timeout)

            startup_duration = time.perf_counter() - t0

            # Discover process hierarchy
            driver_pid = os.getpid()
            scheduler_pid = getattr(self.cluster.scheduler, "pid", driver_pid)
            if not isinstance(scheduler_pid, int):
                scheduler_pid = driver_pid

            nanny_pids: List[int] = []
            worker_pids: List[int] = []

            # Extract nannies and workers
            for worker_key, nanny in self.cluster.workers.items():
                if hasattr(nanny, "pid") and nanny.pid:
                    nanny_pids.append(nanny.pid)
                if hasattr(nanny, "process") and hasattr(nanny.process, "pid") and nanny.process.pid:
                    worker_pids.append(nanny.process.pid)

            # Collect unique PIDs for watchdog monitoring
            all_pids = sorted(list(set([driver_pid, scheduler_pid] + nanny_pids + worker_pids)))

            self.topology = ClusterTopology(
                worker_count_requested=self.worker_count,
                worker_count_observed=len(self.cluster.workers),
                threads_per_worker=self.threads_per_worker,
                driver_pid=driver_pid,
                scheduler_pid=scheduler_pid,
                nanny_pids=nanny_pids,
                worker_pids=worker_pids,
                all_pids=all_pids,
                cluster_startup_s=startup_duration,
                threading_env=applied_env,
            )

            return self.topology

        except Exception as exc:
            # On any startup failure (Client failure, worker registration timeout, etc.),
            # ensure all partially created resources are immediately torn down to prevent orphans.
            self.close(timeout_s=min(5.0, timeout_s))
            raise exc

    def close(self, timeout_s: float = 5.0) -> None:
        """
        Orderly closes the Client and LocalCluster within a shared deadline budget,
        and forcefully terminates any orphaned child processes (workers, nannies).
        """
        driver_pid = os.getpid()
        cleanup_start = time.perf_counter()
        deadline = cleanup_start + timeout_s

        # 1. Discover child processes before closing cluster.
        # Explicitly capture process identity: mapping of PID -> expected creation time (float)
        target_proc_identities: Dict[int, float] = {}

        candidate_pids: List[int] = []
        if self.cluster is not None and hasattr(self.cluster, "workers"):
            try:
                for worker_key, nanny in self.cluster.workers.items():
                    if hasattr(nanny, "pid") and nanny.pid:
                        candidate_pids.append(nanny.pid)
                    if hasattr(nanny, "process") and hasattr(nanny.process, "pid") and nanny.process.pid:
                        candidate_pids.append(nanny.process.pid)
            except Exception:
                pass

        if self.topology is not None:
            candidate_pids.extend(self.topology.nanny_pids)
            candidate_pids.extend(self.topology.worker_pids)

        # Exclude driver PID, invalid PIDs, and deduplicate
        for pid in sorted(list(set(candidate_pids))):
            if pid != driver_pid and pid > 0:
                try:
                    if psutil.pid_exists(pid):
                        p = psutil.Process(pid)
                        if p.is_running():
                            target_proc_identities[pid] = p.create_time()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Helper to verify whether process still exists with matching identity
        def _get_verified_process(pid: int, expected_ctime: float) -> Optional[psutil.Process]:
            try:
                if not psutil.pid_exists(pid):
                    return None
                proc = psutil.Process(pid)
                if not proc.is_running():
                    return None
                # Verify creation time matches within 100 microseconds (protecting against PID reuse)
                proc_ctime = proc.create_time()
                if isinstance(proc_ctime, (int, float)) and isinstance(expected_ctime, (int, float)):
                    if abs(proc_ctime - expected_ctime) < 1e-4:
                        return proc
                    return None
                elif proc_ctime == expected_ctime:
                    return proc
                return None  # PID was reused by another process
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return None

        # 2. Orderly close Client with shared remaining deadline (actual remaining time)
        if self.client is not None:
            rem = deadline - time.perf_counter()
            try:
                if rem > 0.0:
                    self.client.close(timeout=rem)
            except Exception:
                pass
            finally:
                self.client = None

        # 3. Orderly close LocalCluster with shared remaining deadline (actual remaining time)
        if self.cluster is not None:
            rem = deadline - time.perf_counter()
            try:
                if rem > 0.0:
                    self.cluster.close(timeout=rem)
            except Exception:
                pass
            finally:
                self.cluster = None

        # 4. Check process termination: Send terminate() only to alive processes with matching identity
        alive_procs: List[psutil.Process] = []
        for pid, ctime in target_proc_identities.items():
            proc = _get_verified_process(pid, ctime)
            if proc is not None:
                try:
                    proc.terminate()
                    alive_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # 5. Wait for termination within remaining deadline budget; kill only confirmed alive matching processes
        rem = deadline - time.perf_counter()
        if rem > 0.05 and alive_procs:
            try:
                gone, alive = psutil.wait_procs(alive_procs, timeout=rem)
                for proc in alive:
                    expected_ctime = target_proc_identities.get(proc.pid)
                    if expected_ctime is not None and _get_verified_process(proc.pid, expected_ctime) is not None:
                        try:
                            proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except Exception:
                for proc in alive_procs:
                    expected_ctime = target_proc_identities.get(proc.pid)
                    if expected_ctime is not None and _get_verified_process(proc.pid, expected_ctime) is not None:
                        try:
                            proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
        else:
            for proc in alive_procs:
                expected_ctime = target_proc_identities.get(proc.pid)
                if expected_ctime is not None and _get_verified_process(proc.pid, expected_ctime) is not None:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
