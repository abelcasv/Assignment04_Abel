"""
src.dask_matrix.cluster_manager — LocalCluster Lifecycle and Topology Validation

Manages Dask Distributed LocalCluster and Client instances:
- Validates positive integral worker counts (p >= 1) and observed cluster topology.
- Enforces strict single-threading per worker (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc.).
- Measures cluster startup time separately from multiplication time.
- Inspects and records the process hierarchy (Driver/Client/Scheduler, Nannies, Workers).
"""

from dataclasses import dataclass, field
import math
import numbers
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import psutil


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


def _extract_valid_pid(value: Any) -> Optional[int]:
    """Returns a positive integer PID, filtering out booleans, mocks, and non-positive values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    pid = int(value)
    return pid if pid > 0 else None


def _extract_worker_hierarchy_pids(workers_map: Any) -> Tuple[List[int], List[int]]:
    """Safely extracts valid positive nanny and worker PIDs from a cluster workers mapping."""
    nanny_pids: List[int] = []
    worker_pids: List[int] = []
    if not workers_map or not hasattr(workers_map, "items"):
        return nanny_pids, worker_pids
    try:
        for _, nanny in workers_map.items():
            proc = getattr(nanny, "process", None)
            w_pid = _extract_valid_pid(getattr(proc, "pid", None)) if proc is not None else None
            if w_pid is not None:
                worker_pids.append(w_pid)
            n_pid = _extract_valid_pid(getattr(nanny, "pid", None))
            # LocalCluster's Nanny.pid returns the worker subprocess PID when in-process
            if n_pid is not None and n_pid != w_pid:
                nanny_pids.append(n_pid)
    except Exception:
        pass
    return nanny_pids, worker_pids


def verify_multiworker_permission(p: int) -> bool:
    """
    Validates that the requested worker count p is a positive integer (p >= 1).

    Args:
        p: Number of workers requested. Must be a positive integer (p >= 1).

    Returns:
        True if p is a valid positive integer (p >= 1).

    Raises:
        TypeError: If p is a boolean or not an integral number.
        ValueError: If p < 1.
    """
    if isinstance(p, bool) or not isinstance(p, numbers.Integral):
        raise TypeError(f"Worker count p must be an integer, got {type(p).__name__}.")

    if p < 1:
        raise ValueError(f"Worker count p must be a positive integer (p >= 1), got {p}.")

    return True


def validate_injected_cluster_manager(cm: Any, expected_worker_count: int = 1) -> None:
    """
    Validates that an injected cluster manager satisfies the expected worker-count contract.

    Enforces fail-closed validation:
    - If cm is None, validation passes (the runner instantiates its own ClusterManager).
    - cm must expose a 'worker_count' attribute (or equivalent configured attribute on duck-typed test doubles).
    - worker_count must be an integral type (numbers.Integral) and not a boolean.
    - worker_count must be positive (>= 1).
    - worker_count must exactly equal expected_worker_count.

    Args:
        cm: Injected cluster manager instance or test double.
        expected_worker_count: Expected worker count p (default: 1).

    Raises:
        AttributeError: If cm lacks a 'worker_count' attribute.
        TypeError: If worker_count is a boolean, float, None, or non-integral type.
        ValueError: If worker_count < 1 or worker_count != expected_worker_count.
    """
    if cm is None:
        return

    if not hasattr(cm, "worker_count"):
        raise AttributeError("Injected cluster manager must declare a 'worker_count' attribute.")

    worker_count = cm.worker_count

    if isinstance(worker_count, bool) or not isinstance(worker_count, numbers.Integral):
        raise TypeError(
            f"Injected cluster manager worker_count must be an integer, got {type(worker_count).__name__}."
        )

    if worker_count < 1:
        raise ValueError(
            f"Injected cluster manager worker_count must be a positive integer (p >= 1), got {worker_count}."
        )

    if worker_count != expected_worker_count:
        raise ValueError(
            f"Injected cluster manager requires worker_count={expected_worker_count}, got {worker_count}."
        )


def validate_cluster_topology(
    topology: Any,
    expected_worker_count: int = 1,
    expected_threads_per_worker: Optional[int] = None,
) -> "ClusterTopology":
    """
    Validates a started cluster topology descriptor using fail-closed checks.

    Ensures the topology descriptor is non-None and exposes valid, integral,
    expected worker counts and (optionally) threads per worker.

    Raises:
        TypeError: If expected counts or reported topology counts are boolean or non-integral.
        ValueError: If topology is None, missing required attributes, or mismatches expectations.
    """
    if isinstance(expected_worker_count, bool) or not isinstance(expected_worker_count, numbers.Integral):
        raise TypeError(
            f"expected_worker_count must be an integer, got {type(expected_worker_count).__name__}."
        )
    if expected_worker_count < 1:
        raise ValueError(
            f"expected_worker_count must be a positive integer (p >= 1), got {expected_worker_count}."
        )

    if expected_threads_per_worker is not None:
        if isinstance(expected_threads_per_worker, bool) or not isinstance(expected_threads_per_worker, numbers.Integral):
            raise TypeError(
                f"expected_threads_per_worker must be an integer, got {type(expected_threads_per_worker).__name__}."
            )
        if expected_threads_per_worker < 1:
            raise ValueError(
                f"expected_threads_per_worker must be a positive integer, got {expected_threads_per_worker}."
            )

    if topology is None:
        raise ValueError("Cluster startup failed worker count verification: topology descriptor is None.")

    checks = [
        ("worker_count_requested", expected_worker_count, "worker_count_requested"),
        ("worker_count_observed", expected_worker_count, "worker count"),
    ]
    if expected_threads_per_worker is not None:
        checks.append(("threads_per_worker", expected_threads_per_worker, "threads_per_worker"))

    for attr_name, expected_val, label in checks:
        if not hasattr(topology, attr_name):
            raise ValueError(
                f"Cluster startup failed {label} verification: topology missing '{attr_name}' attribute."
            )
        val = getattr(topology, attr_name)
        if isinstance(val, bool) or not isinstance(val, numbers.Integral):
            raise TypeError(
                f"Cluster topology {attr_name} must be an integer, got {type(val).__name__}."
            )
        if val != expected_val:
            raise ValueError(
                f"Cluster startup failed {label} verification: expected {expected_val}, got {val}."
            )

    return topology


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
        if isinstance(worker_count, bool) or not isinstance(worker_count, numbers.Integral):
            raise TypeError(
                f"worker_count must be an integer, got {type(worker_count).__name__}."
            )
        if worker_count < 1:
            raise ValueError(
                f"worker_count must be a positive integer (p >= 1), got {worker_count}."
            )

        if isinstance(threads_per_worker, bool) or not isinstance(threads_per_worker, numbers.Integral):
            raise TypeError(
                f"threads_per_worker must be an integer, got {type(threads_per_worker).__name__}."
            )
        if threads_per_worker < 1:
            raise ValueError(
                f"threads_per_worker must be a positive integer, got {threads_per_worker}."
            )

        self.worker_count = int(worker_count)
        self.threads_per_worker = int(threads_per_worker)
        self.worker_memory_limit = worker_memory_limit
        self.cluster: Optional[Any] = None
        self.client: Optional[Any] = None
        self.topology: Optional[ClusterTopology] = None

    def start(self, timeout_s: float = 60.0) -> ClusterTopology:
        """
        Starts the LocalCluster and Client, enforcing thread limits and topology validation.

        Args:
            timeout_s: Maximum duration in seconds to wait for workers to register (default: 60.0).

        Returns:
            ClusterTopology descriptor.

        Raises:
            RuntimeError: If the cluster manager is already started.
            TypeError: If timeout_s is boolean or non-real.
            TimeoutError: If timeout_s is non-positive or non-finite, or startup times out.
            ImportError: If dask.distributed is not installed.
        """
        if self.cluster is not None or self.client is not None or self.topology is not None:
            raise RuntimeError(
                "ClusterManager is already started; call close() before calling start() again."
            )

        verify_multiworker_permission(self.worker_count)

        if isinstance(timeout_s, bool) or not isinstance(timeout_s, numbers.Real):
            raise TypeError(
                f"Cluster startup timeout_s must be a real number, got {type(timeout_s).__name__}."
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise TimeoutError(
                f"Cluster startup timeout_s must be positive and finite, got {timeout_s!r}."
            )

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
            scheduler_pid = _extract_valid_pid(getattr(self.cluster.scheduler, "pid", None))
            if scheduler_pid is None:
                scheduler_pid = driver_pid

            nanny_pids, worker_pids = _extract_worker_hierarchy_pids(
                getattr(self.cluster, "workers", None)
            )

            # Collect unique PIDs for watchdog monitoring
            all_pids = sorted(list(set([driver_pid, scheduler_pid] + nanny_pids + worker_pids)))

            topology = ClusterTopology(
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
            validate_cluster_topology(
                topology,
                expected_worker_count=self.worker_count,
                expected_threads_per_worker=self.threads_per_worker,
            )
            self.topology = topology

            return self.topology

        except Exception as exc:
            # On any startup failure (Client failure, worker registration timeout, etc.),
            # ensure all partially created resources are immediately torn down to prevent orphans.
            self.close(timeout_s=max(1.0, min(5.0, float(timeout_s))))
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
            c_nannies, c_workers = _extract_worker_hierarchy_pids(self.cluster.workers)
            candidate_pids.extend(c_nannies)
            candidate_pids.extend(c_workers)

        if self.topology is not None:
            for pid in list(self.topology.nanny_pids) + list(self.topology.worker_pids):
                valid_pid = _extract_valid_pid(pid)
                if valid_pid is not None:
                    candidate_pids.append(valid_pid)

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

        self.topology = None
