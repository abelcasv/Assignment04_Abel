"""
src.dask_matrix.watchdog — Process-Tree Resource Watchdog

Strictly implements the operational safety thresholds and the 5-step safe stop
procedure defined in SPEC.md:
- Tracks individual unique PIDs across Driver, Scheduler, Nannies, and Workers.
- Polls at 250 ms sampling interval on an isolated background thread.
- Enforces the provisional 2.5 GiB sampled aggregate RSS ceiling.
- Enforces the 1.0 GiB host available RAM floor.
- Executes formal 5-step cancellation, shutdown, and forceful termination on breach.
"""

from dataclasses import dataclass, field
import numbers
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import psutil


AGGREGATE_RSS_CEILING_MIB = 2560.0    # 2.5 GiB operational safety ceiling
HOST_AVAILABLE_RAM_FLOOR_MIB = 1024.0 # 1.0 GiB OS safety floor
WATCHDOG_SAMPLE_INTERVAL_S = 0.25     # 250 ms sampling interval


@dataclass
class ProcessMemorySample:
    """Instantaneous snapshot of memory usage across tracked processes."""
    timestamp_epoch_s: float
    elapsed_s: float
    rss_driver_mib: float
    rss_process_sum_mib: float
    host_available_ram_mib: float
    per_pid_rss_mib: Dict[int, float] = field(default_factory=dict)


class ResourceWatchdog:
    """
    Background thread resource monitor watching process memory and host RAM.
    """

    def __init__(
        self,
        pids_to_track: List[int],
        client: Optional[Any] = None,
        cluster: Optional[Any] = None,
        rss_ceiling_mib: float = AGGREGATE_RSS_CEILING_MIB,
        host_floor_mib: float = HOST_AVAILABLE_RAM_FLOOR_MIB,
        sample_interval_s: float = WATCHDOG_SAMPLE_INTERVAL_S,
        global_deadline_s: Optional[float] = None,
        on_breach_callback: Optional[Callable[[str, ProcessMemorySample], None]] = None,
    ):
        self.pids = sorted(list(set(pids_to_track)))
        self.client = client
        self.cluster = cluster
        self.rss_ceiling_mib = rss_ceiling_mib
        self.host_floor_mib = host_floor_mib
        self.sample_interval_s = sample_interval_s
        self.global_deadline_s = global_deadline_s
        self.on_breach_callback = on_breach_callback

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.samples: List[ProcessMemorySample] = []
        self.peak_process_sum_mib: float = 0.0
        self.peak_working_set_per_pid_mib: Dict[int, float] = {}
        self.breach_detected: bool = False
        self.breach_type: Optional[str] = None
        self.breach_reason: Optional[str] = None
        self.start_time_epoch_s: float = 0.0

        self._safe_stop_lock = threading.Lock()
        self._safe_stop_executed: bool = False
        self._samples_lock = threading.Lock()

    def start(self) -> None:
        """Starts the watchdog monitoring thread."""
        self._stop_event.clear()
        with self._samples_lock:
            self.samples.clear()
            self.peak_process_sum_mib = 0.0
            self.peak_working_set_per_pid_mib.clear()
        self.breach_detected = False
        self.breach_type = None
        self.breach_reason = None
        self._safe_stop_executed = False
        self.start_time_epoch_s = time.time()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceWatchdogThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stops the watchdog monitoring thread, allowing any active safe-stop to finish."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            join_timeout = 6.0 if self.breach_detected else 1.0
            self._thread.join(timeout=join_timeout)
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None

    def mark_sample_start(self) -> int:
        """Marks the starting sample index for an upcoming execution window."""
        with self._samples_lock:
            return len(self.samples)

    def get_window_metrics(
        self, start_idx: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Extracts (peak_process_sum_mib, latest_driver_rss_mib, peak_single_pid_rss_mib)
        strictly within the sample window starting at `start_idx`.
        For sub-interval (< 250 ms) runs with no background sample, takes one boundary sample.
        """
        with self._samples_lock:
            window = list(self.samples[start_idx:])
        if not window:
            sample = self._sample_once()
            with self._samples_lock:
                self.samples.append(sample)
                window = [sample]

        peak_rss = max(s.rss_process_sum_mib for s in window)
        driver_rss = window[-1].rss_driver_mib
        per_pid_values = [v for s in window for v in s.per_pid_rss_mib.values()]
        peak_single_pid = max(per_pid_values) if per_pid_values else None
        return peak_rss, driver_rss, peak_single_pid

    def _sample_once(self) -> ProcessMemorySample:
        """Takes a single memory sample across all tracked PIDs."""
        now = time.time()
        elapsed = now - self.start_time_epoch_s
        per_pid: Dict[int, float] = {}
        total_rss_bytes = 0
        driver_rss_bytes = 0
        driver_pid = os.getpid()

        for pid in list(self.pids):
            try:
                proc = psutil.Process(pid)
                rss = proc.memory_info().rss
                per_pid[pid] = rss / (1024.0 * 1024.0)
                total_rss_bytes += rss
                if pid == driver_pid:
                    driver_rss_bytes = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        total_rss_mib = total_rss_bytes / (1024.0 * 1024.0)
        driver_rss_mib = driver_rss_bytes / (1024.0 * 1024.0)
        host_avail_mib = psutil.virtual_memory().available / (1024.0 * 1024.0)

        with self._samples_lock:
            for pid, rss_mib in per_pid.items():
                prev_peak = self.peak_working_set_per_pid_mib.get(pid, 0.0)
                if rss_mib > prev_peak:
                    self.peak_working_set_per_pid_mib[pid] = rss_mib
            if total_rss_mib > self.peak_process_sum_mib:
                self.peak_process_sum_mib = total_rss_mib

        return ProcessMemorySample(
            timestamp_epoch_s=now,
            elapsed_s=elapsed,
            rss_driver_mib=driver_rss_mib,
            rss_process_sum_mib=total_rss_mib,
            host_available_ram_mib=host_avail_mib,
            per_pid_rss_mib=per_pid,
        )

    def _monitor_loop(self) -> None:
        """Internal polling loop executing every 250 ms."""
        while not self._stop_event.is_set():
            if self.breach_detected:
                break
            sample = self._sample_once()
            with self._samples_lock:
                self.samples.append(sample)

            # Check Threshold 1: Aggregate RSS Ceiling (2.5 GiB) - Priority 1 (Physical Memory Safety)
            if sample.rss_process_sum_mib >= self.rss_ceiling_mib:
                self.breach_detected = True
                self.breach_type = "OUT_OF_MEMORY"
                self.breach_reason = (
                    f"Aggregate RSS ceiling breached: {sample.rss_process_sum_mib:.1f} MiB "
                    f">= {self.rss_ceiling_mib:.1f} MiB limit."
                )
                self._execute_safe_stop(self.breach_reason, sample)
                break

            # Check Threshold 2: Host Available RAM Floor (1.0 GiB) - Priority 2 (OS System Safety)
            if sample.host_available_ram_mib < self.host_floor_mib:
                self.breach_detected = True
                self.breach_type = "ABORTED_WATCHDOG"
                self.breach_reason = (
                    f"Host available RAM floor breached: {sample.host_available_ram_mib:.1f} MiB "
                    f"< {self.host_floor_mib:.1f} MiB floor."
                )
                self._execute_safe_stop(self.breach_reason, sample)
                break

            # Check Threshold 3: Global Session Deadline - Priority 3 (Administrative Wall-Clock Budget)
            if self.global_deadline_s is not None and time.perf_counter() >= self.global_deadline_s:
                self.breach_detected = True
                self.breach_type = "ABORTED_BUDGET"
                self.breach_reason = (
                    f"Global session deadline reached ({time.perf_counter():.2f} >= {self.global_deadline_s:.2f})."
                )
                self._execute_safe_stop(self.breach_reason, sample)
                break

            self._stop_event.wait(self.sample_interval_s)

    def _execute_safe_stop(self, reason: str, sample: Optional[ProcessMemorySample] = None) -> None:
        """
        Executes the formal 5-step safe stop procedure bounded by a single shared deadline:
        1. Log detection and alert.
        2. Cancel tasks in client.
        3. Orderly close client and cluster (sharing one unified <= 5.0 s deadline).
        4. Forceful concurrent termination of child processes (terminate then kill).
        5. Invoke on_breach callback.
        """
        # Concurrency guard: Ensure single execution across watchdog and caller threads
        with self._safe_stop_lock:
            if self._safe_stop_executed:
                return
            self._safe_stop_executed = True

        stop_start = time.perf_counter()
        total_shutdown_budget = 5.0
        deadline = stop_start + total_shutdown_budget

        # Step 1: Alert
        sys.stderr.write(f"\n[WATCHDOG ALARM] {reason}\n")
        sys.stderr.write(f"[WATCHDOG ALARM] Executing 5-step safe stop procedure (budget <= {total_shutdown_budget:.1f} s)...\n")

        # Step 2: Cancel in-flight futures
        if self.client is not None:
            try:
                futures_map = getattr(self.client, "futures", {})
                if isinstance(futures_map, dict):
                    try:
                        from distributed import Future
                        from distributed.client import FutureState
                    except Exception:
                        Future = None  # type: ignore
                        FutureState = None  # type: ignore

                    to_cancel: List[Any] = []
                    for key, val in list(futures_map.items()):
                        if Future is not None and isinstance(val, Future):
                            to_cancel.append(val)
                        elif Future is not None and FutureState is not None and isinstance(val, FutureState):
                            to_cancel.append(Future(key, client=self.client))
                        else:
                            to_cancel.append(val)
                else:
                    to_cancel = list(futures_map.values())

                if to_cancel or not isinstance(futures_map, dict):
                    try:
                        self.client.cancel(to_cancel, force=True)
                    except TypeError:
                        self.client.cancel(to_cancel)
            except Exception as e:
                sys.stderr.write(f"[WATCHDOG] Client task cancellation error: {e}\n")

        # Step 3: Orderly close sharing a single actual deadline
        if self.client is not None:
            rem = deadline - time.perf_counter()
            if rem > 0.05:
                try:
                    self.client.close(timeout=rem)
                except Exception as e:
                    sys.stderr.write(f"[WATCHDOG] Client orderly shutdown error: {e}\n")

        if self.cluster is not None:
            rem = deadline - time.perf_counter()
            if rem > 0.05:
                try:
                    self.cluster.close(timeout=rem)
                except Exception as e:
                    sys.stderr.write(f"[WATCHDOG] Cluster orderly shutdown error: {e}\n")

        # Step 4: Forceful termination of all child processes (excluding driver PID)
        driver_pid = os.getpid()
        target_pids = [pid for pid in self.pids if pid != driver_pid]

        procs_to_terminate: List[psutil.Process] = []
        for pid in target_pids:
            try:
                p = psutil.Process(pid)
                if p.is_running():
                    # Collect grandchildren if worker/nanny spawned child processes
                    try:
                        for child in p.children(recursive=True):
                            if child.pid != driver_pid:
                                procs_to_terminate.append(child)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    procs_to_terminate.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Deduplicate processes by PID
        unique_procs = {p.pid: p for p in procs_to_terminate}

        # Send SIGTERM / terminate() to all targets
        for proc in unique_procs.values():
            try:
                if proc.is_running():
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait for all processes concurrently up to remaining budget
        rem = deadline - time.perf_counter()
        if rem > 0.05 and unique_procs:
            gone, alive = psutil.wait_procs(list(unique_procs.values()), timeout=rem)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        else:
            # Deadline expired, force immediate kill
            for proc in unique_procs.values():
                try:
                    if proc.is_running():
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Step 5: Notify callback
        if self.on_breach_callback is not None:
            try:
                if sample is None:
                    sample = self._sample_once()
                self.on_breach_callback(reason, sample)
            except Exception:
                pass

    def __enter__(self) -> "ResourceWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def _is_valid_metric(value: Any) -> bool:
    return value is None or (not isinstance(value, bool) and isinstance(value, numbers.Real))


def begin_sample_memory_window(watchdog: Any) -> int:
    """Safely marks the sample window start index across real or mock watchdogs."""
    if watchdog is None:
        return 0
    mark_fn = getattr(watchdog, "mark_sample_start", None)
    if callable(mark_fn):
        idx = mark_fn()
        if not isinstance(idx, bool) and isinstance(idx, numbers.Integral):
            return int(idx)
    return 0


def extract_sample_memory_metrics(
    watchdog: Any, start_idx: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extracts per-sample memory metrics (peak_process_sum_mib, driver_rss_mib,
    peak_single_pid_rss_mib) from a ResourceWatchdog, falling back cleanly for
    older mock doubles that only populate summary attributes.
    """
    if watchdog is None:
        return None, None, None
    if isinstance(watchdog, ResourceWatchdog):
        return watchdog.get_window_metrics(start_idx)

    get_window = getattr(watchdog, "get_window_metrics", None)
    if callable(get_window):
        candidate = get_window(start_idx)
        if (
            isinstance(candidate, tuple)
            and len(candidate) == 3
            and all(_is_valid_metric(v) for v in candidate)
        ):
            return candidate

    peak_rss = getattr(watchdog, "peak_process_sum_mib", None)
    samples = getattr(watchdog, "samples", None)
    driver_rss = samples[-1].rss_driver_mib if samples else None
    per_pid_peaks = getattr(watchdog, "peak_working_set_per_pid_mib", None)
    peak_ws = max(per_pid_peaks.values()) if per_pid_peaks else None
    return peak_rss, driver_rss, peak_ws
