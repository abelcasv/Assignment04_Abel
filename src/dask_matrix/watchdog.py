"""
src.dask_matrix.watchdog — Process-Tree Resource Watchdog

Strictly implements the operational safety thresholds and the 5-step safe stop
procedure defined in docs/experimental_protocol.md Section 10.3:
- Tracks individual unique PIDs across Driver, Scheduler, Nannies, and Workers.
- Polls at 250 ms sampling interval on an isolated background thread.
- Enforces the provisional 2.5 GiB sampled aggregate RSS ceiling.
- Enforces the 1.0 GiB host available RAM floor.
- Executes formal 5-step cancellation, shutdown, and forceful termination on breach.
"""

from dataclasses import dataclass, field
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional
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
        on_breach_callback: Optional[Callable[[str, ProcessMemorySample], None]] = None,
    ):
        self.pids = sorted(list(set(pids_to_track)))
        self.client = client
        self.cluster = cluster
        self.rss_ceiling_mib = rss_ceiling_mib
        self.host_floor_mib = host_floor_mib
        self.sample_interval_s = sample_interval_s
        self.on_breach_callback = on_breach_callback

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.samples: List[ProcessMemorySample] = []
        self.peak_process_sum_mib: float = 0.0
        self.peak_working_set_per_pid_mib: Dict[int, float] = {}
        self.breach_detected: bool = False
        self.breach_reason: Optional[str] = None
        self.start_time_epoch_s: float = 0.0

    def start(self) -> None:
        """Starts the watchdog monitoring thread."""
        self._stop_event.clear()
        self.samples.clear()
        self.peak_process_sum_mib = 0.0
        self.peak_working_set_per_pid_mib.clear()
        self.breach_detected = False
        self.breach_reason = None
        self.start_time_epoch_s = time.time()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceWatchdogThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stops the watchdog monitoring thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

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

                # Track historical peak per process
                prev_peak = self.peak_working_set_per_pid_mib.get(pid, 0.0)
                if per_pid[pid] > prev_peak:
                    self.peak_working_set_per_pid_mib[pid] = per_pid[pid]

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        total_rss_mib = total_rss_bytes / (1024.0 * 1024.0)
        driver_rss_mib = driver_rss_bytes / (1024.0 * 1024.0)
        host_avail_mib = psutil.virtual_memory().available / (1024.0 * 1024.0)

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
            sample = self._sample_once()
            self.samples.append(sample)

            # Check Threshold 1: Aggregate RSS Ceiling (2.5 GiB)
            if sample.rss_process_sum_mib >= self.rss_ceiling_mib:
                self.breach_detected = True
                self.breach_reason = (
                    f"Aggregate RSS ceiling breached: {sample.rss_process_sum_mib:.1f} MiB "
                    f">= {self.rss_ceiling_mib:.1f} MiB limit."
                )
                self._execute_safe_stop(self.breach_reason, sample)
                break

            # Check Threshold 2: Host Available RAM Floor (1.0 GiB)
            if sample.host_available_ram_mib < self.host_floor_mib:
                self.breach_detected = True
                self.breach_reason = (
                    f"Host available RAM floor breached: {sample.host_available_ram_mib:.1f} MiB "
                    f"< {self.host_floor_mib:.1f} MiB floor."
                )
                self._execute_safe_stop(self.breach_reason, sample)
                break

            self._stop_event.wait(self.sample_interval_s)

    def _execute_safe_stop(self, reason: str, sample: ProcessMemorySample) -> None:
        """
        Executes the formal 5-step safe stop procedure:
        1. Log detection and alert.
        2. Cancel tasks in client.
        3. Orderly close client and cluster (5 s grace period).
        4. Forcefully terminate lingering PIDs.
        5. Invoke on_breach callback.
        """
        # Step 1: Alert
        sys.stderr.write(f"\n[WATCHDOG ALARM] {reason}\n")
        sys.stderr.write(f"[WATCHDOG ALARM] Executing 5-step safe stop procedure...\n")

        # Step 2: Cancel in-flight futures
        if self.client is not None:
            try:
                # Cancel all futures tracked by the client
                self.client.cancel(list(getattr(self.client, "futures", {}).values()))
            except Exception as e:
                sys.stderr.write(f"[WATCHDOG] Client task cancellation error: {e}\n")

        # Step 3: Orderly close with 5 s grace period
        try:
            if self.client is not None:
                self.client.close(timeout=5.0)
            if self.cluster is not None:
                self.cluster.close(timeout=5.0)
        except Exception as e:
            sys.stderr.write(f"[WATCHDOG] Orderly shutdown error: {e}\n")

        # Step 4: Forceful termination of child processes
        driver_pid = os.getpid()
        for pid in self.pids:
            if pid != driver_pid:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except psutil.TimeoutExpired:
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Step 5: Notify callback
        if self.on_breach_callback is not None:
            try:
                self.on_breach_callback(reason, sample)
            except Exception:
                pass

    def __enter__(self) -> "ResourceWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
