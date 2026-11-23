"""
src.dask_matrix.telemetry — Dask Diagnostic and Telemetry Extraction

Extracts diagnostic task events, Dask managed object sizes, and task execution
breakdowns from Dask Distributed:
- Strictly distinguishes object nbytes from physical network transfer.
- Distinguishes local IPC/loopback communication from physical multi-node networks.
- Provides hooks to capture task streams and performance reports for separate
  diagnostic characterization without contaminating primary benchmark timings.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import numbers
import os
from typing import Any, Dict, List, Optional


def extract_object_nbytes(raw_nbytes: Any) -> Optional[int]:
    """
    Normalizes Dask Client.nbytes() output into a scalar byte total.

    In Dask Distributed, Client.nbytes() typically returns a dict mapping
    worker addresses (or task keys) to byte counts. Scalar integral values
    (used by some mocks) are also supported. Unrecognized types (such as
    unconfigured mocks or booleans) fail cleanly to None.
    """
    if raw_nbytes is None:
        return None
    if isinstance(raw_nbytes, Mapping):
        total = 0
        for value in raw_nbytes.values():
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                return None
            total += int(value)
        return total
    if isinstance(raw_nbytes, bool) or not isinstance(raw_nbytes, numbers.Real) or not math.isfinite(raw_nbytes):
        return None
    return int(raw_nbytes)


@dataclass
class TaskDiagnostics:
    """Summary of events collected via Dask's get_task_stream."""
    total_tasks: int
    matmul_tasks_count: int
    reduction_tasks_count: int
    mean_matmul_duration_s: float
    mean_reduction_duration_s: float
    total_compute_s: float
    total_transfer_s: float
    total_deserialization_s: float


class TelemetryCollector:
    """Extracts and formats telemetry metrics from an active Dask client."""

    def __init__(self, client: Any):
        self.client = client

    def get_object_nbytes(self) -> Optional[int]:
        """Returns the total estimated memory of cluster-resident objects tracked by Dask."""
        try:
            return extract_object_nbytes(self.client.nbytes())
        except Exception:
            return None

    def get_worker_memory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns managed memory and process memory metrics reported by workers."""
        stats: Dict[str, Dict[str, Any]] = {}
        try:
            worker_info = self.client.scheduler_info().get("workers", {})
            for worker_addr, info in worker_info.items():
                metrics = info.get("metrics")
                metrics_dict = metrics if isinstance(metrics, Mapping) else {}
                legacy_mem = info.get("memory")
                legacy_mem_dict = legacy_mem if isinstance(legacy_mem, Mapping) else {}

                managed = metrics_dict.get("managed_bytes", legacy_mem_dict.get("managed", 0))
                process = metrics_dict.get("memory", legacy_mem_dict.get("process", 0))
                spilled_raw = metrics_dict.get("spilled_bytes", legacy_mem_dict.get("spilled", 0))
                if isinstance(spilled_raw, Mapping):
                    spilled = sum(int(v) for v in spilled_raw.values() if isinstance(v, numbers.Real) and not isinstance(v, bool))
                elif isinstance(spilled_raw, numbers.Real) and not isinstance(spilled_raw, bool):
                    spilled = int(spilled_raw)
                else:
                    spilled = 0

                stats[worker_addr] = {
                    "managed_bytes": int(managed) if isinstance(managed, numbers.Real) and not isinstance(managed, bool) else 0,
                    "process_bytes": int(process) if isinstance(process, numbers.Real) and not isinstance(process, bool) else 0,
                    "spilled_bytes": spilled,
                    "nthreads": info.get("nthreads", 1),
                }
        except Exception:
            pass
        return stats

    def get_worker_transfer_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Returns cumulative worker-to-worker transfer counters reported by Dask 2026.8.0
        in scheduler_info()['workers'][addr]['metrics']['transfer'].
        """
        stats: Dict[str, Dict[str, int]] = {}
        try:
            worker_info = self.client.scheduler_info().get("workers", {})
            for worker_addr, info in worker_info.items():
                metrics = info.get("metrics")
                metrics_dict = metrics if isinstance(metrics, Mapping) else {}
                transfer = metrics_dict.get("transfer")
                transfer_dict = transfer if isinstance(transfer, Mapping) else {}

                def _as_int(val: Any) -> int:
                    if isinstance(val, bool) or not isinstance(val, numbers.Real):
                        return 0
                    return int(val)

                stats[worker_addr] = {
                    "incoming_bytes": _as_int(transfer_dict.get("incoming_bytes", 0)),
                    "outgoing_bytes": _as_int(transfer_dict.get("outgoing_bytes", 0)),
                    "incoming_count_total": _as_int(transfer_dict.get("incoming_count_total", 0)),
                    "outgoing_count_total": _as_int(transfer_dict.get("outgoing_count_total", 0)),
                }
        except Exception:
            pass
        return stats

    def parse_task_stream(self, raw_task_stream: List[Dict[str, Any]]) -> TaskDiagnostics:
        """
        Parses task stream events returned by get_task_stream into aggregate metrics.

        Supports both Dask Distributed's native `startstops` sequence (list or tuple) schema and simplified
        flat (`start`, `stop`, `transfer`, `deserialization_duration`) dictionaries.

        Args:
            raw_task_stream: List of task event dictionaries from Dask.

        Returns:
            TaskDiagnostics summary.
        """
        matmul_durations = []
        reduction_durations = []
        other_compute_s = 0.0
        total_transfer = 0.0
        total_deser = 0.0

        for task in raw_task_stream:
            key = str(task.get("key", ""))
            startstops = task.get("startstops")

            if isinstance(startstops, (list, tuple)):
                compute_duration = 0.0
                for ss in startstops:
                    if not isinstance(ss, Mapping):
                        continue
                    action = ss.get("action")
                    ss_start = float(ss.get("start", 0.0) or 0.0)
                    ss_stop = float(ss.get("stop", 0.0) or 0.0)
                    dur = max(0.0, ss_stop - ss_start)
                    if action == "compute":
                        compute_duration += dur
                    elif action == "transfer":
                        total_transfer += dur
                    elif action == "deserialize":
                        total_deser += dur
            else:
                start = float(task.get("start", 0.0) or 0.0)
                stop = float(task.get("stop", 0.0) or 0.0)
                compute_duration = max(0.0, stop - start)

                for transfer in task.get("transfer", []):
                    total_transfer += float(transfer.get("duration", 0.0) or 0.0)

                total_deser += float(task.get("deserialization_duration", 0.0) or 0.0)

            if "matmul" in key:
                matmul_durations.append(compute_duration)
            elif "add" in key:
                reduction_durations.append(compute_duration)
            else:
                other_compute_s += compute_duration

        mean_matmul = (sum(matmul_durations) / len(matmul_durations)) if matmul_durations else 0.0
        mean_red = (sum(reduction_durations) / len(reduction_durations)) if reduction_durations else 0.0

        return TaskDiagnostics(
            total_tasks=len(raw_task_stream),
            matmul_tasks_count=len(matmul_durations),
            reduction_tasks_count=len(reduction_durations),
            mean_matmul_duration_s=mean_matmul,
            mean_reduction_duration_s=mean_red,
            total_compute_s=sum(matmul_durations) + sum(reduction_durations) + other_compute_s,
            total_transfer_s=total_transfer,
            total_deserialization_s=total_deser,
        )
