"""
src.dask_matrix.telemetry — Dask Diagnostic and Telemetry Extraction

Extracts diagnostic task events, Dask managed object sizes, and task execution
breakdowns from Dask Distributed:
- Strictly distinguishes object nbytes from physical network transfer.
- Distinguishes local IPC/loopback communication from physical multi-node networks.
- Provides hooks to capture task streams and performance reports for separate
  diagnostic characterization without contaminating primary benchmark timings.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional


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
            return int(self.client.nbytes())
        except Exception:
            return None

    def get_worker_memory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns managed memory and process memory metrics reported by workers."""
        stats: Dict[str, Dict[str, Any]] = {}
        try:
            worker_info = self.client.scheduler_info().get("workers", {})
            for worker_addr, info in worker_info.items():
                stats[worker_addr] = {
                    "managed_bytes": info.get("memory", {}).get("managed", 0),
                    "process_bytes": info.get("memory", {}).get("process", 0),
                    "spilled_bytes": info.get("memory", {}).get("spilled", 0),
                    "nthreads": info.get("nthreads", 1),
                }
        except Exception:
            pass
        return stats

    def parse_task_stream(self, raw_task_stream: List[Dict[str, Any]]) -> TaskDiagnostics:
        """
        Parses task stream events returned by get_task_stream into aggregate metrics.

        Args:
            raw_task_stream: List of task event dictionaries from Dask.

        Returns:
            TaskDiagnostics summary.
        """
        matmul_durations = []
        reduction_durations = []
        total_transfer = 0.0
        total_deser = 0.0

        for task in raw_task_stream:
            key = str(task.get("key", ""))
            start = task.get("start", 0.0)
            stop = task.get("stop", 0.0)
            duration = max(0.0, stop - start)

            if "matmul" in key:
                matmul_durations.append(duration)
            elif "add" in key:
                reduction_durations.append(duration)

            # Check transfer/deserialization sub-events if recorded
            for transfer in task.get("transfer", []):
                total_transfer += transfer.get("duration", 0.0)

            total_deser += task.get("deserialization_duration", 0.0)

        mean_matmul = (sum(matmul_durations) / len(matmul_durations)) if matmul_durations else 0.0
        mean_red = (sum(reduction_durations) / len(reduction_durations)) if reduction_durations else 0.0

        return TaskDiagnostics(
            total_tasks=len(raw_task_stream),
            matmul_tasks_count=len(matmul_durations),
            reduction_tasks_count=len(reduction_durations),
            mean_matmul_duration_s=mean_matmul,
            mean_reduction_duration_s=mean_red,
            total_compute_s=sum(matmul_durations) + sum(reduction_durations),
            total_transfer_s=total_transfer,
            total_deserialization_s=total_deser,
        )
