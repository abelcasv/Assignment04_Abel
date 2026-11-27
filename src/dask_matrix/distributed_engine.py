"""
src.dask_matrix.distributed_engine — Dask Distributed 2D Block Matrix Multiplication

Implements the end-to-end distributed execution contract defined in SPEC.md:

Timing contract (t0 to t1):
- t0: Immediately before partitioning and scattering blocks for the current repetition.
- Inside timer:
  1. Partition A and B into 2D blocks (C-order, truncated edge blocks).
  2. Scatter blocks into the Dask cluster.
  3. Submit Nb^3 block multiplication tasks (numpy.matmul).
  4. Submit Nb^2 binary-tree reductions (Nb - 1 additions per block).
  5. Gather all output blocks to driver memory.
  6. Assemble contiguous dense C matrix.
  7. Consume every element via checksum = float(np.sum(C)).
- t1: Immediately after checksum completes.
- T_Dask_E2E = t1 - t0 measured via time.perf_counter().
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.dask_matrix.partitioning import (
    compute_grid_dimensions,
    partition_matrix_2d,
    assemble_matrix_2d,
)
from src.dask_matrix.kernels import (
    block_multiply,
    compute_matrix_checksum,
)
from src.dask_matrix.reduction import (
    build_binary_tree_reduction,
)
from src.dask_matrix.telemetry import (
    TelemetryCollector,
    extract_object_nbytes,
)


@dataclass
class DaskExecutionResult:
    """Holds the timing observations, outputs, and telemetry from a Dask run."""
    status: str                         # COMPLETED, TIMEOUT, OUT_OF_MEMORY, FAILED
    n: int
    block_size: int
    nb: int
    time_e2e_s: Optional[float]
    checksum: Optional[float]
    c_matrix: Optional[np.ndarray]
    error_reason: Optional[str] = None
    elapsed_until_abort_s: Optional[float] = None
    object_nbytes: Optional[int] = None
    phase_breakdown_s: Optional[Dict[str, float]] = None
    dask_managed_memory_mib: Optional[float] = None
    worker_memory_stats: Optional[Dict[str, Dict[str, Any]]] = None


class DistributedBlockMatmul:
    """Orchestrates 2D block matrix multiplication over Dask Distributed."""

    def __init__(self, client: Any, block_size: int = 250):
        self.client = client
        self.block_size = block_size
        self._call_seq = 0

    def multiply(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        repetition_id: int = 0,
        timeout_s: float = 120.0,
        watchdog: Optional[Any] = None,
        global_deadline_s: Optional[float] = None,
    ) -> DaskExecutionResult:
        """
        Executes distributed 2D block matrix multiplication within the strict timing boundary.

        Args:
            a_matrix: Dense square numpy array A (float64).
            b_matrix: Dense square numpy array B (float64).
            repetition_id: Repetition counter used to guarantee unique task keys.
            timeout_s: Maximum execution duration before timing out (default: 120 s).
            watchdog: Optional ResourceWatchdog instance for immediate breach detection.
            global_deadline_s: Optional monotonic timestamp for session budget limit.

        Returns:
            DaskExecutionResult containing timing, checksum, and status.
        """
        try:
            from distributed import wait, TimeoutError as DaskTimeoutError
        except ImportError as e:
            raise ImportError("dask.distributed is required for distributed execution.") from e

        if not isinstance(a_matrix, np.ndarray) or not isinstance(b_matrix, np.ndarray):
            raise TypeError(
                f"Input matrices must be numpy.ndarray instances, got {type(a_matrix).__name__} and {type(b_matrix).__name__}."
            )
        if a_matrix.ndim != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise ValueError(
                f"Input matrix A must be a square 2D array, got shape {a_matrix.shape}."
            )
        if b_matrix.ndim != 2 or b_matrix.shape[0] != b_matrix.shape[1]:
            raise ValueError(
                f"Input matrix B must be a square 2D array, got shape {b_matrix.shape}."
            )
        if a_matrix.shape != b_matrix.shape:
            raise ValueError(
                f"Input matrix shape mismatch: A has shape {a_matrix.shape}, B has shape {b_matrix.shape}."
            )
        if a_matrix.shape[0] <= 0:
            raise ValueError(
                f"Input matrices must be non-empty, got shape {a_matrix.shape}."
            )

        n = a_matrix.shape[0]
        nb = compute_grid_dimensions(n, self.block_size)
        self._call_seq += 1
        key_namespace = f"n{n}-c{self._call_seq}"

        # -------------------------------------------------------------
        # PRIMARY TIMING BOUNDARY START (t0)
        # -------------------------------------------------------------
        t0 = time.perf_counter()

        try:
            # Check early watchdog or deadline breach
            if watchdog is not None and watchdog.breach_detected:
                elapsed_abort = time.perf_counter() - t0
                return DaskExecutionResult(
                    status=watchdog.breach_type or "FAILED",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=watchdog.breach_reason,
                    elapsed_until_abort_s=elapsed_abort,
                )
            if global_deadline_s is not None and time.perf_counter() >= global_deadline_s:
                elapsed_abort = time.perf_counter() - t0
                return DaskExecutionResult(
                    status="ABORTED_BUDGET",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=f"Global session deadline reached ({time.perf_counter():.2f} >= {global_deadline_s:.2f}).",
                    elapsed_until_abort_s=elapsed_abort,
                )

            # 1. 2D Block Partitioning
            t_part_start = time.perf_counter()
            a_blocks = partition_matrix_2d(a_matrix, self.block_size)
            b_blocks = partition_matrix_2d(b_matrix, self.block_size)
            t_part_end = time.perf_counter()

            # 2. Scatter blocks to cluster (with unique keys per repetition)
            t_scatter_start = time.perf_counter()
            a_scattered: List[List[Any]] = []
            for i in range(nb):
                row_scat = []
                for k in range(nb):
                    fut = self.client.scatter(
                        a_blocks[i][k],
                        broadcast=False,
                        hash=False,
                    )
                    row_scat.append(fut)
                a_scattered.append(row_scat)

            b_scattered: List[List[Any]] = []
            for k in range(nb):
                col_scat = []
                for j in range(nb):
                    fut = self.client.scatter(
                        b_blocks[k][j],
                        broadcast=False,
                        hash=False,
                    )
                    col_scat.append(fut)
                b_scattered.append(col_scat)
            t_scatter_end = time.perf_counter()

            # 3. Schedule Nb^3 Multiplication Tasks
            t_graph_start = time.perf_counter()
            # partial_products[i][j] will hold list of Nb futures for k=0..Nb-1
            partial_products: List[List[List[Any]]] = [
                [[] for _ in range(nb)] for _ in range(nb)
            ]

            for i in range(nb):
                for j in range(nb):
                    for k in range(nb):
                        mult_key = f"matmul-{key_namespace}-r{repetition_id}-({i},{j},{k})"
                        fut = self.client.submit(
                            block_multiply,
                            a_scattered[i][k],
                            b_scattered[k][j],
                            key=mult_key,
                            pure=False,
                        )
                        partial_products[i][j].append(fut)

            # 4. Schedule Binary-Tree Reductions across k for each (i, j)
            c_block_futures: List[List[Any]] = [[None for _ in range(nb)] for _ in range(nb)]
            all_final_futures = []

            for i in range(nb):
                for j in range(nb):
                    red_future = build_binary_tree_reduction(
                        client=self.client,
                        partial_product_futures=partial_products[i][j],
                        block_row=i,
                        block_col=j,
                        repetition_id=repetition_id,
                        key_namespace=key_namespace,
                    )
                    c_block_futures[i][j] = red_future
                    all_final_futures.append(red_future)
            t_graph_end = time.perf_counter()

            # 5. Wait for computation with timeout and budget limit
            elapsed_so_far = time.perf_counter() - t0
            sample_rem = timeout_s - elapsed_so_far
            if global_deadline_s is not None:
                session_rem = global_deadline_s - time.perf_counter()
                remaining_timeout = max(0.1, min(sample_rem, session_rem))
            else:
                remaining_timeout = max(0.1, sample_rem)

            t_wait_start = time.perf_counter()
            done, not_done = wait(all_final_futures, timeout=remaining_timeout)
            t_wait_end = time.perf_counter()

            if not_done:
                # Timeout or budget limit occurred!
                self.client.cancel(not_done)
                elapsed_abort = time.perf_counter() - t0
                if watchdog is not None and watchdog.breach_detected:
                    status = watchdog.breach_type or "FAILED"
                    reason = watchdog.breach_reason or "Watchdog breach during computation."
                elif global_deadline_s is not None and time.perf_counter() >= global_deadline_s:
                    status = "ABORTED_BUDGET"
                    reason = f"Global session deadline reached ({time.perf_counter():.2f} >= {global_deadline_s:.2f})."
                else:
                    status = "TIMEOUT"
                    reason = f"Operation timed out after {elapsed_abort:.2f} s (> {timeout_s:.1f} s limit)."
                return DaskExecutionResult(
                    status=status,
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=reason,
                    elapsed_until_abort_s=elapsed_abort,
                )

            # Check watchdog again after wait
            if watchdog is not None and watchdog.breach_detected:
                elapsed_abort = time.perf_counter() - t0
                return DaskExecutionResult(
                    status=watchdog.breach_type or "FAILED",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=watchdog.breach_reason,
                    elapsed_until_abort_s=elapsed_abort,
                )

            # 6. Gather all C_ij output blocks to driver memory
            t_gather_start = time.perf_counter()
            gathered_blocks: List[List[np.ndarray]] = []
            for i in range(nb):
                row_gathered = []
                for j in range(nb):
                    block_data = self.client.gather(c_block_futures[i][j])
                    row_gathered.append(block_data)
                gathered_blocks.append(row_gathered)
            t_gather_end = time.perf_counter()

            # 7. Assemble full dense matrix C
            t_assemble_start = time.perf_counter()
            c_matrix = assemble_matrix_2d(gathered_blocks, n, self.block_size)
            t_assemble_end = time.perf_counter()

            # 8. Force full materialization via Checksum Consumption
            t_check_start = time.perf_counter()
            checksum = compute_matrix_checksum(c_matrix)
            t_check_end = time.perf_counter()

            # -------------------------------------------------------------
            # PRIMARY TIMING BOUNDARY STOP (t1)
            # -------------------------------------------------------------
            t1 = time.perf_counter()
            time_e2e_s = t1 - t0

            # Post-checksum watchdog and timeout checks
            if watchdog is not None and watchdog.breach_detected:
                return DaskExecutionResult(
                    status=watchdog.breach_type or "FAILED",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=watchdog.breach_reason,
                    elapsed_until_abort_s=time_e2e_s,
                )

            if global_deadline_s is not None and t1 >= global_deadline_s:
                return DaskExecutionResult(
                    status="ABORTED_BUDGET",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=f"Global session deadline reached ({t1:.2f} >= {global_deadline_s:.2f}).",
                    elapsed_until_abort_s=time_e2e_s,
                )

            if time_e2e_s >= timeout_s:
                return DaskExecutionResult(
                    status="TIMEOUT",
                    n=n,
                    block_size=self.block_size,
                    nb=nb,
                    time_e2e_s=None,
                    checksum=None,
                    c_matrix=None,
                    error_reason=f"Operation exceeded timeout after {time_e2e_s:.2f} s (> {timeout_s:.1f} s limit).",
                    elapsed_until_abort_s=time_e2e_s,
                )

            # Estimate object nbytes and worker managed memory strictly AFTER t1
            object_nbytes = None
            try:
                object_nbytes = extract_object_nbytes(self.client.nbytes())
            except Exception:
                pass

            worker_memory_stats: Optional[Dict[str, Dict[str, Any]]] = None
            dask_managed_memory_mib: Optional[float] = None
            try:
                raw_mem_stats = TelemetryCollector(self.client).get_worker_memory_stats()
                if raw_mem_stats:
                    worker_memory_stats = raw_mem_stats
                    dask_managed_memory_mib = sum(
                        w["managed_bytes"] for w in raw_mem_stats.values()
                    ) / (1024.0 * 1024.0)
            except Exception:
                pass

            # Optional driver-side wall-clock phase intervals (worker compute/transfer overlap internally)
            breakdown = {
                "partitioning_s": t_part_end - t_part_start,
                "scatter_s": t_scatter_end - t_scatter_start,
                "graph_submission_s": t_graph_end - t_graph_start,
                "wait_s": t_wait_end - t_wait_start,
                "gather_s": t_gather_end - t_gather_start,
                "assembly_s": t_assemble_end - t_assemble_start,
                "checksum_s": t_check_end - t_check_start,
            }

            return DaskExecutionResult(
                status="COMPLETED",
                n=n,
                block_size=self.block_size,
                nb=nb,
                time_e2e_s=time_e2e_s,
                checksum=checksum,
                c_matrix=c_matrix,
                object_nbytes=object_nbytes,
                phase_breakdown_s=breakdown,
                dask_managed_memory_mib=dask_managed_memory_mib,
                worker_memory_stats=worker_memory_stats,
            )

        except DaskTimeoutError:
            elapsed_abort = time.perf_counter() - t0
            if watchdog is not None and watchdog.breach_detected:
                status = watchdog.breach_type or "FAILED"
                reason = watchdog.breach_reason or "Watchdog breach during computation."
            elif global_deadline_s is not None and time.perf_counter() >= global_deadline_s:
                status = "ABORTED_BUDGET"
                reason = f"Global session deadline reached ({time.perf_counter():.2f} >= {global_deadline_s:.2f})."
            else:
                status = "TIMEOUT"
                reason = f"Operation timed out after {elapsed_abort:.2f} s."
            return DaskExecutionResult(
                status=status,
                n=n,
                block_size=self.block_size,
                nb=nb,
                time_e2e_s=None,
                checksum=None,
                c_matrix=None,
                error_reason=reason,
                elapsed_until_abort_s=elapsed_abort,
            )

        except Exception as e:
            elapsed_abort = time.perf_counter() - t0
            status = "FAILED"
            reason = str(e)
            if watchdog is not None and watchdog.breach_detected:
                status = watchdog.breach_type or "FAILED"
                reason = watchdog.breach_reason or str(e)
            elif global_deadline_s is not None and time.perf_counter() >= global_deadline_s:
                status = "ABORTED_BUDGET"
                reason = f"Global session deadline reached ({time.perf_counter():.2f} >= {global_deadline_s:.2f})."
            return DaskExecutionResult(
                status=status,
                n=n,
                block_size=self.block_size,
                nb=nb,
                time_e2e_s=None,
                checksum=None,
                c_matrix=None,
                error_reason=reason,
                elapsed_until_abort_s=elapsed_abort,
            )
