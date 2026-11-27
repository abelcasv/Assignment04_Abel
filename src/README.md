# Assignment 4 Implementation Architecture (`src/`)

**Assignment 4:** Distributed Matrix Multiplication
**Course:** BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC (2026–2027)

---

## 1. Architectural Overview

This package implements the distributed matrix multiplication pipeline specified in [`SPEC.md`](../SPEC.md). It uses **Dask Distributed** with a **2D block decomposition** and **binary-tree reduction**.

```text
src/
├── __init__.py                   # Package exports
├── cli.py                        # Unified execution CLI and parameter checker
├── dask_matrix/                  # Core distributed multiplication package
│   ├── __init__.py
│   ├── partitioning.py           # 2D block decomposition and edge-block policy
│   ├── kernels.py                # Elementary block GEMM and addition tasks
│   ├── reduction.py              # Binary-tree reduction graph builder
│   ├── distributed_engine.py     # End-to-end distributed multiplication orchestrator
│   ├── cluster_manager.py        # LocalCluster lifecycle, topology validation, thread caps
│   ├── pilot_runner.py           # Bounded single-worker pilot orchestrator
│   ├── campaign_runner.py        # Multi-repetition campaign orchestrator
│   ├── watchdog.py               # Process-tree resource monitor and 5-step safe stop
│   ├── input_verifier.py         # SHA-256 dataset verification and fail-closed loader
│   ├── validation.py             # Numerical correctness checking against tolerances
│   ├── telemetry.py              # Dask diagnostic stream and memory telemetry
│   ├── results_logger.py         # Append-only raw_measurements.csv logger
│   └── sidecar.py                # Campaign environment metadata generator
└── adapters/                     # Non-invasive baseline execution adapters
    ├── __init__.py
    ├── a1_adapter.py             # Execution adapter for Assignment 1 sequential binary
    └── a3_adapter.py             # Execution adapter for Assignment 3 OpenMP binary
```

---

## 2. Mathematical Formulation & 2D Block Reduction

For two dense matrices $A, B \in \mathbb{R}^{n \times n}$ in IEEE 754 `binary64` (`float64`), partitioned into $N_b \times N_b$ blocks using block size $b$ ($N_b = \lceil n/b \rceil$):

$$C_{i, j} = \sum_{k=0}^{N_b-1} A_{i, k} \times B_{k, j}$$

1. **Multiplication Stage:**
   For each block coordinate $(i, j, k) \in \{0, \dots, N_b-1\}^3$, Dask schedules an independent task:
   $$P_{i, j, k} = \mathrm{block\_multiply}(A_{i, k}, B_{k, j}) = A_{i, k} \times B_{k, j}$$
   executed via `numpy.matmul` on worker processes with internal BLAS/OpenMP threading restricted to 1 (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
   Total multiplication tasks: $N_b^3$.

2. **Binary-Tree Reduction Stage:**
   For each output block $(i, j)$, the $N_b$ partial products $\{P_{i, j, 0}, \dots, P_{i, j, N_b-1}\}$ are accumulated via a pairwise binary tree:
   - Round 0 pairs $(P_0, P_1), (P_2, P_3), \dots$ and submits `block_add` tasks.
   - Any odd unpaired block carries forward untouched to the subsequent round.
   - Reduction continues until a single accumulated future remains.
   - Total depth: $\lceil \log_2 N_b \rceil$ stages.
   - Additions per output block: exactly $N_b - 1$ tasks.
   - Total reduction tasks: $N_b^2 (N_b - 1)$.
   - Total graph tasks: $N_b^3 + N_b^2 (N_b - 1) = 2 N_b^3 - N_b^2$.

---

## 3. Edge-Block Policy

When matrix dimension $n$ is not divisible by block size $b$ (e.g., during irregular edge validation $n=4, b=3$):
- **Exact Truncated Dimensions:** The final block in each row/column uses exact dimensions:
  $$\mathrm{block\_rows} = \min(b, n - i \cdot b), \quad \mathrm{block\_cols} = \min(b, n - j \cdot b)$$
- **Zero-Padding Forbidden:** Zero-padding is strictly prohibited by [`SPEC.md`](../SPEC.md).
- **Inner Dimension Invariant:** For every multiplication $A_{i, k} \times B_{k, j}$, the column count of $A_{i, k}$ matches the row count of $B_{k, j}$ identically ($\min(b, n - k \cdot b)$).
- **C-Order Memory Contiguity:** Every extracted block and assembled matrix is forced to contiguous C-order layout (`np.ascontiguousarray(..., dtype=np.float64)`).

---

## 4. Primary Timing Boundary Contract

The Dask multiplication timer ($T_{\mathrm{Dask\_E2E}}$) measures wall-clock time via `time.perf_counter()` adhering to [`SPEC.md`](../SPEC.md):

```text
[Excluded from Timer]
1. Cluster creation & worker registration (cluster_startup_s).
2. CSV reading from disk to driver memory (input_load_s).

[START TIMER t0]
3. 2D Block partitioning of A and B.
4. Scattering blocks to the Dask cluster.
5. Submitting Nb^3 multiplication tasks.
6. Submitting Nb^2 tree reduction graphs.
7. Waiting for completion (enforcing 120 s timeout).
8. Gathering output blocks C_ij to driver memory.
9. Assembling dense matrix C.
10. Checksum consumption: checksum = float(np.sum(C)).
[STOP TIMER t1]

T_Dask_E2E = t1 - t0

[Excluded from Timer]
11. Element-by-element numerical correctness validation vs reference.
12. Raw result logging and telemetry serialization.
```

---

## 5. Worker Count & Single-Host Topology Validation

- [`src/dask_matrix/cluster_manager.py`](dask_matrix/cluster_manager.py) validates that requested worker counts $p$ are positive integers ($p \ge 1$), enforcing single-threaded worker processes (`threads_per_worker = 1` and single-thread BLAS environment variables).
- After cluster startup, `validate_cluster_topology` verifies fail-closed that `worker_count_observed` matches `worker_count_requested` identically, preventing silent fallback to a different worker count, and all local runs record `physical_machine_count = 1`.

---

## 6. Resource Watchdog & 5-Step Safe Stop Procedure

[`src/dask_matrix/watchdog.py`](dask_matrix/watchdog.py) enforces operational host safety during runs:
- **Sampling Interval:** 250 ms on an isolated daemon thread.
- **Tracked Processes:** Discovers and monitors unique PIDs across Driver, Scheduler, Nannies, and Workers using `psutil`.
- **Ceilings:**
  1. Aggregate RSS $\ge 2.5$ GiB ($2560.0$ MiB).
  2. Host available RAM $< 1.0$ GiB ($1024.0$ MiB).
- **5-Step Safe Stop:**
  1. Alert and detection log.
  2. Cancel all in-flight Dask client futures.
  3. Orderly close client and cluster (5 s grace period).
  4. Forcefully terminate lingering child worker/nanny PIDs via `proc.terminate() / proc.kill()`.
  5. Log observation as `OUT_OF_MEMORY` or `ABORTED_WATCHDOG`, leaving `time_e2e_s = NULL`.

---

## 7. Raw Results Logger Schema

[`src/dask_matrix/results_logger.py`](dask_matrix/results_logger.py) records every observation to [`results/raw/raw_measurements.csv`](../results/raw/raw_measurements.csv) conforming to the 44-column schema defined in [`SPEC.md`](../SPEC.md):
- Append-only integrity with thread-safe file locking.
- Nullable fields are recorded as empty strings (`""`), never fabricated zeros.
- Status values: `COMPLETED`, `TIMEOUT`, `OUT_OF_MEMORY`, `FAILED`, `ABORTED_WATCHDOG`, `ABORTED_BUDGET`, `NOT_RUN_AFTER_OOM`, `NOT_RUN_AFTER_TIMEOUT`.

---

## 8. CLI Usage

Static configuration verification:
```powershell
python -B -m src.cli --n 1000 --block-size 250 --workers 1 --check-config
python -B -m src.cli --n 1000 --block-size 250 --workers 2 --check-config
```

Offline $4 \times 4$ and $n=250$ distributed-block fixture verification:
```powershell
python -B -m src.cli --validate-fixtures
```
