# Assignment 4: Distributed Matrix Multiplication

**Course:** BIG DATA · Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)  
**Academic Year:** 2026–2027  
**Author:** Abel Castro Villar  
**Repository:** [Assignment04_Abel](https://github.com/abelcasv/Assignment04_Abel)  
**Status:** Certified Final Reproducibility Package  

---

## 1. Executive Summary & Research Scope

This repository presents a controlled, empirical investigation into the scalability, memory footprint, and scheduling overhead of dense square matrix multiplication ($C = A \times B$) executed via **Dask Distributed** on an asymmetric hybrid x86-64 processor. The implementation employs a **2D block decomposition** with a **pairwise binary-tree reduction**, operating on IEEE 754 `binary64` (`float64`) matrices.

In rigorous accordance with the official course specification (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`), the technical requirements in `SPEC.md`, and the approved experimental design in `docs/experimental_protocol.md`, this study addresses:
1. **End-to-End Distributed Latency:** Measuring complete workflow execution time under an explicit, pre-registered timing boundary ($T_{\text{Dask\_E2E}}$).
2. **Local Multi-Worker Strong Scaling & Framework Overhead:** Quantifying speedup $S_{\text{dist}}(p, n)$, parallel efficiency $E_{\text{dist}}(p, n)$, per-process RSS memory scaling, task distribution, loopback inter-process communication (IPC), and scheduler contention across $p \in \{1, 2, 4, 8, 10\}$ workers on a single physical host following instructor authorization (`docs/decisions/DEC-01-instructor-permission.md`).
3. **Contextual Cross-System Comparisons:** Evaluating Dask ($p=1$ and multi-worker $p$) against frozen, certified historical baseline summaries from Assignment 1 (C, Java, Python) and Assignment 3 (OpenMP, C Sequential $i\text{-}k\text{-}j$, and AVX2-tiled) tracked in `data/baselines/` across matching matrix dimensions and numeric types, alongside deterministic common-input datasets and runtime adapters (`scripts/prepare_a1_runtime.py`, `scripts/prepare_a3_runtime.py`) for optional local re-execution.
4. **Administrative & Methodological Transparency:** Recording the complete DEC-01 traceability (blocked during Phases 1–10, authorized on 2026-11-23 for local single-host multi-worker scaling while multi-node cluster execution remains **BLOCKED / NOT AVAILABLE**), distinguishing single-host process emulation (`1 + p` OS PIDs over loopback `127.0.0.1`) from genuine multi-machine clustering, documenting telemetry contracts and historical vs. remediated fields, and reporting the structural validation deficiency identified in the Assignment 1 benchmark harness.

---

## 2. Research Questions and Hypotheses

### 2.1 Central Research Question
> How does the end-to-end wall-clock performance, process memory footprint, and scheduling/communication overhead of a distributed 2D block matrix multiplication implementation using Dask Distributed scale as matrix dimensions increase and across worker topologies, and how does it compare contextually with sequential (Assignment 1) and shared-memory parallel OpenMP (Assignment 3) implementations across matching dimensions and numeric types?

*Experimental Scope Note (DEC-01):* During Phases 1–10, Gate G1 / DEC-01 restricted execution to $p=1$ pending instructor permission, producing the frozen 48-row single-worker dataset (`campaign_20261114_205134_...`, preserved byte-for-byte in lines 1–49 of `results/raw/raw_measurements.csv`). On 2026-11-23, following confirmed instructor authorization (`docs/decisions/DEC-01-instructor-permission.md`), a controlled $p=1$ vs. $p=2$ pilot, a 90-row local multi-worker strong-scaling campaign ($p \in \{1, 2, 4, 8, 10\}$, $n \in \{1000, 1500, 2000\}$, $b=250$, `physical_machine_count=1`), and a separate $n=1000$ diagnostic characterization pass were executed on the same physical host. No accredited physical multi-node cluster was available, so **multi-node execution remains BLOCKED / NOT AVAILABLE**, and all multi-worker measurements are strictly labeled as **local single-host multi-process execution**.

### 2.2 Formal Hypotheses
* **$H_1$ (Small-Dimension Framework Overhead, $n \le 500$):** For small matrices ($n \le 500$), Dask graph generation, task scheduling, socket serialization, and result gathering introduce substantial coordination latency, hypothesized to make Dask slower than compiled native sequential baselines ($\text{CTR} = \tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask}} < 1$). Empirically **partially supported (baseline-dependent)**: supported against all optimized A3 C kernels ($0.17\times\text{–}0.18\times$ at $n=250$; $0.22\times\text{–}0.31\times$ at $n=500$) and `A1_Seq_Java` at $n=250$ ($0.82\times$), but refuted in net duration against unoptimized stride-$n$ `A1_Seq_C_ijk` ($2.71\times, 3.75\times$), Java at $n=500$ ($1.67\times$), and Python ($329.83\times, 484.75\times$).
* **$H_2$ (Larger-Dimension Scaling, Baseline Crossovers & Multi-Worker Granularity, $n \ge 1000$):** Evaluates both whether cubic arithmetic growth ($2n^3$) amortizes fixed framework costs at $p=1$ and whether larger task graphs enable positive multi-worker speedup. Empirically **partially supported**: at $p=1$, crossover occurs vs. `A1_Seq_Java` between $n=250$ ($0.82\times$) and $n=500$ ($1.67\times$) and vs. untiled `A3_OpenMP_ikj_p1` / `A3_Seq_ikj` at $n=1000$ ($1.06\times$ / $1.32\times$, reaching $2.18\times$ / $2.04\times$ at $n=2000$), with **no crossover through $n=2000$** against AVX2-tiled A3 ($0.63\times$ / $0.57\times$); in the local multi-worker strong-scaling campaign, positive median speedup emerges **only** at $n=2000, p=2$ ($S_{\text{dist}}=1.22\times$, $E_{\text{dist}}=60.8\%$, with overlapping per-repetition ranges `[8.49, 12.75] s` at $p=1$ vs. `[9.01, 11.68] s` at $p=2$), while $n \in \{1000, 1500\}$ remain overhead-bound even at $p=2$ ($S=0.66\times$ and $0.76\times$).
* **$H_3$ (Worker Saturation & Scaling Limits on Asymmetric Single Host):** Increasing worker process counts beyond the 2 physical performance cores induces severe coordination contention and non-monotonic scaling limits. Empirically **supported**: for $p \ge 4$, end-to-end runtime degrades sharply across all tested sizes ($S(4) = 0.21\times\text{–}0.77\times$, $S(8) = 0.12\times\text{–}0.34\times$, $\text{IQR}$ up to $24.06\text{ s}$), culminating in a $120\text{ s}$ per-sample timeout at $p=10$ ($131.92\text{ s}$ at $n=1500$). Separate $n=1000$ diagnostics rule out RAM exhaustion and disk spilling (max RSS $1{,}468.1\text{ MiB} \ll 2{,}560\text{ MiB}$ cap, 0 disk spills) as well as BLAS compute ($\le 0.335\text{ s}$ aggregate), showing that driver `scatter_s` + `wait_s` + `gather_s` account for `96.11%` of wall-clock latency at $p=8$ (`98.70%` including `graph_submission_s`), consistent with architectural hypotheses of centralized Tornado scheduler contention, Windows loopback TCP IPC overhead, and hybrid P/E-core scheduling (not separately isolated by kernel/event-loop profiling).
* **$H_4$ (System-Level Contextual Boundaries):** Differences between Dask/NumPy/BLAS and C/OpenMP reflect differing runtime engines, memory layouts, and timing boundaries, representing a cross-system contextual comparison rather than an isolated algorithmic speedup. Empirically **supported** and reinforced by comparing Dask's best local multi-worker median ($9.5635\text{ s}$ at $n=2000, p=2$) against zero-copy shared-memory OpenMP $t=10$ (`omp_tiled_b32`: $0.6250\text{ s}$; `omp_ikj`: $1.5443\text{ s}$).

---

## 3. System Architecture & Algorithms

The distributed multiplication engine is implemented in Python under `src/dask_matrix/` using Dask Distributed:

```
src/
├── __init__.py                   # Package exports
├── cli.py                        # Unified CLI and configuration validator
├── dask_matrix/                  # Core distributed multiplication package
│   ├── __init__.py
│   ├── partitioning.py           # 2D block decomposition & edge-block truncation
│   ├── kernels.py                # Elementary block GEMM and addition tasks
│   ├── reduction.py              # Binary-tree reduction DAG builder
│   ├── distributed_engine.py     # End-to-end distributed multiplication orchestrator
│   ├── cluster_manager.py        # LocalCluster lifecycle, Gate G1 validation, thread caps
│   ├── pilot_runner.py           # Phase 7 bounded pilot campaign orchestrator
│   ├── campaign_runner.py        # Phase 8 full 36-sample campaign orchestrator
│   ├── watchdog.py               # Process-tree resource monitor & 5-step safe stop
│   ├── input_verifier.py         # SHA-256 dataset verification and loader
│   ├── validation.py             # Numerical correctness checking against tolerances
│   ├── telemetry.py              # Dask diagnostic stream & memory telemetry
│   ├── results_logger.py         # Append-only raw_measurements.csv logger
│   └── sidecar.py                # Environment metadata sidecar generator
└── adapters/                     # Non-invasive baseline execution adapters
    ├── __init__.py
    ├── a1_adapter.py             # Execution adapter for Assignment 1 sequential binary
    └── a3_adapter.py             # Execution adapter for Assignment 3 OpenMP binary
```

### 3.1 2D Block Partitioning
For dense matrices $A, B \in \mathbb{R}^{n \times n}$ in IEEE 754 `binary64` (`float64`), the matrices are partitioned into an $N_b \times N_b$ grid of sub-blocks using a fixed block dimension $b = 250$ ($N_b = \lceil n / b \rceil$):
$$C_{i, j} = \sum_{k=0}^{N_b - 1} A_{i, k} \times B_{k, j}$$

* **Edge-Block Truncation Policy:** When $n$ is not an exact multiple of $b$, edge blocks adopt exact truncated dimensions:
  $$\text{block\_rows} = \min(b, n - i \cdot b), \quad \text{block\_cols} = \min(b, n - j \cdot b)$$
  Zero-padding is strictly prohibited by `SPEC.md` §5.1 to preserve exact computational complexity.
* **Memory Contiguity:** All extracted slices and assembled arrays are forced to contiguous C-order layout (`np.ascontiguousarray(..., dtype=np.float64)`).

### 3.2 Task Graph Construction & Binary-Tree Reduction
1. **Multiplication Stage:** For each block triple $(i, j, k) \in \{0, \dots, N_b - 1\}^3$, Dask schedules an independent task:
   $$P_{i, j, k} = A_{i, k} \times B_{k, j}$$
   executed via `numpy.matmul` on worker processes with internal BLAS multi-threading strictly pinned to 1. Total multiplication tasks: $N_b^3$.
2. **Pairwise Binary-Tree Reduction:** For each output block $(i, j)$, the $N_b$ partial products are accumulated hierarchically:
   - Round 0 pairs adjacent futures $(P_{i, j, 0}, P_{i, j, 1}), (P_{i, j, 2}, P_{i, j, 3}), \dots$ submitting `block_add` tasks.
   - Any odd un-paired block automatically carries forward untouched to the subsequent stage.
   - Reduction continues until a single accumulated block future remains.
   - Tree depth: $\lceil \log_2 N_b \rceil$ stages.
   - Reduction additions per block: exactly $N_b - 1$ tasks.
   - Total DAG tasks: $N_b^3 + N_b^2 (N_b - 1) = 2 N_b^3 - N_b^2$.

### 3.3 Output Array Consumption & Checksum Barrier
Within the timed execution boundary, the assembled dense output matrix $C$ is consumed by a full-matrix checksum calculation:
$$\text{checksum} = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} C_{i, j} = \text{float}(\text{np.sum}(C, \text{dtype}=\text{np.float64}))$$

This aggregation ensures that all elements of the assembled array are accessed within the timed boundary ($T_{\text{Dask\_E2E}}$). However, checksum consumption aggregates array values and confirms checksum determinism across repetitions (zero observed checksum divergence); it does not prove that the complete output matrices are identical (as distinct matrices may yield identical aggregate sums) and does not constitute independent numerical correctness validation of individual elements. Independent element-by-element numerical validation is performed exclusively at $n=250$ against an independent calculated numerical reference matrix outside the timed boundary.

---

## 4. Experimental Environment & Runtime Topology

All experiments were conducted on a dedicated host without external background load.

### 4.1 Hardware Specifications
* **CPU:** Intel Core i5-1335U (Raptor Lake, Intel64 Family 6 Model 186 Stepping 3).
  - Physical Cores: 10 (2 Performance cores, 8 Efficient cores).
  - Logical Processors: 12 (Hyper-Threading on P-cores: $2 \times 2 + 8 = 12$).
* **Physical RAM:** 16.0 GB total (16,009.14 MiB / 15.63 GiB; 3,688.8 MiB / ~3.60 GiB available at idle baseline).
* **Operating System:** Microsoft Windows 11 Home 64-bit (Version 10.0.26300, AMD64).

### 4.2 Software Stack
* **Python Runtime:** 3.14.0 (MSC v.1944 64-bit AMD64).
* **Distributed Framework:** Dask 2026.8.0, Distributed 2026.8.0.
* **Numerical Libraries:** NumPy 2.5.4, SciPy OpenBLAS 0.3.34.106.0 (`libscipy_openblas64`).
* **Process & Plotting Tooling:** psutil 7.2.2, Matplotlib 3.11.2, threadpoolctl 3.7.0.
* **C Compilers:** GCC 13.2.0 (MinGW-W64 / MSYS2, C11 standard).

### 4.3 Concurrency & Process Mapping
* **Threading Isolation:** Strictly 1 Dask worker thread (`threads_per_worker = 1`). OpenBLAS internal multi-threading was pinned to 1 thread via environment variables (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) verified dynamically via `threadpoolctl`.
* **Process Hierarchy:** On Windows `LocalCluster(processes=True, n_workers=p, threads_per_worker=1)`:
  - **Driver Process (1 PID):** Hosts `Client`, `LocalCluster` orchestrator, in-process `Scheduler`, and in-process `Nanny` controller (`nanny_pids = []` after PID deduplication; in the frozen Nov 14 Phase 8 sidecar `nanny_pids: [21036]` duplicated the worker PID).
  - **Worker Processes ($p$ distinct child OS PIDs):** Dedicated OS processes executing Python/OpenBLAS block tasks.
  - **Observed OS Process Mapping:** Exactly **`1 + p` unique OS PIDs** verified empirically across all strong-scaling configurations: `2` ($p=1$), `3` ($p=2$), `5` ($p=4$), `9` ($p=8$), and `11` ($p=10$) on `physical_machine_count = 1`.

---

## 5. Administrative Status & Strong Scaling Compliance

### 5.1 Institutional Permission Gate (DEC-01 / A4-11)
Official course requirement A4-11 (PDF page 7) states:
> *"A local execution with multiple workers may be used only if the instructor permits it. It can investigate framework behaviour and process overhead, but does not demonstrate inter-machine network performance or multi-machine memory capacity. Label the environment clearly."*

In accordance with `SPEC.md`, `docs/decisions/DEC-01-instructor-permission.md`, and `docs/experimental_protocol.md`:
* **Historical Phase 1–10 Baseline ($p = 1$):** During Phases 1–10, local multi-worker execution ($p > 1$) remained **BLOCKED** pending written instructor approval; the resulting 48-row single-worker dataset (`campaign_20261114_205134_...`) is preserved intact in lines 1–49 of `results/raw/raw_measurements.csv`.
* **Authorized Local Multi-Worker Execution ($p \in \{1, 2, 4, 8, 10\}$, 2026-11-23):** Following confirmed instructor permission documented in `docs/decisions/DEC-01-instructor-permission.md`, Gate G1 natively unlocked local multi-process strong scaling on **1 physical machine** (`physical_machine_count = 1`) under the unchanged 2.5 GiB RSS ceiling, 1.0 GiB host RAM floor, and 120.0 s per-sample timeout.
* **Strong Scaling Status (Single-Host Local):** **COMPLETED (Local Single-Host Multi-Process)** for $p \in \{1, 2, 4, 8\}$ across all three strong-scaling sizes ($n \in \{1000, 1500, 2000\}$, $b = 250$), with $p = 10$ empirically hitting the 120.0 s timeout limit on `n = 1500` repetition 5 (`131.92 s`; 4/5 completed, 1 `TIMEOUT`, 12 `NOT_RUN_AFTER_TIMEOUT`).
* **Physical Multi-Node Infrastructure:** No external university HPC or multi-machine cluster was accredited; genuine **multi-node execution remains BLOCKED / NOT AVAILABLE**.
* **Network Cost Isolation & Weak Scaling (DEC-07):** Inter-machine physical network costs were not isolated because all multi-worker IPC traversed Windows loopback (`127.0.0.1`). Optional dense cubic weak scaling ($W(n) = 2n^3 \implies n(p) = n(1) \cdot p^{1/3}$) remains **deferred** (DEC-07) because single-host strong scaling already saturates at $p=2$ and times out at $p=10$.

---

## 6. Timing Contracts & Operational Safeguards

### 6.1 Primary Timing Boundary ($T_{\text{Dask\_E2E}}$)
Adhering to `docs/experimental_protocol.md` §6.2, the primary measurement boundary is monotonic end-to-end wall-clock time ($t_0 \to t_1$):
```
[EXCLUDED FROM TIMER]
1. ClusterManager.start() — LocalCluster & Client creation (cluster_startup_s = 4.816 s in Phase 8; 3.11 s to 20.80 s across strong-scaling p=1..10).
2. Input matrix CSV loading from disk (input_load_s: 0.161 s at N=250 to 6.478 s at N=2000).

[START TIMER t0]
3. 2D Block partitioning of matrices A and B.
4. Scattering blocks to Dask worker memory.
5. Submitting Nb^3 multiplication tasks.
6. Submitting Nb^2 pairwise binary-tree reduction DAG.
7. Waiting for graph completion (client.gather).
8. Assembling dense n x n result matrix C.
9. Full-matrix checksum consumption: float(np.sum(C)).
[STOP TIMER t1]

T_Dask_E2E = t1 - t0

[EXCLUDED FROM TIMER]
10. Correctness validation against tolerances.
11. Results logging to CSV and metadata sidecar generation.
```

### 6.2 Operational Resource Safeguards
* **ResourceWatchdog:** Background daemon polling active cluster PIDs every 250 ms.
  - Aggregate Process RSS Ceiling: 2.5 GiB ($2560.0\text{ MiB}$).
  - Host Available RAM Floor: 1.0 GiB ($1024.0\text{ MiB}$).
  - Safe 5-Step Stop: Immediately cancels in-flight client tasks, closes cluster within 5 s, forcefully terminates lingering child PIDs (`terminate() / kill()`), and records `OUT_OF_MEMORY` (RSS ceiling breach), `ABORTED_WATCHDOG` (host RAM floor breach), or `ABORTED_BUDGET` (session budget exhaustion).
* **Execution Budgets:** 120.0 s per-sample timeout; 10.0-hour total campaign envelope; 60.0 s cluster startup timeout.

---

## 7. Baseline Provenance & Compiler Optimization Finding

### 7.1 Certified Reused Baselines
Snapshots from earlier assignments are frozen and immutable in `baselines/`:
* **Assignment 1 Sequential:** Commit `90681e85949b6d65bdbcfef702b6306b0003f971`.
  - Canonical $i\text{-}j\text{-}k$ triple nested loop in C11 (`benchmark.c`).
  - Compiled under `-Wall -Wextra -O3 -std=c11`.
* **Assignment 3 OpenMP:** Commit `23cffbc82c155b6c3de24b3c149b509b63fc870b`.
  - Loop-interchanged $i\text{-}k\text{-}j$ parallel kernel (`parallel_benchmark.c`).
  - Compiled under `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native`.

### 7.2 A1 Benchmark Compiler-Optimization Finding
During Phase 5 assembly inspection (`build/bin/a1_benchmark.s`), a critical methodological characteristic was identified:
* In `baselines/a1_sequential/benchmark.c`, the benchmark harness consumed only `c[0]` after multiplication (`result_sink = c[0]`).
* Disassembly analysis of the compiled binary (`build/bin/a1_benchmark.exe` / `build/bin/a1_benchmark.s`) confirmed that the complete $O(n^3)$ loop body survived dead-code elimination under GCC 13.2.0 (`-O3`), with inner-loop vectorization using packed SSE/AVX instructions (`movupd`, `mulpd`, `unpckhpd`). The binary did not demonstrate elimination of the multiplication loop.
* **Methodological Vulnerability:** However, the benchmark harness consumed only $c[0]$ via a volatile sink (`result_sink = c[0]`). Consequently, the remaining $n^2 - 1$ output elements were never validated for numerical correctness during benchmarking. In contrast, Assignment 4 evaluates a full-matrix checksum across the assembled array within the timed boundary, ensuring that all elements are accessed in memory. For numerical validation, the independent element-wise validation at $n=250$ used a tolerance-based acceptance criterion ($\text{atol}=10^{-9}, \text{rtol}=10^{-12}$) against an independent calculated numerical reference (`data/validation/reference/C_250.csv`, generated via `scripts/generate_dataset.py`'s `generate_reference_product()` using an independent standard-library Python scalar loop without BLAS or Dask). Across all 62,500 elements, OpenBLAS `np.matmul` ($b=250$) achieves $\max|\Delta| = 1.137 \times 10^{-13}$ ($\max_{\text{rel}} = 1.896 \times 10^{-15}$, matching floating-point AVX2/FMA vs. scalar summation rounding; `verify_correctness.py` with $b=100$ achieves $1.208 \times 10^{-13}$, $\max_{\text{rel}} = 2.015 \times 10^{-15}$), four orders of magnitude below the acceptance tolerances, while exact $0.0$ difference holds for the $4 \times 4$ integer fixtures, the $n=250$ aggregate checksum difference, and the $C_{500}$ ($p=2$ vs. $p=1$) cross-validation. For $n \ge 500$, campaign evidence confirms checksum determinism across repetitions (zero observed checksum divergence), while noting that identical aggregate checksums do not prove complete matrix identity.
* **Consequence for Interpreting A1 Timing:** While the measured A1 runtime reflects execution of the complete $O(n^3)$ loop in the inspected binary, the harness suffered from a structural validation deficiency because it failed to certify that the computed result matrix was numerically correct beyond $c[0]$. Direct comparisons between A1/A3 and A4 represent **contextual duration ratios** ($\text{CTR} = \tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask\_E2E}}$), not isolated algorithmic speedups, due to differing timing boundaries (`TB_KERNEL_ONLY` vs `TB_DASK_E2E`), runtime backends, and memory layouts.

---

## 8. Verified Empirical Results

All results are derived deterministically from the 138 raw records in `results/raw/raw_measurements.csv` (full SHA-256 `831A731D9DF6073F3D19AD4A7D4E4F8B54E04D39CFE463D298857F26F4ED426C`): the original 48 Phase 7–8 single-worker rows are preserved byte-for-byte in lines 1–49 (prefix SHA-256 `3D9D721343D379C26F61DD6B4669979F6E7E5262E0D8F6E96CD4D26B1F0FE92C`), followed by 90 non-destructively appended local multi-worker strong-scaling rows (indexed in `results/metadata/strong_scaling_manifest.json`).

### 8.1 Single-Worker Dask Execution Time Scaling ($p = 1, b = 250$)
Each configuration evaluates 1 unmeasured warm-up followed by 5 measured repetitions, summarized using Median and linear Interquartile Range (IQR):

| Dimension $n$ | Blocks $N_b$ | Tasks (Mult + Add = Total) | Mean $T_{\text{E2E}}$ (s) | Median $T_{\text{E2E}}$ (s) | Std Dev (s) | IQR (s) | Min (s) | Max (s) | Peak RSS (MiB) | Checksum Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250** | 1 | $1 + 0 = 1$ | 0.0349 | **0.0336** | 0.0057 | 0.0032 | 0.0300 | 0.0446 | 796.7* | Validated vs Reference ($n=250$) |
| **500** | 2 | $8 + 4 = 12$ | 0.2024 | **0.2062** | 0.0270 | 0.0160 | 0.1621 | 0.2362 | 796.7* | Checksum Deterministic ($\Delta = 0.0$) |
| **750** | 3 | $27 + 18 = 45$ | 0.4480 | **0.4449** | 0.0521 | 0.0733 | 0.3798 | 0.5063 | 327.3 | Checksum Deterministic ($\Delta = 0.0$) |
| **1000** | 4 | $64 + 48 = 112$ | 0.9804 | **0.9318** | 0.1140 | 0.1228 | 0.8695 | 1.1528 | 525.6* | Checksum Deterministic ($\Delta = 0.0$) |
| **1500** | 6 | $216 + 180 = 396$ | 3.2977 | **3.0392** | 0.5197 | 0.7948 | 2.7801 | 3.9470 | 525.6 | Checksum Deterministic ($\Delta = 0.0$) |
| **2000** | 8 | $512 + 448 = 960$ | 7.3006 | **7.4963** | 0.6079 | 0.0952 | 6.2562 | 7.8475 | 796.7 | Checksum Deterministic ($\Delta = 0.0$) |

*\*All 30 measured campaign repetitions completed successfully across all attempted dimensions (`largest_attempted_n = 2000`, `largest_completed_n = 2000`; $n > 2000$ not attempted) with 0 timeouts and 0 memory breaches. Note on memory & telemetry completeness: (1) Peak RSS reports the maximum sampled `rss_process_sum_peak_sampled_mib` across repetitions per dimension (on Windows `psutil` `.rss` measures Working Set, and `peak_working_set_mib` in the frozen CSV records the max 250 ms-sampled single-PID RSS rather than OS `PeakWorkingSetSize`). Because the frozen campaign ran before `ResourceWatchdog` added per-sample windowing (`mark_sample_start()`/`get_window_metrics()`), `n=1000` ($525.6\text{ MiB}$, driver `244.7–260.4 MiB`) inherits the $n=1500$ high-water mark and $n \in \{250, 500\}$ ($796.7\text{ MiB}$, driver `246.8–253.5 MiB`) inherit the $n=2000$ high-water mark in the randomized sequence ($750 \to 1500 \to 1000 \to 2000 \to 250 \to 500$), whereas Pilot 2's isolated runs recorded `162.9 MiB` ($n=250$) and `178.3 MiB` ($n=500$). (2) Online validation (`campaign_runner.py`) checked full IEEE-754 `float64` checksums (`atol = 1e-6`), while `results_logger.py` formatted `checksum_sum >= 1e8` (`n >= 750`) as `:.6e` (7 significant digits) in the frozen CSV (remediated to `%.17g` for future runs). (3) In the frozen raw CSV, `object_nbytes` is `NULL` because `Client.nbytes()` returns a dictionary in Dask Distributed, which raised `TypeError` in `int(self.client.nbytes())` (since fixed via `extract_object_nbytes()`); directional transfer columns (`client_to_worker_payload_bytes`, `worker_to_worker_payload_bytes`, `worker_to_client_payload_bytes`, `compressed_transfer_bytes`, `transfer_metric_source`, `dask_managed_memory_mib`) remain `NULL` because diagnostic transfer hooks were omitted to avoid perturbing `TB_DASK_E2E` ($3 \times 8n^2 = 24n^2$ bytes describes structural storage complexity, not measured loopback payload). (4) `DistributedBlockMatmul.multiply()` captures in-memory `phase_breakdown_s` (`partitioning_s`, `scatter_s`, `graph_submission_s`, `gather_s`, `assembly_s`, `checksum_s`), which is not serialized to the 44-column `raw_measurements.csv` schema.*

### 8.2 Contextual Baseline Comparisons
*Baseline provenance:* Contextual ratios below pair the A4 campaign medians against frozen historical baseline medians from `data/baselines/a1_summary.csv` and `data/baselines/a3_parallel_results_summary.csv` (originally recorded on each prior assignment's own random `float64`/`double` matrices of the same dimensions); `scripts/prepare_a1_runtime.py` and `scripts/prepare_a3_runtime.py` supply common A4 inputs for optional local re-execution.  
**Contextual Duration Ratio:** $\text{CTR} = \tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask\_p1}}$  
- $\text{CTR} > 1.0$: Baseline duration was longer than Dask E2E duration.  
- $\text{CTR} < 1.0$: Baseline duration was shorter than Dask E2E duration.  
*These ratios reflect cross-system contextual comparisons under differing timing boundaries (`TB_KERNEL_ONLY` vs `TB_DASK_E2E`), not algorithmic speedups.*

| Dimension $n$ | Baseline Method | Baseline Median (s) | Dask $p=1$ Median (s) | Contextual Ratio | Methodological Comparison Note |
|:---:|---|:---:|:---:|:---:|---|
| **250** | A1 C Sequential ($i\text{-}j\text{-}k$) | 0.0912 | 0.0336 | **2.71×** | Baseline duration > Dask duration |
| **250** | A1 Java Sequential | 0.0276 | 0.0336 | **0.82×** | Baseline duration < Dask duration |
| **250** | A1 Python Sequential | 11.0849 | 0.0336 | **329.83×** | Baseline duration > Dask duration |
| **250** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 0.0060 | 0.0336 | **0.18×** | Baseline duration < Dask duration |
| **250** | A3 C Seq Tiled AVX2 ($b=32$) | 0.0059 | 0.0336 | **0.18×** | Baseline duration < Dask duration |
| **500** | A1 C Sequential ($i\text{-}j\text{-}k$) | 0.7726 | 0.2062 | **3.75×** | Baseline duration > Dask duration |
| **500** | A1 Java Sequential | 0.3437 | 0.2062 | **1.67×** | Baseline duration > Dask duration |
| **500** | A1 Python Sequential | 99.9401 | 0.2062 | **484.75×** | Baseline duration > Dask duration |
| **500** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 0.0640 | 0.2062 | **0.31×** | Baseline duration < Dask duration |
| **500** | A3 C Seq Tiled AVX2 ($b=32$) | 0.0447 | 0.2062 | **0.22×** | Baseline duration < Dask duration |
| **750** | A1 C Sequential ($i\text{-}j\text{-}k$) | 3.2241 | 0.4449 | **7.25×** | Baseline duration > Dask duration |
| **750** | A1 Java Sequential | 1.4662 | 0.4449 | **3.30×** | Baseline duration > Dask duration |
| **750** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 0.3099 | 0.4449 | **0.70×** | Baseline duration < Dask duration |
| **750** | A3 C Seq Tiled AVX2 ($b=32$) | 0.1765 | 0.4449 | **0.40×** | Baseline duration < Dask duration |
| **1000** | A1 C Sequential ($i\text{-}j\text{-}k$) | 7.1264 | 0.9318 | **7.65×** | Baseline duration > Dask duration |
| **1000** | A1 Java Sequential | 3.3293 | 0.9318 | **3.57×** | Baseline duration > Dask duration |
| **1000** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 0.9893 | 0.9318 | **1.06×** | Baseline duration > Dask duration |
| **1000** | A3 C Seq Tiled AVX2 ($b=32$) | 0.5276 | 0.9318 | **0.57×** | Baseline duration < Dask duration |
| **1500** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 6.5549 | 3.0392 | **2.16×** | Baseline duration > Dask duration |
| **1500** | A3 C Seq Tiled AVX2 ($b=32$) | 1.8883 | 3.0392 | **0.62×** | Baseline duration < Dask duration |
| **2000** | A3 C OpenMP $i\text{-}k\text{-}j$ ($p=1$) | 16.3745 | 7.4963 | **2.18×** | Baseline duration > Dask duration |
| **2000** | A3 C Seq Tiled AVX2 ($b=32$) | 4.7202 | 7.4963 | **0.63×** | Baseline duration < Dask duration |

*Note on contextual baseline selection & multi-threaded OpenMP context: Table 2 presents 22 representative single-worker/single-thread comparisons selected from the 34 single-thread ($t=1$) records in the underlying contextual baseline CSV (`results/analysis/contextual_baseline_comparison.csv`) to match A4's Phase 8 $p=1$ concurrency. For each of the six dimensions, it omits the alternative single-threaded representation `A3_Seq_ikj` in favor of `A3_OpenMP_ikj_p1`, and `A3_OpenMP_tiled_b32_p1` in favor of `A3_Seq_tiled_b32`. Separately, the frozen Assignment 3 summary (`data/baselines/a3_parallel_results_summary.csv`) also records multi-threaded shared-memory OpenMP medians—specifically at $t=10$ threads matching the 10 physical cores: untiled `omp_ikj` runs in $0.0019\text{ s}$ ($n=250$) to $1.5443\text{ s}$ ($n=2000$, $4.85\times$ shorter than Dask $p=1$ and $6.19\times$ shorter than Dask's best local multi-worker result $9.5635\text{ s}$ at $p=2$), and `omp_tiled_b32` runs in $0.0016\text{ s}$ ($n=250$) to $0.6250\text{ s}$ ($n=2000$, $11.99\times$ shorter than Dask $p=1$ and $15.30\times$ shorter than Dask $p=2$). These shared-memory timings are reported as contextual cross-system evidence demonstrating the advantage of zero-copy shared-memory threading on a single host.*

### 8.3 Local Multi-Worker Strong Scaling ($p \in \{1, 2, 4, 8, 10\}, n \in \{1000, 1500, 2000\}, b = 250$)
Following DEC-01 authorization (`docs/decisions/DEC-01-instructor-permission.md`), a controlled $p=1$ vs. $p=2$ pilot (`results/metadata/pilot_multiworker_p1_p2.json`; full 250,000-element $C_{500}$ match `max_abs_diff = 0.0`, `n=250` `max_abs_diff = 1.137e-13`), a 90-row strong-scaling sweep (`scripts/run_strong_scaling.py --campaign`: 5 consecutive single-`p` sub-campaigns for $p = 1 \to 2 \to 4 \to 8 \to 10$, each with its own `run_id`, fresh `LocalCluster`, and sidecar inside one continuous 26.6 min invocation on 2026-11-23 under `driver_pid = 15160`, 1 warm-up + 5 measured repetitions per cell, seed `20261113`), and a separate $n=1000$ diagnostic pass (`results/metadata/strong_scaling_diagnostics.json`) were executed on 1 physical machine (`1 + p` unique OS PIDs). All strong-scaling speedups use the dedicated Nov 23 $p=1$ sub-campaign (`campaign_20261123_132555_...`) as denominator rather than the Nov 14 Phase 8 run, noting that because `p` order was fixed ascending (not interleaved with `p=1`), residual intra-sweep thermal/frequency drift on Windows is not fully eliminated:

| Dimension $n$ | Workers $p$ | OS PIDs | Completed | Median $T_{\text{E2E}}$ (s) | IQR (s) | Mean (s) | Speedup $S_{\text{dist}}(p)$ | Efficiency $E_{\text{dist}}(p)$ | Peak RSS (MiB) | Dask Managed (MiB) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1000** | 1 | 2 | 5/5 | **1.5287** | 0.5568 | 1.7350 | 1.00× | 100.0% | 407.7 | 53.4 | `COMPLETED` ($\Delta=0.0$) |
| **1000** | 2 | 3 | 5/5 | **2.3203** | 0.4401 | 2.5033 | 0.66× | 32.9% | 535.9 | 69.1 | `COMPLETED` ($\Delta=0.0$) |
| **1000** | 4 | 5 | 5/5 | **7.3915** | 0.6233 | 7.5670 | 0.21× | 5.2% | 733.7 | 70.1 | `COMPLETED` ($\Delta=0.0$) |
| **1000** | 8 | 9 | 5/5 | **12.9262** | 14.7408 | 18.3671 | 0.12× | 1.5% | 1075.2 | 63.4 | `COMPLETED` ($\Delta=0.0$) |
| **1000** | 10 | 11 | 0/5 | — | — | — | N/A | N/A | — | — | `NOT_RUN_AFTER_TIMEOUT` |
| **1500** | 1 | 2 | 5/5 | **6.0131** | 0.8975 | 6.3114 | 1.00× | 100.0% | 520.1 | 154.5 | `COMPLETED` ($\Delta=0.0$) |
| **1500** | 2 | 3 | 5/5 | **7.9502** | 1.8595 | 7.9982 | 0.76× | 37.8% | 692.5 | 205.5 | `COMPLETED` ($\Delta=0.0$) |
| **1500** | 4 | 5 | 5/5 | **7.8444** | 1.1721 | 9.1675 | 0.77× | 19.2% | 866.5 | 206.5 | `COMPLETED` ($\Delta=0.0$) |
| **1500** | 8 | 9 | 5/5 | **17.7094** | 2.9573 | 17.0143 | 0.34× | 4.2% | 1173.2 | 165.9 | `COMPLETED` ($\Delta=0.0$) |
| **1500** | 10 | 11 | 4/5 | *40.3655\** | *41.7125\** | *54.8795\** | N/A | N/A | 1360.0 | 154.5 | `PARTIAL_TIMEOUT` (rep 5: 131.92 s) |
| **2000** | 1 | 2 | 5/5 | **11.6333** | 1.2076 | 11.2135 | 1.00× | 100.0% | 787.1 | 335.7 | `COMPLETED` ($\Delta=0.0$) |
| **2000** | 2 | 3 | 5/5 | **9.5635** | 2.4293 | 10.1923 | **1.22×** | **60.8%** | 967.8 | 408.2 | `COMPLETED` ($\Delta=0.0$) |
| **2000** | 4 | 5 | 5/5 | **39.1173** | 9.6631 | 35.7538 | 0.30× | 7.4% | 1133.5 | 356.7 | `COMPLETED` ($\Delta=0.0$) |
| **2000** | 8 | 9 | 5/5 | **43.8998** | 24.0615 | 41.2992 | 0.26× | 3.3% | 1468.1 | 337.6 | `COMPLETED` ($\Delta=0.0$) |
| **2000** | 10 | 11 | 0/5 | — | — | — | N/A | N/A | — | — | `NOT_RUN_AFTER_TIMEOUT` |

*\*Protocol §11.3 note: At $p=10, n=1500$, repetition 5 exceeded the 120.0 s timeout (`131.92 s`), triggering immediate safe halt and marking subsequent $n=2000$ and $n=1000$ samples `NOT_RUN_AFTER_TIMEOUT`. Partial 4-sample timing is shown for transparency; official $S(10)$ and $E(10)$ are omitted (`N/A`). Separate $n=1000$ diagnostics (`results/metadata/strong_scaling_diagnostics.json`) confirmed balanced task distribution (`[57, 55]` at $p=2$; `[29, 27, 28, 28]` at $p=4$; `[14, 17, 13, 19, 12, 14, 14, 9]` at $p=8$), real inter-worker transfers (`0 → 48 → 68 → 101` incoming transfer events; `0 B` spilled to disk), and driver wall-clock domination by `scatter_s` / `wait_s` / `gather_s` ($0.102\text{ s}/0.185\text{ s}/0.036\text{ s}$ at $p=1$ vs. $27.309\text{ s}/27.834\text{ s}/24.028\text{ s}$ at $p=8$, totaling `96.11%` of $82.379\text{ s}$, or `98.70%` including `graph_submission_s = 2.133 s`, where `wait_s` also encloses worker execution) while overlapping aggregate worker compute stayed $\le 0.335\text{ s}$.*

---

## 9. Key Scientific Findings

1. **Framework Coordination Latency at Small Dimensions ($H_1$ Partially Supported — Baseline-Dependent):**
   At $n = 250$ and $n = 500$, Dask end-to-end execution ($0.0336\text{ s}$ and $0.2062\text{ s}$ median) is substantially slower than all optimized A3 C kernels ($\text{CTR} = 0.17\times\text{–}0.18\times$ at $n=250$; $0.22\times\text{–}0.31\times$ at $n=500$) and slower than `A1_Seq_Java` at $n=250$ ($\text{CTR} = 0.82\times$), confirming that distributed orchestration costs dominate against optimized native kernels, whereas OpenBLAS still outperforms stride-$n$ `A1_Seq_C_ijk` ($2.71\times, 3.75\times$), Java at $n=500$ ($1.67\times$), and Python ($\ge 329.8\times$).
2. **Larger-Dimension Crossovers & Multi-Worker Granularity ($H_2$ Partially Supported):**
   At $p=1$, Dask/OpenBLAS overtakes A1 Java by $n=500$ ($1.67\times$) and untiled A3 $i\text{-}k\text{-}j$ at $n=1000$ ($1.06\times$, reaching $2.18\times$ at $n=2000$), with **no crossover through $n=2000$** against AVX2-tiled A3 ($0.63\times$). Across multiple local workers, **positive median strong-scaling speedup occurs only at $n=2000, p=2$ ($S_{\text{dist}} = 1.22\times$, $E_{\text{dist}} = 60.8\%$)**—noting that per-repetition ranges at $n=2000$ partially overlap between the consecutive $p=1$ (`[8.49, 12.75] s`, $\text{IQR}=1.21\text{ s}$) and $p=2$ (`[9.01, 11.68] s`, $\text{IQR}=2.43\text{ s}$) sub-campaigns—while at $n=1000$ and $n=1500$, even $p=2$ incurs net slowdown ($0.66\times$ and $0.76\times$).
3. **Worker Saturation & Coordination Overhead Beyond Two Workers ($H_3$ Empirically Supported):**
   Scaling beyond 2 workers on a single Windows host causes steep, non-monotonic degradation ($S(4)=0.21\times\text{–}0.77\times$, $S(8)=0.12\times\text{–}0.34\times$, $\text{IQR}$ up to $24.06\text{ s}$) and a `120.0 s` per-sample timeout at $p=10$ (`131.92 s` on `n=1500` rep 5; also observed in the $n=1000$ diagnostic warm-up at `121.38 s`). Separate $n=1000$ diagnostics rule out memory exhaustion and disk spilling (max RSS $1{,}468.1\text{ MiB} \ll 2{,}560\text{ MiB}$ cap, 0 disk spills) and BLAS compute ($\le 0.335\text{ s}$ aggregate), measuring `96.11%` of $T_{\text{E2E}}$ at $p=8$ in `scatter_s` + `wait_s` + `gather_s` (`98.70%` including `graph_submission_s`). As an architectural hypothesis consistent with these measurements (though not independently isolated via OS/event-loop profiling), this behavior aligns with centralized Tornado scheduler contention, Windows loopback TCP IPC overhead, and hybrid P/E-core scheduling.
4. **Memory Footprint & Process Scaling:**
   With per-sample `ResourceWatchdog` windowing in the strong-scaling runs, aggregate sampled RSS scales linearly with worker count (+80–160 MiB per Python worker process), from $407.7\text{ MiB}$ ($n=1000, p=1$) to $1{,}468.1\text{ MiB}$ ($n=2000, p=8$), while Dask managed worker memory peaks at $408.2\text{ MiB}$ ($n=2000, p=2$) with zero watchdog breaches or spills.
5. **Pilot 1 Startup Diagnosis & Architectural Scope ($H_4$ Supported):**
   Raising startup timeout from $30.0\text{ s}$ (Pilot 1) to $60.0\text{ s}$ resolved Windows cold-start failures. Comparing Dask's best local multi-worker result ($9.5635\text{ s}$ at $n=2000, p=2$) against zero-copy shared-memory OpenMP $t=10$ ($0.6250\text{ s}$ tiled / $1.5443\text{ s}$ untiled) confirms $H_4$: distributed multi-process frameworks trade substantial IPC and orchestration overhead for multi-machine scalability that cannot be demonstrated on a single host.

---

## 10. Step-by-Step Reproducibility Guide

### 10.1 Environment Setup
Requirements: Python 3.12+ (numpy 2.5.4 requires Python $\ge 3.12$; tested on Python 3.14.0 64-bit AMD64), Git, and MSYS2/MinGW GCC (for C baselines).

```bash
# 1. Clone repository
git clone https://github.com/abelcasv/Assignment04_Abel.git
cd Assignment04_Abel

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

*(Note: pinned versions in `requirements.txt` are `dask==2026.8.0`, `distributed==2026.8.0`, `numpy==2.5.4`, `psutil==7.2.2`, `matplotlib==3.11.2`, and `threadpoolctl==3.7.0`)*

### 10.2 Dataset Verification & Reconstruction
In accordance with Storage Policy Option B (`SPEC.md` DEC-05), dense benchmark matrices (`data/dense/`) can be generated deterministically on-demand:

```bash
# Generate dense CSV benchmark matrices
python scripts/generate_dataset.py

# Verify cryptographic integrity of all matrices against SHA256SUMS.txt
python scripts/verify_dataset.py
```

### 10.3 Running the Test Suite
The repository includes 198 unit and contract tests verifying matrix partitioning, tree reduction, cluster lifecycle, DEC-01 permission gate, telemetry contracts, adapters, reproducibility scripts, strong-scaling analysis, and input integrity:

```bash
python -m unittest discover tests -v
```
*Expected result: `Ran 198 tests ... OK`.*

### 10.4 CLI Usage & Parameter Checking
Verify execution parameters, fixture correctness, and DEC-01 authorization without launching benchmarks:

```bash
# Verify valid single-worker and authorized local multi-worker configurations
python -m src.cli --n 1000 --block-size 250 --workers 1 --check-config
python -m src.cli --n 1000 --block-size 250 --workers 2 --check-config
python -m src.cli --campaign --workers 2 --check-config

# Run small-fixture and n=250 numerical correctness verification
python -m src.cli --validate-fixtures
```

*(Authorized local strong-scaling execution is orchestrated via `scripts/run_strong_scaling.py --pilot-p1-p2`, `--campaign`, or `--diagnostics-only`.)*

### 10.5 Reproducing Statistical Analysis & Figures
Generate all Phase 8 and strong-scaling summary CSV tables, publication-quality plots, and SHA-256 provenance strictly offline:

```bash
python scripts/analyze_campaign.py
python scripts/verify_provenance.py
```
*Outputs generated and verified (13/13 PASS) in `results/analysis/`:*
- `campaign_measurements.csv` (SHA-256: `5C5F9866...`)
- `campaign_summary_statistics.csv` (SHA-256: `B82F8EE5...`)
- `contextual_baseline_comparison.csv` (SHA-256: `FAEE4E52...`)
- `plot_dask_p1_scaling.png` (SHA-256: `F000AAAB...`)
- `plot_contextual_comparison.png` (SHA-256: `AA810C44...`)
- `strong_scaling_measurements.csv` (SHA-256: `A92F74E8...`)
- `strong_scaling_summary.csv` (SHA-256: `C3B11DB5...`)
- `plot_strong_scaling.png` (SHA-256: `E213132C...`)
- `analysis_provenance.json`

### 10.6 Compiling the English Research Report
The final report is available as both Markdown (`report/report.md`) and LaTeX (`report/Assignment04_Report.tex`):

```bash
cd report
pdflatex -interaction=nonstopmode Assignment04_Report.tex
pdflatex -interaction=nonstopmode Assignment04_Report.tex
```

---

## 11. Certified Artifact Checksums

| Artifact File | Description | SHA-256 Digest |
|---|---|---|
| `results/raw/raw_measurements.csv` | Append-only raw measurements (138 rows: 48 historical + 90 strong-scaling; lines 1–49 prefix preserved = `3D9D7213...`) | `831A731D9DF6073F3D19AD4A7D4E4F8B54E04D39CFE463D298857F26F4ED426C` |
| `results/metadata/sidecar_campaign_20261114_205134_1d126401af244671b4ca8708818549f3.json` | Frozen Phase 8 single-worker campaign sidecar | `94EABF800B581408E540B9DEB103B93D3CD3724CF166EB3DEFDE02D5DE421E89` |
| `results/metadata/strong_scaling_manifest.json` | DEC-01 strong-scaling campaign run manifest (5 sessions) | `22635DFA7B84FB23D339EB56273487F3E12CB0B9BB6466CC090775CF366F2EAD` |
| `results/metadata/strong_scaling_diagnostics.json` | Separate $n=1000$ task-stream & phase diagnostic pass | `BB8A769C7175435311E6AAB9FEEE621EA7F283B8E3D004BEFBFB6C2700596811` |
| `data/validation/reference/C_250.csv` | Independent calculated numerical reference matrix ($n=250$) | `598E9A9FC8F0C8DB1D5DA277F4D93561AD83E2DF77E3FCB5F968F67B8033909E` |
| `scripts/analyze_campaign.py` | Offline analysis & visualization pipeline | `1080CFEC929284FDC567DE3460FF1D1484177A95FE62CA292851FE014BA22892` |
| `results/analysis/campaign_measurements.csv` | Filtered Phase 8 $p=1$ campaign subset (36 rows) | `5C5F9866D14F189853AAEAA0733ACBA344856763EF1E8D8E5C968896C5BDBD30` |
| `results/analysis/campaign_summary_statistics.csv` | Phase 8 $p=1$ descriptive statistics summary table | `B82F8EE5A22FB44D32274D60C90420BA9137D83AC3B592D7DD9D11CD314669F4` |
| `results/analysis/contextual_baseline_comparison.csv` | Cross-system contextual comparison dataset | `FAEE4E526D3C8808A54DDFE28EA8CDDAB708D7CC2D8C6AA72064D40ED662E204` |
| `results/analysis/strong_scaling_measurements.csv` | Extracted strong-scaling subset (90 rows) | `A92F74E888B87CA2B60D8D7381B717F865260E8296EF083EB4E21662968CB60D` |
| `results/analysis/strong_scaling_summary.csv` | Strong-scaling medians, $S_{\text{dist}}(p)$, $E_{\text{dist}}(p)$, RSS | `C3B11DB5CE1B0C4C4DDF3FC43913029726C48CA47CB53ECB4CB0D50DD55C2396` |
| `results/analysis/plot_dask_p1_scaling.png` | Phase 8 $p=1$ scaling figure (Linear & Log-Log) | `F000AAAB091D051A6221B8478EF2DD79F01AD9D335D21FB47519A05B90B42098` |
| `results/analysis/plot_contextual_comparison.png` | Contextual comparison figure | `AA810C44894482E9C763A46771953704BB6D35F1DF48D0B1516A5A1396E73D79` |
| `results/analysis/plot_strong_scaling.png` | Local multi-worker strong-scaling figure | `E213132C3C31C186AB420AEE498427AC4FA2EB2D5C9559CB185F822A36DE2E58` |

---

## 12. Generative AI Assistance Disclosure

In compliance with the ULPGC Big Data course Generative AI policy (PDF page 9), the author discloses that an AI tool (**Google Antigravity / Gemini**) was utilized under explicit supervision for:
1. **Architectural Scaffolding:** Drafting modular class structures for Dask Distributed graph generation, binary-tree reduction, and process-tree watchdog monitoring.
2. **Quality Engineering & Test Construction:** Authoring mock-based unit and regression test suites (`tests/test_cli.py`, `tests/test_pilot_runner.py`, `tests/test_campaign_runner.py`).
3. **Statistical Analysis Scripting:** Implementing offline NumPy/Matplotlib analysis pipelines (`scripts/analyze_campaign.py`).
4. **Language & Documentation Revision:** Assisting with technical English drafting and LaTeX typesetting for the research report.

**Author Certification:** All source code was reviewed and validated; all physical benchmarks were executed locally on the documented hardware without data fabrication; all experimental records trace directly to verifiable raw CSV records; and the author assumes full academic responsibility for the integrity, results, and conclusions of this study.

---

## 13. License & Academic Integrity

This project is submitted as coursework for the **Grado en Ciencia e Ingeniería de Datos** at the **Universidad de Las Palmas de Gran Canaria (ULPGC)**. The codebase is released for academic evaluation under the terms of the MIT License ([LICENSE](LICENSE)).
