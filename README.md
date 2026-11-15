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
2. **Framework & Scheduling Overhead:** Quantifying graph construction, task serialization, local inter-process communication (IPC), and tree reduction costs.
3. **Contextual Cross-System Comparisons:** Evaluating Dask against frozen, certified sequential baselines from Assignment 1 (C, Java, Python) and shared-memory parallel baselines from Assignment 3 (OpenMP, C Sequential $i\text{-}k\text{-}j$, and AVX2-tiled) across identical logical inputs.
4. **Administrative & Methodological Transparency:** Disclosing why multi-worker scaling ($p > 1$) was **BLOCKED / NOT COMPLETED** under institutional policy DEC-01, distinguishing single-host process emulation from genuine physical clustering, and reporting the structural dead-code vulnerability identified in the Assignment 1 benchmark harness.

---

## 2. Research Questions and Hypotheses

### 2.1 Central Research Question
> How does the end-to-end wall-clock performance, process memory footprint, and scheduling/communication overhead of a distributed 2D block matrix multiplication implementation using Dask Distributed scale as matrix dimensions increase on a single heterogeneous host, and how does it compare contextually with validated sequential (Assignment 1) and shared-memory parallel OpenMP (Assignment 3) implementations on identical logical input matrices and numeric types?

### 2.2 Formal Hypotheses
* **$H_1$ (Framework Overhead Dominance at Small Dimensions):** For small matrices ($n \le 500$), Dask graph generation, task scheduling, socket serialization, and result gathering will dominate execution time, causing Dask to exhibit lower throughput than native compiled sequential baselines ($R_{\text{seq}} = T_{\text{seq}} / T_{\text{Dask}} < 1$).
* **$H_2$ (Asymptotic Computational Domination Hypothesis):** As problem dimension grows ($n \ge 1000$), increasing computational work is hypothesized to reduce the relative impact of fixed framework coordination costs, resulting in shorter observed end-to-end execution times for the Dask/OpenBLAS system compared to unoptimized sequential baseline implementations within the tested scope.
* **$H_3$ (Worker Saturation & Scaling Limits):** On a shared-memory host, increasing worker process counts beyond physical performance cores would induce memory bandwidth contention and core-heterogeneity imbalance (retained as a theoretical hypothesis for future research; unverified empirically because multi-worker execution with $p > 1$ was BLOCKED under DEC-01).
* **$H_4$ (System-Level Contextual Boundaries):** Differences between Dask/NumPy/BLAS and C/OpenMP reflect differing runtime engines, memory layouts, and timing boundaries, representing a cross-system contextual comparison rather than an isolated algorithmic speedup.

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
* **Physical RAM:** 16.0 GB total (16,009.14 MiB, ~3.68 GiB available at baseline).
* **Operating System:** Microsoft Windows 11 Home 64-bit (Version 10.0.26300, AMD64).

### 4.2 Software Stack
* **Python Runtime:** 3.14.0 (MSC v.1944 64-bit AMD64).
* **Distributed Framework:** Dask 2026.8.0, Distributed 2026.8.0.
* **Numerical Libraries:** NumPy 2.5.4, SciPy OpenBLAS 0.3.34.106.0 (`libscipy_openblas64`).
* **Process Instrumentation:** psutil 7.2.2.
* **C Compilers:** GCC 13.2.0 (MinGW-W64 / MSYS2, C11 standard).

### 4.3 Concurrency & Process Mapping
* **Threading Isolation:** Strictly 1 Dask worker thread (`threads_per_worker = 1`). OpenBLAS internal multi-threading was pinned to 1 thread via environment variables (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) verified dynamically via `threadpoolctl`.
* **Process Hierarchy:** On Windows `LocalCluster(processes=True, n_workers=1)`:
  - **Driver Process (PID 33560):** Hosts Client, LocalCluster orchestrator, in-process Scheduler, and in-process Nanny management controller.
  - **Worker Process (PID 21036):** Dedicated child OS process executing task computation.
  - **Total OS Processes:** Exactly 2 operating system PIDs.

---

## 5. Administrative Status & Strong Scaling Compliance

### 5.1 Institutional Permission Gate (DEC-01 / A4-11)
Official course requirement A4-11 (PDF page 7) states:
> *"A local execution with multiple workers may be used only if the instructor permits it. It can investigate framework behaviour and process overhead, but does not demonstrate inter-machine network performance or multi-machine memory capacity. Label the environment clearly."*

In accordance with `SPEC.md` (DEC-01) and `docs/experimental_protocol.md`:
* **Single-Worker Execution ($p = 1$):** Fully validated and permitted as the baseline distributed framework measurement.
* **Multi-Worker Execution ($p \in \{2, 4, 8, 10\}$):** Strictly **BLOCKED** pending explicit written instructor authorization stored in `docs/decisions/`.
* **Physical Multi-Node Infrastructure:** No external university HPC or multi-machine cluster was accredited; genuine multi-node execution was **BLOCKED / NOT AVAILABLE**.
* **Strong Scaling Status:** Because written instructor permission for local multi-worker execution was not granted and no multi-node cluster existed, mandatory strong scaling across multiple workers is formally recorded as:
  $$\mathbf{BLOCKED\ /\ NOT\ COMPLETED}$$
* **Methodological Transparency:** Single-worker execution ($p = 1$) measures single-worker application latency and framework coordination; **$p = 1$ does NOT constitute a strong-scaling sweep**. Under no circumstances is $p = 1$ presented as satisfying the strong-scaling requirement.
* **Network Cost Isolation:** Network costs between physical machines were not isolated because all execution was confined to a single host over loopback IPC.

---

## 6. Timing Contracts & Operational Safeguards

### 6.1 Primary Timing Boundary ($T_{\text{Dask\_E2E}}$)
Adhering to `docs/experimental_protocol.md` §6.2, the primary measurement boundary is monotonic end-to-end wall-clock time ($t_0 \to t_1$):
```
[EXCLUDED FROM TIMER]
1. ClusterManager.start() — LocalCluster & Client creation (cluster_startup_s = 4.816 s).
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
  - Safe 5-Step Stop: Immediately cancels in-flight client tasks, closes cluster within 5 s, forcefully terminates lingering child PIDs (`terminate() / kill()`), and records `OUT_OF_MEMORY`.
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
* **Methodological Vulnerability:** However, the benchmark harness consumed only $c[0]$ via a volatile sink (`result_sink = c[0]`). Consequently, the remaining $n^2 - 1$ output elements were never validated for numerical correctness during benchmarking. In contrast, Assignment 4 evaluates a full-matrix checksum across the assembled array within the timed boundary, ensuring that all elements are accessed in memory. For numerical validation, the independent element-wise validation at $n=250$ used a tolerance-based acceptance criterion ($\text{atol}=10^{-9}, \text{rtol}=10^{-12}$) against an independent calculated numerical reference (`data/validation/reference/C_250.csv`, generated via `scripts/generate_dataset.py`'s `generate_reference_product()` using an independent standard-library Python scalar loop without BLAS or Dask). The observed maximum absolute difference was $0.0$ across all 62,500 elements, indicating exact numerical equality for this validation run. For $n \ge 500$, campaign evidence confirms checksum determinism across repetitions (zero observed checksum divergence), while noting that identical aggregate checksums do not prove complete matrix identity.
* **Consequence for Interpreting A1 Timing:** While the measured A1 runtime reflects execution of the complete $O(n^3)$ loop in the inspected binary, the harness suffered from a structural validation deficiency because it failed to certify that the computed result matrix was numerically correct beyond $c[0]$. Direct comparisons between A1/A3 and A4 represent **contextual duration ratios** ($\text{CTR} = \tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask\_E2E}}$), not isolated algorithmic speedups, due to differing timing boundaries (`TB_KERNEL_ONLY` vs `TB_DASK_E2E`), runtime backends, and memory layouts.

---

## 8. Verified Empirical Results

All results are derived deterministically from the 48 raw records in `results/raw/raw_measurements.csv` (SHA-256 `3D9D721343D379C26F61DD6B4669979F6E7E5262E0D8F6E96CD4D26B1F0FE92C`).

### 8.1 Single-Worker Dask Execution Time Scaling ($p = 1, b = 250$)
Each configuration evaluates 1 unmeasured warm-up followed by 5 measured repetitions, summarized using Median and linear Interquartile Range (IQR):

| Dimension $n$ | Blocks $N_b$ | Tasks (Mult + Add = Total) | Mean $T_{\text{E2E}}$ (s) | Median $T_{\text{E2E}}$ (s) | Std Dev (s) | IQR (s) | Min (s) | Max (s) | Checksum Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250** | 1 | $1 + 0 = 1$ | 0.0349 | **0.0336** | 0.0057 | 0.0032 | 0.0300 | 0.0446 | Validated vs Reference ($n=250$) |
| **500** | 2 | $8 + 4 = 12$ | 0.2024 | **0.2062** | 0.0270 | 0.0160 | 0.1621 | 0.2362 | Checksum Deterministic ($\Delta = 0.0$) |
| **750** | 3 | $27 + 18 = 45$ | 0.4480 | **0.4449** | 0.0521 | 0.0733 | 0.3798 | 0.5063 | Checksum Deterministic ($\Delta = 0.0$) |
| **1000** | 4 | $64 + 48 = 112$ | 0.9804 | **0.9318** | 0.1140 | 0.1228 | 0.8695 | 1.1528 | Checksum Deterministic ($\Delta = 0.0$) |
| **1500** | 6 | $216 + 180 = 396$ | 3.2977 | **3.0392** | 0.5197 | 0.7948 | 2.7801 | 3.9470 | Checksum Deterministic ($\Delta = 0.0$) |
| **2000** | 8 | $512 + 448 = 960$ | 7.3006 | **7.4963** | 0.6079 | 0.0952 | 6.2562 | 7.8475 | Checksum Deterministic ($\Delta = 0.0$) |

*All 30 measured campaign repetitions completed successfully with 0 timeouts and 0 memory breaches. Note on memory telemetry: In the campaign runner implementation (`campaign_runner.py`), the `ResourceWatchdog` is initialized once at cluster startup and maintains a cumulative running peak (`peak_process_sum_mib`) across the persistent cluster session without resetting between samples. The campaign executed dimensions in randomized order ($750 \to 1500 \to 1000 \to 2000 \to 250 \to 500$). Consequently, the recorded `rss_process_sum_peak_sampled_mib` represents the cumulative session high-water mark up to that sample; the values for $n=250$ and $n=500$ ($796.7\text{ MiB}$) reflect the persistent session high-water mark established during the preceding $n=2000$ execution, rather than an isolated per-sample memory footprint.*

### 8.2 Contextual Baseline Comparisons
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

*Note on contextual baseline selection: Table 2 presents 22 representative comparisons selected from the 34 records in the underlying contextual baseline CSV (`results/analysis/contextual_baseline_comparison.csv`). For each of the six dimensions, it omits the alternative single-threaded representation `A3_Seq_ikj` in favor of `A3_OpenMP_ikj_p1`, and `A3_OpenMP_tiled_b32_p1` in favor of `A3_Seq_tiled_b32`. These are equivalent algorithm/configuration variants rather than exact duplicate measurements; their measured execution times differ slightly due to OpenMP runtime library call overhead versus direct function invocation. The selection avoids redundant presentation of the same algorithmic kernels while preserving all distinct computational methods across the tested dimensions.*

---

## 9. Key Scientific Findings

1. **Framework Coordination Latency at Small Dimensions ($H_1$ Supported):**
   At $n = 250$, Dask end-to-end execution takes $33.6\text{ ms}$ (median), compared to $6.0\text{ ms}$ for the single-threaded A3 C $i\text{-}k\text{-}j$ in-process kernel (a contextual duration ratio of $0.18\times$). This difference is consistent with additional framework coordination and lifecycle activities (2D block slicing, scattering data over loopback IPC, dynamic DAG scheduling, gathering futures, dense array assembly, and checksum consumption), demonstrating that for small problem sizes with low computational intensity, distributed framework coordination introduces substantial latency overhead relative to in-process execution. Note that individual phase contributions were not separately instrumented or isolated within the end-to-end timer.
2. **Observed Scaling Trends at Large Dimensions ($H_2$ Supported as System-Level Trend):**
   The observed scaling trend is consistent with increasing computational work reducing the relative impact of fixed framework coordination costs. At large dimensions ($n \ge 1000$), Dask end-to-end duration was shorter than unoptimized sequential baselines ($0.93\text{ s}$ for Dask vs $7.13\text{ s}$ for A1 C at $n=1000$, $\text{CTR} = 7.65\times$; and $7.50\text{ s}$ for Dask vs $16.37\text{ s}$ for A3 OpenMP $p=1$ at $n=2000$, $\text{CTR} = 2.18\times$). However, when compared against A3's dedicated cache-blocked AVX2 implementation (Seq Tiled $b=32$, $4.72\text{ s}$ at $n=2000$), the baseline duration was shorter than Dask ($\text{CTR} = 0.63\times$). Because individual execution phases were not separately isolated, the experiment does not establish a measured phase crossover or prove that arithmetic computation becomes the dominant runtime contributor; rather, the data support $H_2$ strictly as an observed system-level duration trend under differing timing boundaries, rather than an isolated algorithmic speedup or pure kernel benchmark.
3. **Pilot 1 Startup Failure Analysis:**
   The initial pilot execution encountered a timeout during `LocalCluster` initialization because the default timeout ($30.0\text{ s}$) was insufficient for simultaneous Windows socket binding, nanny process creation, and worker registration under cold system conditions. Increasing the startup timeout to $60.0\text{ s}$ resolved the issue, and all subsequent Pilot 2 and Campaign runs completed with 100% success.
4. **Memory Footprint & Telemetry:**
   Peak sampled aggregate RSS remained strictly within operational safety limits, rising from $310.5\text{ MiB}$ at $n=750$ to $775.2\text{ MiB}$ at $n=2000$, well below the 2.5 GiB watchdog threshold.
5. **Strong Scaling Status ($H_3$ Unconfirmed):**
   Mandatory strong scaling across multiple workers remains **BLOCKED / NOT COMPLETED** under institutional decision DEC-01. Single-worker execution ($p=1$) does not constitute strong scaling, and Hypothesis 3 could not be empirically evaluated. Theoretical considerations regarding memory bus saturation and hybrid core asymmetry remain unverified hypotheses for future research under approved multi-worker configurations.

---

## 10. Step-by-Step Reproducibility Guide

### 10.1 Environment Setup
Requirements: Python 3.10+ (tested on Python 3.14.0 64-bit AMD64), Git, and MSYS2/MinGW GCC (for C baselines).

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

*(Note: dependencies are `dask[distributed]>=2026.8.0`, `numpy>=2.0.0`, `psutil>=5.9.0`, `matplotlib>=3.8.0`, `threadpoolctl>=3.0.0`)*

### 10.2 Dataset Verification & Reconstruction
In accordance with Storage Policy Option B (`SPEC.md` DEC-05), dense benchmark matrices (`data/dense/`) can be generated deterministically on-demand:

```bash
# Generate dense CSV benchmark matrices
python scripts/generate_dataset.py

# Verify cryptographic integrity of all matrices against SHA256SUMS.txt
python scripts/verify_dataset.py
```

### 10.3 Running the Test Suite
The repository includes 145 unit and contract tests verifying matrix partitioning, tree reduction, cluster lifecycle, DEC-01 permission gate, and input integrity:

```bash
python -m unittest discover tests -v
```
*Expected result: `Ran 145 tests in ~20s ... OK`.*

### 10.4 CLI Usage & Parameter Checking
Verify execution parameters and DEC-01 permission enforcement without launching benchmarks:

```bash
# Verify valid single-worker configuration
python -m src.cli --n 1000 --block-size 250 --workers 1 --check-config

# Test DEC-01 permission gate enforcement (must fail with exit code 1)
python -m src.cli --n 1000 --workers 2 --check-config
```

### 10.5 Reproducing Statistical Analysis & Figures
Generate all summary CSV tables and publication-quality plots from certified raw measurements:

```bash
python scripts/analyze_campaign.py
```
*Outputs generated in `results/analysis/`:*
- `campaign_measurements.csv` (SHA-256: `5C5F9866...`)
- `campaign_summary_statistics.csv` (SHA-256: `B82F8EE5...`)
- `contextual_baseline_comparison.csv` (SHA-256: `FAEE4E52...`)
- `plot_dask_p1_scaling.png` (SHA-256: `F000AAAB...`)
- `plot_contextual_comparison.png` (SHA-256: `AA810C44...`)
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
| `results/raw/raw_measurements.csv` | Immutable raw campaign measurements (48 rows) | `3D9D721343D379C26F61DD6B4669979F6E7E5262E0D8F6E96CD4D26B1F0FE92C` |
| `results/metadata/sidecar_campaign_20261114_205134_1d126401af244671b4ca8708818549f3.json` | Full campaign execution sidecar | `94EABF800B581408E540B9DEB103B93D3CD3724CF166EB3DEFDE02D5DE421E89` |
| `data/validation/reference/C_250.csv` | Independent calculated numerical reference matrix ($n=250$) | `598E9A9FC8F0C8DB1D5DA277F4D93561AD83E2DF77E3FCB5F968F67B8033909E` |
| `scripts/analyze_campaign.py` | Analysis & visualization pipeline | `88E36C023D74ED5D6C4B7F15FF897CFD774DB4657B808684A8E1133AD5B13ABF` |
| `results/analysis/campaign_summary_statistics.csv` | Descriptive statistics summary table | `B82F8EE5A22FB44D32274D60C90420BA9137D83AC3B592D7DD9D11CD314669F4` |
| `results/analysis/contextual_baseline_comparison.csv` | Cross-system comparison dataset | `FAEE4E526D3C8808A54DDFE28EA8CDDAB708D7CC2D8C6AA72064D40ED662E204` |
| `results/analysis/plot_dask_p1_scaling.png` | Runtime scaling figure (Linear & Log-Log) | `F000AAAB091D051A6221B8478EF2DD79F01AD9D335D21FB47519A05B90B42098` |
| `results/analysis/plot_contextual_comparison.png` | Contextual comparison figure | `AA810C44894482E9C763A46771953704BB6D35F1DF48D0B1516A5A1396E73D79` |

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

This project is submitted as coursework for the **Grado en Ciencia e Ingeniería de Datos** at the **Universidad de Las Palmas de Gran Canaria (ULPGC)**. The codebase is released for academic evaluation under the terms of the MIT License.
