# Assignment 4: Distributed Matrix Multiplication

**Course:** BIG DATA · Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)  
**Academic Year:** 2026–2027  
**Author:** Abel Castro Villar  
**Repository:** [Assignment04_Abel](https://github.com/abelcasv/Assignment04_Abel)  
**Primary Deliverable:** `report/Assignment04_Report.pdf` (compiled from `report/Assignment04_Report.tex`)

---

## 1. Executive Summary & Research Scope

This repository presents a controlled, empirical investigation into the scalability, memory footprint, and scheduling/communication overhead of dense square matrix multiplication ($C = A \times B$) executed via **Dask Distributed** on an asymmetric hybrid x86-64 processor. The implementation employs a **2D block decomposition** ($b = 250$) with a **pairwise binary-tree reduction**, operating on IEEE 754 `binary64` (`float64`) matrices.

In strict compliance with the official course assignment specification (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`, pages 2–3 and 7–9), complemented by student-authored technical notes (`SPEC.md`, `docs/experimental_protocol.md`), this study provides:
1. **End-to-End Distributed Latency ($p = 1$):** Measuring complete distributed workflow execution time across $n \in \{250, 500, 750, 1000, 1500, 2000\}$ under an explicit pre-registered timing boundary ($T_{\text{Dask\_E2E}}$).
2. **Controlled Common-Input Cross-Assignment Comparisons (A1, A3, and A4):** Executing the frozen Assignment 1 sequential C implementation (`baselines/a1_sequential/`) and Assignment 3 sequential and OpenMP C implementations (`baselines/a3_openmp/`) on the **exact same deterministic `data/dense/` IEEE 754 `binary64` input matrices** consumed by Assignment 4 (`scripts/run_common_input_baselines.py`), verified via SHA-256 parity (`results/analysis/common_input_comparison.csv`) while preserving historical contextual summaries (`data/baselines/`) separately without mixing datasets.
3. **Local Multi-Worker Strong Scaling & Per-Worker Overhead ($p \in \{1, 2, 4, 8, 10\}$):** Quantifying speedup $S_{\text{dist}}(p, n)$, parallel efficiency $E_{\text{dist}}(p, n)$, per-worker OS process and Dask-managed memory, task distribution, loopback inter-process communication (IPC), and scheduler contention across `1 + p` OS processes on a single physical host (`physical_machine_count = 1`).
4. **Methodological & Provenance Rigor:** Preserving historical raw measurements byte-for-byte (`results/raw/raw_measurements.csv`), distinguishing single-host distributed multi-worker execution (`127.0.0.1` loopback) from multi-machine cluster deployment (not used in this campaign), documenting exact checksum serialization precision across historical and remediated rows, and reporting the structural $c[0]$-only validation limitation verified in the GCC 16.1.0 `-O3` assembly of Assignment 1 (`build/bin/a1_benchmark.s`).

---

## 2. Research Questions and Hypotheses

### 2.1 Central Research Question
> How does the end-to-end wall-clock performance, per-process/per-worker memory footprint, and scheduling/communication overhead of a distributed 2D block matrix multiplication implementation using Dask Distributed scale as matrix dimensions increase and across local worker configurations, and how does it compare on identical IEEE 754 `binary64` input matrices against sequential (Assignment 1) and shared-memory OpenMP (Assignment 3) implementations?

*Experimental Scope Note (Official PDF "Local execution limitation", p. 7):* All distributed experiments were conducted on a single physical machine (`physical_machine_count = 1`) using Dask Distributed with one driver process, a Dask scheduler, and $p$ independent worker processes (`1 + p` OS PIDs) communicating through localhost (`127.0.0.1`). Initial single-worker benchmarking on 2026-11-14 produced the frozen 48-row single-worker dataset (`campaign_20261114_205134_...`, preserved byte-for-byte in lines 1–49 of `results/raw/raw_measurements.csv`), followed on 2026-11-23 by a controlled $p=1$ vs. $p=2$ pilot, a 90-row multi-worker strong-scaling campaign ($p \in \{1, 2, 4, 8, 10\}$, $n \in \{1000, 1500, 2000\}$, $b=250$), and a separate $n=1000$ diagnostic pass. On 2026-11-26, controlled common-input A1 (`results/raw/a1_common_input_raw.csv`, 25 rows) and A3 (`results/raw/a3_common_input_raw.csv`, 540 rows) baseline campaigns were executed on the exact `data/dense/` matrices used by Dask. This single-host configuration enabled distributed task execution and strong-scaling evaluation across worker counts; no multi-machine deployment was used in this experimental campaign.

### 2.2 Formal Hypotheses
* **$H_1$ (Small-Dimension Framework Overhead, $n \le 500$):** For small matrices ($n \le 500$), Dask graph construction, task scheduling, loopback serialization, and result gathering introduce fixed coordination overhead relative to native in-process C kernels ($\text{Ratio} = \tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask}}$). Empirically **supported at $n=250$ across all native C baselines and baseline-dependent at $n=500$**: on identical `data/dense/` inputs at $n=250$, **every** native C kernel is faster than Dask `p=1` ($0.0336\text{ s}$), including `A1 C Sequential i-j-k` ($0.0195\text{ s}$, $0.58\times$) and all A3 C kernels ($0.0023\text{–}0.0085\text{ s}$, $0.07\times\text{–}0.25\times$). At $n=500$, all A3 sequential and OpenMP kernels remain faster than Dask ($0.0129\text{–}0.0782\text{ s}$ vs. $0.2062\text{ s}$, $0.06\times\text{–}0.38\times$), whereas unblocked stride-$n$ `A1 C Sequential i-j-k` crosses above Dask ($0.2278\text{ s}$, $1.11\times$) once $B$-column accesses exceed L1/L2 cache locality.
* **$H_2$ (Larger-Dimension Scaling, Baseline Crossovers & Multi-Worker Granularity, $n \ge 1000$):** Evaluates whether cubic arithmetic growth ($2n^3$) amortizes fixed framework costs at $p=1$ and enables positive multi-worker speedup. Empirically **partially supported**: on identical inputs against Dask `p=1`, `A1 C Sequential i-j-k` reaches $4.31\times$ slower at $n=1000$ ($4.0157\text{ s}$ vs. $0.9318\text{ s}$), and untiled single-thread A3 (`seq_ikj` / `omp_ikj t=1`) crosses above Dask `p=1` at $n=1500$ ($1.24\times$ / $1.36\times$) and $n=2000$ ($15.5009\text{ s}$ / $17.3955\text{ s}$, $2.07\times$ / $2.32\times$) once the $51.5\text{–}91.6\text{ MiB}$ operands exceed the $12\text{ MB}$ L3 cache—yet cache-blocked `seq_tiled_b32` remains faster than Dask through $n=2000$ ($4.6706\text{ s}$, $0.62\times$ vs. Phase 8 `p=1` and $0.49\times$ vs. strong-scaling `p=2`). Across local Dask workers, positive median strong-scaling speedup occurs **only** at $n=2000, p=2$ ($S_{\text{dist}}=1.22\times$, $E_{\text{dist}}=60.8\%$, $9.5635\text{ s}$ vs. $11.6333\text{ s}$, with overlapping repetition ranges `[8.49, 12.75] s` vs. `[9.01, 11.68] s`), while $n \in \{1000, 1500\}$ slow down even at $p=2$ ($0.66\times, 0.76\times$).
* **$H_3$ (Worker Saturation & Scaling Limits on Asymmetric Single Host):** Increasing worker processes beyond the 2 Performance cores induces severe coordination contention and non-monotonic degradation. Empirically **supported**: for $p \ge 4$, runtime degrades sharply across all tested sizes ($S(4) = 0.21\times\text{–}0.77\times$, $S(8) = 0.12\times\text{–}0.34\times$, $\text{IQR}$ up to $24.06\text{ s}$), reaching the $120.0\text{ s}$ per-sample timeout at $p=10$ ($131.92\text{ s}$ at $n=1500$ rep 5). Separate $n=1000$ diagnostics rule out RAM exhaustion, disk spilling (`0 B`; per-worker OS RSS `84.60–113.62 MiB`, max aggregate RSS $1{,}468.1\text{ MiB} \ll 2{,}560\text{ MiB}$ cap), and BLAS compute ($\le 0.335\text{ s}$ aggregate), measuring `scatter_s` + `wait_s` + `gather_s` at `96.11%` of wall-clock latency at $p=8$ (`98.70%` including `graph_submission_s`), consistent with centralized Tornado scheduler contention, Windows loopback TCP IPC overhead, and hybrid P/E-core scheduling.
* **$H_4$ (System-Level & Timing-Boundary Qualification):** Even when consuming identical input matrices, runtime ratios between C/OpenMP (`TB_A1_KERNEL_QPC`, `TB_A3_KERNEL_QPC`) and Dask/OpenBLAS (`TB_DASK_E2E`) reflect non-equivalent timing boundaries, runtime engines, and memory models rather than isolated algorithmic speedups. Empirically **supported** and underscored by common-input shared-memory OpenMP at $n=2000$ (`omp_tiled_b32`: $0.8571\text{ s}$ at $t=8$ [$11.16\times$ shorter than Dask `p=2` $9.5635\text{ s}$] and $3.2325\text{ s}$ at $t=10$ [$2.96\times$ shorter]).

---

## 3. System Architecture & Algorithms

The distributed multiplication engine is implemented in Python under `src/dask_matrix/` using low-level Dask Distributed primitives (`Client.scatter`, `Client.submit`, `Client.gather`):

```
src/
├── __init__.py                   # Package exports
├── cli.py                        # Unified CLI and configuration validator
├── dask_matrix/                  # Core distributed multiplication package
│   ├── __init__.py
│   ├── partitioning.py           # 2D block decomposition & edge-block truncation
│   ├── kernels.py                # Elementary block GEMM and addition tasks
│   ├── reduction.py              # Pairwise binary-tree reduction DAG builder
│   ├── distributed_engine.py     # End-to-end distributed multiplication orchestrator
│   ├── cluster_manager.py        # LocalCluster lifecycle, DEC-01 gate, thread caps
│   ├── pilot_runner.py           # Bounded pilot campaign orchestrator
│   ├── campaign_runner.py        # Randomized multi-repetition campaign orchestrator
│   ├── watchdog.py               # Process-tree resource monitor & 5-step safe stop
│   ├── input_verifier.py         # SHA-256 dataset verification and loader
│   ├── validation.py             # Numerical correctness checking against tolerances
│   ├── telemetry.py              # Dask diagnostic stream & per-worker memory telemetry
│   ├── results_logger.py         # Append-only raw_measurements.csv logger
│   └── sidecar.py                # Environment metadata sidecar generator
└── adapters/                     # Non-invasive baseline execution adapters
    ├── __init__.py
    ├── a1_adapter.py             # Execution adapter for Assignment 1 sequential binary
    └── a3_adapter.py             # Execution adapter for Assignment 3 OpenMP binary
```

### 3.1 2D Block Partitioning & Pairwise Binary-Tree Reduction
For dense matrices $A, B \in \mathbb{R}^{n \times n}$ in IEEE 754 `binary64` (`float64`), operands are partitioned into an $N_b \times N_b$ grid using block dimension $b = 250$ ($N_b = \lceil n / b \rceil$):
$$C_{i, j} = \sum_{k=0}^{N_b - 1} A_{i, k} \times B_{k, j}$$
* **Edge-Block Truncation:** When $n$ is not divisible by $b$, edge blocks use exact truncated dimensions $\min(b, n - i \cdot b) \times \min(b, n - j \cdot b)$ without zero-padding, enforced in contiguous C-order (`np.ascontiguousarray(..., dtype=np.float64)`).
* **Task Graph & Tree Depth:** Each block triple $(i, j, k)$ submits a single-threaded `np.matmul` task ($N_b^3$ multiplications). For each output block $(i, j)$, partial products reduce pairwise across $\lceil \log_2 N_b \rceil$ tree stages ($N_b - 1$ additions per output block; total DAG tasks $= 2N_b^3 - N_b^2$, reaching $960$ tasks at $n=2000$).
* **Output Checksum Barrier:** Within $T_{\text{Dask\_E2E}}$, the reassembled matrix $C$ is materialized via $\text{float}(\text{np.sum}(C, \text{dtype}=\text{np.float64}))$. Aggregate checksum consumption ensures full-array memory access and tests cross-repetition checksum determinism, while independent element-wise accuracy is validated separately at $n=250$ against `data/validation/reference/C_250.csv` ($\text{atol}=10^{-9}, \text{rtol}=10^{-12}$) and at $n=500$ across $p=2$ vs. $p=1$.

---

## 4. Experimental Environment & Runtime Topology

### 4.1 Hardware & Software Stack
* **CPU:** Intel Core i5-1335U (Raptor Lake, 10 physical cores: 2 Performance cores with Hyper-Threading + 8 Efficient cores; 12 logical processors; 12 MB shared L3 cache).
* **Physical RAM:** 16.0 GB DDR4 (`16,009.14 MiB` / `15.63 GiB`; `3.60–4.78 GiB` available across idle benchmark baselines).
* **Operating System:** Microsoft Windows 11 Home 64-bit (Build `10.0.26300`, AMD64).
* **Python & Distributed Stack:** Python `3.14.0` (MSC v.1944 64-bit), Dask `2026.8.0`, Distributed `2026.8.0`, NumPy `2.5.4`, SciPy OpenBLAS `0.3.34.106.0` (`libscipy_openblas64`), psutil `7.2.2`, Matplotlib `3.11.2`, threadpoolctl `3.7.0`.
* **C / OpenMP Compiler:** MSYS2 MinGW-W64 UCRT64 **GCC `16.1.0`** (`gcc.EXE (Rev5, Built by MSYS2 project) 16.1.0`, C11, OpenMP `202111` / 5.2)—verified across historical A1/A3 logs, `build/bin/a1_benchmark.s`, and embedded `.ident` strings in the common-input binaries.

### 4.2 Process Mapping & Concurrency Isolation
* **Single-Thread Worker Isolation:** `LocalCluster(processes=True, n_workers=p, threads_per_worker=1)` with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` verified dynamically via `threadpoolctl`.
* **OS Process Count:** 1 driver PID (hosting `Client`, `LocalCluster`, in-process `Scheduler`, and in-process `Nanny`) + $p$ child worker PIDs = **`1 + p` unique OS PIDs** (`2, 3, 5, 9, 11` for $p \in \{1, 2, 4, 8, 10\}$) on `physical_machine_count = 1`.

---

## 5. Official Compliance & Administrative Scope

Under the official specification's *"Local execution limitation"* and *"Scaling and overhead"* sections (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`, page 7):
* **Single-Host Local Multi-Worker Configuration:** Separate worker processes ran with `p ∈ {1, 2, 4, 8, 10}` on one physical host (`physical_machine_count = 1`), using Windows loopback `127.0.0.1` under the `2.5 GiB` aggregate RSS ceiling, `1.0 GiB` host free-RAM floor, and `120.0 s` per-sample timeout.
* **Single-Host Distributed Scope vs. Multi-Machine Deployment:** All distributed experiments ran on one physical machine with multiple independent Dask worker processes communicating over localhost (`127.0.0.1`), enabling distributed task execution and strong-scaling evaluation across worker counts; no multi-machine deployment was used in this experimental campaign.
* **Optional Weak Scaling:** **Deferred** (`SPEC.md` DEC-07; dense cubic equal-work scaling $n(p) = n(1) \cdot p^{1/3}$ on the same single host cannot isolate multi-node behavior once local strong scaling saturates at $p=2$ and times out at $p=10$).

---

## 6. Timing Boundaries & Baseline Provenance

### 6.1 Non-Equivalent Timing Boundaries
* **`TB_DASK_E2E` (Assignment 4):** `time.perf_counter()` spanning 2D block slicing, `Client.scatter`, DAG submission ($N_b^3$ `matmul` + $N_b^2(N_b-1)$ `add`), worker execution/IPC, `Client.gather`, dense reassembly, and full-matrix checksum `float(np.sum(C))`. Excludes `LocalCluster` startup (`3.11–20.80 s`) and CSV disk loading (`0.161–6.478 s`).
* **`TB_A1_KERNEL_QPC` (Assignment 1 C):** `QueryPerformanceCounter` spanning `memset(c, 0, n^2 * 8)` and the triple `i-j-k` loop (`-Wall -Wextra -O3 -std=c11`, commit `90681e85949b6d65bdbcfef702b6306b0003f971`). Excludes CSV loading, allocation, and `result_sink = c[0]`. Disassembly of `build/bin/a1_benchmark.s` (GCC `16.1.0 -O3`) confirms the full $\mathcal{O}(n^3)$ loop survived dead-code elimination (`movupd`/`mulpd`/`unpckhpd`), while documenting the harness's $c[0]$-only validation limitation.
* **`TB_A3_KERNEL_QPC` (Assignment 3 C / OpenMP):** `QueryPerformanceCounter` spanning `memset(c, 0, n^2 * 8)`, OpenMP region management (for `omp_*`), and the `i-k-j` or $32 \times 32$ tiled multiplication loop (`-Wall -Wextra -O3 -std=c11 -fopenmp -march=native`, commit `23cffbc82c155b6c3de24b3c149b509b63fc870b`). Full $n^2$ checksum into `result_sink` occurs immediately after the stop timer.

---

## 7. Verified Empirical Results

All experimental datasets are strictly separated by provenance and never mixed:
1. **Dask Raw Measurements (`results/raw/raw_measurements.csv`, 138 rows):** 48 preserved historical single-worker rows (lines 1–49, SHA-256 prefix `3D9D7213...`) + 90 local multi-worker strong-scaling rows (lines 50–139).
2. **Controlled Common-Input Baselines (`results/raw/a1_common_input_raw.csv` [25 rows] & `results/raw/a3_common_input_raw.csv` [540 rows]):** Executed on the exact `data/dense/` matrices used by Dask (`results/metadata/common_input_baselines_manifest.json`).
3. **Historical Contextual Summaries (`data/baselines/`):** Preserved original A1/A3 summaries on prior random inputs, reported separately for historical context only.

### 7.1 Single-Worker Dask Scaling ($p = 1, b = 250$, Certified `data/dense/` Inputs)

| Dimension $n$ | Blocks $N_b$ | Total Tasks | Mean $T_{\text{E2E}}$ (s) | Median $T_{\text{E2E}}$ (s) | Std Dev (s) | IQR (s) | Min (s) | Max (s) | Peak RSS (MiB) | Verification & CSV Checksum Precision |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **250** | 1 | 1 | 0.0349 | **0.0336** | 0.0057 | 0.0032 | 0.0300 | 0.0446 | 796.7* | Element-wise vs `C_250.csv` ($\max|\Delta|=1.14\times 10^{-13}$); CSV `:.6f` |
| **500** | 2 | 12 | 0.2024 | **0.2062** | 0.0270 | 0.0160 | 0.1621 | 0.2362 | 796.7* | Online $|\Delta|\le 10^{-6}$ (`C_500` $p=2$ vs $p=1$: `0.0`); CSV `:.6f` |
| **750** | 3 | 45 | 0.4480 | **0.4449** | 0.0521 | 0.0733 | 0.3798 | 0.5063 | 327.3 | Online $|\Delta|\le 10^{-6}$; historical CSV rounded to `:.6e` |
| **1000** | 4 | 112 | 0.9804 | **0.9318** | 0.1140 | 0.1228 | 0.8695 | 1.1528 | 525.6* | Online $|\Delta|\le 10^{-6}$ (`%.17g` exact $\Delta=0.0$ in SS); CSV `:.6e` |
| **1500** | 6 | 396 | 3.2977 | **3.0392** | 0.5197 | 0.7948 | 2.7801 | 3.9470 | 525.6 | Online $|\Delta|\le 10^{-6}$ (`%.17g` exact $\Delta=0.0$ in SS); CSV `:.6e` |
| **2000** | 8 | 960 | 7.3006 | **7.4963** | 0.6079 | 0.0952 | 6.2562 | 7.8475 | 796.7 | Online $|\Delta|\le 10^{-6}$ (`%.17g` exact $\Delta=0.0$ in SS); CSV `:.6e` |

*\*Note on Phase 8 memory & checksum serialization:* (1) In the randomized Nov 14 sequence ($750 \to 1500 \to 1000 \to 2000 \to 250 \to 500$), `ResourceWatchdog` tracked cumulative cluster high-water marks prior to per-sample windowing, so $n=1000$ ($525.6\text{ MiB}$) and $n \in \{250, 500\}$ ($796.7\text{ MiB}$) reflect earlier $n=1500$/$2000$ peaks (Pilot 2 isolated runs measured $162.9\text{ MiB}$ at $n=250$ and $178.3\text{ MiB}$ at $n=500$). (2) Across the **119 completed Dask rows** in `raw_measurements.csv`, the **77** strong-scaling rows (lines 50–139) record full `%.17g` `binary64` checksums with exact `Δ = 0.0`, whereas the **42** historical Pilot 2 / Phase 8 rows (lines 8–49) passed online in-memory `float64` checks (`|Δ| ≤ 1e-6`) before formatting `n ∈ {250, 500}` (18 rows) as `:.6f` and `n ∈ {750..2000}` (24 rows) as 7-digit `:.6e` strings (from which sub-rounding differences cannot be reconstructed).

### 7.2 Primary Controlled Common-Input Comparison (Identical `data/dense/` Matrices, $N = 5$ Reps)

All rows below verified `input_sha256_verified_identical = True` against `data/SHA256SUMS.txt` and `results/raw/raw_measurements.csv` (`results/analysis/common_input_comparison.csv`). Cross-boundary ratios $\tilde{T}_{\text{baseline}} / \tilde{T}_{\text{Dask}}$ compare `TB_A1_KERNEL_QPC` / `TB_A3_KERNEL_QPC` against `TB_DASK_E2E` (`< 1.0×` = baseline faster than Dask; `> 1.0×` = baseline slower than Dask):

| Tier & $n$ | Method (`threads`) | Boundary | Baseline Med (s) | Baseline IQR (s) | Baseline memory* (MiB) | Phase 8 Dask (p=1) Med (s) | Ratio vs Phase 8 Dask (p=1) | Dask SS $p=2$ Med (s) | Ratio vs SS $p=2$ |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3-Way 250** | A1 C Seq `i-j-k` (`1`) | `TB_A1` | 0.0195 | 0.0006 | 5.81 | 0.0336 | **0.58×** | — | — |
| **3-Way 250** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 0.0069 | 0.0006 | 8.01 | 0.0336 | **0.21×** | — | — |
| **3-Way 250** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 0.0085 | 0.0023 | 8.01 | 0.0336 | **0.25×** | — | — |
| **3-Way 250** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 0.0057 / 0.0023 / 0.0027 | 0.0002 / 0.0001 / 0.0004 | 8.01 | 0.0336 | **0.17× / 0.07× / 0.08×** | — | — |
| **3-Way 250** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 0.0063 / 0.0028 / 0.0023 | 0.0029 / 0.0002 / 0.0004 | 8.01 | 0.0336 | **0.19× / 0.08× / 0.07×** | — | — |
| **3-Way 500** | A1 C Seq `i-j-k` (`1`) | `TB_A1` | 0.2278 | 0.0047 | 10.22 | 0.2062 | **1.11×** | — | — |
| **3-Way 500** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 0.0631 | 0.0063 | 13.38 | 0.2062 | **0.31×** | — | — |
| **3-Way 500** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 0.0754 | 0.0367 | 13.38 | 0.2062 | **0.37×** | — | — |
| **3-Way 500** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 0.0782 / 0.0239 / 0.0135 | 0.0252 / 0.0078 / 0.0020 | 13.38 | 0.2062 | **0.38× / 0.12× / 0.07×** | — | — |
| **3-Way 500** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 0.0521 / 0.0129 / 0.0133 | 0.0016 / 0.0013 / 0.0006 | 13.38 | 0.2062 | **0.25× / 0.06× / 0.06×** | — | — |
| **3-Way 750** | A1 C Seq `i-j-k` (`1`) | `TB_A1` | 0.9797 | 0.4965 | 17.37 | 0.4449 | **2.20×** | — | — |
| **3-Way 750** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 0.3756 | 0.0286 | 20.53 | 0.4449 | **0.84×** | — | — |
| **3-Way 750** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 0.2343 | 0.0064 | 20.53 | 0.4449 | **0.53×** | — | — |
| **3-Way 750** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 0.2904 / 0.0443 / 0.0404 | 0.1117 / 0.0055 / 0.0027 | 20.53 | 0.4449 | **0.65× / 0.10× / 0.09×** | — | — |
| **3-Way 750** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 0.2148 / 0.0388 / 0.0438 | 0.0160 / 0.0023 / 0.0073 | 20.53 | 0.4449 | **0.48× / 0.09× / 0.10×** | — | — |
| **3-Way 1000** | A1 C Seq `i-j-k` (`1`) | `TB_A1` | 4.0157 | 1.6171 | 27.39 | 0.9318 | **4.31×** | 2.3203 | **1.73×** |
| **3-Way 1000** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 0.7972 | 0.0832 | 30.54 | 0.9318 | **0.86×** | 2.3203 | **0.34×** |
| **3-Way 1000** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 0.4827 | 0.0985 | 30.54 | 0.9318 | **0.52×** | 2.3203 | **0.21×** |
| **3-Way 1000** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 0.9248 / 0.1351 / 0.1133 | 0.1156 / 0.0054 / 0.0161 | 30.54 | 0.9318 | **0.99× / 0.15× / 0.12×** | 2.3203 | **0.40× / 0.06× / 0.05×** |
| **3-Way 1000** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 1.9963 / 0.6635 / 0.8128 | 0.5570 / 0.3513 / 0.4213 | 30.54 | 0.9318 | **2.14× / 0.71× / 0.87×** | 2.3203 | **0.86× / 0.29× / 0.35×** |
| **2-Way 1500** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 3.7641 | 0.0731 | 59.10 | 3.0392 | **1.24×** | 7.9502 | **0.47×** |
| **2-Way 1500** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 2.6413 | 3.0955 | 59.10 | 3.0392 | **0.87×** | 7.9502 | **0.33×** |
| **2-Way 1500** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 4.1195 / 0.7595 / 0.7235 | 0.6079 / 0.0213 / 0.0437 | 59.10 | 3.0392 | **1.36× / 0.25× / 0.24×** | 7.9502 | **0.52× / 0.10× / 0.09×** |
| **2-Way 1500** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 1.6692 / 0.2518 / 3.1990 | 0.0976 / 0.0137 / 0.6726 | 59.11 | 3.0392 | **0.55× / 0.08× / 1.05×** | 7.9502 | **0.21× / 0.03× / 0.40×** |
| **2-Way 2000** | A3 C Seq `i-k-j` (`1`) | `TB_A3` | 15.5009 | 17.2869 | 99.28 | 7.4963 | **2.07×** | 9.5635 | **1.62×** |
| **2-Way 2000** | A3 C Seq Tiled `b=32` (`1`) | `TB_A3` | 4.6706 | 0.6875 | 99.28 | 7.4963 | **0.62×** | 9.5635 | **0.49×** |
| **2-Way 2000** | A3 OpenMP `i-k-j` (`1` / `8` / `10`) | `TB_A3` | 17.3955 / 1.9567 / 8.5514 | 0.4122 / 0.0497 / 0.5237 | 99.28 | 7.4963 | **2.32× / 0.26× / 1.14×** | 9.5635 | **1.82× / 0.20× / 0.89×** |
| **2-Way 2000** | A3 OpenMP Tiled `b=32` (`1` / `8` / `10`) | `TB_A3` | 6.6864 / 0.8571 / 3.2325 | 2.1897 / 2.5477 / 0.0567 | 99.28 | 7.4963 | **0.89× / 0.11× / 0.43×** | 9.5635 | **0.70× / 0.09× / 0.34×** |

*Baseline memory note:* A1 reports a post-run `WorkingSetSize` snapshot; A3 reports process-lifetime `PeakWorkingSetSize`. These measurements have different boundaries and are not directly equivalent.

*(Note: Frozen `benchmark.c` also executed auxiliary `n=100` in `0.0010 s` [`IQR=0.0000 s`, `4.59 MiB`], and frozen `parallel_benchmark.c` also swept `t ∈ {2, 4, 12, 16, 24}`, all preserved in `results/raw/a1_common_input_raw.csv`, `results/raw/a3_common_input_raw.csv`, and `results/analysis/common_input_baseline_summary.csv`. In the fixed-order 17-min A3 sweep, `t=8` measured faster than `t=10` at $n=2000$. The tiled `t=1` run measured `1.9963 s` at $n=1000$ versus `0.4827 s` for sequential tiled A3. No temperature or CPU-frequency telemetry was collected, so thermal, scheduling, and other run-order explanations remain unverified.)*

### 7.3 Historical Contextual Reference (`data/baselines/`, Non-Identical Random Inputs)
Preserved separately in `results/analysis/contextual_baseline_comparison.csv` (`34` single-thread records from prior assignments' own random `float64` matrices), historical medians include **A1 Java** (`0.0276 s` [`0.82×`], `0.3437 s` [`1.67×`], `1.4662 s` [`3.30×`], `3.3293 s` [`3.57×`] for $n \in \{250, 500, 750, 1000\}$), **A1 Python** (`11.0849 s` [`329.83×`] at $n=250$, `99.9401 s` [`484.75×`] at $n=500$), historical **A1 C** (`0.0912–7.1264 s`), and historical **A3 C/OpenMP** (`omp_tiled_b32 t=10`: `0.6250 s` at $n=2000$). These values provide multi-language context only and are **not** substitutes for Section 7.2's controlled common-input benchmarks.

### 7.4 Local Multi-Worker Strong Scaling & Per-Worker Memory ($p \in \{1, 2, 4, 8, 10\}$)

Speedups use the dedicated Nov 23 $p=1$ sub-campaign (`campaign_20261123_132555_...`) as denominator inside the continuous 26.6 min ascending-$p$ sweep (`p = 1 → 2 → 4 → 8 → 10`, 1 physical host, `1 + p` OS PIDs):

| Dimension $n$ | Workers $p$ | OS PIDs | Completed | Median $T_{\text{E2E}}$ (s) | IQR (s) | Mean (s) | Speedup $S_{\text{dist}}(p)$ | Efficiency $E_{\text{dist}}(p)$ | Peak Cluster RSS (MiB) | Dask Managed Sum (MiB) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1000** | 1 | 2 | 5/5 | **1.5287** | 0.5568 | 1.7350 | 1.00× | 100.0% | 407.7 | 53.4 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1000** | 2 | 3 | 5/5 | **2.3203** | 0.4401 | 2.5033 | 0.66× | 32.9% | 535.9 | 69.1 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1000** | 4 | 5 | 5/5 | **7.3915** | 0.6233 | 7.5670 | 0.21× | 5.2% | 733.7 | 70.1 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1000** | 8 | 9 | 5/5 | **12.9262** | 14.7408 | 18.3671 | 0.12× | 1.5% | 1075.2 | 63.4 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1000** | 10 | 11 | 0/5 | — | — | — | N/A | N/A | — | — | `NOT_RUN_AFTER_TIMEOUT` |
| **1500** | 1 | 2 | 5/5 | **6.0131** | 0.8975 | 6.3114 | 1.00× | 100.0% | 520.1 | 154.5 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1500** | 2 | 3 | 5/5 | **7.9502** | 1.8595 | 7.9982 | 0.76× | 37.8% | 692.5 | 205.5 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1500** | 4 | 5 | 5/5 | **7.8444** | 1.1721 | 9.1675 | 0.77× | 19.2% | 866.5 | 206.5 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1500** | 8 | 9 | 5/5 | **17.7094** | 2.9573 | 17.0143 | 0.34× | 4.2% | 1173.2 | 165.9 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **1500** | 10 | 11 | 4/5 | *40.3655\** | *41.7125\** | *54.8795\** | N/A | N/A | 1360.0 | 154.5 | `PARTIAL_TIMEOUT` (rep 5: 131.92 s) |
| **2000** | 1 | 2 | 5/5 | **11.6333** | 1.2076 | 11.2135 | 1.00× | 100.0% | 787.1 | 335.7 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **2000** | 2 | 3 | 5/5 | **9.5635** | 2.4293 | 10.1923 | **1.22×** | **60.8%** | 967.8 | 408.2 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **2000** | 4 | 5 | 5/5 | **39.1173** | 9.6631 | 35.7538 | 0.30× | 7.4% | 1133.5 | 356.7 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **2000** | 8 | 9 | 5/5 | **43.8998** | 24.0615 | 41.2992 | 0.26× | 3.3% | 1468.1 | 337.6 | `COMPLETED` (`%.17g` $\Delta=0.0$) |
| **2000** | 10 | 11 | 0/5 | — | — | — | N/A | N/A | — | — | `NOT_RUN_AFTER_TIMEOUT` |

*Per-Worker Memory & Phase Breakdown (`results/metadata/strong_scaling_diagnostics.json`, $n=1000, b=250$):* Post-gather scheduler `worker_memory_stats` show per-worker OS `process_bytes` of **`96.84 MiB`** ($p=1$), **`106.38–113.62 MiB`** ($p=2$), **`94.82–96.68 MiB`** ($p=4$), and **`84.60–87.48 MiB`** ($p=8$), while per-worker Dask `managed_bytes` partition cleanly (`6.68 MiB` at $p=1$; `32.42–33.38 MiB` at $p=2$; `13.83 MiB` uniform across all 4 workers at $p=4$; `5.72–7.15 MiB` at $p=8$; `0 B` spilled to disk) alongside driver RSS of `372.15–386.62 MiB`. Task distribution is balanced (`[112]`, `[57, 55]`, `[29, 27, 28, 28]`, `[14, 17, 13, 19, 12, 14, 14, 9]`), incoming inter-worker transfers grow `0 → 48 → 68 → 101`, overlapping aggregate worker compute stays $\le 0.335\text{ s}$, and driver `scatter_s` + `wait_s` + `gather_s` represent `96.11%` of $82.379\text{ s}$ at $p=8$ (`98.70%` including `graph_submission_s = 2.133 s`).

---

## 8. Step-by-Step Reproducibility Guide

### 8.1 Environment Setup
Requirements: Python 3.12+ (tested on Python `3.14.0` 64-bit AMD64), MSYS2 MinGW-W64 UCRT64 GCC (`16.1.0`) for C/OpenMP baselines, and `pdflatex` (MiKTeX / TeX Live) for PDF compilation.

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify deterministic dataset (20/20 SHA-256 checks) & numerical fixtures (6/6 PASS)
python -B scripts/verify_dataset.py
python -B scripts/verify_correctness.py

# 3. Run unit and contract test suite (200 tests)
python -B -m unittest discover tests -v

# 4. Regenerate all analysis CSVs, 4 figures (synced to report/plots/), and verify provenance (19/19 PASS)
python -B scripts/analyze_campaign.py
python -B scripts/verify_provenance.py
```

*(Optional re-execution of experimental campaigns: `python -B scripts/run_common_input_baselines.py` re-runs the controlled common-input A1 & A3 benchmarks; `python -B scripts/run_strong_scaling.py --pilot-p1-p2 | --campaign | --diagnostics-only` runs the DEC-01 local multi-worker Dask suites.)*

### 8.2 Compiling the Official PDF Report
```powershell
cd report
pdflatex -interaction=nonstopmode Assignment04_Report.tex
pdflatex -interaction=nonstopmode Assignment04_Report.tex
```

---

## 9. Certified Artifact Checksums (`verify_provenance.py`: 19/19 PASS)

| Artifact File | Description | SHA-256 Digest |
|---|---|---|
| `results/raw/raw_measurements.csv` | Dask raw measurements (138 rows; lines 1–49 historical prefix = `3D9D7213...`) | `831A731D9DF6073F3D19AD4A7D4E4F8B54E04D39CFE463D298857F26F4ED426C` |
| `results/raw/a1_common_input_raw.csv` | Controlled common-input Assignment 1 C raw measurements (25 rows) | `D62258F3CC6CC3387D57B61E9CD0AAD40F2674E3BECB592EA48C43D4BDEF1728` |
| `results/raw/a3_common_input_raw.csv` | Controlled common-input Assignment 3 C/OpenMP raw measurements (540 rows) | `107F7E94E119A01778F9BD31151BA0E202D7749431D2BA30EBEA7D85D05B1399` |
| `results/metadata/common_input_baselines_manifest.json` | Common-input baseline provenance, compiler, and input SHA-256 parity manifest | `33F50B0EA6227E2ECCE0693814323D581D8A890F77F7ED220532723DC697D3F3` |
| `results/metadata/sidecar_campaign_20261114_205134_1d126401af244671b4ca8708818549f3.json` | Frozen Phase 8 single-worker campaign sidecar | `94EABF800B581408E540B9DEB103B93D3CD3724CF166EB3DEFDE02D5DE421E89` |
| `results/metadata/strong_scaling_manifest.json` | Local multi-worker strong-scaling campaign manifest | `22635DFA7B84FB23D339EB56273487F3E12CB0B9BB6466CC090775CF366F2EAD` |
| `results/metadata/strong_scaling_diagnostics.json` | Separate $n=1000$ task-stream, phase, and per-worker memory diagnostics | `BB8A769C7175435311E6AAB9FEEE621EA7F283B8E3D004BEFBFB6C2700596811` |
| `data/validation/reference/C_250.csv` | Independent scalar-Python numerical reference matrix ($n=250$) | `598E9A9FC8F0C8DB1D5DA277F4D93561AD83E2DF77E3FCB5F968F67B8033909E` |
| `scripts/analyze_campaign.py` | Offline statistical analysis & figure generation pipeline | `B96117975445901F1117EF679F42974FF0B4E072DFC3AB135CC3CCD86402E517` |
| `results/analysis/campaign_measurements.csv` | Filtered Phase 8 $p=1$ subset (36 rows) | `5C5F9866D14F189853AAEAA0733ACBA344856763EF1E8D8E5C968896C5BDBD30` |
| `results/analysis/campaign_summary_statistics.csv` | Phase 8 $p=1$ descriptive statistics summary | `B82F8EE5A22FB44D32274D60C90420BA9137D83AC3B592D7DD9D11CD314669F4` |
| `results/analysis/common_input_baseline_summary.csv` | Controlled common-input A1 & A3 summary statistics (113 rows) | `FB183B7822D0E2C272248B8C5279F815CD573BC8F89CD24DEF25B689C0551FF9` |
| `results/analysis/common_input_comparison.csv` | Controlled common-input A1/A3 vs. Dask comparison table (52 rows) | `F2A968101FD404393733A0D1820AFE8BE96570EEC333AF66C6E1925BB024147F` |
| `results/analysis/contextual_baseline_comparison.csv` | Historical contextual baseline comparison dataset (34 rows) | `FAEE4E526D3C8808A54DDFE28EA8CDDAB708D7CC2D8C6AA72064D40ED662E204` |
| `results/analysis/strong_scaling_measurements.csv` | Extracted strong-scaling subset (90 rows) | `A92F74E888B87CA2B60D8D7381B717F865260E8296EF083EB4E21662968CB60D` |
| `results/analysis/strong_scaling_summary.csv` | Strong-scaling medians, $S_{\text{dist}}(p)$, $E_{\text{dist}}(p)$, RSS | `C3B11DB5CE1B0C4C4DDF3FC43913029726C48CA47CB53ECB4CB0D50DD55C2396` |
| `results/analysis/plot_dask_p1_scaling.png` | Single-worker Dask scaling figure (synced to `report/plots/`) | `2A8097DF719143DA7E011CC898460C27D7F9DCABA8E3BBA4BD74AB3866E5FE8F` |
| `results/analysis/plot_common_input_comparison.png` | Controlled common-input comparison figure (synced to `report/plots/`) | `E95FFFE77D3E8E7BCB52D12B67DFA662FD96D8058421BFECD213FC7032A8D49C` |
| `results/analysis/plot_contextual_comparison.png` | Historical contextual comparison figure (synced to `report/plots/`) | `934330C8D998E3D435CDEA93A61E8AAE0D4A9DA0F27C12DE48EE3CFF831C9350` |
| `results/analysis/plot_strong_scaling.png` | Local multi-worker strong-scaling figure (synced to `report/plots/`) | `E213132C3C31C186AB420AEE498427AC4FA2EB2D5C9559CB185F822A36DE2E58` |

---

## 10. Generative AI Assistance & Academic Integrity

In compliance with the ULPGC Big Data course policy on Generative Artificial Intelligence (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`, page 9), the author discloses that an AI assistant (**Google Antigravity / Gemini**) was used under direct supervision for modular architectural scaffolding, unit/contract test suites, offline NumPy/Matplotlib analysis pipelines, and technical English/LaTeX editing. All code was audited and executed locally on the documented hardware; all numerical claims trace to unfabricated raw CSV measurements; and this repository is submitted exclusively as individual academic coursework for the **Grado en Ciencia e Ingeniería de Datos** at the **Universidad de Las Palmas de Gran Canaria (ULPGC)**, with full academic responsibility assumed by the author.
