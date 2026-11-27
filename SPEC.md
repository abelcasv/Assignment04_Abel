# SPEC.md — Assignment 4: Technical Specification and Experimental Contract

**Course:** BIG DATA — Grado en Ciencia e Ingeniería de Datos, ULPGC
**Academic Year:** 2026–2027
**Assignment:** 4 — Distributed Matrix Multiplication
**Canonical Report:** [`report/Assignment04_Report.pdf`](report/Assignment04_Report.pdf) (compiled from [`report/Assignment04_Report.tex`](report/Assignment04_Report.tex))

---

## 1. Purpose and Experimental Scope

This document specifies the architecture, data contracts, baseline provenance, timing and memory boundaries, operational safeguards, and verification criteria for the Assignment 4 empirical study of distributed dense matrix multiplication ($C = A \times B$) using **Dask Distributed**.

The investigation evaluates:
1. **Computation, Scheduling, and Communication Overhead:** End-to-end distributed latency ($T_{\mathrm{Dask\_E2E}}$), task-graph submission, loopback serialization/transfer, and result gathering across increasing matrix dimensions $n \in \{250, 500, 750, 1000, 1500, 2000\}$ with block dimension $b = 250$.
2. **Controlled Common-Input Cross-Assignment Comparisons:** Comparing Dask Distributed against the reused Assignment 1 sequential C implementation ([`baselines/a1_sequential/`](baselines/a1_sequential/)) and Assignment 3 sequential and OpenMP C implementations ([`baselines/a3_openmp/`](baselines/a3_openmp/)) on identical deterministic IEEE 754 `binary64` input matrices in `data/dense/` (`input_sha256_verified_identical = True`), while keeping historical prior-assignment summaries ([`data/baselines/`](data/baselines/)) strictly separate.
3. **Single-Host Multi-Worker Strong Scaling and Per-Worker Memory:** Evaluating fixed matrix dimensions $n \in \{1000, 1500, 2000\}$ across $p \in \{1, 2, 4, 8, 10\}$ independent Dask worker processes (`1 + p` OS PIDs, `threads_per_worker = 1`, `OPENBLAS_NUM_THREADS = 1`) communicating over `127.0.0.1` loopback TCP on one physical host (`physical_machine_count = 1`), together with a separate $n = 1000$ diagnostic pass measuring per-worker OS RSS (`process_bytes`), Dask-managed memory (`managed_bytes`), disk spill, task distribution, and driver phase durations.
4. **Scope Boundaries:** All distributed experiments ran on a single physical machine (`Intel Core i5-1335U`, `2` Performance cores + `8` Efficient cores, `16.0 GB` RAM, Windows 11 Home 64-bit). No physical multi-node cluster deployment was used, and optional weak-scaling experiments ($n(p) = n(1) \cdot p^{1/3}$) were not included in this strong-scaling-focused campaign.

---

## 2. Research Question and Hypotheses

### 2.1 Research Question
> How does the end-to-end wall-clock performance, per-process/per-worker memory footprint, and scheduling/communication overhead of a distributed 2D block matrix multiplication implementation using Dask Distributed scale as matrix dimensions increase and across local worker configurations, and how does it compare on identical IEEE 754 `binary64` input matrices against sequential (Assignment 1) and shared-memory parallel OpenMP (Assignment 3) implementations?

### 2.2 Hypotheses
- **$H_1$ (Small-Dimension Framework Overhead, $n \le 500$):** For small matrices ($n \le 500$), DAG construction, centralized scheduling, loopback serialization, and driver gathering exceed raw kernel computation time, making Dask slower than native in-process C baselines ($\mathrm{Ratio} = \tilde{T}_{\mathrm{baseline}} / \tilde{T}_{\mathrm{Dask}} < 1$).
- **$H_2$ (Larger-Dimension Scaling, Crossovers & Multi-Worker Granularity, $n \ge 1000$):** Cubic arithmetic growth ($2n^3$) amortizes fixed framework costs against unblocked baselines at $p = 1$ once working sets exceed the shared `12 MB` L3 cache, and enables positive multi-worker speedup ($S_{\mathrm{dist}}(p, n) > 1$) only when task granularity is sufficiently coarse.
- **$H_3$ (Single-Host Worker Saturation):** Increasing worker processes past the `2` Performance cores on a single asymmetric host is expected to yield non-monotonic scaling limits as centralized scheduler contention, loopback IPC overhead, and hybrid P/E-core scheduling effects grow.
- **$H_4$ (Non-Equivalent Timing & Architectural Boundaries):** Even on byte-identical input matrices, runtime ratios between C/OpenMP (`TB_A1_KERNEL_QPC`, `TB_A3_KERNEL_QPC`) and Dask/OpenBLAS (`TB_DASK_E2E`) reflect distinct timing boundaries, memory spaces, and runtime backends rather than isolated algorithmic speedups.

---

## 3. Distributed Architecture and Mathematical Contract

### 3.1 Matrix Representation and 2D Block Partitioning
- Input and output operands $A, B, C \in \mathbb{R}^{n \times n}$ are square dense matrices stored in IEEE 754 `binary64` (`float64`).
- Matrices are partitioned into an $N_b \times N_b$ grid using reference block size $b = 250$, where $N_b = \lceil n/b \rceil$.
- **Edge-Block Truncation:** When $n$ is not divisible by $b$, edge blocks adopt exact truncated dimensions $\mathrm{rows} = \min(b, n - i \cdot b)$ and $\mathrm{cols} = \min(b, n - j \cdot b)$ without zero-padding, enforced in contiguous C-order (`np.ascontiguousarray(..., dtype=np.float64)`).

### 3.2 Scheduler-Managed DAG and Pairwise Binary-Tree Reduction
- Each output block $(i, j) \in \{0, \dots, N_b - 1\}^2$ is computed as:
  $$C_{i, j} = \sum_{k=0}^{N_b - 1} A_{i, k} \times B_{k, j}$$
- Using low-level `Client.scatter`, `Client.submit`, and `Client.gather` futures coordinated by Dask Distributed's centralized scheduler:
  1. **Multiplication Stage:** $N_b^3$ independent `block_multiply` tasks compute partial products $P_{i, j, k} = A_{i, k} \times B_{k, j}$ via single-threaded `numpy.matmul`.
  2. **Reduction Stage:** For each output block $(i, j)$, partial products are reduced pairwise via `block_add` across $\lceil \log_2 N_b \rceil$ binary-tree stages ($N_b - 1$ additions per output block; total DAG tasks $= N_b^3 + N_b^2(N_b - 1) = 2N_b^3 - N_b^2$, reaching `960` tasks at $n = 2000$).
  3. **Assembly & Checksum Barrier:** Gathered output blocks are assembled into dense matrix $C$ in driver memory and consumed before timer stop via:
     $$\mathrm{checksum} = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} C_{i, j} = \mathrm{float}(\mathrm{np.sum}(C, \mathrm{dtype}=\mathrm{np.float64}))$$

### 3.3 Process Topology and Thread Isolation
- `LocalCluster(processes=True, n_workers=p, threads_per_worker=1, memory_limit=None)` spawns `1` driver process (`Client`, `LocalCluster`, in-process `Scheduler`, and `Nanny`) plus $p$ child worker processes (`1 + p` unique OS PIDs: `2, 3, 5, 9, 11` for $p \in \{1, 2, 4, 8, 10\}$) on `physical_machine_count = 1`.
- Internal numerical library threading is pinned to `1` per worker via `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`, verified dynamically via `threadpoolctl` (`libscipy_openblas64` v`0.3.34.106.0`).

---

## 4. Reused Baseline Provenance and Runtime Adapters

Machine-readable provenance is recorded in [`results/metadata/common_input_baselines_manifest.json`](results/metadata/common_input_baselines_manifest.json) and enforced by [`scripts/run_common_input_baselines.py`](scripts/run_common_input_baselines.py).

### 4.1 Frozen Source Snapshots
Unmodified source snapshots from the author's prior assignments are vendored under [`baselines/`](baselines/) with `.gitattributes` binary line-ending preservation (`baselines/** -text`):

| Local Snapshot Path | Origin Repository & Commit SHA | File SHA-256 Digest |
|---|---|---|
| [`baselines/a1_sequential/benchmark.c`](baselines/a1_sequential/benchmark.c) | `Assignment01_Abel` (`90681e85949b6d65bdbcfef702b6306b0003f971`) | `4893fc553e97a0ec596c3c1cf9d5d8f582fc49308fc5971b338f48fd0268236f` |
| [`baselines/a1_sequential/Makefile`](baselines/a1_sequential/Makefile) | `Assignment01_Abel` (`90681e85949b6d65bdbcfef702b6306b0003f971`) | `b35b8cf8671d4cb0bdc42b59f0708dd8e55ebe51842034d3046c60da82f8c31f` |
| [`baselines/a3_openmp/parallel_benchmark.c`](baselines/a3_openmp/parallel_benchmark.c) | `Assignment03_Abel` (`23cffbc82c155b6c3de24b3c149b509b63fc870b`) | `6da20a3aed0862adc296fe9f4c8625bc51cd777959a5403088f7960c8741dbc8` |
| [`baselines/a3_openmp/Makefile`](baselines/a3_openmp/Makefile) | `Assignment03_Abel` (`23cffbc82c155b6c3de24b3c149b509b63fc870b`) | `474e4d2f2b57a5696bd8202ac0e125b101dbe6bcd5f58bd92df2a23f8a37e364` |

### 4.2 Compilation, Runtime Staging, and Assembly Verification
- **Compiler & Flags:** MSYS2 MinGW-W64 UCRT64 **GCC `16.1.0`** (`Rev5, Built by MSYS2 project`, C11, OpenMP `202111` / 5.2).
  - Assignment 1 (`a1_benchmark.exe`): `-Wall -Wextra -O3 -std=c11 -lpsapi` (canonical `i-j-k` triple loop).
  - Assignment 3 (`a3_parallel_benchmark.exe`): `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native -lpsapi` (`seq_ikj`, `seq_tiled_b32`, `omp_ikj`, `omp_tiled_b32` across $t \in \{1, 2, 4, 8, 10, 12, 16, 24\}$).
- **Non-Invasive Runtime Adapters:** [`scripts/prepare_a1_runtime.py`](scripts/prepare_a1_runtime.py) and [`scripts/prepare_a3_runtime.py`](scripts/prepare_a3_runtime.py) stage the exact `data/dense/` matrices and validation fixtures into isolated `build/a1_runtime/` and `build/a3_runtime/` directories ( verifying SHA-256 parity). Because frozen `benchmark.c` hardcodes $n \in \{100, 250, 500, 750, 1000\}$, auxiliary $n = 100$ is staged for A1 binary compatibility and excluded from 3-way comparisons ($n \in \{250, 500, 750, 1000\}$), while extended 2-way comparisons cover $n \in \{1500, 2000\}$ for A3 and Dask.
- **A1 `-O3` Disassembly Verification:** Inspection of GCC `16.1.0 -O3` assembly (`build/bin/a1_benchmark.s`) confirms that the full $\mathcal{O}(n^3)$ multiplication loop survived dead-code elimination (`.L63`, `.L69`, `.L65` using `movupd`/`mulpd`/`unpckhpd`), while documenting that `result_sink = c[0]` leaves $n^2 - 1$ output elements unvalidated during timing.

---

## 5. Common Input Dataset and Numerical Correctness Contract

### 5.1 Deterministic Dataset Policy
- Generated by [`scripts/generate_dataset.py`](scripts/generate_dataset.py) using Python's standard library `random.Random(42 + n)` (MT19937), uniform $[0.0, 1.0)$, IEEE 754 `binary64` formatted as `%.17g` with Unix LF (`\n`) line endings, and certified by `20` SHA-256 digests in [`data/SHA256SUMS.txt`](data/SHA256SUMS.txt) (verified by [`scripts/verify_dataset.py`](scripts/verify_dataset.py)).
- To avoid repository bloat and large-file limits, the `14` dense CSV matrices in `data/dense/` (~310 MB) are excluded from Git via `.gitignore` and deterministically reconstructed on fresh clones (see [`data/README.md`](data/README.md)).

### 5.2 Numerical Correctness Fixtures and Tolerances
Validated by [`scripts/verify_correctness.py`](scripts/verify_correctness.py) and `python -B -m src.cli --validate-fixtures` (`6/6 PASS`):
1. Analytical $4 \times 4$ known-answer product ($b = 2$).
2. Irregular edge-block truncation ($n = 4, b = 3$).
3. Multiplicative identity ($A \times I = A$, $n = 4, b = 2$).
4. Multiplicative zero ($A \times 0 = 0$, $n = 4, b = 2$).
5. Element-wise cross-check at $n = 250$ against [`data/validation/reference/C_250.csv`](data/validation/reference/C_250.csv) (pure-Python scalar triple-loop reference, SHA-256 `598e9a9fc8f0c8db1d5da277f4d93561ad83e2df77e3fcb5f968f67b8033909e`) under:
   $$|C_{\mathrm{test}}[i, j] - C_{\mathrm{ref}}[i, j]| \le \mathrm{atol} + \mathrm{rtol} \cdot |C_{\mathrm{ref}}[i, j]|, \quad \mathrm{atol} = 10^{-9},\ \mathrm{rtol} = 10^{-12}$$
   achieving $\max|\Delta| = 1.137 \times 10^{-13}$ ($b = 250$) and $1.208 \times 10^{-13}$ ($b = 100$), plus exact `0.0` element-wise difference across all `250,000` elements of $C_{500}$ ($p = 2$ vs. $p = 1$).
6. Full-matrix checksum consumption and finiteness verification (zero `NaN` or `Inf` values).

---

## 6. Timing Boundaries, Memory Metrics, and Resource Watchdog

### 6.1 Timing Boundaries
- **`TB_DASK_E2E` (Assignment 4):** Monotonic `time.perf_counter()` spanning 2D block slicing, `Client.scatter`, DAG submission, distributed worker execution and loopback IPC, `Client.gather`, dense assembly, and full-matrix checksum `float(np.sum(C, dtype=np.float64))`. Excludes `LocalCluster` startup (`cluster_startup_s`) and CSV disk loading (`input_load_s`).
- **`TB_A1_KERNEL_QPC` (Assignment 1 C):** `QueryPerformanceCounter` spanning `memset(c, 0, n^2 * 8)` and the `i-j-k` loop; excludes CSV loading and post-stop `result_sink = c[0]`.
- **`TB_A3_KERNEL_QPC` (Assignment 3 C/OpenMP):** `QueryPerformanceCounter` spanning `memset(c, 0, n^2 * 8)`, OpenMP region management, and the `i-k-j` or $32 \times 32$ tiled loop; excludes CSV loading and the post-stop $n^2$ checksum loop into `result_sink`.

### 6.2 Memory Metrics and Operational Watchdog
Memory metrics across systems have different boundaries and are reported separately:
- **Dask Sampled Aggregate RSS (`rss_process_sum_peak_sampled_mib`):** Simultaneous sum of Windows Working Set (`.rss` via `psutil`) across all unique cluster PIDs (driver + workers) sampled every `250 ms`.
- **Dask Per-Worker Memory (`strong_scaling_diagnostics.json`):** Post-gather scheduler `worker_memory_stats` reporting per-worker OS RSS (`process_bytes`), Dask-managed block payload (`managed_bytes`), and disk spill (`spilled_bytes = 0`).
- **Baseline Memory:** Assignment 1 records a post-run `WorkingSetSize` snapshot; Assignment 3 records process-lifetime `PeakWorkingSetSize`.
- **ResourceWatchdog ([`src/dask_matrix/watchdog.py`](src/dask_matrix/watchdog.py)):** Samples cluster PIDs every `250 ms` and enforces a `2,560.0 MiB` (`2.5 GiB`) aggregate RSS ceiling (`OUT_OF_MEMORY`), a `1,024.0 MiB` (`1.0 GiB`) free host RAM floor (`ABORTED_WATCHDOG`), a `120.0 s` per-sample timeout (`TIMEOUT`), and a 5-step safe teardown (task cancellation, client/cluster close, and child PID termination).

---

## 7. Experimental Datasets, Statistical Analysis, and Provenance

- **Repetition Protocol:** Each primary configuration targets `1` unmeasured warm-up (`is_warmup = true`, `rep = 0`) and `5` measured repetitions (`rep = 1..5`) in randomized order (`seed = 20261113`).
- **Raw Data Separation:**
  1. [`results/raw/raw_measurements.csv`](results/raw/raw_measurements.csv) (`138` Dask rows in a 44-column append-only schema: `48` single-worker pilot/campaign rows in lines 1–49 + `90` strong-scaling rows in lines 50–139).
  2. [`results/raw/a1_common_input_raw.csv`](results/raw/a1_common_input_raw.csv) (`25` rows) and [`results/raw/a3_common_input_raw.csv`](results/raw/a3_common_input_raw.csv) (`540` rows) measured on identical `data/dense/` matrices.
  3. [`data/baselines/a1_summary.csv`](data/baselines/a1_summary.csv) and [`data/baselines/a3_parallel_results_summary.csv`](data/baselines/a3_parallel_results_summary.csv) preserved strictly for historical contextual reference.
- **Statistical Formulas ([`scripts/analyze_campaign.py`](scripts/analyze_campaign.py)):** Central tendency and spread use the median ($\tilde{T}$) and linear interquartile range ($\mathrm{IQR} = Q_3 - Q_1$). Strong-scaling speedup $S_{\mathrm{dist}}(p, n) = \tilde{T}(1, n) / \tilde{T}(p, n)$ and efficiency $E_{\mathrm{dist}}(p, n) = S_{\mathrm{dist}}(p, n) / p$ are computed relative to the dedicated $p = 1$ strong-scaling sub-campaign and omitted (`N/A`) whenever a configuration has fewer than `5` completed measured repetitions (such as $p = 10$).
- **Cryptographic Provenance ([`results/analysis/analysis_provenance.json`](results/analysis/analysis_provenance.json)):** Indexes all `19` raw CSV, metadata JSON, analysis script, derived CSV, and PNG plot artifacts by SHA-256 digest, verified read-only via [`scripts/verify_provenance.py`](scripts/verify_provenance.py).

---

## 8. Repository Structure

```text
Assignment04_Abel/
├── README.md                         # Project overview, setup, and reproducibility commands
├── SPEC.md                           # Technical specification and experimental contract
├── requirements.txt                  # Pinned Python dependencies
├── baselines/                        # Frozen unmodified A1 (C) and A3 (C/OpenMP) source snapshots
├── data/                             # Deterministic dataset guide, SHA256SUMS.txt, fixtures, and baselines
├── src/                              # Distributed Dask multiplication engine, CLI, watchdog, and adapters
├── scripts/                          # Dataset generator, verifiers, baseline/scaling runners, and analysis
├── tests/                            # Automated unit and contract test suite (200 tests)
├── results/                          # Immutable raw CSVs, metadata manifests/sidecars, and analysis outputs
└── report/                           # Canonical academic report (Assignment04_Report.pdf, .tex, plots/)
```
