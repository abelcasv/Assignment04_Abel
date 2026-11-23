# SPEC.md — Assignment 4: Distributed Matrix Multiplication

**Course:** BIG DATA — Grado en Ciencia e Ingeniería de Datos, ULPGC
**Academic year:** 2026–2027
**Assignment:** 4 — Distributed Matrix Multiplication
**Specification basis:** `BD_Individual_Assignments_2026_2027.pdf`, revised proposal 2026–2027.
**Status:** Binding Technical Contract and Implementation Specification. Formally synchronized with the accepted experimental protocol in `docs/experimental_protocol.md`. Framework, architecture, provenance, and experimental protocol are decision-gated; they must not be guessed or altered by an AI agent.

---

## 1. Purpose and scope

The study must investigate how a distributed matrix-multiplication implementation scales with workers or nodes, considering:

- Computation cost.
- Communication and data-transfer cost.
- Distributed-framework and process/scheduling overhead.
- Resource consumption, especially memory per worker or node where measurable.

The distributed implementation must be compared, for feasible common problem sizes, with:

1. A **validated basic sequential implementation** reused from an earlier assignment (Assignment 1).
2. A **validated parallel implementation** reused from an earlier assignment (Assignment 3).

The comparison must use the same logical input matrices wherever feasible and document differences in hardware, algorithm, numeric type, storage, library backend and timing boundary that prevent a fully isolated comparison.

The goal is a defensible controlled empirical study, not a novel distributed algorithm or a publication-level contribution.

## 2. Source-derived requirements and interpretation

### 2.1 Assignment 4 requirements (official PDF, page 7)

| ID | Requirement | Acceptance evidence |
|---|---|---|
| A4-01 | Implement matrix multiplication using a distributed computing framework. | Working, documented distributed implementation; framework and version recorded. |
| A4-02 | Explain matrix partitioning, placement of computation, and collection or storage of the result. | Architecture and data-flow explanation, supported by code/configuration. |
| A4-03 | Compare against a validated basic sequential version and a parallel version for feasible common sizes. | Matched-input measurements and a clear comparison table/plot; baseline provenance and differences documented. |
| A4-04 | Test multiple worker or node configurations where resources allow. | Raw measurements identify workers, processes and physical machines separately. Feasibility limits are explained. |
| A4-05 | Use sufficiently large feasible inputs to investigate scaling. | Declared increasing size sequence, runtime and memory budgets, largest feasible size and attempted sizes reported. |
| A4-06 | Keep total problem size fixed when measuring strong scaling. | Strong-scaling results vary resources while keeping `n`, inputs, dtype and algorithm/configuration fixed. |
| A4-07 | Report end-to-end time and define the timing boundary. | Explicit start/stop events and raw end-to-end wall-clock observations. |
| A4-08 | Where supported, report computation, setup, transfer and result-collection costs. | Separate phase measurements or a documented statement that a phase cannot be isolated. Overlaps are explained. |
| A4-09 | Report resource counts, memory per worker/node where measurable, data-transfer measurements where available, and tools. | Resource metadata, named memory/transfer metrics and tool/version documentation. |
| A4-10 | If network cost cannot be isolated, state this limitation and analyze available observations without inventing measurements. | Explicit limitations in results/discussion; no fabricated network metrics. |
| A4-11 | A local execution with multiple workers is permitted only if the instructor permits it. | Written/otherwise explicit instructor permission recorded before any local multi-worker experiment. The environment is labelled as single-machine. |
| A4-12 | Do not describe a local multi-worker experiment as inter-machine networking or multi-machine memory. | Accurate terminology throughout metadata, figures, README and report. |

### 2.2 Common protocol requirements (official PDF, pages 2–3)

| ID | Requirement | Acceptance evidence |
|---|---|---|
| COM-01 | Focused research question; distinguish hypotheses from results. | Introduction states question, changed factor, outcomes and predictions. |
| COM-02 | Same logical matrices for each method comparison where possible. | Shared input files or verified common generator; seed, range, dimensions, dtype and hashes recorded. |
| COM-03 | Record CPU, physical/logical cores, RAM, OS, language/library versions, compiler settings and relevant runtime configuration. Include numerical-library internal threads. | Environment manifest and run metadata. |
| COM-04 | Make algorithm/arithmetic comparable where required; report unavoidable storage/runtime differences. Do not treat different machines as a language-only comparison. | Method comparison table and limitations. |
| COM-05 | Define timer scope; for end-to-end runs document data loading, setup, transfers and result collection. Never merge different timing boundaries without qualification. | Timing contract and method-specific boundary notes. |
| COM-06 | Use a suitable monotonic timer and ensure results are consumed or checked. | Timer implementation and output-consumption/checksum code. |
| COM-07 | Use warm-ups when framework startup, compilation or caching matters; exclude and count warm-ups. | Warm-up protocol and raw metadata. |
| COM-08 | Recommended starting point: at least five measured repetitions per configuration; justify fewer and report uncertainty. | Five or more timed repetitions/configuration, or a written budget-based justification. |
| COM-09 | Retain every raw measurement; report central value (preferably median), variability (e.g. IQR), units and repeat count. Explain exclusions. | Immutable machine-readable raw data plus analysis scripts and summary tables. |
| COM-10 | Validate before benchmarking with known small cases, including identity and zero, and a trusted reference. Define absolute/relative tolerances for floating point. | Correctness fixtures, reference comparison and documented tolerances. |
| COM-11 | Choose feasible increasing sizes and declare runtime/memory budgets before large runs. | Pre-registered experiment plan, budget limits and size sequence. |
| COM-12 | Report largest tested size that meets limits separately from largest attempted; report timeouts and memory failures without inventing execution times. | Dataset status fields and report statement. |
| COM-13 | Measure time and memory; identify metric and tool and discuss limitations. Distinguish different memory quantities. | Per-worker/node/driver metric definitions and limitations. |
| COM-14 | Reproducibility: source, exact README commands, dependencies/versions, configuration, input generation/acquisition instructions, raw results and analysis scripts. | Repository and independent reproduction instructions. |
| COM-15 | Label every configuration with matrix size, method, numeric type, repetition and relevant worker/thread count; for distributed runs also capture topology. | Raw-data schema and metadata validation. |

### 2.3 Report and submission requirements (official PDF, pages 8–9)

The final report must be in English and use the course research-report structure:

1. Title and short abstract: question, method, main result and scope.
2. Introduction and objective: motivation, focused question and hypothesis.
3. Background and references: relevant distributed algorithms/framework concepts and supporting sources.
4. Methodology: implementations, inputs, environment, correctness, measurement protocol and budgets.
5. Results: labelled tables/plots, units, repetitions and variability.
6. Discussion: evidence-based explanations and limitations.
7. Conclusions: what the study establishes within its tested scope and what remains unresolved.
8. References and materials: full citations and location/description of reproducibility artifacts.

Submission package:
- PDF report with student name, assignment number and research title.
- Source code, README, dependencies/environment details, exact build and run instructions.
- Raw results, input generation/acquisition instructions and scripts that reproduce analysis and figures.
- Reused materials identified by exact version.
- GitHub repository is mandatory under the course proposal.

Figures must have labelled axes and units, identify configurations and explain what they show. Borrowed ideas, code, text and figures must be cited consistently (e.g. IEEE), and each source must exist and support the claim.

AI use must be disclosed by tool, purpose and affected parts of the work. The student remains responsible for understanding, validating and defending the implementation, measurements and conclusions. Generated content is not evidence. Fabricated results or references are unacceptable.

The course-plan window is **Week 12, 23–27 November 2026**. The PDF says the exact closing day and time are still to be confirmed; do not invent them.

### 2.4 Domain-Specific Authority Model and Normative Precedence

To eliminate ambiguities where procedural engineering rules might appear to override experimental parameters, this project strictly applies a **domain-specific authority model**:

1. **Official Course Document (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`) and Explicit Written Instructor Clarifications:** Highest academic authority; governs all learning objectives, submission deliverables, assessment criteria, and administrative restrictions (including the mandatory requirement for instructor permission prior to executing multiple local workers).
2. **`SPEC.md` (This Document):** Normative project specification; governs project architecture, technical contracts, baseline interfaces, and acceptance requirements.
3. **`docs/experimental_protocol.md`:** The **single detailed source of truth** for all experimental parameters, matrix dimensions, block partitioning, timing boundaries, measurement contracts, telemetry metrics, numerical correctness tolerances, resource budgets, and failure handling policies, within the boundaries of the PDF and `SPEC.md`. Formally accepted under Gate G4.
4. **`CLAUDE.md`:** Binding agent engineering rules, code quality, safety constraints, permission gates, Git discipline, and operational procedures. It **cannot** silently redefine or weaken experimental parameters.
5. **Task Prompts:** Authorize only the operational actions and deliverables explicitly assigned to the immediate phase; cannot override or weaken higher-tier documents.

**Conflict Resolution Rule:** If any genuine contradiction arises across these documents, the agent must stop immediately, document the conflicting passages, and request clarification from the student before proceeding.

### 2.5 Canonical and Archival Document Definitions

- **Canonical Documents:** `CLAUDE.md`, `SPEC.md`, and `docs/experimental_protocol.md` are the sole authoritative, canonical project documents.
- **Archival Drafts:** Historical drafts `CLAUDE_UPDATED.md` and `SPEC_UPDATED.md` have been fully consolidated into the canonical governance documents and removed from the active repository; they are superseded and are not active repository documents.
- **Superseded Drafts:** Any prior protocol working draft (including `docs/Assignment04_Experimental_Protocol_Final.md`) is fully superseded by `docs/experimental_protocol.md` and must never be treated as a competing source of truth.

---

## 3. Research question and hypotheses

### 3.1 Research question

> How does the end-to-end wall-clock performance, process memory footprint, and scheduling/communication overhead of a distributed 2D block matrix multiplication implementation using Dask Distributed scale as worker resources increase on a single heterogeneous host, and how does it compare with validated sequential (Assignment 1) and shared-memory parallel OpenMP (Assignment 3) implementations on identical logical input matrices and numeric types?

### 3.2 Hypotheses (predictions, not assumed results)

- **$H_1$ (Framework Overhead at Small Sizes):** For small matrices ($n \le 500$), Dask framework overhead (graph construction, serialization, loopback IPC, and scheduling) will dominate execution time, causing Dask to be slower than the basic sequential baseline ($R_{\text{seq}} = T_{\text{seq}} / T_{\text{dist}} < 1$).
- **$H_2$ (Strong Scaling at Large Sizes):** For larger matrices ($n \ge 1000$), cubic arithmetic work will overtake graph overhead, resulting in positive speedup ($S_{\text{dist}}(p) > 1$). However, parallel efficiency will decay ($E_{\text{dist}}(p) < 1$) due to memory bandwidth contention, loopback serialization, and thread/core heterogeneity.
- **$H_3$ (Worker Saturation and Non-Monotonic Scaling):** Beyond an optimal worker count, adding further workers on a single host will cease to improve performance or will degrade it due to scheduler contention, loopback socket latency, and scheduling onto lower-throughput Efficient cores.
- **$H_4$ (Contextual System Comparison):** Differences between Dask/NumPy/BLAS and C/OpenMP reflect differing runtime backends, memory layouts, and timing boundaries, representing a contextual system comparison rather than an isolated algorithmic speedup.

---

## 4. Architectural and experimental decisions register

No AI agent may silently alter these decisions without explicit student authorization.

### DEC-01 — Instructor permission for local multiple workers
- **Status:** **AUTHORIZED (2026-11-23) for local single-host multi-worker execution ($p > 1$)** via [`docs/decisions/DEC-01-instructor-permission.md`](docs/decisions/DEC-01-instructor-permission.md). *(Historical record: previously `BLOCKED / NOT COMPLETED` for $p > 1$ during Phases 1–10 while written instructor permission was absent; single-worker $p=1$ pilot and Phase 8 campaign completed under Gates G7–G9 and remain preserved intact.)*
- **Rule:** Local multi-process execution with multiple workers ($p \in \{2, 4, 8, 10\}$) is permitted only when written authorization evidence exists in `docs/decisions/` (enforced fail-closed by `verify_multiworker_permission(p)`).
- **Strong-Scaling Compliance & Topology Boundary:** `p=1` alone does not constitute a strong-scaling sweep. Authorized multi-worker runs ($p \in \{1, 2, 4, 8, 10\}$) execute on **one physical machine** (`physical_machine_count = 1`) over loopback TCP sockets (`127.0.0.1`) and shared DRAM/L3; they must be labeled strictly as local multi-process execution and **never** as multi-node cluster scaling or inter-machine network throughput. Multi-node execution remains **BLOCKED / NOT AVAILABLE**.

### DEC-02 — Available deployment topology
- **Status:** **Confirmed Single Physical Machine**.
- **Hardware:** Intel Core i5-1335U (10 physical cores: 2 P-cores, 8 E-cores; 12 logical processors), 16 GB RAM, Windows 11 Home 64-bit.
- **Infrastructure:** No physical multi-node cluster, university HPC, or cloud infrastructure is currently accredited. Multi-node execution is **BLOCKED / NOT AVAILABLE**. Multi-worker local runs must be labeled accurately as single-machine multi-process execution and never as physical network or cluster scaling.

### DEC-03 — Distributed framework selection
- **Status:** **Approved (ADR-001)**: **Dask Distributed**.
- **Architecture:** 2D block matrix multiplication with intermediate binary-tree reduction for output blocks.
- **Local Kernel:** `numpy.matmul` executing on contiguous `float64` blocks.
- **Threading Control:** Strictly 1 thread per worker (`threads_per_worker=1`). Internal BLAS/OpenMP threading is restricted via environment variables: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`. Phase 6 dynamically verified via `threadpoolctl` that Worker 0 loaded `libscipy_openblas64` (v0.3.34.106.0) with internal thread count pinned to 1, worker framework `nthreads=1`, and environment variables inherited prior to worker process launch. No OpenMP runtime was loaded in the worker.

### DEC-04 — Baseline provenance and immutability
- **Status:** **Frozen and Certified in Phase 2.1**.
- **Assignment 1 Sequential:** Commit SHA `90681e85949b6d65bdbcfef702b6306b0003f971`. Frozen snapshot in `baselines/a1_sequential/` (`benchmark.c`, `Makefile`).
- **Assignment 3 OpenMP:** Commit SHA `23cffbc82c155b6c3de24b3c149b509b63fc870b`. Frozen snapshot in `baselines/a3_openmp/` (`parallel_benchmark.c`, `Makefile`).
- **Immutability Contract:** Snapshots are strictly read-only. No source files, algorithms, or flags may be altered.
- **Runtime Directories:** Populated via adapters `scripts/prepare_a1_runtime.py` (`build/a1_runtime/data/`) and `scripts/prepare_a3_runtime.py` (`build/a3_runtime/data/dense/` and `data/validation/`).
- **Execution Paths:** Executables must compile to `build/bin/`. Invoked from repository root as:
  ```powershell
  .\build\bin\a1_benchmark.exe build/a1_runtime
  .\build\bin\a3_parallel_benchmark.exe build/a3_runtime
  ```
- **A1 Optimization Inspection (Phase 5 Assignment):** The A1 harness consumes only `c[0]`. Phase 4 remains strictly implementation-only without compilation. In **Phase 5 (Correctness / Build Validation)**, upon authorized compilation of `baselines/a1_sequential/benchmark.c` under original `-O3` flags, inspect the generated assembly/disassembly (`gcc -S` or `objdump -d`) of `build/bin/a1_benchmark.exe` to verify whether the complete $O(n^3)$ multiplication survived dead-code elimination. Do not alter A1 source code or compiler flags. Do not benchmark.

### DEC-05 — Input and common-size policy
- **Status:** **Frozen and Certified in Phase 2.1; Storage Policy Option B Approved**.
- **Dataset:** Benchmark matrices stored locally in `data/dense/` and verified with 0 errors via `scripts/verify_dataset.py` against `data/SHA256SUMS.txt`.
- **Storage Decision (Option B):** The 14 generated dense CSV matrices (~310 MB) are deliberately excluded from Git via `.gitignore` (`/data/dense/`) to prevent repository bloat and avoid GitHub file size warnings (>50 MB). Their deterministic generator (`scripts/generate_dataset.py`) and cryptographic manifest (`data/SHA256SUMS.txt`) are version-controlled. See `data/README.md` for clean-clone reconstruction instructions.
- **Properties:** Generated by `scripts/generate_dataset.py`, seed $42+n$, uniform $[0.0, 1.0)$, IEEE 754 binary64 `float64`, `%.17g` precision, Unix LF line endings.
- **Size Allocation:**
  - Common 3-Way Benchmark: $n \in \{250, 500, 750, 1000\}$ (A1, A3, Dask).
  - Extended Scaling: $n \in \{1500, 2000\}$ (A3, Dask; A1 excluded due to code limit $n \le 1000$).
  - Auxiliary: $n = 100$ (A1 compatibility check only; excluded from 3-way comparisons).
- **Immutability:** Existing frozen datasets in an active workspace must never be regenerated or overwritten. A fresh clone must materialize and verify the dataset before running benchmarks.

### DEC-06 — Timing contract, resource budgets, and watchdog semantics
- **Status:** **Approved (Gate G4)** in `docs/experimental_protocol.md`.
- **Timing Boundary:** Primary Dask metric is end-to-end wall-clock time ($t_0 \to t_1$), starting immediately before 2D block partitioning/scattering, including task graph submission, computation, inter-worker transfer, tree reduction, gather to driver, dense assembly, and full-matrix checksum consumption (`float(np.sum(C, dtype=np.float64))`).
- **Excluded Setup:** Cluster startup (`cluster_startup_s`) and CSV input loading (`input_load_s`) are measured and recorded outside the multiplication timer.
- **Correctness Check:** Element-by-element numerical verification is performed outside the timed region.
- **Budget Values and Explicit Status Classification:**
  - **Per-sample timeout (120 s):** *Student-selected operational safety limit* and *provisional limit requiring pilot validation*. Not an empirically validated budget.
  - **Campaign ceiling (10 h):** *Student-selected operational planning envelope*. Not an empirical measurement.
  - **Process aggregate RSS ceiling (2.5 GiB):** *Student-selected operational safety limit* and *provisional limit requiring pilot validation*.
  - **Host available-memory floor (1.0 GiB):** *Student-selected operational OS safety limit*.
  - **Watchdog sampling interval (250 ms):** *Student-selected operational monitoring parameter*. Measures sampled peak aggregate RSS; may miss transient peaks shorter than 250 ms.
- **Watchdog Accounting Semantics:**
  - The sum of process RSS is not equivalent to unique physical RAM usage because shared pages are counted across multiple PIDs.
  - Process identities must be tracked using unique operating-system PIDs mapped to logical roles (driver, scheduler, nanny, worker). In the verified Windows in-process `LocalCluster` deployment, the Driver process hosts the Client, Scheduler, and LocalCluster Nanny management instance, while Worker 0 runs in an independent child process (2 distinct OS PIDs observed for $p=1$). Project accounting records actual unique PIDs observed per run rather than imposing an unverified universal formula.
  - When `memory_limit=None`, Dask's internal auto-spill/pause mechanisms are inactive; the project `ResourceWatchdog` (2.5 GiB sampled aggregate RSS ceiling, 1.0 GiB host available-RAM floor) is the specified active operational safeguard (a sampled check, not a hard OS process limit).
  - **Safe Stop Procedure:** On threshold breach, immediately cancel client tasks (`client.cancel`), gracefully close cluster/client within 5 s, forcefully terminate lingering child PIDs (`terminate() / kill()`), and log status as `OUT_OF_MEMORY` or `ABORTED_WATCHDOG`.
  - **Phase 7 Pilot Specification (Stage 1 Formalization Approved):** Formally defined in `docs/experimental_protocol.md` §10.5. Evaluates `dask_block_2d` exclusively across $n \in \{250, 500\}$ with $b=250$, single worker $p=1$ (single host, 2 OS PIDs, 1 thread/worker, OpenBLAS `num_threads=1`), 1 warm-up (`is_warmup=true`) + 2 measured repetitions per size (6 runs total). Enforces:
    1. **15-Minute Global Session Budget (900.0 s):** Overall monotonic deadline covering input loading, cluster startup, runs, validation, and sidecar; maps to `ABORTED_BUDGET` if reached.
    2. **120.0 s Per-Sample Timeout:** Enforced on end-to-end distributed execution ($T_{\text{Dask\_e2e}}$); maps to `TIMEOUT` for that sample.
    3. **60.0 s Startup Timeout:** Governs Client connection and worker registration (`wait_for_workers`); maps to `FAILED` if exceeded while session budget remains.
    4. **Uninterruptible Synchronous LocalCluster Construction:** `LocalCluster(...)` has no native Dask timeout parameter; overruns are detected post-return and trigger immediate cleanup.
    5. **Nominal Cleanup Timeout (5.0 s) vs OS Behavior:** Shared budget across Client, Cluster, and child termination (`terminate() / kill()`, excluding driver PID); operational software envelope, not a hard real-time kernel limit.
    Results logged append-only with warm-ups identified and excluded from statistics. Pilot execution requires independent Stage 2 authorization.

### DEC-07 — Optional weak scaling
- **Status:** **Deferred and BLOCKED**.
- **Relation:** Theoretical equal-work formula for dense cubic multiplication is $n(p) = n(1) \cdot p^{1/3}$.
- **Condition:** Requires completion and audit of the strong scaling study, approval of new input sizes, generation of verified input files, and explicit authorization.

---

## 5. Logical architecture and distributed implementation

Detailed architecture specifications are defined normatively in `docs/experimental_protocol.md`.

### 5.1 Matrix representation and 2D block partitioning
- Matrices are square dense $n \times n$ arrays in IEEE 754 binary64 (`float64`).
- Partitioned into 2D blocks using reference block size $b = 250$.
- Number of blocks per axis: $N_b = \lceil n/b \rceil$.
- Blocks are contiguous in C-order in memory.
- **Edge Blocks:** When $n$ is not divisible by $b$, edge blocks use exact truncated dimensions:
  $\text{block\_rows} = \min(b, n - \text{row\_start})$, $\text{block\_cols} = \min(b, n - \text{col\_start})$. Zero-padding is strictly forbidden.

### 5.2 Distributed computation and tree reduction
- Output block algebraic formulation:
  $$C_{i, j} = \sum_{k=0}^{N_b-1} A_{i, k} \times B_{k, j}$$
- For each output block $(i, j)$, $N_b$ independent multiplication tasks produce partial products $P_{i, j, k} = A_{i, k} \times B_{k, j}$ using `numpy.matmul`.
- Partial products are combined via a **binary-tree reduction**, producing $\lceil \log_2 N_b \rceil$ reduction stages and $N_b - 1$ addition tasks per output block.
- Final output blocks $C_{i, j}$ are gathered to driver memory and assembled into a contiguous dense $n \times n$ matrix $C$.

### 5.3 Output consumption and validation
- The timer stops only after a full-matrix consumption operation forces materialization of every element:
  `checksum = float(np.sum(C, dtype=np.float64))`
- Numerical validation against reference data occurs after timer stop.

---

## 6. Experimental design contract

Normative reference: `docs/experimental_protocol.md`.

### 6.1 Comparison methods and mapping
- `seq_basic`: A1 basic triple loop ($i$-$j$-$k$).
- `seq_ikj`: A3 optimized loop ($i$-$k$-$j$).
- `omp_ikj`: A3 OpenMP parallel loop ($i$-$k$-$j$). Historical summaries in `data/baselines/a3_parallel_results_summary.csv` span $t \in \{1, 2, 4, 8, 10\}$ threads (with single-threaded $t=1$ paired against A4 $p=1$ in Table 2 and multi-threaded $t=10$ matching the 10 physical cores reported as shared-memory context).
- `omp_tiled_b32`: A3 OpenMP tiled variant ($32 \times 32$ tiles, optional secondary).
- `dask_block_2d`: Dask Distributed 2D block implementation.

### 6.2 Strong scaling sweep
- Evaluated on $n \in \{1000, 1500, 2000\}$ with fixed $b = 250$.
- Worker configurations: $p \in \{1, 2, 4, 8, 10\}$ (subject to DEC-01 permission for $p > 1$).
- All problem parameters remain invariant during each sweep.

### 6.3 Repetition and run-order protocol
- Exactly 1 unmeasured warm-up per configuration (`is_warmup=true`).
- Exactly 5 measured repetitions per configuration ($N = 5$).
- Configurations executed in a reproducible randomized sequence using seed `20261113`.
- Outlier trimming or deletion is strictly prohibited; all raw samples are preserved.

---

## 7. Timing, overhead, and telemetry definitions

### 7.1 Primary distributed timing boundary
- **Start ($t_0$):** Sampled via `time.perf_counter()` immediately before input matrices $A$ and $B$ are sliced into blocks for the current repetition.
- **Includes:** 2D slicing, data scatter, task graph submission, worker multiplication, tree reduction, gathering blocks, dense assembly, and full-matrix checksum summation.
- **Stop ($t_1$):** Sampled via `time.perf_counter()` immediately after checksum completion.
- **Duration:** $T_{\text{dist\_e2e}} = t_1 - t_0$.

### 7.2 Telemetry and transfer metrics
- Object `nbytes` represents in-memory uncompressed array sizes, NOT physical network bytes.
- Dask transfer telemetry (e.g. `incoming_transfer_bytes`, `outgoing_transfer_bytes`) represents application payload over loopback sockets and must be labeled as "Dask-reported payload bytes".
- Physical network interface metrics must not be reported on a single host.
- Overlapping task durations must never be summed into wall-clock time.

---

## 8. Correctness and numerical contract

Normative reference: `docs/experimental_protocol.md`, Section 9.

### 8.1 Correctness fixtures
Prior to benchmarking, implementations must pass all validation fixtures:
1. Identity matrix: $A \times I = A$.
2. Zero matrix: $A \times 0 = 0$.
3. Analytical $4 \times 4$ integer matrix with exact known output.
4. Irregular edge-block fixture: $4 \times 4$ matrix with $b = 3$.
5. Benchmark reference: $n = 250$ element-by-element cross-check against `data/validation/reference/C_250.csv`.
6. Non-finite check: Zero `NaN` or `Inf` values permitted.

### 8.2 Trusted reference `C_250.csv`
- Certified SHA-256: `598e9a9fc8f0c8db1d5da277f4d93561ad83e2df77e3fcb5f968f67b8033909e`.
- Provenance: Generated via `scripts/generate_dataset.py` using pure Python standard library loops, independently of NumPy, BLAS, or Dask. It is a calculated numerical reference.

### 8.3 Acceptance tolerances
Floating-point comparison criterion:
$$|C_{\text{test}}[i, j] - C_{\text{ref}}[i, j]| \le \text{atol} + \text{rtol} \cdot |C_{\text{ref}}[i, j]|$$
- Absolute tolerance: $\text{atol} = 10^{-9}$.
- Relative tolerance: $\text{rtol} = 10^{-12}$.
- Tolerances must be validated in Gate G6 and never relaxed after observing failures.

---

## 9. Memory and resource measurement contract

Normative reference: `docs/experimental_protocol.md`, Section 7.4.

Process memory metrics must be clearly distinguished:
1. **Per-Process RSS:** Physical resident memory of an individual PID.
2. **Sampled Aggregate RSS Sum:** Simultaneous sum of RSS across all benchmark processes (driver, scheduler, nannies, workers), sampled every 250 ms. Conservative proxy; double-counts shared pages.
3. **Peak Sampled Aggregate RSS:** Maximum observed simultaneous aggregate RSS during the run.
4. **Peak Working Set (`peak_working_set_mib`):** In `A1`/`A3` C baselines, the OS process-lifetime peak Working Set from `GetProcessMemoryInfo()`; in Dask Windows runs (`psutil`), `.rss` reports the Windows process Working Set, and `peak_working_set_mib` records the maximum single-PID sampled Working Set within the sample window (cannot be summed across processes).
5. **Dask Managed Memory:** Memory tracked by Dask workers for task results.
6. **Structural Memory:** Theoretical $3 \cdot 8n^2$ bytes for matrices $A, B, C$.
7. **Host Available Memory:** System-wide available physical memory.
8. **Dask `memory_limit`:** Configured as `None` in `ClusterManager`, leaving Dask per-worker spill/pause inactive. Primary operational safety is actively enforced across all cluster PIDs by the global 2.5 GiB aggregate RSS watchdog and 1.0 GiB host available-RAM floor.

---

## 10. Raw data, metadata, and analysis

Normative schema and formulas: `docs/experimental_protocol.md`, Section 11.

- Raw data stored in `results/raw/raw_measurements.csv` (1 row per sample, normative 44-column schema, append-only, preserving all statuses: `COMPLETED`, `TIMEOUT`, `OUT_OF_MEMORY`, `ABORTED_WATCHDOG`, `ABORTED_BUDGET`, `NOT_RUN_AFTER_OOM`, `NOT_RUN_AFTER_TIMEOUT`, `FAILED`).
- Campaign metadata stored in companion JSON manifests (with unique `run_id` formatted as `pilot_<timestamp>_<uuid.uuid4().hex>` or `campaign_<timestamp>_<uuid.uuid4().hex>`, providing 122 random bits of entropy in a 32-hexadecimal character representation per RFC 4122 §4.4, and recording process hierarchy, stack versions, status, and failure reason).
- Statistical summaries use Median ($\tilde{T}$) and linear Interquartile Range ($\text{IQR} = Q_3 - Q_1$).
- Strong scaling speedup: $S_{\text{dist}}(p, n) = \tilde{T}_{\text{dist}}(1, n) / \tilde{T}_{\text{dist}}(p, n)$.
- Strong scaling efficiency: $E_{\text{dist}}(p, n) = S_{\text{dist}}(p, n) / p$.
- Cross-method ratios: $R_{\text{seq}}(p, n) = \tilde{T}_{\text{seq}}(n) / \tilde{T}_{\text{dist}}(p, n)$ (reported as contextual system comparisons, not pure algorithmic speedups).

---

## 11. Development quality, testing, and version control

Normative standards: `CLAUDE.md`, Section 9.

- **Engineering Quality:** Clean Code, KISS, YAGNI, sensible DRY, Boy Scout rule, separation of concerns, explicitness, and reproducibility. Immutable baselines and datasets.
- **Source Code Standards:** Explicit interfaces, input validation at boundaries, structured error handling, PEP 8 / type hints for Python, strict memory and compiler safety for C.
- **Testing & Static Checks:** Focused deterministic unit/contract tests for new features and bug fixes; edge cases and invalid inputs; only authorized checks executed; tests unexecuted due to phase constraints marked `NOT RUN / NOT VERIFIED`; no Husky or default Node tooling; no unapproved dependencies.
- **Incremental Local Commits:** Commit completed logical units progressively; standing authorization for ordinary local commits under the 10-step safety procedure in `CLAUDE.md`, Section 9.4; no blanket staging (`git add .` prohibited); explicit paths only.
- **Commit Messages:** Conventional Commits (`<type>(<scope>): <imperative summary>` in English, max 72 chars).
- **Branching & History Safety:** Stable `main` branch; short-lived branches for isolated/risky tasks; no unauthorized branch switching, merging, history rewriting, amending, rebasing, resetting, or force-pushing; pushing always requires separate explicit authorization.
- **Artifact Handling:** Build outputs, executables, object files, caches, venvs, temporary runtime directories (`build/a1_runtime/`, `build/a3_runtime/`), secrets, PDFs, and private prompts excluded from Git; frozen baselines and inputs strictly immutable; dense benchmark matrices in `data/dense/` excluded from Git under Option B (see `data/README.md`); raw measurements append-only.

---

## 12. Repository structure

```text
Assignment04/
├── CLAUDE.md                 # Mandatory engineering rules
├── LICENSE                   # MIT License
├── SPEC.md                   # Technical specification and contract
├── README.md                 # Project documentation and reproducibility guide
├── requirements.txt          # Pinned Python dependencies for clean-clone setup
├── docs/
│   ├── experimental_protocol.md  # Single detailed source of truth for experiment
│   ├── decisions/            # ADR-001 and instructor permission record
│   ├── provenance/           # Baseline and input provenance records
│   └── reference/            # Course specification PDF
├── baselines/                # Frozen immutable A1/A3 snapshots
│   ├── a1_sequential/
│   └── a3_openmp/
├── build/                    # Runtime directories and compiled binaries
│   ├── bin/                  # Target compiled executables
│   ├── a1_runtime/           # Prepared A1 runtime data
│   └── a3_runtime/           # Prepared A3 runtime data
├── data/                     # Certified common input datasets
│   ├── baselines/            # Tracked copies of frozen A1/A3 summary CSVs
│   ├── dense/                # Common benchmark matrices (generated on-demand, .gitignored)
│   ├── validation/           # Correctness fixtures and reference C_250.csv (tracked)
│   ├── README.md             # Operational dataset reconstruction guide
│   └── SHA256SUMS.txt        # Certified checksum manifest (tracked)
├── src/                      # Distributed Dask implementation and adapters
├── scripts/                  # Dataset generator, verification, and runtime adapters
├── tests/                    # Focused correctness and contract tests
├── results/                  # Experimental results
│   ├── raw/                  # Immutable raw measurements CSV
│   ├── metadata/             # Run manifests and environment sidecars
│   └── analysis/             # Derived summary tables and figures
└── report/                   # Final research report and materials
```

---

## 13. Acceptance gates

| Gate | Identifier | Status | Prerequisites to Pass |
|---|---|---|---|
| G0 | Discovery | **PASSED** | Environment, host, baseline repos, and course requirements inventoried. |
| G1 | Permission / Topology | **PASSED (Single-Host Local `p >= 1`, 2026-11-23) / BLOCKED (Multi-Node)** | Historically blocked for $p > 1$ during Phases 1–10; authorized on 2026-11-23 via `docs/decisions/DEC-01-instructor-permission.md` for single-host local multi-process strong scaling (`physical_machine_count=1`). Multi-node remains **BLOCKED / NOT AVAILABLE**. |
| G2 | Framework / Architecture | **PASSED** | Dask Distributed and 2D block reduction architecture approved (ADR-001). |
| G3 | Provenance / Inputs | **PASSED** | Baseline snapshots and common inputs frozen and certified. |
| G4 | Protocol / Budgets | **PASSED** | `docs/experimental_protocol.md` approved and synchronized. |
| G5 | Implementation (Prompt 4) | **PASSED** | Dask implementation (`src/dask_matrix/`), CLI, watchdog, logger, sidecar, and runtime adapters implemented and unit-tested. |
| G6 | Correctness & Build (Prompt 5) | **PASSED** | Baselines compiled to `build/bin/`, A1 `-O3` assembly inspected, correctness fixtures (`4×4` exact `0.0`) and $n=250$ reference (`max_abs_diff = 1.137e-13` for $b=250$ / `1.208e-13` for $b=100$, well within `atol=1e-9, rtol=1e-12`; aggregate checksum diff `0.0`) verified. |
| G7 | Environment (Prompt 6) | **PASSED** | Windows LocalCluster 2-PID process mapping, 1 thread/worker, single-thread OpenBLAS via `threadpoolctl`, and idle RAM telemetry calibrated. |
| G8 | Pilot & Benchmark (Prompts 7–8 + DEC-01 Strong Scaling) | **PASSED** | Phase 7/8 `p=1` campaign (48 historical rows preserved byte-for-byte) plus DEC-01 authorized local multi-worker pilot (`p=1` vs `p=2`), strong-scaling sweep (`p ∈ {1, 2, 4, 8, 10}`, `n ∈ {1000, 1500, 2000}`, 90 rows appended), and separate `n=1000` diagnostic pass logged cleanly. |
| G9 | Data Audit and Analysis | **PASSED** | 138 raw records (48 historical + 90 strong-scaling) audited; derived CSVs (Phase 8 + strong-scaling summary), 3 publication figures, and 13/13 SHA-256 provenance checks generated and verified cleanly by `scripts/analyze_campaign.py` and `scripts/verify_provenance.py`. |
| G10 | Final Audit | **PASSED** | Code contracts, unit tests, clean-clone reproducibility, `README.md`, `report/report.md`, `report/Assignment04_Report.tex`/`.pdf`, and specs reconciled. |

---

## 14. Definition of done

The project is complete and ready for submission only when:

1. The distributed Dask implementation performs 2D block matrix multiplication with tree reduction, and its partitioning, placement, and result gathering are documented.
2. Sequential, OpenMP, and distributed methods are compared on feasible common sizes: under the frozen-baseline and DEC-01 rules, the report presents cross-system contextual comparisons against preserved Assignment 1/3 summaries (`data/baselines/`), while common deterministic input matrices in `data/dense/` and runtime adapters remain available for authorized execution.
3. Strong scaling is evaluated with fixed problem size across permitted worker configurations ($p=1$ baseline, and $p \in \{2, 4, 8, 10\}$ if permitted).
4. Workers, OS processes, and physical machines are distinguished accurately throughout code, metadata, and report.
5. End-to-end timing boundaries, optional phase timings, and communication limitations are reported transparently.
6. Execution time, process memory, data transfer, and budgets are measured and documented.
7. Correctness validation, numerical reference provenance, and tolerances are verified.
8. The English report follows the course structure, cites genuine sources, and discusses findings and limitations with academic rigor.
9. All code, raw data, scripts, and documentation form an independently reproducible package.
10. Generative AI tool use is fully disclosed in accordance with course policy.
