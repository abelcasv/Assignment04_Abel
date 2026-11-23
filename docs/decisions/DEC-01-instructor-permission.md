# DEC-01 — Instructor Permission for Local Multi-Worker Execution (`p > 1`)

- **Decision ID:** DEC-01 (Gate G1)
- **Original Status (Phases 1–10):** `BLOCKED / NOT COMPLETED` for `p > 1` (single-worker `p = 1` exclusively permitted while written instructor authorization was pending)
- **Updated Status (2026-11-23):** **AUTHORIZED — Local Single-Host Multi-Process Execution (`p > 1`) Permitted**
- **Multi-Node Cluster Status:** Remains **BLOCKED / NOT AVAILABLE** (no physical multi-node cluster, university HPC, or cloud cluster accredited)

---

## 1. Historical Traceability

During Phases 1–10 (Gates G1–G10), local multi-worker execution (`p > 1`) was strictly blocked in compliance with official course requirement **A4-11** (*"Local multi-process execution with multiple workers requires explicit instructor permission"*). All Phase 7 pilot runs and the Phase 8 single-worker campaign (`run_id = campaign_20261114_205134_1d126401af244671b4ca8708818549f3`, 36 rows in `results/raw/raw_measurements.csv`) were therefore restricted strictly to `p = 1` and transparently reported multi-worker strong scaling as `BLOCKED / NOT COMPLETED`.

All historical Phase 7 and Phase 8 records, sidecars, and frozen baseline snapshots are preserved intact and never overwritten.

---

## 2. Updated Authorization Scope (2026-11-23)

On **2026-11-23**, the student confirmed explicit authorization from the course instructor to execute the permission-gated local multi-worker experiments (`p > 1`) using **Dask Distributed**, following the same course framework as prior assignments.

This file serves as the formal written record required by Gate G1 (`src/dask_matrix/cluster_manager.py::verify_multiworker_permission`).

### Permitted Scope
- Local distributed multi-process execution via `distributed.LocalCluster(n_workers=p, threads_per_worker=1, processes=True, memory_limit=None)` on the single accredited student host.
- Controlled `p = 1` vs `p = 2` validation pilot and strong-scaling experimental sweep on fixed matrix dimensions.
- Separate diagnostic characterization passes (`get_task_stream`, per-worker transfer/memory telemetry) run strictly outside primary timed measurements.

### Preserved Prohibitions
- **No multi-node claims:** All executions occur on **1 physical machine** (`physical_machine_count = 1`). Under no circumstances may local multi-process results be presented as physical cluster scaling, inter-machine network performance, or multi-node aggregate RAM scaling.
- **No silent worker fallback:** `verify_multiworker_permission(p)` remains fail-closed if this authorization record is absent; requested `p` must match observed `worker_count_observed` identically.
- **No thread oversubscription per worker:** Each worker remains pinned to `threads_per_worker = 1` and single-threaded BLAS/OpenMP (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
- **Operational safety ceilings remain mandatory:** `120.0 s` per-sample timeout, `2,560.0 MiB` (`2.5 GiB`) sampled aggregate RSS ceiling (`OUT_OF_MEMORY`), and `1,024.0 MiB` (`1.0 GiB`) host available RAM floor (`ABORTED_WATCHDOG`).

---

## 3. Physical Host Topology vs. Multi-Node Distinction

| Dimension | Local Multi-Process Execution (Authorized Here) | True Physical Multi-Node Cluster (Not Available) |
|---|---|---|
| Physical machines (`physical_machine_count`) | **1** (Intel Core i5-1335U, 10 physical cores: 2 P-cores + 8 E-cores, 12 logical processors, 16 GB DDR4 RAM, Windows 11 Home 64-bit) | $\ge 2$ physically distinct hosts |
| Communication medium | Local loopback TCP sockets (`127.0.0.1`) in OS kernel memory | Physical NICs, switches, MTU framing, wire latency |
| Memory subsystem | Single shared physical DRAM pool (`16,009.14 MiB`) and shared L3 cache (`12 MB`), subject to bus contention and OS page sharing | Physically independent node RAM pools |
| Scientific scope | Evaluates Dask DAG scheduling, serialization, loopback IPC overhead, worker contention, and heterogeneous P/E-core strong scaling | Evaluates physical network bottlenecks and distributed memory scaling |

---

## 4. Justified Worker Configurations (`p`) and Strong-Scaling Matrix

Following `SPEC.md` §6.2 and `docs/experimental_protocol.md` §3.2, the strong-scaling campaign evaluates fixed total problem sizes $n \in \{1000, 1500, 2000\}$ with fixed block size $b = 250$ ($N_b \in \{4, 6, 8\}$; $112$, $396$, and $960$ logical DAG tasks respectively) across:

- **`p = 1`** (2 unique OS PIDs on Windows `LocalCluster`): Dedicated single-worker reference sub-campaign executed first inside the continuous strong-scaling sweep ($S(1) = 1.0$, $E(1) = 1.0$; separate `run_id`, fresh `LocalCluster`, and sidecar from each subsequent $p > 1$ sub-campaign, with no interleaved $p = 1$ re-baseline inside individual $p > 1$ clusters).
- **`p = 2`**: Matches the 2 Performance cores (P-cores) and introduces the minimum multi-worker loopback communication boundary.
- **`p = 4`**: Spans 2 P-cores + 2 Efficient cores (E-cores), exposing core heterogeneity and moderate loopback/memory contention.
- **`p = 8`**: Occupies 8 of 10 physical cores (2 P-cores + 6 E-cores), testing high multi-process scheduling and memory bandwidth contention.
- **`p = 10`**: Occupies all 10 physical cores (maximum non-oversubscribed physical core count on i5-1335U; executed strictly under the `2.5 GiB` aggregate RSS ceiling and `1.0 GiB` host available RAM floor—any watchdog breach is recorded honestly rather than relaxing limits).

Weak scaling remains optional and is not executed as an additional campaign; its theoretically sound dense cubic formulation ($W(n) = 2n^3 \implies n(p) = n(1) \cdot p^{1/3}$) is retained for methodological completeness in `docs/experimental_protocol.md` §4.
