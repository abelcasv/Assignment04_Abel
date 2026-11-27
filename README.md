# Assignment 4: Distributed Matrix Multiplication (Dask Distributed)

- **Course:** BIG DATA · Grado en Ciencia e Ingeniería de Datos
- **Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC) · Academic Year 2026–2027
- **Author:** Abel Castro Villar
- **Canonical Academic Report:** [`report/Assignment04_Report.pdf`](report/Assignment04_Report.pdf) (compiled from [`report/Assignment04_Report.tex`](report/Assignment04_Report.tex))

---

## 1. Overview & Experimental Scope

This repository implements and evaluates dense square matrix multiplication ($C = A \times B$) in **Dask Distributed** using a **2D block decomposition** ($b = 250$) and a **pairwise binary-tree reduction** over IEEE 754 `binary64` (`float64`) matrices.

- **Single-Worker & Cross-Assignment Common-Input Comparison:** Compares Dask (`p = 1` and `p = 2`) against the frozen Assignment 1 sequential C baseline ([`baselines/a1_sequential/`](baselines/a1_sequential/), origin commit `90681e85949b6d65bdbcfef702b6306b0003f971`) and Assignment 3 sequential and OpenMP C baselines ([`baselines/a3_openmp/`](baselines/a3_openmp/), origin commit `23cffbc82c155b6c3de24b3c149b509b63fc870b`) on the **exact same deterministic `data/dense/` matrices** (`input_sha256_verified_identical = True`), while keeping non-equivalent timing boundaries explicit (`TB_DASK_E2E` vs. `TB_A1_KERNEL_QPC` and `TB_A3_KERNEL_QPC`) and separating historical prior-assignment summaries ([`data/baselines/`](data/baselines/)).
- **Single-Host Distributed Strong Scaling:** Evaluates fixed matrix dimensions $n \in \{1000, 1500, 2000\}$ across $p \in \{1, 2, 4, 8, 10\}$ independent Dask worker processes (`1 + p` OS PIDs, `threads_per_worker = 1`, `OPENBLAS_NUM_THREADS = 1`) communicating over `127.0.0.1` loopback TCP on **one physical machine** (`physical_machine_count = 1`). No multi-machine deployment was used, and optional weak-scaling experiments were not included in this campaign.

Full methodology, hypotheses, statistical tables, figures, discussion, and references are presented in [`report/Assignment04_Report.pdf`](report/Assignment04_Report.pdf).

---

## 2. Architecture Summary

Implemented under [`src/`](src/) (detailed module contracts in [`src/README.md`](src/README.md) and [`SPEC.md`](SPEC.md)):

- **Partitioning & Scheduler-Managed DAG:** [`partitioning.py`](src/dask_matrix/partitioning.py) slices $A$ and $B$ into $N_b \times N_b$ contiguous `float64` blocks ($N_b = \lceil n/b \rceil$) using exact edge truncation without zero-padding. [`distributed_engine.py`](src/dask_matrix/distributed_engine.py) and [`reduction.py`](src/dask_matrix/reduction.py) coordinate $N_b^3$ `np.matmul` tasks and $N_b^2(N_b - 1)$ binary-tree `block_add` tasks ($2N_b^3 - N_b^2$ total tasks) through Dask's centralized scheduler via `Client.scatter`, `Client.submit`, and `Client.gather`.
- **Timing & Safety Safeguards:** End-to-end wall-clock time (`TB_DASK_E2E`) spans 2D slicing, scatter, DAG submission, worker execution/IPC, gather, dense reassembly, and full-matrix checksum materialization (`float(np.sum(C, dtype=np.float64))`), excluding cluster startup and CSV disk loading. [`watchdog.py`](src/dask_matrix/watchdog.py) enforces a `250 ms` sampled `2.5 GiB` aggregate RSS cap, a `1.0 GiB` host free-RAM floor, and a `120.0 s` per-sample timeout.

---

## 3. Key Verified Results

- **Numerical Correctness:** `6/6` validation checks pass (`atol = 1e-9`, `rtol = 1e-12`), achieving $\max|\Delta| = 1.137 \times 10^{-13}$ ($b = 250$) against the scalar-Python reference [`data/validation/reference/C_250.csv`](data/validation/reference/C_250.csv), exact `0.0` element-wise difference on $C_{500}$ (`p = 2` vs. `p = 1`), and exact `%.17g` checksum invariance across all `77` completed strong-scaling rows.
- **Single-Worker Scaling (`p = 1, b = 250`):** Median $T_{\text{Dask\_E2E}}$ grows from `0.0336 s` (`IQR = 0.0032 s`) at `n = 250` to `0.9318 s` at `n = 1000` and `7.4963 s` (`IQR = 0.0952 s`) at `n = 2000`.
- **Controlled Common-Input Comparison (`data/dense/`):** At `n = 250` (`1.43 MiB` across three matrices, within the shared `12 MB` L3 cache), every native C baseline runs faster than Dask `p = 1` (`0.0195 s` A1 C; `0.0023–0.0085 s` A3 C/OpenMP). Dask `p = 1` overtakes unblocked `A1 C` from `n = 500` (`0.2278 s` $\to$ `4.0157 s` at `n = 1000`) and untiled single-thread A3 from `n = 1500` (`3.7641 s` $\to$ `15.5009 s` at `n = 2000`), while cache-blocked `seq_tiled_b32` (`4.6706 s` at `n = 2000`) and shared-memory OpenMP (`omp_tiled_b32`: `0.8571 s` at `t = 8`, `3.2325 s` at `t = 10`) remain faster.
- **Single-Host Multi-Worker Strong Scaling (`p` in `{1, 2, 4, 8, 10}`):** Positive median speedup occurs solely at `n = 2000, p = 2` (`9.5635 s` vs. `11.6333 s`, $S_{\text{dist}} = 1.22\times$, $E_{\text{dist}} = 60.8\%$). Configurations with `p` in `{4, 8}` show steep degradation across all tested matrix sizes ($S(4) = 0.21\times\text{--}0.77\times$, $S(8) = 0.12\times\text{--}0.34\times$), with `scatter + wait + gather` representing `96.11%` of wall-clock latency at `p = 8, n = 1000` (`0 B` disk spill; `84.60–113.62 MiB` per-worker OS RSS), a pattern consistent with centralized scheduler contention, loopback IPC overhead, and asymmetric P/E-core behavior contributing to the slowdown (not individually isolated). The `p = 10` campaign encountered the `120.0 s` timeout (`131.92 s` on `n = 1500` rep 5), leaving `n = 1500` partially completed (`4/5`) and `n` in `{1000, 2000}` unexecuted (`0/5`).

---

## 4. Environment & Installation

- **Benchmark Host:** Intel Core i5-1335U (`2` P-cores + `8` E-cores, `12` logical processors, `12 MB` shared L3), `16.0 GB` DDR4 RAM, Windows 11 Home 64-bit (`10.0.26300`).
- **Software Toolchain:** Python 3.12+ (tested with Python `3.14.0` AMD64) with pinned dependencies in [`requirements.txt`](requirements.txt) (`dask==2026.8.0`, `distributed==2026.8.0`, `numpy==2.5.4`, `psutil==7.2.2`, `matplotlib==3.11.2`, `threadpoolctl==3.7.0`); MSYS2 MinGW-W64 UCRT64 GCC (`16.1.0`) for C/OpenMP baselines; `pdflatex` (MiKTeX or TeX Live) for report compilation.

```powershell
# Create and activate a virtual environment from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 5. Verification, Execution & Report Build (PowerShell)

### 5.1 Dataset Reconstruction (Fresh Clones Only)
The `14` dense CSV matrices in `data/dense/` (~310 MB) are excluded from Git via `.gitignore` and certified by [`data/SHA256SUMS.txt`](data/SHA256SUMS.txt) (see [`data/README.md`](data/README.md)). On a fresh clone where `data/dense/` is absent, materialize and verify them first:

```powershell
python -B scripts/generate_dataset.py
python -B scripts/verify_dataset.py data/SHA256SUMS.txt
```

### 5.2 Fast Local Verification Suite
Run all read-only integrity checks, numerical fixtures, CLI checks, and the unit/contract test suite:

```powershell
python -B scripts/verify_dataset.py
python -B scripts/verify_correctness.py
python -B -m src.cli --n 1000 --block-size 250 --workers 1 --check-config
python -B -m src.cli --validate-fixtures
python -B -m unittest discover -s tests -v
python -B scripts/verify_provenance.py
```

### 5.3 Offline Analysis & Benchmark Execution Modes
Regenerate derived summary CSVs and plots in [`results/analysis/`](results/analysis/) (updating `generated_utc` in [`results/analysis/analysis_provenance.json`](results/analysis/analysis_provenance.json)):

```powershell
python -B scripts/analyze_campaign.py
```

Optional benchmark runners (do not run unless intentionally collecting new measurements; full campaigns are time-consuming):

```powershell
# Single-worker Dask modes via CLI (alternative modes: bounded pilot vs. full p=1 campaign)
python -B -m src.cli --pilot
python -B -m src.cli --campaign

# Multi-worker single-host Dask modes (alternative execution modes; --campaign takes ~27 min)
python -B scripts/run_strong_scaling.py --pilot-p1-p2
python -B scripts/run_strong_scaling.py --campaign
python -B scripts/run_strong_scaling.py --diagnostics-only

# Controlled common-input A1 & A3 C/OpenMP suite (~18 min; pass --overwrite only when deliberately replacing existing results)
python -B scripts/run_common_input_baselines.py
```

### 5.4 Compiling the Canonical PDF Report
```powershell
Push-Location report
pdflatex -interaction=nonstopmode -halt-on-error Assignment04_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Assignment04_Report.tex
Remove-Item -Force Assignment04_Report.aux, Assignment04_Report.log, Assignment04_Report.out -ErrorAction SilentlyContinue
Pop-Location
```

---

## 6. Repository Structure & Reproducibility Artifacts

```text
Assignment04_Abel/
├── README.md                         # Concise project overview and execution guide
├── SPEC.md                           # Technical specification, provenance, and module contracts
├── requirements.txt                  # Pinned Python dependencies
├── baselines/                        # Frozen unmodified A1 (C) and A3 (C/OpenMP) snapshots
├── data/
│   ├── README.md                     # Deterministic dataset generation & verification guide
│   ├── SHA256SUMS.txt                # SHA-256 manifest for all 20 dense & validation CSVs
│   ├── baselines/                    # Preserved historical A1/A3 summary CSVs (contextual only)
│   ├── dense/                        # Deterministic benchmark matrices (generated via generate_dataset.py)
│   └── validation/                   # Tracked 4x4 fixtures and scalar-Python reference C_250.csv
├── src/                              # Dask 2D block engine, watchdog, telemetry, CLI, and adapters
├── scripts/                          # Dataset, verification, baseline runner, and offline analysis scripts
├── tests/                            # Automated unit and contract test suite (200 tests)
├── results/
│   ├── raw/                          # Immutable raw CSVs (Dask 138 rows, A1 25 rows, A3 540 rows)
│   ├── metadata/                     # Environment sidecars, manifests, and n=1000 diagnostics JSON
│   └── analysis/                     # Derived summary CSVs, plots, and analysis_provenance.json
└── report/
    ├── Assignment04_Report.pdf       # Canonical academic research report for submission
    ├── Assignment04_Report.tex       # Editable LaTeX source of the canonical report
    └── plots/                        # Publication figures included by Assignment04_Report.tex
```

All `19` raw, metadata, script, derived CSV, and figure artifacts are cryptographically indexed in [`results/analysis/analysis_provenance.json`](results/analysis/analysis_provenance.json) (with baseline origin commits and frozen source SHA-256 checksums recorded in [`results/metadata/common_input_baselines_manifest.json`](results/metadata/common_input_baselines_manifest.json) and [`SPEC.md`](SPEC.md)) and verified byte-for-byte via `python -B scripts/verify_provenance.py`.

---

## 7. Generative AI Assistance Disclosure

In compliance with the ULPGC Big Data course policy on Generative Artificial Intelligence, the author discloses the supervised use of three AI tools:
1. **ChatGPT (OpenAI):** Planning, technical review, and workflow/prompt preparation ([`SPEC.md`](SPEC.md)).
2. **Google Antigravity (Gemini):** Modular Python implementation ([`src/`](src/)), execution and verification scripts ([`scripts/`](scripts/)), unit and contract test suites ([`tests/`](tests/)), offline statistical analysis and figure generation ([`scripts/analyze_campaign.py`](scripts/analyze_campaign.py)), and English/LaTeX report and documentation preparation ([`report/Assignment04_Report.tex`](report/Assignment04_Report.tex), [`README.md`](README.md)).
3. **Codex (OpenAI):** Repository review, technical auditing, and documentation refinement ([`README.md`](README.md), [`SPEC.md`](SPEC.md), [`report/Assignment04_Report.tex`](report/Assignment04_Report.tex)).

All code, scripts, and tests were reviewed, validated, and executed locally on the documented hardware; all numerical claims trace directly to unfabricated raw CSV records; all references were verified; and the author assumes full academic responsibility for understanding, validating, and defending the implementation, measurements, and conclusions.
