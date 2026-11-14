# CLAUDE.md — Assignment 4 Engineering Rules

**Project:** Assignment 4 — Distributed Matrix Multiplication
**Course:** BIG DATA, Grado en Ciencia e Ingeniería de Datos, ULPGC
**Academic year:** 2026–2027
**Purpose of this file:** Mandatory permanent working rules for every AI agent and human contributor working in this repository.

---

## 1. Project objective

This repository implements and evaluates **Assignment 4: Distributed Matrix Multiplication** for the ULPGC Big Data course.

The project must be a controlled, empirical and reproducible study of how a distributed matrix-multiplication implementation scales with workers or nodes, taking computation, communication, data movement, scheduling and framework overhead into account.

For feasible common input sizes, the distributed implementation must be compared with:

1. A validated basic sequential implementation reused from Assignment 1.
2. A validated shared-memory parallel implementation reused from Assignment 3.

The project must follow the official assignment, the technical requirements in `SPEC.md`, the accepted experimental protocol in `docs/experimental_protocol.md`, and the approval gates defined below.

---

## 2. Mandatory documentation and reading procedure

### 2.1 Documents to read before every task

Before starting **each task**, every AI agent must read:

1. `CLAUDE.md` — project-wide working rules, Git discipline, and agent behaviour.
2. `SPEC.md` — the complete technical specification, requirements and acceptance criteria.
3. `docs/experimental_protocol.md` — the **single detailed source of truth** for all experimental parameters, configurations, timing boundaries, correctness fixtures, tolerances, budgets, and failure criteria. Mandatory reading for all phases involving experimental design, implementation, correctness, environment validation, benchmarking, data analysis, or reporting (Phases 3–9).
4. The specific task prompt supplied by the student.

These are mandatory working contracts. Do not rely on memory, previous conversations, summaries, or assumptions instead of reading the current files.

### 2.2 Official assignment PDF

The authoritative course document is:

`docs/reference/BD_Individual_Assignments_2026_2027.pdf`

During **initial project onboarding (Prompt 0)**, read the complete PDF, not merely a summary. Extract all applicable common-protocol requirements, Assignment 4 requirements, report/submission requirements, assessment criteria and AI-use policy. `SPEC.md` is intended to encode these requirements in an actionable form.

The PDF does **not** need to be re-read in full before every later task. Consult the relevant pages or passages whenever:

- a requirement is being interpreted or checked;
- a design choice may affect course compliance;
- an ambiguity or conflict is found in `SPEC.md` or `docs/experimental_protocol.md`;
- a new experiment, optional extension or submission decision is proposed; or
- the student asks for a compliance review.

Do not invent, weaken or silently reinterpret an official requirement.

### 2.3 Domain-specific authority model and conflict resolution

To prevent ambiguities where procedural rules appear to override experimental design, the repository operates under a **domain-specific authority model**:

1. **Official Assignment PDF (`docs/reference/BD_Individual_Assignments_2026_2027.pdf`) & Explicit Written Clarification from the Instructor:** Highest academic authority; governs learning goals, evaluation criteria, and institutional restrictions (including the requirement for instructor permission prior to running multiple local workers).
2. **`SPEC.md`:** Governs normative project architecture, technical contracts, baseline interfaces, and acceptance requirements.
3. **`docs/experimental_protocol.md`:** The **single detailed source of truth** for all experimental parameters, matrix dimensions, block partitioning, timing boundaries, measurement contracts, telemetry metrics, numerical correctness tolerances, resource budgets, and failure handling policies, within the boundaries of the PDF and `SPEC.md`. Formally accepted under Gate G4.
4. **`CLAUDE.md`:** Governs permanent agent behaviour, safety constraints, permission gates, engineering standards, Git discipline, and operational workflow. It **cannot** silently redefine or weaken experimental parameters.
5. **The current task prompt:** Governs the immediate scope and requested deliverables for a specific phase; authorizes only actions explicitly assigned to that phase, and cannot override or weaken higher-tier documents.

**Conflict Resolution Rule:** If any genuine contradiction arises across these documents, the agent must stop immediately, document the conflicting passages, and request clarification from the student before proceeding.

### 2.4 Canonical and archival document definitions

- **Canonical Documents:** `CLAUDE.md`, `SPEC.md`, and `docs/experimental_protocol.md` are the sole authoritative, canonical project documents.
- **Archival Drafts:** Historical drafts `CLAUDE_UPDATED.md` and `SPEC_UPDATED.md` have been fully consolidated into the canonical governance documents and removed from the active repository; they are superseded and are not active repository documents.
- **Superseded Drafts:** Any prior protocol working draft (including `docs/Assignment04_Experimental_Protocol_Final.md`) is fully superseded by `docs/experimental_protocol.md` and must never be treated as a competing source of truth.

### 2.5 Private prompt sequence

`PROMPTS.md` is the student's private task sequence and is intentionally kept outside the Git repository.

- Do not search for, read, reconstruct, or follow `PROMPTS.md` unless the student explicitly supplies its contents in the current task.
- A task prompt supplied by the student is the immediate instruction for that task.
- Do not advance to the next prompt or phase automatically.

---

## 3. Local reference repositories and source materials

The previous assignments are available as read-only local clones at:

```text
external_repos/
├── Assignment01/   # Assignment 1 repository clone (read-only)
└── Assignment03/   # Assignment 3 repository clone (read-only)
```

Rules:
- Treat both prior repositories as strictly read-only. Never edit their files, run formatters, commit, checkout branches, reset, or push to them.
- Do not add `external_repos/` as submodules or copied Git history inside Assignment 4.
- Vendor snapshots extracted in Phase 2.1 reside in `baselines/a1_sequential/` and `baselines/a3_openmp/`. These snapshots are **immutable**.
- Frozen common inputs reside in `data/dense/` and `data/validation/`. They are **immutable**.

The intended local layout is:

```text
Assignment04/
├── CLAUDE.md
├── SPEC.md
├── README.md
├── docs/
│   ├── experimental_protocol.md
│   ├── decisions/
│   ├── provenance/
│   └── reference/
│       └── BD_Individual_Assignments_2026_2027.pdf
├── baselines/
├── build/
├── data/
├── src/
├── scripts/
├── tests/
├── results/
└── external_repos/
```

`.gitignore` must exclude `external_repos/` and the local official PDF from the Assignment 4 Git repository.

---

## 4. Non-negotiable rules

- **Truthfulness and Evidence:** Never fabricate measurements, metadata, hardware facts, framework capabilities, citations, references, permissions, or successful checks. Clearly distinguish **observed facts**, **source-code inferences**, **assumptions**, **proposals**, and **unknowns**.
- **Protocol Adherence:** Never modify settled experimental protocol parameters, matrix dimensions, block sizes, worker allocations, timing boundaries, tolerances, or resource budgets without explicit student authorization. `docs/experimental_protocol.md` is binding.
- **Local Multi-Worker Permission Gate & Strong Scaling (DEC-01):** Local multi-worker execution ($p \in \{2, 4, 8, 10\}$) is strictly **BLOCKED** until explicit written instructor permission is stored in `docs/decisions/`. The single-worker configuration ($p = 1$) is the only permitted distributed configuration prior to instructor approval. **`p=1` alone does NOT constitute a strong-scaling sweep.** If instructor permission is unavailable and no approved multi-node cluster exists, mandatory strong scaling must be recorded as **BLOCKED / NOT COMPLETED**. Under no circumstances may $p=1$ be substituted for a multi-worker sweep to claim that the strong-scaling requirement was met.
- **Accurate Topology Labeling:** Never describe local worker processes on a single physical host as physical machines/nodes, and never describe loopback IPC as inter-machine network scaling or multi-machine memory.
- **Immutability of Baselines and Datasets:** Never edit the source files, Makefiles, or compiler flags of the frozen snapshots in `baselines/`. Never regenerate or overwrite the frozen common matrices in `data/dense/` or validation fixtures in `data/validation/` in an existing experimental workspace. In a clean repository clone where `data/dense/` is excluded from Git under Option B, materialize and certify matrices using `scripts/generate_dataset.py` and `scripts/verify_dataset.py` (see `data/README.md`).
- **Strict Distinction of Gates:** Respect the sequential progression and distinct criteria of Gates G0 through G10. Never conflate static code inspection with dynamic runtime execution. Label unexecuted checks as **NOT RUN / NOT VERIFIED**.
- **Timing Boundary Integrity:** Follow the timing contract in `docs/experimental_protocol.md`. Cluster startup and CSV input loading must be outside the main multiplication timer. The main timer must end only after a full-matrix checksum consumption forces complete materialization. Numerical correctness checks must be outside the timer.
- **Overlapping Metrics:** In parallel/distributed systems, never sum overlapping task durations to reconstruct wall-clock time. Do not report object `nbytes` as physical network bytes.
- **Resource Budgets & Watchdog Semantics:**
  - Budgets (120 s timeout, 2.5 GiB sampled aggregate RSS cap) are **student-selected operational safety limits** and **provisional limits requiring pilot validation**, NOT empirically validated budgets. The 10 h campaign cap is an operational planning envelope; 1.0 GiB host available RAM floor is an OS safety limit; 250 ms watchdog interval measures sampled peak RSS and may miss transient spikes.
  - The sum of process RSS is not equivalent to unique physical RAM usage (shared pages).
  - Track unique PIDs (driver, scheduler, nannies, workers). Dask `memory_limit` is best-effort, not a hard process cap.
  - Safe watchdog stop procedure: cancel client tasks immediately, close cluster within 5 s, forcefully terminate lingering PIDs (`terminate() / kill()`), and log status as `OUT_OF_MEMORY` or `ABORTED_WATCHDOG`.
- **Measurement Immutability:** Raw measurements in `results/raw/` are append-only. Never silently edit, overwrite, or delete raw observations. Retain failed, timed-out, and memory-limited samples.
- **A1 Optimization Verification (Phase 5 Assignment):** Do not attempt compilation or assembly inspection in Phase 4. In **Phase 5 (Correctness / Build Validation)**, upon authorized compilation of `baselines/a1_sequential/benchmark.c` under original `-O3` flags, inspect the generated assembly/disassembly (`gcc -S` or `objdump -d`) to verify if the full $O(n^3)$ kernel survived optimization given that the harness consumes only `c[0]`. Do not alter A1 source code or compiler flags. Do not benchmark.
- **Scope Discipline:** Do not make changes outside the scope of the current task. Do not refactor working code for style alone. Do not install packages, compile code, or run benchmarks unless explicitly authorized by the task prompt.
- **Git Discipline:** Ordinary local commits are authorized for completed, reviewed logical units under Section 9. Pushing, history rewriting, rebasing, amending, and branch switching require explicit authorization. Blanket staging (`git add .`) is strictly forbidden.

---

## 5. Work in gated phases

Work only in the phase requested by the student. Do not advance automatically, combine phases, or treat a proposal as approval.

### Phase 0 — Repository and environment discovery
- Read complete official PDF, `CLAUDE.md`, and `SPEC.md`.
- Inspect repository structure, Git status, prior assignment repositories, input artifacts, host environment, and installed tools.
- Read-only inspection only.

### Phase 1 — Framework and architecture decision
- Compare viable distributed frameworks against operating system, hardware, and course requirements.
- Produce ADR-001 recommending framework and architecture.
- Approved by student: Dask Distributed, 2D block matrix multiplication, binary-tree reduction, `numpy.matmul`, single thread per worker.

### Phase 2 — Baseline and input provenance freeze
- Audit A1 and A3 implementations, source paths, commits, and input generators.
- Materialize frozen snapshots in `baselines/` and common datasets in `data/dense/`.
- Generate `data/SHA256SUMS.txt` and provenance documentation. Completed and frozen in Phase 2.1.

### Phase 3 — Experimental protocol and budgets
- Design complete experimental protocol, research question, hypotheses, configurations, timing boundaries, telemetry, correctness fixtures, tolerances, and resource budgets.
- Synchronized, audited, and approved in `docs/experimental_protocol.md`. **Gate G4 PASSED.**

### Phase 4 (Prompt 4) — Implementation
- Implementation-only phase: Implement the approved distributed 2D block matrix multiplication in `src/` using Dask Distributed.
- Implement runtime adapters for A1 and A3 to populate `build/a1_runtime/` and `build/a3_runtime/`.
- Perform static review of Makefiles, includes, relative paths, and macro quoting.
- **Strictly prohibited: compilation of code, execution of tests, launching of pilots, or running of benchmarks.**

### Phase 5 (Prompt 5) — Correctness and build validation
- Authorizes compiling the immutable A1 and A3 snapshots to `build/bin/` using their original documented compiler flags.
- Authorizes generating and inspecting A1 compiler output / assembly (`gcc -S` or `objdump -d`) to verify if the $O(n^3)$ kernel was eliminated by dead-code elimination.
- Execute authorized correctness fixtures (Identity, Zero, $4 \times 4$ analytical, edge block $n=4, b=3$, and $n=250$ element-by-element vs. `data/validation/reference/C_250.csv`).
- Verify finite values and candidate tolerances ($\text{atol}=10^{-9}, \text{rtol}=10^{-12}$).
- **Strictly prohibited: execution of performance benchmarks.**

### Phase 6 (Prompt 6) — Environment and topology validation
- Verify runtime environment, host specs, CPU core topology, RAM, OS build, and package versions.
- Verify Dask process tree (nominal $1 + 2p$ processes: Client/Scheduler, Nannies, Workers) and verify that internal library threading is strictly constrained to 1 thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
- Verify idle memory footprint and calibrate worker `memory_limit`.

### Phase 7 (Prompt 7) — Pilot experiment
- A pilot requires separate explicit authorization of a bounded execution plan to validate operational budgets (120 s timeout, 2.5 GiB aggregate RSS).
- **Instructor permission is required ONLY if the pilot matrix includes local $p>1$ configurations. An approved pilot consisting exclusively of $p=1$ configurations is NOT blocked by the absence of $p>1$ permission.**

### Phase 8 (Prompt 8) — Full benchmark campaign
- Full benchmark campaign requires separate explicit authorization following pilot review.
- **Under no circumstances may $p=1$ be substituted for a blocked $p>1$ configuration to claim strong scaling. If $p>1$ remains unpermitted, strong scaling across multiple workers must be reported as BLOCKED / NOT COMPLETED.**
- Enforce the 120 s timeout, 10 h campaign cap, 2.5 GiB sampled aggregate RSS cap, and 1.0 GiB host available RAM floor.
- Follow randomized execution order (seed `20261113`) and append all raw observations to `results/raw/raw_measurements.csv`.

### Phase 9 (Prompt 9) — Data audit and analysis
- Derive results deterministically from preserved raw data.
- Compute medians and linear IQRs. Compute strong-scaling speedup and efficiency for Dask (if multi-worker runs were permitted).
- Compute cross-method contextual ratios ($T_{\text{kernel}} / T_{\text{Dask\_e2e}}$) with explicit boundary qualifications.
- Generate planned figures and tables without rerunning benchmarks.

### Phase 10 (Prompts 10–12) — Report, reproducibility package, and final audit
- Prepare README and final research report in English, following the course structure.
- Document framework, topology, baseline provenance, input generation, timing boundaries, budgets, limitations, and AI usage.
- Audit source code, raw data, analysis, figures, report, and reproducibility instructions against `SPEC.md` and the official PDF.
- Preserve strict distinction between `BLOCKED`, `NOT STARTED`, `PASSED`, `NOT RUN`, and `NOT VERIFIED`.

At the end of each task, report:
1. Phase and task completed.
2. Files inspected and files changed.
3. Commands executed and their purpose.
4. Checks completed (static vs. dynamic).
5. Findings, assumptions, and open blockers.
6. Deviations from `SPEC.md` or approved decisions.
7. Exact next decision or approval needed from the student.
8. Git status and commit information (if a local commit was created).

---

## 6. Implementation and scientific-quality rules

- **2D Block Slicing:** Partition matrices into $b \times b$ blocks ($b = 250$). Truncate edge blocks without zero-padding.
- **Local Kernel:** Use `numpy.matmul` on contiguous `float64` block arrays.
- **Threading Isolation:** Force 1 thread per worker in Dask. Avoid hidden multi-threading in BLAS/LAPACK.
- **Accurate Resource Counting:** Distinguish framework workers, worker processes, scheduler/driver processes, and physical machines.
- **Timing Boundary:** Start timer immediately before block partitioning/scatter; stop timer immediately after full-matrix checksum consumption. Exclude cluster startup and CSV loading.
- **Output Integrity:** Preserve all raw samples, including failure statuses.

---

## 7. Optional extensions and quality priorities

- Mandatory requirements always take priority over optional extensions.
- **Weak Scaling:** Evaluated under $n(p) = n(1) \cdot p^{1/3}$. Requires approved new inputs and separate permission; remains **BLOCKED**.
- **Multi-Machine Execution:** Only to be undertaken if genuine physical multi-node infrastructure is accredited. Never simulate by relabeling local processes.
- Do not let optional extensions compromise core correctness or project deadlines.

---

## 8. Documentation and communication

- Write README and final research report in English.
- Cite genuine, verifiable sources (e.g. IEEE format).
- Label all figure axes, units, configurations, and sample sizes.
- Disclose generative AI tool usage, purpose, and affected sections.
- Keep conclusions scoped strictly to the tested hardware, sizes, and measurement conditions.

---

## 9. Code quality, incremental delivery and Git discipline

The student has instructed that development must follow professional software engineering practices throughout the project, not be postponed until the end. This includes Clean Code, KISS, YAGNI, sensible DRY, the Boy Scout Rule, and a reviewable Git history. The student has also given standing authorization for ordinary **local commits** under the rules below. This is not authorization to push or rewrite history.

### 9.1 General engineering principles

Apply these principles pragmatically; do not turn them into unnecessary ceremony:

- **Clean Code:** Use clear names, cohesive functions/modules, explicit interfaces, predictable control flow, and visible error handling. Keep each function focused on one responsibility. Split long or complex logic when doing so improves comprehension or testability.
- **KISS (Keep It Simple, Stupid):** Choose the simplest design that satisfies the approved requirements. Prefer straightforward code and data flow over clever abstractions, unnecessary frameworks, generic factories, or premature extensibility.
- **YAGNI (You Aren't Gonna Need It):** Do not implement speculative features, configuration options, premature abstractions, or infrastructure that the current approved phase does not need.
- **Sensible DRY (Don't Repeat Yourself):** Centralize genuinely shared rules and avoid duplicated business logic, but do not create an abstraction for two superficially similar operations with different semantics. A little explicit duplication can be clearer than a misleading abstraction.
- **Boy Scout Rule:** Leave touched code clearer or safer when a small adjacent improvement is obviously justified. Keep the improvement local to the current task; do not use it as permission for broad refactoring or scope expansion.
- **Separation of Concerns:** Keep matrix computation, data loading, orchestration, timing, telemetry, correctness validation, and reporting distinguishable. Avoid mixing benchmark measurement logic into numerical kernels unless the approved timing contract requires it.
- **No Premature Optimization:** Preserve approved baseline algorithms. Optimize the new implementation only to satisfy measured or justified requirements, and keep algorithmic changes separate from framework/configuration changes where practical.
- **Explicitness and Reproducibility:** Prefer explicit parameters, validated configuration, deterministic behavior, and traceable paths over hidden global state, implicit working-directory assumptions, or environment-dependent defaults.
- **Preserve Immutability:** Never alter frozen baseline snapshots in `baselines/` or common datasets in `data/dense/` and `data/validation/` in an existing experimental workspace. On a clean clone, follow `data/README.md` to materialize and certify `data/dense/`.

### 9.2 Source-code standards

For new or modified Assignment 4 code:

- Use meaningful, consistent identifiers. Avoid unexplained abbreviations, misleading names, unnecessary mutable global state, and magic numbers. Put legitimate constants and experiment settings in one clearly identified configuration source.
- Keep functions and modules cohesive. Separate input parsing/validation, numerical kernels, distributed scheduling, result assembly, correctness checks, timing, and metrics collection.
- Validate external inputs and configuration at module boundaries. Fail clearly and early on invalid dimensions, missing files, incompatible shapes, unsupported settings, or non-finite values. Do not silently substitute defaults that change the experiment.
- Handle errors explicitly. Do not swallow exceptions, use broad catch-all handlers as normal control flow, or report success after partial failure. Include useful context while preserving the original cause.
- Document public interfaces, non-obvious algorithms, numerical assumptions, ownership/lifetime constraints, and reasons for non-obvious decisions. Comments should explain why or clarify a non-obvious invariant, not merely repeat the code.
- For Python, use type hints for public functions and meaningful module boundaries where they improve clarity; use standard-library facilities or already-approved dependencies before introducing new packages. Adhere to PEP 8.
- For C, preserve clear ownership and cleanup paths, check allocation and I/O failures, use appropriate const-correctness, avoid undefined behavior, and keep compiler warnings visible (`-Wall -Wextra`). Do not weaken warning or optimization flags merely to make a build pass.
- Keep style consistent within each language and with existing project conventions. Do not reformat unrelated files or entire frozen baseline snapshots.
- Do not add dependencies, formatters, linters, code generators, or build systems without checking whether an existing tool already solves the need and obtaining approval when installation or dependency changes are required.
- Do not edit the frozen A1/A3 baseline algorithms or reformat their snapshots. Any compatibility layer, wrapper, or adapter must remain separate, minimal, and attributed.

### 9.3 Tests, static checks and automation

- Add or update focused automated tests for new behavior and bug fixes when a meaningful test can be written. Keep tests deterministic, bounded, and independent of performance measurements.
- A bug fix should normally include a regression test that fails for the defect and passes for the correction, where feasible.
- Test edge cases and invalid inputs at module boundaries, especially dimensions, partial/edge blocks, file parsing, dtype, finite values, result assembly, and failure handling.
- Run only checks authorized by the current phase. Never bypass the correctness, local-worker permission, pilot, or benchmark approval gates to satisfy a commit checklist.
- Before committing code, run the applicable authorized focused tests and static checks. If execution is prohibited or the environment is unavailable, perform the allowed static review and clearly record the unexecuted checks as **NOT RUN / NOT VERIFIED**; never claim or imply they passed.
- Do not make a commit while a known relevant check is failing. If a failure is unrelated or cannot be resolved within scope, stop and report it rather than hiding or weakening the check.
- CI and Git hooks are optional engineering aids, not mandatory requirements. Introduce them only after stable commands and dependencies exist and the student approves any new workflow/dependency. CI may run linting, unit/contract tests, and safe build checks; it must not silently run benchmarks, launch unapproved workers, access paid infrastructure, or mutate frozen/raw data.
- Do not add Husky or introduce Node tooling by default in this Python/C project. Use the simplest already-available mechanism. A lightweight pre-commit setup or native Git hook may be considered later if its benefit justifies its maintenance and dependency cost.
- Do not add new dependencies, test tools, formatters, or automation without appropriate student authorization.

### 9.4 Incremental local-commit policy

**Do not accumulate all project changes for one final commit.** Create local commits progressively as the work progresses, after completing a coherent, reviewable logical unit.

The standing student instruction authorizes ordinary local commits without requesting approval for each individual commit, provided all of the following conditions are satisfied:

1. The current task changes files inside the Assignment 4 repository and is not marked read-only, no-commit, or proposal-only.
2. The change is complete enough to form a meaningful logical unit, not a temporary half-written state.
3. The relevant diff has been reviewed and contains no accidental, unrelated, or generated files.
4. Applicable authorized checks have passed, or prohibited/unavailable checks are explicitly identified as not run, and there is no known relevant failure.
5. The commit contains only the intended files and does not alter protected baselines, immutable data, or raw measurements outside the approved task.

**Commit granularity:**
- Prefer one commit per completed logical change (e.g., a feature, bug fix, test set, documentation decision, or reproducibility artifact group).
- A task/phase may produce multiple commits when it contains independently meaningful units. Conversely, several small edits that form one coherent change should be committed together.
- Do not create a commit for every save, typo, generated temporary file, or trivial edit.
- Avoid incomplete `WIP` commits on the stable branch (`main`). Do not delay commits until the end of the whole project.
- Keep implementation and its relevant tests together where that makes the commit independently understandable. Separate unrelated documentation or maintenance changes when that improves reviewability.

**Mandatory 10-step pre-commit procedure:**
Before each commit, the agent must:

1. **Verify repository root, branch, and HEAD:** Confirm working on the intended branch (`main`) at the correct root.
2. **Inspect working-tree status:** Run `git status --short`, `git diff`, and `git diff --cached`.
3. **Preserve unrelated changes:** Ensure pre-existing user-staged or unstaged changes are preserved and not accidentally committed.
4. **Stage explicit paths only:** Run `git add -- <explicit-paths>`. Never use `git add .` or `git add -A` as a shortcut.
5. **Review the entire staged diff:** Inspect `git diff --cached` line-by-line to verify only intended changes are included.
6. **Check for prohibited content:** Check for secrets, credentials, access tokens, private correspondence, personal data, local paths, build outputs, binaries, caches, external repositories (`external_repos/`), course PDFs, private prompts (`PROMPTS.md`), and accidental changes to protected baselines or datasets.
7. **Check whitespace and formatting:** Run `git diff --cached --check` to ensure no whitespace errors exist.
8. **Run authorized checks:** Run all relevant checks authorized by the current phase. If dynamic execution is prohibited, ensure static checks pass and record runtime checks as `NOT RUN / NOT VERIFIED`.
9. **Commit only the intended logical unit:** Create the commit with a descriptive message adhering to Section 9.5.
10. **Verify the commit:** Run `git show --stat --oneline HEAD` and verify the remaining working-tree status with `git status --short --branch`. Do not discard unrelated changes to obtain a clean tree.

If pre-existing staged changes, uncertain ownership, an unexpected branch state, a merge conflict, or an unclear file scope is found, do not commit those changes. Report the blocker and ask the student.

A task-specific instruction such as "read-only", "do not commit", or "return a proposal only" overrides the standing local-commit authorization for that task.

### 9.5 Commit-message convention (Conventional Commits)

Use **Conventional Commits** in English with a concise, descriptive subject:

`<type>(<scope>): <imperative summary>`

Use the following types where appropriate:
- `feat`: new user-visible or project functionality.
- `fix`: correction of a defect.
- `refactor`: structure change without intended behavior change.
- `perf`: measured or explicitly justified performance improvement.
- `test`: test-only changes or test infrastructure.
- `docs`: documentation or specification changes.
- `build`: compiler, build, or dependency configuration.
- `ci`: continuous-integration workflow changes.
- `chore`: repository maintenance that does not fit another type.

Examples:
```text
feat(data): add deterministic common matrix generator
fix(runtime): resolve Assignment 1 input directory
feat(dask): implement 2d block multiplication
test(blocks): cover irregular edge blocks
docs(protocol): define strong-scaling timing boundary
build(c): record reproducible compiler settings
```

The subject line must:
- Be written in English to match the repository's technical documentation;
- Normally be no longer than 72 characters;
- Use imperative wording, start with a lowercase type/scope, and omit the final period;
- Describe the actual change rather than a vague phrase such as `updates`, `changes`, or `fixes`.

Add a commit body when the purpose, trade-off, or validation is not clear from the subject:
- Explain **why** the change was needed and summarize important behavior changes;
- Record relevant validation commands and their actual outcome in the task report; do not claim tests passed if they were not run;
- Reference an issue or decision ID (e.g., `DEC-01`, `DEC-06`) when one exists;
- Do not invent authorship or add misleading `Co-authored-by` trailers for tools or people who did not author the change.

### 9.6 Branching and history safety

- Keep `main` as the stable, reviewable line of work.
- For a substantial independent feature, risky refactor, or isolated experiment, prefer a short-lived branch (e.g., `feat/dask-block-multiply`, `fix/a1-runtime-adapter`, `docs/experiment-protocol`). Do not create branches for every tiny change or solely to simulate a team workflow in this single-student project.
- Branch creation, checkout/switching, merging, cherry-picking, and deletion require explicit student authorization. The standing authorization in Section 9.4 covers ordinary local commits only.
- Review the branch diff and relevant tests before merging. Squashing may be appropriate for a temporary branch with noisy intermediate commits, but only when explicitly authorized and before its history is shared.
- Never rewrite published or shared history. Do not amend, rebase, reset, force-push, delete branches, tag releases, or push unless the student explicitly authorizes the exact action.
- A local commit does not imply permission to push. Pushes always require separate explicit authorization.

### 9.7 Generated, frozen and raw artifacts

- Track source code, tests, configuration, documentation, and small reproducibility manifests when they belong to the project.
- Keep build outputs, executables, object files, caches, virtual environments, temporary runtime copies (`build/a1_runtime/`, `build/a3_runtime/`), and machine-specific secrets/configuration out of Git by default.
- Never alter frozen baseline snapshots (`baselines/`) or the canonical input dataset (`data/dense/`, `data/validation/`) merely to make a later task convenient. Changes require a new version or explicit approval and provenance. Generated dense benchmark matrices (`data/dense/`) are excluded from Git under Option B; their deterministic generator, checksum manifest, and operational reconstruction guide (`data/README.md`) are version-controlled.
- Raw measurements (`results/raw/`) are append-only source evidence. A correction requires a new, traceable run/version; never silently rewrite or delete a previous sample.
- Before tracking large datasets or raw result bundles, assess repository size and choose a documented storage approach (ordinary Git, Git LFS, release artifact, or external archival location). Do not silently omit required reproducibility data, and do not introduce Git LFS or another storage dependency without approval.
- Always track or preserve the generator/provenance instructions and SHA-256 manifests needed to verify external or separately stored artifacts.
- Never commit credentials, access tokens, private instructor correspondence, personal data, or local machine secrets. Store only a necessary, sanitized record of approvals.

---

## 10. Definition of done

The project is complete only when:
- The distributed Dask implementation is documented, validated, and reproducible.
- Sequential and OpenMP baselines are compared on common sizes using identical logical inputs.
- Strong scaling is evaluated with fixed problem size across permitted configurations.
- Workers, processes, and physical machines are accurately identified.
- Timing boundaries, phase timings, memory metrics, and communication limitations are honestly reported.
- Correctness evidence, raw data, analysis, figures, README, and report agree.
- All acceptance criteria in `SPEC.md` and the official PDF are satisfied.
