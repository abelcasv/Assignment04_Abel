#!/usr/bin/env python3
"""
scripts/run_common_input_baselines.py — Controlled Common-Input Baseline Runner (A1 & A3)
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

Executes the frozen, unmodified Assignment 1 sequential C baseline and Assignment 3
sequential/OpenMP C baseline on the certified Assignment 4 deterministic common
input matrices (`data/dense/A_<n>.csv`, `data/dense/B_<n>.csv`), preserving raw
observations and complete experimental provenance without altering historical
baseline summaries (`data/baselines/`) or Dask raw measurements (`results/raw/raw_measurements.csv`).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.prepare_a1_runtime import A1_SIZES, prepare_a1_runtime
from scripts.prepare_a3_runtime import A3_SIZES, prepare_a3_runtime
from scripts.verify_dataset import verify_sha256_manifest

FROZEN_SOURCE_HASHES = {
    "baselines/a1_sequential/benchmark.c": "4893fc553e97a0ec596c3c1cf9d5d8f582fc49308fc5971b338f48fd0268236f",
    "baselines/a1_sequential/Makefile": "b35b8cf8671d4cb0bdc42b59f0708dd8e55ebe51842034d3046c60da82f8c31f",
    "baselines/a3_openmp/parallel_benchmark.c": "6da20a3aed0862adc296fe9f4c8625bc51cd777959a5403088f7960c8741dbc8",
    "baselines/a3_openmp/Makefile": "474e4d2f2b57a5696bd8202ac0e125b101dbe6bcd5f58bd92df2a23f8a37e364",
}

A1_ORIGIN_COMMIT = "90681e85949b6d65bdbcfef702b6306b0003f971"
A3_ORIGIN_COMMIT = "23cffbc82c155b6c3de24b3c149b509b63fc870b"

A1_CFLAGS = "-Wall -Wextra -O3 -std=c11"
A1_LDFLAGS = "-lpsapi"
A3_CFLAGS = "-Wall -Wextra -O3 -std=c11 -fopenmp -march=native"
A3_LDFLAGS = "-lpsapi" if os.name == "nt" else "-lm"

COMMON_COMPARISON_SIZES_3WAY = [250, 500, 750, 1000]
EXTENDED_COMPARISON_SIZES_A3_DASK = [1500, 2000]


def sha256_file(filepath: str) -> str:
    """Computes lower-case SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_embedded_gcc_ident(binary_path: str) -> List[str]:
    """Extracts embedded GCC .ident strings from a compiled executable."""
    with open(binary_path, "rb") as f:
        blob = f.read()
    matches = re.findall(rb"GCC: \([^\x00\r\n]{1,100}", blob)
    return sorted({m.decode("ascii", errors="ignore").strip() for m in matches})


def resolve_gcc_binary() -> Optional[str]:
    """Finds gcc on PATH or standard Windows MSYS2 UCRT64 path."""
    found = shutil.which("gcc")
    if found:
        return found
    msys_gcc = r"C:\msys64\ucrt64\bin\gcc.exe"
    if os.path.isfile(msys_gcc):
        return msys_gcc
    return None


def build_runtime_env(gcc_path: Optional[str]) -> Dict[str, str]:
    """Builds subprocess environment ensuring compiler runtime DLLs are discoverable."""
    env = os.environ.copy()
    # Keep OpenMP environment unpinned so A3's explicit per-region num_threads(threads)
    # clauses control thread counts identically to the validated Assignment 3 protocol.
    for key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE", "OMP_THREAD_LIMIT"):
        env.pop(key, None)
    if gcc_path:
        gcc_dir = os.path.dirname(os.path.abspath(gcc_path))
        env["PATH"] = gcc_dir + os.pathsep + env.get("PATH", "")
    elif os.name == "nt" and os.path.isdir(r"C:\msys64\ucrt64\bin"):
        env["PATH"] = r"C:\msys64\ucrt64\bin" + os.pathsep + env.get("PATH", "")
    return env


def verify_frozen_sources_and_dataset(repo_root: str) -> Dict[str, str]:
    """Verifies frozen baseline C snapshots and data/SHA256SUMS.txt bit-for-bit."""
    for rel_path, expected_sha in FROZEN_SOURCE_HASHES.items():
        abs_path = os.path.join(repo_root, *rel_path.split("/"))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Frozen baseline snapshot missing: {abs_path}")
        actual_sha = sha256_file(abs_path)
        if actual_sha.lower() != expected_sha.lower():
            raise ValueError(
                f"Frozen snapshot SHA-256 mismatch for {rel_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    manifest_path = os.path.join(repo_root, "data", "SHA256SUMS.txt")
    passed, failed, errors = verify_sha256_manifest(manifest_path)
    if failed > 0 or passed != 20:
        raise ValueError(
            f"Dataset SHA-256 verification failed (passed={passed}, failed={failed}): {errors}"
        )
    dataset_hashes: Dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            expected_sha, rel_data_path = stripped.split(maxsplit=1)
            dataset_hashes[f"data/{rel_data_path.strip()}"] = expected_sha.lower()

    return dataset_hashes


def verify_dask_input_hash_parity(
    repo_root: str, dataset_hashes: Dict[str, str]
) -> Dict[str, Any]:
    """Verifies that existing Dask measurements in raw_measurements.csv used identical input hashes."""
    dask_csv = os.path.join(repo_root, "results", "raw", "raw_measurements.csv")
    with open(dask_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parity_by_n: Dict[str, Any] = {}
    for n in COMMON_COMPARISON_SIZES_3WAY + EXTENDED_COMPARISON_SIZES_A3_DASK:
        expected_a = dataset_hashes[f"data/dense/A_{n}.csv"]
        expected_b = dataset_hashes[f"data/dense/B_{n}.csv"]
        matching_completed = [
            r
            for r in rows
            if int(r["n"]) == n
            and r["status"] == "COMPLETED"
            and r["is_warmup"].lower() == "false"
            and r["input_A_sha256"].lower() == expected_a
            and r["input_B_sha256"].lower() == expected_b
        ]
        if not matching_completed:
            raise ValueError(f"No completed Dask rows with matching input SHA-256 found for n={n}.")
        parity_by_n[str(n)] = {
            "input_A_sha256": expected_a,
            "input_B_sha256": expected_b,
            "dask_completed_measured_rows_matching": len(matching_completed),
            "dask_worker_counts_present": sorted({int(r["worker_count"]) for r in matching_completed}),
        }
    return parity_by_n


def ensure_binaries_compiled(
    repo_root: str, gcc_path: Optional[str], compile_if_missing: bool
) -> Dict[str, str]:
    """Ensures A1 and A3 binaries exist in build/bin/, compiling from frozen snapshots if requested."""
    bin_dir = os.path.join(repo_root, "build", "bin")
    a1_exe = os.path.join(bin_dir, "a1_benchmark.exe" if os.name == "nt" else "a1_benchmark")
    a3_exe = os.path.join(
        bin_dir, "a3_parallel_benchmark.exe" if os.name == "nt" else "a3_parallel_benchmark"
    )

    if (not os.path.isfile(a1_exe) or not os.path.isfile(a3_exe)) and not compile_if_missing:
        raise FileNotFoundError(
            f"Compiled baseline binaries missing ({a1_exe}, {a3_exe}). Pass --compile-if-missing."
        )

    env = build_runtime_env(gcc_path)
    if not os.path.isfile(a1_exe):
        if not gcc_path:
            raise RuntimeError("GCC compiler not found to build a1_benchmark.")
        os.makedirs(bin_dir, exist_ok=True)
        src_a1 = os.path.join(repo_root, "baselines", "a1_sequential", "benchmark.c")
        cmd_a1 = [gcc_path, *A1_CFLAGS.split(), src_a1, "-o", a1_exe, A1_LDFLAGS]
        subprocess.run(cmd_a1, check=True, env=env)

    if not os.path.isfile(a3_exe):
        if not gcc_path:
            raise RuntimeError("GCC compiler not found to build a3_parallel_benchmark.")
        os.makedirs(bin_dir, exist_ok=True)
        src_a3 = os.path.join(repo_root, "baselines", "a3_openmp", "parallel_benchmark.c")
        cmd_a3 = [
            gcc_path,
            *A3_CFLAGS.split(),
            f'-DBUILD_FLAGS="{A3_CFLAGS}"',
            src_a3,
            "-o",
            a3_exe,
            A3_LDFLAGS,
        ]
        subprocess.run(cmd_a3, check=True, env=env)

    return {"a1_exe": a1_exe, "a3_exe": a3_exe}


def parse_kv_file(filepath: str) -> Dict[str, str]:
    """Parses simple key: value metadata text emitted by A3."""
    result: Dict[str, str] = {}
    if not os.path.isfile(filepath):
        return result
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()
    return result


def get_git_head(repo_root: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute frozen A1 and A3 baselines on certified common A4 input matrices."
    )
    parser.add_argument(
        "--compile-if-missing",
        action="store_true",
        default=True,
        help="Compile frozen A1/A3 sources to build/bin/ if executables are absent (default: True).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting existing common-input raw CSV artifacts.",
    )
    parser.add_argument(
        "--a1-timeout-s",
        type=float,
        default=600.0,
        help="Timeout in seconds for full A1 suite invocation across n in {100..1000} (default: 600s).",
    )
    parser.add_argument(
        "--a3-timeout-s",
        type=float,
        default=3600.0,
        help="Timeout in seconds for full A3 suite invocation across n in {250..2000} (default: 3600s).",
    )
    args = parser.parse_args()

    out_a1_csv = os.path.join(REPO_ROOT, "results", "raw", "a1_common_input_raw.csv")
    out_a3_csv = os.path.join(REPO_ROOT, "results", "raw", "a3_common_input_raw.csv")
    out_manifest = os.path.join(
        REPO_ROOT, "results", "metadata", "common_input_baselines_manifest.json"
    )

    if not args.overwrite:
        for existing in (out_a1_csv, out_a3_csv):
            if os.path.isfile(existing):
                sys.stderr.write(
                    f"[ERROR] Output raw file already exists: {existing}. "
                    "Pass --overwrite only if re-execution is explicitly authorized.\n"
                )
                return 1

    print("=" * 80)
    print("ASSIGNMENT 4 — CONTROLLED COMMON-INPUT BASELINE EXECUTION (A1 & A3)")
    print("=" * 80)

    # 1. Verify frozen baseline snapshots, dataset SHA-256s, and Dask input parity
    dataset_hashes = verify_frozen_sources_and_dataset(REPO_ROOT)
    dask_parity = verify_dask_input_hash_parity(REPO_ROOT, dataset_hashes)
    print("[OK] Verified 4 frozen baseline source files and 20 dataset SHA-256 hashes.")
    print("[OK] Verified bit-for-bit input matrix SHA-256 parity with Dask raw_measurements.csv.")

    # 2. Ensure compiled executables exist and inspect compiler metadata
    gcc_path = resolve_gcc_binary()
    gcc_version_line = "not_invoked"
    if gcc_path:
        gcc_proc = subprocess.run([gcc_path, "--version"], capture_output=True, text=True, check=True)
        gcc_version_line = gcc_proc.stdout.splitlines()[0].strip()

    bins = ensure_binaries_compiled(REPO_ROOT, gcc_path, compile_if_missing=args.compile_if_missing)
    a1_exe = bins["a1_exe"]
    a3_exe = bins["a3_exe"]
    a1_ident = extract_embedded_gcc_ident(a1_exe)
    a3_ident = extract_embedded_gcc_ident(a3_exe)
    print(f"[OK] A1 binary: {a1_exe} | ident={a1_ident}")
    print(f"[OK] A3 binary: {a3_exe} | ident={a3_ident}")

    # 3. Prepare clean isolated runtime directories with common matrices
    if not prepare_a1_runtime():
        sys.stderr.write("[ERROR] prepare_a1_runtime() failed.\n")
        return 1
    if not prepare_a3_runtime():
        sys.stderr.write("[ERROR] prepare_a3_runtime() failed.\n")
        return 1

    env = build_runtime_env(gcc_path)
    run_id = f"common_baselines_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
    vm_before = psutil.virtual_memory()
    started_utc = datetime.now(timezone.utc).isoformat()
    t_suite_start = time.perf_counter()

    # 4. Execute Assignment 1 Sequential Baseline on common inputs
    a1_runtime_dir = os.path.join(REPO_ROOT, "build", "a1_runtime")
    print("\n[RUN 1/2] Executing Assignment 1 Sequential C baseline on common inputs...")
    t_a1_0 = time.perf_counter()
    a1_started_utc = datetime.now(timezone.utc).isoformat()
    proc_a1 = subprocess.run(
        [a1_exe, a1_runtime_dir],
        cwd=a1_runtime_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.a1_timeout_s,
    )
    a1_wall_s = time.perf_counter() - t_a1_0
    a1_completed_utc = datetime.now(timezone.utc).isoformat()
    if proc_a1.returncode != 0:
        sys.stderr.write(f"[ERROR] A1 baseline failed (exit {proc_a1.returncode}):\n{proc_a1.stderr}\n")
        return 1
    print(f"[OK] A1 completed in {a1_wall_s:.2f} s.")

    a1_src_csv = os.path.join(a1_runtime_dir, "results", "c_results.csv")
    with open(a1_src_csv, "r", encoding="utf-8", newline="") as f:
        a1_rows = list(csv.DictReader(f))
    if len(a1_rows) != len(A1_SIZES) * 5:
        raise RuntimeError(f"Expected {len(A1_SIZES) * 5} A1 rows, found {len(a1_rows)}.")

    # Brief thermal/OS settle pause before A3
    time.sleep(2.0)

    # 5. Execute Assignment 3 Sequential + OpenMP Baseline on common inputs
    a3_runtime_dir = os.path.join(REPO_ROOT, "build", "a3_runtime")
    a3_results_csv_in_runtime = os.path.join(a3_runtime_dir, "results", "parallel_results.csv")
    print("\n[RUN 2/2] Executing Assignment 3 Sequential & OpenMP C baseline on common inputs...")
    t_a3_0 = time.perf_counter()
    a3_started_utc = datetime.now(timezone.utc).isoformat()
    proc_a3 = subprocess.run(
        [a3_exe, a3_runtime_dir, "--output", a3_results_csv_in_runtime],
        cwd=a3_runtime_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.a3_timeout_s,
    )
    a3_wall_s = time.perf_counter() - t_a3_0
    a3_completed_utc = datetime.now(timezone.utc).isoformat()
    if proc_a3.returncode != 0:
        sys.stderr.write(f"[ERROR] A3 baseline failed (exit {proc_a3.returncode}):\n{proc_a3.stderr}\n")
        return 1
    print(f"[OK] A3 completed in {a3_wall_s:.2f} s.")

    with open(a3_results_csv_in_runtime, "r", encoding="utf-8", newline="") as f:
        a3_rows = list(csv.DictReader(f))
    expected_a3_rows = len(A3_SIZES) * (2 + 2 * 8) * 5  # 6 sizes * 18 configs * 5 reps = 540
    if len(a3_rows) != expected_a3_rows:
        raise RuntimeError(f"Expected {expected_a3_rows} A3 rows, found {len(a3_rows)}.")

    a3_env_meta = parse_kv_file(a3_results_csv_in_runtime + ".environment.txt")
    total_wall_s = time.perf_counter() - t_suite_start
    completed_utc = datetime.now(timezone.utc).isoformat()
    vm_after = psutil.virtual_memory()

    # 6. Copy raw CSVs to results/raw/ with normalized LF line endings for cross-platform hashing
    os.makedirs(os.path.dirname(out_a1_csv), exist_ok=True)
    with open(a1_src_csv, "r", encoding="utf-8", newline="") as src, open(
        out_a1_csv, "w", encoding="utf-8", newline="\n"
    ) as dst:
        dst.write(src.read().replace("\r\n", "\n"))

    with open(a3_results_csv_in_runtime, "r", encoding="utf-8", newline="") as src, open(
        out_a3_csv, "w", encoding="utf-8", newline="\n"
    ) as dst:
        dst.write(src.read().replace("\r\n", "\n"))

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "description": (
            "Controlled common-input execution of frozen Assignment 1 (sequential C) and "
            "Assignment 3 (sequential & OpenMP C) baselines using the exact deterministic "
            "IEEE 754 binary64 matrices in data/dense/ used by Assignment 4 Dask campaigns."
        ),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "total_wall_duration_s": total_wall_s,
        "git_commit_sha_at_execution": get_git_head(REPO_ROOT),
        "host_environment": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_processors": psutil.cpu_count(logical=True),
            "total_ram_mib": vm_before.total / (1024.0 * 1024.0),
            "available_ram_before_mib": vm_before.available / (1024.0 * 1024.0),
            "available_ram_after_mib": vm_after.available / (1024.0 * 1024.0),
            "os_platform": platform.platform(),
            "python_orchestrator_version": sys.version,
        },
        "compiler_and_build_provenance": {
            "gcc_cli_version": gcc_version_line,
            "a1_binary_embedded_ident": a1_ident,
            "a3_binary_embedded_ident": a3_ident,
            "a1_cflags": A1_CFLAGS,
            "a1_ldflags": A1_LDFLAGS,
            "a3_cflags": A3_CFLAGS,
            "a3_ldflags": A3_LDFLAGS,
            "a3_runtime_environment_record": a3_env_meta,
        },
        "source_and_binary_provenance": {
            "a1_origin_commit": A1_ORIGIN_COMMIT,
            "a3_origin_commit": A3_ORIGIN_COMMIT,
            "frozen_source_sha256": FROZEN_SOURCE_HASHES,
            "a1_binary_sha256": sha256_file(a1_exe).upper(),
            "a3_binary_sha256": sha256_file(a3_exe).upper(),
        },
        "common_input_dataset_provenance": {
            "generator_script": "scripts/generate_dataset.py",
            "seed_formula": "42 + n",
            "prng": "Python random.Random (MT19937)",
            "value_range": "[0.0, 1.0)",
            "dtype": "IEEE 754 binary64 (double / float64)",
            "common_3way_dimensions": COMMON_COMPARISON_SIZES_3WAY,
            "extended_2way_a3_dask_dimensions": EXTENDED_COMPARISON_SIZES_A3_DASK,
            "a1_auxiliary_dimension": 100,
            "dask_input_hash_parity_by_n": dask_parity,
        },
        "timing_boundaries": {
            "a1_sequential_c": {
                "boundary_id": "TB_A1_KERNEL_QPC",
                "timer": "Windows QueryPerformanceCounter",
                "includes": ["memset(c, 0, n^2 * 8)", "triple i-j-k multiplication loop"],
                "excludes": ["CSV disk loading", "heap allocation/free", "volatile c[0] sink assignment"],
                "output_consumption": "volatile double result_sink = c[0] (post-stop; full O(n^3) loop verified in GCC 16.1.0 -O3 assembly)",
            },
            "a3_sequential_and_openmp_c": {
                "boundary_id": "TB_A3_KERNEL_QPC",
                "timer": "Windows QueryPerformanceCounter",
                "includes": [
                    "memset(c, 0, n^2 * 8)",
                    "OpenMP parallel region management (for omp_* methods)",
                    "matrix multiplication loop (ikj or 32x32 tiled)",
                ],
                "excludes": [
                    "CSV disk loading",
                    "heap allocation/free",
                    "post-stop full-matrix checksum loop into volatile result_sink",
                ],
                "output_consumption": "full n^2 summation into volatile double result_sink immediately after stop timer",
            },
            "a4_dask_distributed": {
                "boundary_id": "TB_DASK_E2E",
                "timer": "Python time.perf_counter()",
                "includes": [
                    "2D block partitioning",
                    "Client.scatter to workers",
                    "DAG submission (matmul + binary-tree add)",
                    "distributed worker execution & loopback IPC transfers",
                    "Client.gather to driver",
                    "dense matrix reassembly",
                    "full-matrix checksum float(np.sum(C, dtype=np.float64))",
                ],
                "excludes": ["LocalCluster startup", "CSV disk loading", "post-run elementwise validation"],
            },
        },
        "executions": {
            "a1_common_input": {
                "started_utc": a1_started_utc,
                "completed_utc": a1_completed_utc,
                "wall_duration_s": a1_wall_s,
                "dimensions_executed": A1_SIZES,
                "warmups_per_config": 1,
                "measured_repetitions_per_config": 5,
                "total_measured_rows": len(a1_rows),
                "failed_or_timed_out_runs": 0,
                "pre_benchmark_validation_passed": "All correctness checks passed." in proc_a1.stdout,
                "raw_csv_path": "results/raw/a1_common_input_raw.csv",
                "raw_csv_sha256": sha256_file(out_a1_csv).upper(),
            },
            "a3_common_input": {
                "started_utc": a3_started_utc,
                "completed_utc": a3_completed_utc,
                "wall_duration_s": a3_wall_s,
                "dimensions_executed": A3_SIZES,
                "methods_executed": ["seq_ikj", "seq_tiled_b32", "omp_ikj", "omp_tiled_b32"],
                "openmp_threads_executed": [1, 2, 4, 8, 10, 12, 16, 24],
                "warmups_per_config": 1,
                "measured_repetitions_per_config": 5,
                "total_measured_rows": len(a3_rows),
                "failed_or_timed_out_runs": 0,
                "pre_benchmark_validation_passed": "[SUCCESS] All validation tests passed cleanly!" in proc_a3.stdout,
                "raw_csv_path": "results/raw/a3_common_input_raw.csv",
                "raw_csv_sha256": sha256_file(out_a3_csv).upper(),
            },
        },
    }

    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("[SUCCESS] Common-input baselines completed and preserved:")
    print(f"  A1 Raw CSV ({len(a1_rows)} rows): {out_a1_csv}")
    print(f"  A3 Raw CSV ({len(a3_rows)} rows): {out_a3_csv}")
    print(f"  Provenance Manifest:         {out_manifest}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
