#!/usr/bin/env python3
"""
scripts/prepare_a1_runtime.py — Runtime Directory Adapter for Assignment 1 Baseline
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

Purpose:
Prepares the runtime directory structure expected by the frozen Assignment 1
executable without modifying its source code:
  <root>/build/a1_runtime/data/A_<n>.csv
  <root>/build/a1_runtime/data/B_<n>.csv
  <root>/build/a1_runtime/data/<fixtures>

The A1 executable accepts argv[1] as its base directory and looks for data in:
  sprintf(data_dir, "%s/data", base_dir);

This script:
1. Copies the common dense matrices for n in {100, 250, 500, 750, 1000} from data/dense/.
2. Places A1's original validation fixtures into build/a1_runtime/data/.
3. Verifies that SHA-256 checksums of all copied dense matrices match the central dataset.
"""

import hashlib
import os
import shutil
import sys


A1_SIZES = [100, 250, 500, 750, 1000]


def sha256_of_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_a1_fixtures(target_dir: str) -> dict[str, str]:
    """
    Writes the exact validation fixtures required by A1's validate_correctness().
    A1 expects:
      known_A_4.csv, identity_4.csv, zero_4.csv, known_B_4.csv, expected_C_4.csv
    Values correspond exactly to Assignment 1's generate_matrices.py.
    """
    os.makedirs(target_dir, exist_ok=True)
    n = 4

    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]
    known_a = [
        [1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    known_b = [
        [2.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 3.0],
    ]
    # Expected product: known_a x known_b
    expected_c = [
        [5.0, 10.0, 7.0, 14.0],
        [17.0, 26.0, 19.0, 30.0],
        [29.0, 42.0, 31.0, 46.0],
        [41.0, 58.0, 43.0, 62.0],
    ]

    fixtures = {
        "identity_4.csv": identity,
        "zero_4.csv": zero,
        "known_A_4.csv": known_a,
        "known_B_4.csv": known_b,
        "expected_C_4.csv": expected_c,
    }

    hashes = {}
    for name, mat in fixtures.items():
        path = os.path.join(target_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            for row in mat:
                f.write(",".join(f"{val:.6f}" for val in row) + "\n")
        hashes[name] = sha256_of_file(path)

    return hashes


def prepare_a1_runtime() -> bool:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dense_dir = os.path.join(repo_root, "data", "dense")
    runtime_dir = os.path.join(repo_root, "build", "a1_runtime")
    target_data_dir = os.path.join(runtime_dir, "data")

    print("=================================================================")
    print("Preparing Runtime Directory for Assignment 1 Baseline")
    print(f"Target Runtime: {runtime_dir}")
    print("=================================================================\n")

    os.makedirs(target_data_dir, exist_ok=True)

    # 1. Copy dense matrices
    copied_count = 0
    all_matched = True

    for n in A1_SIZES:
        for prefix in ("A", "B"):
            filename = f"{prefix}_{n}.csv"
            src_file = os.path.join(dense_dir, filename)
            dst_file = os.path.join(target_data_dir, filename)

            if not os.path.isfile(src_file):
                print(f"  [ERROR] Source file missing: {src_file}")
                all_matched = False
                continue

            shutil.copyfile(src_file, dst_file)
            src_hash = sha256_of_file(src_file)
            dst_hash = sha256_of_file(dst_file)

            if src_hash == dst_hash:
                copied_count += 1
            else:
                print(f"  [HASH MISMATCH] {filename}: {src_hash} != {dst_hash}")
                all_matched = False

    print(f"  [OK] Copied and verified {copied_count} dense matrix files into {target_data_dir}")

    # 2. Write A1 original validation fixtures
    fixture_hashes = write_a1_fixtures(target_data_dir)
    print(f"  [OK] Installed {len(fixture_hashes)} validation fixtures in {target_data_dir}")

    # 3. Save runtime manifest
    manifest_path = os.path.join(runtime_dir, "A1_RUNTIME_MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Assignment 1 Runtime Adapter Manifest\n")
        f.write(f"# Base Directory Argument to pass to A1: {runtime_dir}\n")
        for n in A1_SIZES:
            for p in ("A", "B"):
                fn = f"{p}_{n}.csv"
                f.write(f"{sha256_of_file(os.path.join(target_data_dir, fn))}  data/{fn}\n")
        for fn in sorted(fixture_hashes):
            f.write(f"{fixture_hashes[fn]}  data/{fn}\n")

    print(f"  [OK] Runtime manifest saved: {manifest_path}")

    if all_matched:
        print("\n[SUCCESS] Assignment 1 runtime directory prepared successfully.")
        return True
    else:
        print("\n[FAILURE] Preparation encountered errors.")
        return False


if __name__ == "__main__":
    success = prepare_a1_runtime()
    sys.exit(0 if success else 1)
