#!/usr/bin/env python3
"""
scripts/prepare_a3_runtime.py — Runtime Directory Adapter for Assignment 3 Baseline
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

Purpose:
Prepares the runtime directory structure expected by the frozen Assignment 3
executable without modifying its source code:
  <root>/build/a3_runtime/data/dense/A_<n>.csv
  <root>/build/a3_runtime/data/dense/B_<n>.csv
  <root>/build/a3_runtime/data/validation/<fixtures>

The A3 executable accepts base_dir as a CLI argument and expects:
  sprintf(val_dir, "%s/data/validation", base_dir);
  sprintf(dense_dir, "%s/data/dense", base_dir);
  sprintf(results_dir, "%s/results", base_dir);

This script:
1. Copies the common dense matrices for n in {250, 500, 750, 1000, 1500, 2000} from data/dense/.
2. Places A3's exact validation fixtures into build/a3_runtime/data/validation/.
3. Verifies that SHA-256 checksums of all copied dense matrices match the central dataset.
4. Verifies that validation fixtures match Assignment 3's pre-registered hashes.
"""

import hashlib
import os
import shutil
import sys


A3_SIZES = [250, 500, 750, 1000, 1500, 2000]

A3_FIXTURE_HASHES = {
    "identity_4.csv": "00e98169ffd7d2fdc0178daf7ee6d96628ba7f979106030a24a02f091b250dd7",
    "zero_4.csv": "835439acc00f5f0d2656b407295672fadee80f10874d3375ede0a945716c9f09",
    "known_A_4.csv": "d401de25f11b07abcf0f4303024f87c8311b11ce0ccc9a5b43481f883646846b",
    "known_B_4.csv": "d3cf711abe4fb27c21f9fc2a99f4c93726be24a9b70f4b1fc7282d7f649043f3",
    "expected_C_4.csv": "7df8b59e00f12c4f587a75ab5f4810f44cf9ebb63c02ddf8483c3da6da9ef1fb",
}

A3_FIXTURE_MATRICES = {
    "identity_4.csv": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "zero_4.csv": [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    "known_A_4.csv": [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ],
    "known_B_4.csv": [
        [7.0, 6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
    ],
    "expected_C_4.csv": [
        [41.0, 39.0, 37.0, 53.0],
        [117.0, 107.0, 97.0, 141.0],
        [85.0, 76.0, 67.0, 67.0],
        [98.0, 90.0, 82.0, 119.0],
    ],
}


def sha256_of_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_synthesized_fixture(dst_path: str, rows: list) -> None:
    """Writes a 4x4 A3 CSV fixture using canonical %.6f precision and CRLF line endings."""
    payload = "".join(",".join(f"{val:.6f}" for val in row) + "\r\n" for row in rows)
    with open(dst_path, "wb") as f:
        f.write(payload.encode("ascii"))


def install_a3_fixtures(target_dir: str, a3_orig_dir: str) -> bool:
    os.makedirs(target_dir, exist_ok=True)
    all_matched = True

    for name, expected_hash in A3_FIXTURE_HASHES.items():
        src_path = os.path.join(a3_orig_dir, name)
        dst_path = os.path.join(target_dir, name)

        if os.path.isfile(src_path):
            shutil.copyfile(src_path, dst_path)
            source_mode = "copied"
        else:
            _write_synthesized_fixture(dst_path, A3_FIXTURE_MATRICES[name])
            source_mode = "synthesized"

        actual_hash = sha256_of_file(dst_path)

        if actual_hash.lower() == expected_hash.lower():
            print(f"  [OK] Installed fixture {name} ({source_mode}, SHA-256 verified)")
        else:
            print(f"  [HASH MISMATCH] Fixture {name}: expected {expected_hash}, got {actual_hash}")
            all_matched = False

    return all_matched


def prepare_a3_runtime() -> bool:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dense_dir = os.path.join(repo_root, "data", "dense")
    runtime_dir = os.path.join(repo_root, "build", "a3_runtime")
    target_dense_dir = os.path.join(runtime_dir, "data", "dense")
    target_val_dir = os.path.join(runtime_dir, "data", "validation")
    a3_orig_val = os.path.join(repo_root, "external_repos", "Assignment03", "data", "validation")

    print("=================================================================")
    print("Preparing Runtime Directory for Assignment 3 Baseline")
    print(f"Target Runtime: {runtime_dir}")
    print("=================================================================\n")

    os.makedirs(target_dense_dir, exist_ok=True)
    os.makedirs(target_val_dir, exist_ok=True)

    # 1. Copy dense matrices
    copied_count = 0
    all_matched = True

    for n in A3_SIZES:
        for prefix in ("A", "B"):
            filename = f"{prefix}_{n}.csv"
            src_file = os.path.join(dense_dir, filename)
            dst_file = os.path.join(target_dense_dir, filename)

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

    print(f"  [OK] Copied and verified {copied_count} dense matrix files into {target_dense_dir}")

    # 2. Install validation fixtures
    fixtures_ok = install_a3_fixtures(target_val_dir, a3_orig_val)
    if not fixtures_ok:
        all_matched = False

    if not all_matched:
        print("\n[FAILURE] Preparation encountered errors; runtime manifest was not finalized.")
        return False

    # 3. Save runtime manifest
    manifest_path = os.path.join(runtime_dir, "A3_RUNTIME_MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Assignment 3 Runtime Adapter Manifest\n")
        f.write(f"# Base Directory Argument to pass to A3: {runtime_dir}\n")
        for n in A3_SIZES:
            for p in ("A", "B"):
                fn = f"{p}_{n}.csv"
                f.write(f"{sha256_of_file(os.path.join(target_dense_dir, fn))}  data/dense/{fn}\n")
        for fn in sorted(A3_FIXTURE_HASHES):
            f.write(f"{A3_FIXTURE_HASHES[fn]}  data/validation/{fn}\n")

    print(f"  [OK] Runtime manifest saved: {manifest_path}")

    if all_matched:
        print("\n[SUCCESS] Assignment 3 runtime directory prepared successfully.")
        return True
    else:
        print("\n[FAILURE] Preparation encountered errors.")
        return False


if __name__ == "__main__":
    success = prepare_a3_runtime()
    sys.exit(0 if success else 1)
