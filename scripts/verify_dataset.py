#!/usr/bin/env python3
"""
scripts/verify_dataset.py — Independent Integrity and Dimension Verifier
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

This script performs strictly read-only verifications:
1. Validates every entry in data/SHA256SUMS.txt against disk files.
2. Verifies matrix dimensions (rows and columns) for all CSV files.
3. Verifies that all parsed numbers are finite IEEE 754 float64 (no NaN or Inf).
4. Verifies line endings (LF) and formatting.
"""

import argparse
import hashlib
import math
import os
import sys


def verify_sha256_manifest(manifest_path: str) -> tuple[int, int, list[str]]:
    data_dir = os.path.dirname(os.path.abspath(manifest_path))
    passed = 0
    failed = 0
    errors = []

    if not os.path.isfile(manifest_path):
        return 0, 1, [f"Manifest not found: {manifest_path}"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"Line {line_num}: malformed entry: {line}")
                failed += 1
                continue

            expected_hash, rel_path = parts
            full_path = os.path.join(data_dir, rel_path.replace("/", os.sep))

            if not os.path.isfile(full_path):
                errors.append(f"Missing file: {rel_path}")
                failed += 1
                continue

            hasher = hashlib.sha256()
            with open(full_path, "rb") as target:
                for chunk in iter(lambda: target.read(65536), b""):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()

            if actual_hash.lower() == expected_hash.lower():
                passed += 1
            else:
                errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")
                failed += 1

    return passed, failed, errors


def verify_matrix_contents(manifest_path: str) -> tuple[int, int, list[str]]:
    data_dir = os.path.dirname(os.path.abspath(manifest_path))
    passed = 0
    failed = 0
    errors = []

    if not os.path.isfile(manifest_path):
        return 0, 1, [f"Manifest not found: {manifest_path}"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"Line {line_num}: malformed entry: {line}")
                failed += 1
                continue

            _, rel_path = parts
            full_path = os.path.join(data_dir, rel_path.replace("/", os.sep))

            if not os.path.isfile(full_path):
                errors.append(f"Missing file: {rel_path}")
                failed += 1
                continue

            # Determine expected dimension from filename
            basename = os.path.basename(rel_path)
            expected_n = None
            if basename.endswith("_4.csv"):
                expected_n = 4
            elif "_" in basename and basename.endswith(".csv"):
                size_str = basename.split("_")[-1].replace(".csv", "")
                if size_str.isdigit():
                    expected_n = int(size_str)

            # Inspect full file for CR ('\r') bytes across all line lengths
            row_count = 0
            has_cr = False
            all_finite = True

            with open(full_path, "rb") as raw_f:
                for chunk in iter(lambda: raw_f.read(65536), b""):
                    if b"\r" in chunk:
                        has_cr = True
                        break

            if has_cr:
                errors.append(f"{rel_path}: Contains Windows CR ('\\r') characters; LF expected.")
                failed += 1
                continue

            with open(full_path, "r", encoding="utf-8") as text_f:
                for r_idx, text_line in enumerate(text_f):
                    row_count += 1
                    cols = text_line.strip().split(",")
                    if expected_n is not None and len(cols) != expected_n:
                        errors.append(f"{rel_path}: Row {r_idx} has {len(cols)} columns, expected {expected_n}")
                        all_finite = False
                        break
                    for val_str in cols:
                        try:
                            val = float(val_str)
                            if not math.isfinite(val):
                                errors.append(f"{rel_path}: Non-finite value encountered in row {r_idx}")
                                all_finite = False
                                break
                        except ValueError:
                            errors.append(f"{rel_path}: Unparseable float '{val_str}' in row {r_idx}")
                            all_finite = False
                            break
                    if not all_finite:
                        break

            if expected_n is not None and row_count != expected_n:
                errors.append(f"{rel_path}: Row count is {row_count}, expected {expected_n}")
                failed += 1
            elif not all_finite:
                failed += 1
            else:
                passed += 1

    return passed, failed, errors


def main(argv: list[str] | None = None) -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_manifest = os.path.join(repo_root, "data", "SHA256SUMS.txt")

    parser = argparse.ArgumentParser(
        description="Verify dataset SHA-256 checksums, LF line endings, dimensions, and finite float64 values."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=default_manifest,
        help="Optional path to SHA256SUMS manifest (default: data/SHA256SUMS.txt)",
    )
    args = parser.parse_args(argv)
    manifest_path = os.path.abspath(args.manifest)

    print("=================================================================")
    print("Assignment 4: Dataset Integrity Verification")
    print(f"Manifest: {manifest_path}")
    print("=================================================================\n")

    p_hash, f_hash, err_hash = verify_sha256_manifest(manifest_path)
    print(f"SHA-256 Checksum Verification: {p_hash} passed, {f_hash} failed.")
    for err in err_hash:
        print(f"  [ERROR] {err}")

    p_cont, f_cont, err_cont = verify_matrix_contents(manifest_path)
    print(f"Matrix Structure & Dimension Verification: {p_cont} passed, {f_cont} failed.")
    for err in err_cont:
        print(f"  [ERROR] {err}")

    if f_hash == 0 and f_cont == 0:
        print("\n[SUCCESS] All data files verified bit-for-bit with correct dimensions and finite values.")
        return 0
    else:
        print("\n[FAILURE] Integrity verification encountered errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
