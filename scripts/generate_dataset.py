#!/usr/bin/env python3
"""
scripts/generate_dataset.py — Deterministic Dataset and Validation Generator
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

This script generates:
1. Dense benchmark matrices A_<n>.csv and B_<n>.csv in data/dense/
   - Isolated seed per dimension: random.Random(42 + n)
   - Value range: [0.0, 1.0)
   - IEEE 754 binary64 precision using '%.17g'
   - Pure Python standard library (no external dependencies)
   - Dimensions: 100 (auxiliary for A1), 250, 500, 750, 1000, 1500, 2000
2. Small validation fixtures (4x4) in data/validation/
   - identity_4.csv, zero_4.csv, known_A_4.csv, known_B_4.csv, expected_C_4.csv
3. Independent deterministic reference product for n=250 in data/validation/reference/C_250.csv
   - Computed with pure Python standard arithmetic (independent of NumPy/BLAS)
4. SHA-256 manifest in data/SHA256SUMS.txt
"""

import csv
import hashlib
import os
import random
import sys


DENSE_SIZES = [100, 250, 500, 750, 1000, 1500, 2000]
AUXILIARY_SIZES = [100]  # Auxiliary for A1 compatibility; excluded from common 3-way benchmark
COMMON_SIZES = [250, 500, 750, 1000]
DISTRIBUTED_SCALING_SIZES = [1500, 2000]
VALIDATION_SIZE = 4
BASE_SEED = 42


def get_dimension_seed(n: int) -> int:
    return BASE_SEED + n


def write_csv_matrix(filepath: str, matrix: list[list[float]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # Explicit LF line endings
        for row in matrix:
            line = ",".join(f"{val:.17g}" for val in row) + "\n"
            f.write(line)


def read_csv_matrix(filepath: str) -> list[list[float]]:
    matrix = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                matrix.append([float(val) for val in row])
    return matrix


def pure_python_matmul(a: list[list[float]], b: list[list[float]], n: int) -> list[list[float]]:
    """
    Standard triple-loop matrix multiplication in pure Python.
    Serves as an independent numerical reference, completely isolated
    from NumPy, OpenBLAS, MKL, or Dask.
    """
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c


def generate_dense_matrices(dense_dir: str) -> list[str]:
    generated_files = []
    print("=== Generating Dense Input Matrices ===")
    print(f"  Generator: Python {sys.version.split()[0]} standard random.Random")
    print(f"  Numeric Format: IEEE 754 binary64, format string '%.17g'")
    print(f"  Line Endings: Unix LF ('\\n')")

    for n in DENSE_SIZES:
        seed = get_dimension_seed(n)
        rng = random.Random(seed)

        # Generation order: A first (row-major), then B (row-major)
        matrix_a = [[rng.random() for _ in range(n)] for _ in range(n)]
        matrix_b = [[rng.random() for _ in range(n)] for _ in range(n)]

        file_a = os.path.join(dense_dir, f"A_{n}.csv")
        file_b = os.path.join(dense_dir, f"B_{n}.csv")

        write_csv_matrix(file_a, matrix_a)
        write_csv_matrix(file_b, matrix_b)

        generated_files.extend([file_a, file_b])
        role = " (Auxiliary A1)" if n in AUXILIARY_SIZES else ""
        print(f"  [OK] Dimension {n}x{n}{role}: seed={seed}, files: A_{n}.csv, B_{n}.csv")

    return generated_files


def generate_validation_fixtures(val_dir: str) -> list[str]:
    print("\n=== Generating Validation Fixtures ===")
    n = VALIDATION_SIZE
    generated_files = []

    # 1. Identity matrix fixture: A x I = A
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    # 2. Zero matrix fixture: A x 0 = 0
    zero = [[0.0] * n for _ in range(n)]

    # 3. Known analytical integer matrices with exact product (zero round-off error)
    known_a = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]
    known_b = [
        [7.0, 6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    expected_c = pure_python_matmul(known_a, known_b, n)

    fixtures = {
        "identity_4.csv": identity,
        "zero_4.csv": zero,
        "known_A_4.csv": known_a,
        "known_B_4.csv": known_b,
        "expected_C_4.csv": expected_c,
    }

    for name, mat in fixtures.items():
        path = os.path.join(val_dir, name)
        write_csv_matrix(path, mat)
        generated_files.append(path)
        print(f"  [OK] Validation fixture: {name}")

    return generated_files


def generate_reference_product(val_dir: str, dense_dir: str) -> str:
    print("\n=== Generating Independent Reference Product (n=250) ===")
    ref_dir = os.path.join(val_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    n = 250
    file_a = os.path.join(dense_dir, f"A_{n}.csv")
    file_b = os.path.join(dense_dir, f"B_{n}.csv")
    ref_path = os.path.join(ref_dir, f"C_{n}.csv")

    print(f"  Loading input matrices A_{n}.csv and B_{n}.csv...")
    mat_a = read_csv_matrix(file_a)
    mat_b = read_csv_matrix(file_b)

    print(f"  Computing independent reference product via pure Python standard arithmetic...")
    mat_c = pure_python_matmul(mat_a, mat_b, n)

    write_csv_matrix(ref_path, mat_c)
    print(f"  [OK] Saved independent reference product: C_{n}.csv")
    return ref_path


def write_sha256_manifest(data_dir: str) -> str:
    print("\n=== Generating SHA-256 Manifest ===")
    data_root = os.path.abspath(data_dir)
    matrix_files = []

    for subdir in ("dense", "validation"):
        subdir_path = os.path.join(data_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for root, _, files in os.walk(subdir_path):
            for f in files:
                if f.endswith(".csv"):
                    matrix_files.append(os.path.join(root, f))

    matrix_files.sort()
    lines = [
        "# SHA-256 Checksums for Assignment 4 Shared Input Matrices and Validation Fixtures",
        f"# Generated with Python {sys.version.split()[0]} using standard library hashlib",
        "# Format: <sha256>  <relative_path>",
    ]

    for filepath in matrix_files:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        rel_path = os.path.relpath(filepath, data_root).replace(os.sep, "/")
        lines.append(f"{digest}  {rel_path}")

    manifest_path = os.path.join(data_root, "SHA256SUMS.txt")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [OK] Manifest created at {manifest_path} ({len(matrix_files)} CSV files indexed).")
    return manifest_path


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    dense_dir = os.path.join(data_dir, "dense")
    val_dir = os.path.join(data_dir, "validation")

    print("=================================================================")
    print("Assignment 4: Dataset and Validation Fixture Generator")
    print(f"Target Data Directory: {data_dir}")
    print("=================================================================\n")

    dense_files = generate_dense_matrices(dense_dir)
    val_files = generate_validation_fixtures(val_dir)
    ref_file = generate_reference_product(val_dir, dense_dir)
    manifest = write_sha256_manifest(data_dir)

    print("\n=================================================================")
    print("[SUCCESS] All dataset artifacts generated and hashed successfully.")
    print("=================================================================")


if __name__ == "__main__":
    main()
