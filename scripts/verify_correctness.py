"""
scripts/verify_correctness.py — Standalone Numerical Correctness Validation Suite

Executes the small-matrix correctness checks mandated for Phase 5:
1. Analytical 4x4 known-answer multiplication (b=2).
2. Non-divisible edge-block partitioning (n=4, b=3).
3. Identity property fixture: A x I = A (n=4, b=2).
4. Zero property fixture: A x 0 = 0 (n=4, b=2).
5. Cross-check n=250 common input (b=100) element-by-element against
   the trusted sequential reference C_250.csv.
6. Checksum consumption verification and finite-value guarantees.

Strictly adheres to tolerances: atol = 1e-9, rtol = 1e-12.
Does NOT execute performance benchmarks or measure execution speed.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.dask_matrix.partitioning import (
    compute_grid_dimensions,
    partition_matrix_2d,
    assemble_matrix_2d,
)
from src.dask_matrix.kernels import (
    block_multiply,
    block_add,
    compute_matrix_checksum,
)
from src.dask_matrix.validation import MatrixValidator

VAL_DIR = os.path.join(REPO_ROOT, "data", "validation")
DENSE_DIR = os.path.join(REPO_ROOT, "data", "dense")


def simulate_block_matmul_2d(a: np.ndarray, b: np.ndarray, block_size: int) -> np.ndarray:
    """
    Executes 2D block matrix multiplication and binary-tree reduction locally.
    Uses exact identical kernel and reduction logic to verify numerical correctness.
    """
    n = a.shape[0]
    nb = compute_grid_dimensions(n, block_size)

    a_blocks = partition_matrix_2d(a, block_size)
    b_blocks = partition_matrix_2d(b, block_size)

    # 1. Multiplication phase: compute all Nb^3 partial products
    c_blocks = []
    for i in range(nb):
        c_row = []
        for j in range(nb):
            # Partial products for block (i, j)
            partials = [block_multiply(a_blocks[i][k], b_blocks[k][j]) for k in range(nb)]

            # 2. Binary-tree reduction phase
            current = list(partials)
            while len(current) > 1:
                next_layer = []
                for idx in range(0, len(current), 2):
                    if idx + 1 < len(current):
                        summed = block_add(current[idx], current[idx + 1])
                        next_layer.append(summed)
                    else:
                        next_layer.append(current[idx])
                current = next_layer

            c_row.append(current[0])
        c_blocks.append(c_row)

    # 3. Assemble full dense matrix
    return assemble_matrix_2d(c_blocks, n, block_size)


def run_correctness_checks() -> bool:
    validator = MatrixValidator(atol=1e-9, rtol=1e-12)
    all_passed = True

    print("=================================================================")
    print("Assignment 4: Phase 5 Numerical Correctness Validation Suite")
    print("Tolerances: atol = 1e-9, rtol = 1e-12")
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # Check 1: Analytical 4x4 Known-Answer (b=2, 2x2 grid)
    # -----------------------------------------------------------------
    print("--- [Check 1] Analytical 4x4 Known Product (b=2, 2x2 grid) ---")
    file_a = os.path.join(VAL_DIR, "known_A_4.csv")
    file_b = os.path.join(VAL_DIR, "known_B_4.csv")
    file_expected = os.path.join(VAL_DIR, "expected_C_4.csv")

    mat_a = np.loadtxt(file_a, delimiter=",", dtype=np.float64)
    mat_b = np.loadtxt(file_b, delimiter=",", dtype=np.float64)
    expected_c = np.loadtxt(file_expected, delimiter=",", dtype=np.float64)

    c_calc = simulate_block_matmul_2d(mat_a, mat_b, block_size=2)
    res = validator.validate_matrices(c_calc, expected_c)
    print(f"  Shape: {c_calc.shape}, Elements: {res.elements_compared}")
    print(f"  Max Absolute Diff: {res.max_abs_diff:.3e}")
    print(f"  Max Relative Diff: {res.max_rel_diff:.3e}")
    print(f"  Finite: {res.all_finite}")
    print(f"  Status: {'[PASSED]' if res.passed else '[FAILED]'}\n")
    all_passed = all_passed and res.passed

    # -----------------------------------------------------------------
    # Check 2: Non-divisible Edge-Block Partitioning (n=4, b=3)
    # -----------------------------------------------------------------
    print("--- [Check 2] Non-divisible Edge Blocks (n=4, b=3) ---")
    c_edge = simulate_block_matmul_2d(mat_a, mat_b, block_size=3)
    res_edge = validator.validate_matrices(c_edge, expected_c)
    print(f"  Shape: {c_edge.shape}, Block grid: 2x2 (shapes: 3x3, 3x1, 1x3, 1x1)")
    print(f"  Max Absolute Diff: {res_edge.max_abs_diff:.3e}")
    print(f"  Max Relative Diff: {res_edge.max_rel_diff:.3e}")
    print(f"  Status: {'[PASSED]' if res_edge.passed else '[FAILED]'}\n")
    all_passed = all_passed and res_edge.passed

    # -----------------------------------------------------------------
    # Check 3: Identity Property Fixture (A x I = A, b=2)
    # -----------------------------------------------------------------
    print("--- [Check 3] Identity Property Fixture (A x I = A, b=2) ---")
    file_id = os.path.join(VAL_DIR, "identity_4.csv")
    mat_id = np.loadtxt(file_id, delimiter=",", dtype=np.float64)
    c_id = simulate_block_matmul_2d(mat_a, mat_id, block_size=2)
    res_id = validator.validate_matrices(c_id, mat_a)
    print(f"  Max Absolute Diff: {res_id.max_abs_diff:.3e}")
    print(f"  Status: {'[PASSED]' if res_id.passed else '[FAILED]'}\n")
    all_passed = all_passed and res_id.passed

    # -----------------------------------------------------------------
    # Check 4: Zero Property Fixture (A x 0 = 0, b=2)
    # -----------------------------------------------------------------
    print("--- [Check 4] Zero Property Fixture (A x 0 = 0, b=2) ---")
    file_zero = os.path.join(VAL_DIR, "zero_4.csv")
    mat_zero = np.loadtxt(file_zero, delimiter=",", dtype=np.float64)
    c_zero = simulate_block_matmul_2d(mat_a, mat_zero, block_size=2)
    res_zero = validator.validate_zero_fixture(c_zero)
    print(f"  Max Absolute Diff: {res_zero.max_abs_diff:.3e}")
    print(f"  Status: {'[PASSED]' if res_zero.passed else '[FAILED]'}\n")
    all_passed = all_passed and res_zero.passed

    # -----------------------------------------------------------------
    # Check 5: Cross-check n=250 vs Trusted Sequential Reference
    # -----------------------------------------------------------------
    print("--- [Check 5] Common Size n=250 Cross-Check vs C_250.csv (b=100) ---")
    file_a_250 = os.path.join(DENSE_DIR, "A_250.csv")
    file_b_250 = os.path.join(DENSE_DIR, "B_250.csv")
    file_ref_250 = os.path.join(VAL_DIR, "reference", "C_250.csv")

    if not (os.path.isfile(file_a_250) and os.path.isfile(file_b_250) and os.path.isfile(file_ref_250)):
        print("  [ERROR] n=250 input or reference files missing!")
        return False

    mat_a_250 = np.loadtxt(file_a_250, delimiter=",", dtype=np.float64)
    mat_b_250 = np.loadtxt(file_b_250, delimiter=",", dtype=np.float64)
    mat_ref_250 = np.loadtxt(file_ref_250, delimiter=",", dtype=np.float64)

    # Use b=100 on n=250 to force a 3x3 grid with edge block sizes of 50
    c_calc_250 = simulate_block_matmul_2d(mat_a_250, mat_b_250, block_size=100)
    res_250 = validator.validate_matrices(c_calc_250, mat_ref_250)

    print(f"  Shape: {c_calc_250.shape}, Total Elements Compared: {res_250.elements_compared}")
    print(f"  Elements Within Tolerance: {res_250.elements_within_tolerance} / {res_250.elements_compared}")
    print(f"  Max Absolute Difference: {res_250.max_abs_diff:.6e} (tolerance limit: 1.000e-09)")
    print(f"  Max Relative Difference: {res_250.max_rel_diff:.6e} (tolerance limit: 1.000e-12)")
    print(f"  All Elements Finite: {res_250.all_finite}")
    print(f"  Status: {'[PASSED]' if res_250.passed else '[FAILED]'}\n")
    all_passed = all_passed and res_250.passed

    # -----------------------------------------------------------------
    # Check 6: Checksum Consumption Verification
    # -----------------------------------------------------------------
    print("--- [Check 6] Full-Matrix Checksum Consumption Verification ---")
    checksum_calc = compute_matrix_checksum(c_calc_250)
    checksum_ref = compute_matrix_checksum(mat_ref_250)
    checksum_diff = abs(checksum_calc - checksum_ref)
    print(f"  Computed Checksum: {checksum_calc:.10f}")
    print(f"  Reference Checksum: {checksum_ref:.10f}")
    print(f"  Absolute Difference: {checksum_diff:.3e}")
    checksum_passed = checksum_diff <= 1e-6
    print(f"  Status: {'[PASSED]' if checksum_passed else '[FAILED]'}\n")
    all_passed = all_passed and checksum_passed

    print("=================================================================")
    if all_passed:
        print("[OVERALL RESULT] ALL CORRECTNESS CHECKS PASSED WITH 0 FAILURES.")
    else:
        print("[OVERALL RESULT] ONE OR MORE CORRECTNESS CHECKS FAILED.")
    print("=================================================================")

    return all_passed


if __name__ == "__main__":
    success = run_correctness_checks()
    sys.exit(0 if success else 1)
