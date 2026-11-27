"""
src.dask_matrix.kernels — Core Computational Kernels

Contains the elementary tasks executed on Dask workers:
- block_multiply: Multiplies block A_ik by B_kj via numpy.matmul.
- block_add: Accumulates partial product blocks during binary-tree reduction.
- compute_matrix_checksum: Full-matrix summation forcing complete array materialization.
"""

from typing import Union
import numpy as np


def block_multiply(a_ik: np.ndarray, b_kj: np.ndarray) -> np.ndarray:
    """
    Computes the partial matrix product of two 2D blocks: P_ijk = A_ik @ B_kj.

    Args:
        a_ik: Block of matrix A with shape (r_i, k_dim).
        b_kj: Block of matrix B with shape (k_dim, c_j).

    Returns:
        Resulting partial product block with shape (r_i, c_j) in C-order.

    Raises:
        ValueError: If inner dimensions do not match or arrays are not 2D float64.
    """
    if a_ik.ndim != 2 or b_kj.ndim != 2:
        raise ValueError(f"Inputs must be 2D arrays, got A:{a_ik.shape}, B:{b_kj.shape}.")
    if a_ik.shape[1] != b_kj.shape[0]:
        raise ValueError(
            f"Inner dimension mismatch: A_ik has {a_ik.shape[1]} cols, B_kj has {b_kj.shape[0]} rows."
        )

    # Use numpy.matmul for optimized BLAS GEMM execution within the worker
    result = np.matmul(a_ik, b_kj, dtype=np.float64)
    return np.ascontiguousarray(result, dtype=np.float64)


def block_add(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Computes the element-wise sum of two partial product blocks: P_sum = P1 + P2.

    Args:
        p1: First block with shape (r, c).
        p2: Second block with shape (r, c).

    Returns:
        Accumulated block with shape (r, c) in C-order.

    Raises:
        ValueError: If block shapes do not match.
    """
    if p1.shape != p2.shape:
        raise ValueError(f"Shape mismatch for block addition: {p1.shape} vs {p2.shape}.")

    result = np.add(p1, p2, dtype=np.float64)
    return np.ascontiguousarray(result, dtype=np.float64)


def compute_matrix_checksum(matrix: np.ndarray) -> float:
    """
    Computes the total sum of all elements in the full dense matrix.

    This function represents the output-consumption operation defined in SPEC.md.
    It forces the CPU and Python memory manager to access every floating-point
    value, ensuring complete materialization before the end-to-end timer stops.

    Args:
        matrix: Assembled 2D dense matrix.

    Returns:
        Scalar float representing the sum of all matrix elements.

    Raises:
        ValueError: If matrix contains NaN or Inf values.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(matrix).__name__}.")

    checksum = float(np.sum(matrix, dtype=np.float64))

    # Fast sanity check that checksum is finite
    if np.isnan(checksum) or np.isinf(checksum):
        raise ValueError(f"Computed checksum is non-finite: {checksum}.")

    return checksum
