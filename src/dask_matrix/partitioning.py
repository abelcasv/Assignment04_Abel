"""
src.dask_matrix.partitioning — 2D Block Partitioning and Assembly

Handles 2D decomposition of square matrices into blocks of reference size b x b.
Strictly implements the edge-block policy defined in SPEC.md:
- Edge blocks are truncated to their exact dimensions: min(b, n - start).
- Zero-padding is strictly forbidden.
- All extracted blocks and assembled matrices are forced to contiguous C-order.
"""

from dataclasses import dataclass
import math
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class BlockSlice:
    """Represents the row and column coordinate bounds of a 2D block."""
    block_row: int
    block_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_end)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_start, self.col_end)


def compute_grid_dimensions(n: int, b: int) -> int:
    """
    Computes the number of blocks per dimension: Nb = ceil(n / b).

    Args:
        n: Matrix dimension (rows and columns).
        b: Reference block size.

    Returns:
        Number of blocks along each axis.

    Raises:
        ValueError: If n <= 0 or b <= 0.
    """
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")
    if b <= 0:
        raise ValueError(f"Block size b must be positive, got {b}.")
    return math.ceil(n / b)


def get_block_slices(n: int, b: int) -> List[List[BlockSlice]]:
    """
    Generates the grid of BlockSlice descriptors for an n x n matrix partitioned with block size b.

    Args:
        n: Matrix dimension.
        b: Reference block size.

    Returns:
        2D list of shape (Nb, Nb) containing BlockSlice metadata.
    """
    nb = compute_grid_dimensions(n, b)
    slices: List[List[BlockSlice]] = []

    for i in range(nb):
        row_start = i * b
        row_end = min(row_start + b, n)
        row_list: List[BlockSlice] = []

        for j in range(nb):
            col_start = j * b
            col_end = min(col_start + b, n)
            row_list.append(
                BlockSlice(
                    block_row=i,
                    block_col=j,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                )
            )
        slices.append(row_list)

    return slices


def partition_matrix_2d(matrix: np.ndarray, b: int) -> List[List[np.ndarray]]:
    """
    Partitions a 2D dense numpy array into a 2D list of contiguous C-order blocks.

    Edge blocks are sliced to their exact truncated dimensions without zero-padding.

    Args:
        matrix: 2D numpy array of shape (n, n) and dtype float64.
        b: Block size.

    Returns:
        2D list of numpy ndarrays representing the block grid of shape (Nb, Nb).

    Raises:
        ValueError: If matrix is not 2D, not square, not float64, or empty.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(matrix).__name__}.")
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {matrix.shape}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}.")
    if matrix.dtype != np.float64:
        raise ValueError(f"Matrix dtype must be float64, got {matrix.dtype}.")

    n = matrix.shape[0]
    nb = compute_grid_dimensions(n, b)
    block_grid: List[List[np.ndarray]] = []

    for i in range(nb):
        row_start = i * b
        row_end = min(row_start + b, n)
        row_blocks: List[np.ndarray] = []

        for j in range(nb):
            col_start = j * b
            col_end = min(col_start + b, n)
            # Slice and enforce C-order memory layout
            block = np.ascontiguousarray(matrix[row_start:row_end, col_start:col_end], dtype=np.float64)
            row_blocks.append(block)

        block_grid.append(row_blocks)

    return block_grid


def assemble_matrix_2d(blocks: List[List[np.ndarray]], n: int, b: int) -> np.ndarray:
    """
    Assembles a 2D grid of blocks back into a contiguous C-order dense matrix of shape (n, n).

    Args:
        blocks: 2D list of numpy ndarrays of shape (Nb, Nb).
        n: Target full matrix dimension.
        b: Block size used for partitioning.

    Returns:
        Assembled numpy ndarray of shape (n, n) and dtype float64 in C-order.

    Raises:
        ValueError: If grid dimensions or block shapes are inconsistent.
    """
    nb = compute_grid_dimensions(n, b)
    if len(blocks) != nb:
        raise ValueError(f"Expected {nb} block rows, got {len(blocks)}.")

    assembled = np.empty((n, n), dtype=np.float64, order="C")

    for i in range(nb):
        if len(blocks[i]) != nb:
            raise ValueError(f"Row {i} has {len(blocks[i])} blocks, expected {nb}.")
        row_start = i * b
        row_end = min(row_start + b, n)

        for j in range(nb):
            col_start = j * b
            col_end = min(col_start + b, n)
            expected_shape = (row_end - row_start, col_end - col_start)
            block = blocks[i][j]

            if block.shape != expected_shape:
                raise ValueError(
                    f"Block ({i}, {j}) has shape {block.shape}, expected {expected_shape}."
                )

            assembled[row_start:row_end, col_start:col_end] = block

    return np.ascontiguousarray(assembled, dtype=np.float64)
