"""
src.dask_matrix.validation — Numerical Correctness Validation and Tolerances

Implements the correctness criteria defined in SPEC.md and docs/experimental_protocol.md (Section 9):
- Tolerances: absolute tolerance atol = 1e-9, relative tolerance rtol = 1e-12.
- Condition: |C_ij - C_ref_ij| <= atol + rtol * |C_ref_ij| for all i, j.
- Validates all values are finite IEEE 754 float64 (no NaN or Inf).
- Supports validation against identity, zero, analytical 4x4, and n=250 numerical reference.
- Executes strictly OUTSIDE the multiplication timer.
"""

from dataclasses import dataclass
import os
from typing import Optional, Tuple
import numpy as np


DEFAULT_ATOL = 1e-9
DEFAULT_RTOL = 1e-12


@dataclass
class ValidationResult:
    """Detailed outcome of element-by-element matrix comparison."""
    passed: bool
    n: int
    elements_compared: int
    elements_within_tolerance: int
    max_abs_diff: float
    max_rel_diff: float
    all_finite: bool
    atol_used: float
    rtol_used: float
    error_message: Optional[str] = None


class MatrixValidator:
    """Validates computed matrix results against reference matrices."""

    def __init__(self, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL):
        self.atol = atol
        self.rtol = rtol

    def validate_matrices(
        self,
        c_computed: np.ndarray,
        c_reference: np.ndarray,
    ) -> ValidationResult:
        """
        Performs element-by-element numerical verification.

        Args:
            c_computed: Computed dense product matrix.
            c_reference: Ground-truth reference matrix.

        Returns:
            ValidationResult with detailed metrics.
        """
        if c_computed.shape != c_reference.shape:
            return ValidationResult(
                passed=False,
                n=c_computed.shape[0] if c_computed.ndim > 0 else 0,
                elements_compared=0,
                elements_within_tolerance=0,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                all_finite=False,
                atol_used=self.atol,
                rtol_used=self.rtol,
                error_message=(
                    f"Shape mismatch: computed {c_computed.shape} vs reference {c_reference.shape}."
                ),
            )

        if c_computed.ndim == 0 or c_computed.size == 0:
            return ValidationResult(
                passed=False,
                n=c_computed.shape[0] if c_computed.ndim > 0 else 0,
                elements_compared=0,
                elements_within_tolerance=0,
                max_abs_diff=float("nan"),
                max_rel_diff=float("nan"),
                all_finite=False,
                atol_used=self.atol,
                rtol_used=self.rtol,
                error_message="Empty or 0-D matrix cannot be validated.",
            )

        n = c_computed.shape[0]
        elements_count = c_computed.size

        # Check finite values
        finite_computed = np.all(np.isfinite(c_computed))
        finite_reference = np.all(np.isfinite(c_reference))
        all_finite = bool(finite_computed and finite_reference)

        if not all_finite:
            return ValidationResult(
                passed=False,
                n=n,
                elements_compared=elements_count,
                elements_within_tolerance=0,
                max_abs_diff=float("nan"),
                max_rel_diff=float("nan"),
                all_finite=False,
                atol_used=self.atol,
                rtol_used=self.rtol,
                error_message="Non-finite values (NaN or Inf) detected in matrix.",
            )

        # Compute element-wise differences
        abs_diff = np.abs(c_computed - c_reference)
        max_abs = float(np.max(abs_diff))

        # Relative difference with epsilon floor to avoid division by zero
        ref_abs = np.abs(c_reference)
        rel_diff = abs_diff / np.maximum(ref_abs, 1e-15)
        max_rel = float(np.max(rel_diff))

        # Check tolerance formula: abs_diff <= atol + rtol * abs(reference)
        tolerance_bound = self.atol + self.rtol * ref_abs
        within_tolerance_mask = abs_diff <= tolerance_bound
        within_count = int(np.sum(within_tolerance_mask))

        passed = bool(within_count == elements_count)
        error_msg = None
        if not passed:
            mismatches = elements_count - within_count
            error_msg = (
                f"{mismatches} of {elements_count} elements exceeded tolerance "
                f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e})."
            )

        return ValidationResult(
            passed=passed,
            n=n,
            elements_compared=elements_count,
            elements_within_tolerance=within_count,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            all_finite=all_finite,
            atol_used=self.atol,
            rtol_used=self.rtol,
            error_message=error_msg,
        )

    def validate_identity_fixture(self, a_matrix: np.ndarray, computed_result: np.ndarray) -> ValidationResult:
        """Verifies A x I = A."""
        return self.validate_matrices(computed_result, a_matrix)

    def validate_zero_fixture(self, computed_result: np.ndarray) -> ValidationResult:
        """Verifies A x 0 = 0."""
        zero_ref = np.zeros_like(computed_result, dtype=np.float64)
        return self.validate_matrices(computed_result, zero_ref)
