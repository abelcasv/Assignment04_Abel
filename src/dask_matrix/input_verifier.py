"""
src/dask_matrix/input_verifier.py — Cryptographic Input Integrity Verifier

Enforces fail-closed SHA-256 integrity verification for all dense input matrices
consumed by PilotRunner and CampaignRunner prior to cluster startup:
1. Validates that every matrix has an entry in data/SHA256SUMS.txt.
2. Validates that the declared digest is a valid 64-character hexadecimal SHA-256.
3. Computes the SHA-256 digest from actual file bytes in memory.
4. Compares actual vs declared digest case-insensitively.
5. Rejects any missing, malformed, or mismatched inputs.
6. Parses the matrix directly from the verified in-memory byte snapshot to eliminate
   time-of-check / time-of-use (TOCTOU) race conditions.
"""

import hashlib
import io
import os
import re
from typing import Dict, Optional, Tuple
import numpy as np

HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class InputIntegrityError(ValueError):
    """Raised when an input matrix fails cryptographic integrity verification."""

    def __init__(
        self,
        message: str,
        rel_path: str,
        reason: str,
        expected_digest: Optional[str] = None,
        actual_digest: Optional[str] = None,
    ):
        super().__init__(message)
        self.rel_path = rel_path
        self.reason = reason
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest


def load_sha256_manifest(manifest_path: str) -> Dict[str, str]:
    """
    Loads and parses recorded SHA-256 checksums from a manifest file.
    Format per line: <sha256_hex>  <relative_path> or <sha256_hex> <relative_path>

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        InputIntegrityError: If a line in the manifest is malformed or duplicate.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"SHA-256 manifest file not found: '{manifest_path}'. "
            f"A valid manifest is mandatory for fail-closed input verification."
        )

    sums: Dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise InputIntegrityError(
                    message=(
                        f"Malformed manifest line {line_num} in '{manifest_path}': '{line}'. "
                        f"Expected '<sha256> <relative_path>'."
                    ),
                    rel_path="",
                    reason="Malformed manifest syntax",
                )
            digest, rel = parts[0].strip(), parts[1].replace("\\", "/").strip()
            if rel in sums:
                raise InputIntegrityError(
                    message=(
                        f"Duplicate manifest entry for '{rel}' at line {line_num} in '{manifest_path}'."
                    ),
                    rel_path=rel,
                    reason="Duplicate entry in SHA-256 manifest",
                )
            sums[rel] = digest
    return sums


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hex digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hex digest of a file reading binary chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def read_and_verify_matrix_bytes(
    filepath: str,
    rel_path: str,
    manifest: Dict[str, str],
) -> Tuple[bytes, str]:
    """
    Verifies that a matrix file matches its declared manifest entry and
    returns the verified in-memory byte snapshot and verified digest.

    Enforces fail-closed integrity:
    1. Rejects if rel_path is missing from manifest.
    2. Rejects if declared digest is not a valid 64-char hex string.
    3. Rejects if matrix file does not exist on disk.
    4. Reads raw bytes into memory snapshot.
    5. Computes actual SHA-256 of the raw bytes.
    6. Rejects if actual digest does not match declared digest (case-insensitive).
    7. Returns (raw_bytes, actual_digest).
    """
    rel_norm = rel_path.replace("\\", "/").strip()

    # 1. Manifest entry presence check
    if rel_norm not in manifest:
        err_msg = (
            f"Input matrix integrity verification failed for '{rel_norm}'. "
            f"Reason: Missing entry in SHA-256 manifest. "
            f"Expected digest: N/A. Actual digest: N/A."
        )
        raise InputIntegrityError(
            message=err_msg,
            rel_path=rel_norm,
            reason="Missing entry in SHA-256 manifest",
            expected_digest=None,
            actual_digest=None,
        )

    expected_digest = manifest[rel_norm].strip()

    # 2. Declared digest format validation
    if not HEX_64_PATTERN.match(expected_digest):
        err_msg = (
            f"Input matrix integrity verification failed for '{rel_norm}'. "
            f"Reason: Malformed SHA-256 digest in manifest (expected 64 hex characters). "
            f"Expected digest: {expected_digest}. Actual digest: N/A."
        )
        raise InputIntegrityError(
            message=err_msg,
            rel_path=rel_norm,
            reason="Malformed SHA-256 digest in manifest",
            expected_digest=expected_digest,
            actual_digest=None,
        )

    # 3. File existence check
    if not os.path.isfile(filepath):
        err_msg = (
            f"Input matrix integrity verification failed for '{rel_norm}'. "
            f"Reason: Required matrix file not found: {filepath}. "
            f"Expected digest: {expected_digest}. Actual digest: NOT_FOUND."
        )
        raise FileNotFoundError(err_msg)

    # 4. Atomic read into memory snapshot
    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    # 5. Compute actual digest from the snapshot bytes
    actual_digest = compute_bytes_sha256(raw_bytes)

    # 6. Compare actual vs declared digest (case-insensitive)
    if actual_digest != expected_digest.lower():
        err_msg = (
            f"Input matrix integrity verification failed for '{rel_norm}'. "
            f"Reason: Actual file SHA-256 does not match declared manifest checksum (checksum mismatch). "
            f"Expected digest: {expected_digest.lower()}. Actual digest: {actual_digest}."
        )
        raise InputIntegrityError(
            message=err_msg,
            rel_path=rel_norm,
            reason="Checksum mismatch",
            expected_digest=expected_digest.lower(),
            actual_digest=actual_digest,
        )

    return raw_bytes, actual_digest


def parse_matrix_bytes(raw_bytes: bytes) -> np.ndarray:
    """Parses a verified byte snapshot into a C-contiguous float64 array."""
    mat = np.loadtxt(io.BytesIO(raw_bytes), delimiter=",", dtype=np.float64)
    return np.ascontiguousarray(mat, dtype=np.float64)


def verify_and_load_matrix(
    filepath: str,
    rel_path: str,
    manifest: Dict[str, str],
) -> Tuple[np.ndarray, str]:
    """
    Atomically verifies a matrix file against the manifest and parses it
    from the verified byte snapshot without TOCTOU window.
    """
    raw_bytes, verified_sha = read_and_verify_matrix_bytes(filepath, rel_path, manifest)
    mat = parse_matrix_bytes(raw_bytes)
    return mat, verified_sha
