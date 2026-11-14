#!/usr/bin/env python3
"""
scripts/verify_provenance.py — Cryptographic Integrity and Provenance Verifier
Assignment 4: Distributed Matrix Multiplication
Course: BIG DATA, Academic Year 2026-2027, ULPGC

This script performs strictly read-only verification of the campaign analysis
provenance manifest (results/analysis/analysis_provenance.json) against actual
disk files:
1. Validates that recorded paths resolve safely within the repository root (no traversal).
2. Validates that recorded SHA-256 strings are valid 64-character hexadecimal digests.
3. Verifies that all mandatory artifacts exist on disk.
4. Computes SHA-256 hashes of the exact file bytes on disk.
5. Verifies that computed hashes match recorded hashes exactly.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def compute_file_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hex digest of a file reading binary chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def resolve_artifact_path(raw_path: str, repo_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Resolve and validate an artifact path against the repository root.
    Rejects any path that escapes repo_root via directory traversal or absolute symlinks.
    """
    # Normalize separators
    normalized = raw_path.replace("\\", "/")
    candidate = Path(normalized)

    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
            resolved.relative_to(repo_root)
            return resolved, None
        except ValueError:
            return None, f"Absolute path escapes repository root: {raw_path}"
    else:
        try:
            combined = (repo_root / candidate).resolve()
            combined.relative_to(repo_root)
            return combined, None
        except ValueError:
            return None, f"Path traversal escapes repository root: {raw_path}"


def extract_artifact_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all artifact verification targets from the provenance manifest schema.
    Returns a list of dicts:
      {
        "name": str,
        "raw_path": str,
        "expected_sha256": str,
        "optional": bool
      }
    """
    entries: List[Dict[str, Any]] = []

    # 1. Source script
    if "source_script" in manifest and "source_script_sha256" in manifest:
        entries.append({
            "name": "source_script",
            "raw_path": manifest["source_script"],
            "expected_sha256": manifest["source_script_sha256"],
            "optional": False
        })

    # 2. Raw measurements CSV
    if "raw_measurements_csv" in manifest:
        item = manifest["raw_measurements_csv"]
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            entries.append({
                "name": "raw_measurements_csv",
                "raw_path": item["path"],
                "expected_sha256": item["sha256"],
                "optional": item.get("optional", False)
            })

    # 3. Campaign sidecar JSON
    if "campaign_sidecar_json" in manifest:
        item = manifest["campaign_sidecar_json"]
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            entries.append({
                "name": "campaign_sidecar_json",
                "raw_path": item["path"],
                "expected_sha256": item["sha256"],
                "optional": item.get("optional", False)
            })

    # 4. Derived artifacts dictionary
    if "derived_artifacts" in manifest and isinstance(manifest["derived_artifacts"], dict):
        for name, item in manifest["derived_artifacts"].items():
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                entries.append({
                    "name": f"derived:{name}",
                    "raw_path": item["path"],
                    "expected_sha256": item["sha256"],
                    "optional": item.get("optional", False)
                })

    return entries


def verify_provenance_manifest(
    manifest_path: Path,
    repo_root: Optional[Path] = None
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """
    Verify all artifacts referenced in a provenance manifest.

    Returns:
      (success: bool, records: List[Dict], errors: List[str])
    """
    records: List[Dict[str, Any]] = []
    errors: List[str] = []

    if repo_root is None:
        repo_root = manifest_path.resolve().parent.parent.parent

    repo_root = repo_root.resolve()

    if not manifest_path.is_file():
        errors.append(f"Provenance manifest not found: {manifest_path}")
        return False, records, errors

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        errors.append(f"Failed to parse manifest JSON ({manifest_path}): {exc}")
        return False, records, errors

    entries = extract_artifact_entries(manifest)
    if not entries:
        errors.append(f"No verifiable artifact entries found in manifest: {manifest_path}")
        return False, records, errors

    for entry in entries:
        name = entry["name"]
        raw_path = entry["raw_path"]
        expected_hash = entry["expected_sha256"]
        is_optional = entry["optional"]

        record = {
            "name": name,
            "path": raw_path,
            "expected_sha256": expected_hash,
            "computed_sha256": None,
            "size_bytes": None,
            "status": "PENDING",
            "message": ""
        }

        # 1. Validate hash format
        if not (isinstance(expected_hash, str) and HEX_64_PATTERN.match(expected_hash)):
            err_msg = f"Invalid SHA-256 format for '{name}': {expected_hash!r}"
            errors.append(err_msg)
            record["status"] = "INVALID_HASH_FORMAT"
            record["message"] = err_msg
            records.append(record)
            continue

        # 2. Path resolution and traversal check
        resolved_path, traversal_err = resolve_artifact_path(raw_path, repo_root)
        if traversal_err or resolved_path is None:
            err_msg = f"Path security violation for '{name}': {traversal_err}"
            errors.append(err_msg)
            record["status"] = "PATH_TRAVERSAL"
            record["message"] = err_msg
            records.append(record)
            continue

        # 3. File existence check
        if not resolved_path.is_file():
            if is_optional:
                record["status"] = "SKIPPED_OPTIONAL_MISSING"
                record["message"] = f"Optional artifact not present: {resolved_path}"
                records.append(record)
                continue
            else:
                err_msg = f"Missing mandatory artifact for '{name}': {resolved_path}"
                errors.append(err_msg)
                record["status"] = "MISSING_FILE"
                record["message"] = err_msg
                records.append(record)
                continue

        # 4. Compute hash and compare
        try:
            size = resolved_path.stat().st_size
            computed = compute_file_sha256(resolved_path)
            record["size_bytes"] = size
            record["computed_sha256"] = computed

            if computed.upper() == expected_hash.upper():
                record["status"] = "PASS"
                record["message"] = "Hash matches exact file bytes."
            else:
                err_msg = (
                    f"Hash mismatch for '{name}' ({raw_path}):\n"
                    f"  Expected: {expected_hash.upper()}\n"
                    f"  Computed: {computed.upper()}"
                )
                errors.append(err_msg)
                record["status"] = "HASH_MISMATCH"
                record["message"] = err_msg

        except Exception as exc:
            err_msg = f"Error reading artifact '{name}' ({resolved_path}): {exc}"
            errors.append(err_msg)
            record["status"] = "READ_ERROR"
            record["message"] = err_msg

        records.append(record)

    all_passed = len(errors) == 0 and all(
        r["status"] in ("PASS", "SKIPPED_OPTIONAL_MISSING") for r in records
    )
    return all_passed, records, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SHA-256 integrity of artifacts referenced in campaign provenance manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/analysis/analysis_provenance.json"),
        help="Path to provenance manifest JSON (default: results/analysis/analysis_provenance.json)"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to repository root (defaults to parent of results/ directory)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-artifact PASS messages"
    )

    args = parser.parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent

    print(f"=== Provenance Integrity Verification ===")
    print(f"Manifest:  {args.manifest}")
    print(f"Repo Root: {repo_root}")
    print("-" * 70)

    success, records, errors = verify_provenance_manifest(args.manifest, repo_root)

    for rec in records:
        status = rec["status"]
        name = rec["name"]
        size_str = f"{rec['size_bytes']:,} B" if rec["size_bytes"] is not None else "N/A"
        if status == "PASS":
            if not args.quiet:
                print(f"[PASS] {name} ({size_str}) -> SHA-256 matches")
        elif status == "SKIPPED_OPTIONAL_MISSING":
            print(f"[SKIP] {name} -> Optional file not present on disk")
        else:
            print(f"[FAIL] {name} ({status}): {rec['message']}")

    print("-" * 70)
    total = len(records)
    passed = sum(1 for r in records if r["status"] == "PASS")
    failed = sum(1 for r in records if r["status"] not in ("PASS", "SKIPPED_OPTIONAL_MISSING"))
    skipped = sum(1 for r in records if r["status"] == "SKIPPED_OPTIONAL_MISSING")

    print(f"Summary: Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    if success:
        print("[SUCCESS] All provenance artifacts verified against actual file bytes.")
        return 0
    else:
        print(f"[FAILED] Integrity check failed with {len(errors)} error(s).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
