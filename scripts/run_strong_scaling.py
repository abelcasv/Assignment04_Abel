"""
scripts/run_strong_scaling.py — Controlled Multi-Worker Pilot & Strong-Scaling Campaign Runner

Implements the local single-host multi-worker experiments on 1 physical host:
- --pilot-p1-p2: Bounded p=1 vs p=2 pilot (n in {250, 500}, b=250) validating numerical
  correctness, real multi-worker task distribution, transfer telemetry, RSS delta, and
  clean worker teardown. Outputs strictly to results/metadata/pilot_multiworker_p1_p2.json
  without touching results/raw/raw_measurements.csv.
- --campaign: Definitive strong-scaling sweep (p in {1, 2, 4, 8, 10}, n in {1000, 1500, 2000},
  b=250, 1 warm-up + 5 measured reps per (n, p), seed=20261113) via CampaignRunner, appending
  non-destructively to results/raw/raw_measurements.csv with one sidecar JSON per p, followed
  by a separate diagnostic task-stream pass (results/metadata/strong_scaling_diagnostics.json)
  outside primary timed measurements per SPEC.md.
"""

import argparse
from collections import defaultdict
import dataclasses
from datetime import datetime, timezone
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dask_matrix.campaign_runner import (
    CampaignRunner,
    CAMPAIGN_BLOCK_SIZE,
    CAMPAIGN_REPETITIONS,
    CAMPAIGN_SAMPLE_TIMEOUT_S,
    CAMPAIGN_STARTUP_TIMEOUT_S,
)
from src.dask_matrix.cluster_manager import (
    ClusterManager,
    validate_cluster_topology,
    verify_multiworker_permission,
)
from src.dask_matrix.distributed_engine import DistributedBlockMatmul
from src.dask_matrix.input_verifier import (
    load_sha256_manifest,
    parse_matrix_bytes,
    read_and_verify_matrix_bytes,
)
from src.dask_matrix.telemetry import TelemetryCollector
from src.dask_matrix.validation import DEFAULT_ATOL, DEFAULT_RTOL, MatrixValidator
from src.dask_matrix.watchdog import (
    AGGREGATE_RSS_CEILING_MIB,
    HOST_AVAILABLE_RAM_FLOOR_MIB,
    ResourceWatchdog,
    begin_sample_memory_window,
    extract_sample_memory_metrics,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
METADATA_DIR = os.path.join(REPO_ROOT, "results", "metadata")
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "raw", "raw_measurements.csv")

STRONG_SCALING_SIZES: Tuple[int, ...] = (1000, 1500, 2000)
STRONG_SCALING_WORKERS: Tuple[int, ...] = (1, 2, 4, 8, 10)


def _load_verified_pair(n: int, manifest: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    a_raw, _ = read_and_verify_matrix_bytes(
        os.path.join(DATA_DIR, "dense", f"A_{n}.csv"), f"dense/A_{n}.csv", manifest
    )
    b_raw, _ = read_and_verify_matrix_bytes(
        os.path.join(DATA_DIR, "dense", f"B_{n}.csv"), f"dense/B_{n}.csv", manifest
    )
    return parse_matrix_bytes(a_raw), parse_matrix_bytes(b_raw)


def _summarize_task_stream(ts_events: List[Dict[str, Any]], tc: TelemetryCollector) -> Dict[str, Any]:
    parsed = tc.parse_task_stream(ts_events)
    per_worker: Dict[str, Dict[str, int]] = defaultdict(lambda: {"matmul": 0, "add": 0, "other": 0, "total": 0})
    actions_seen = set()
    startstops_types = set()

    for ev in ts_events:
        if not isinstance(ev, dict):
            continue
        worker = str(ev.get("worker", "unknown"))
        key = str(ev.get("key", ""))
        if "matmul" in key:
            per_worker[worker]["matmul"] += 1
        elif "add" in key:
            per_worker[worker]["add"] += 1
        else:
            per_worker[worker]["other"] += 1
        per_worker[worker]["total"] += 1

        ss = ev.get("startstops")
        if ss is not None:
            startstops_types.add(type(ss).__name__)
            if isinstance(ss, (list, tuple)):
                for item in ss:
                    if isinstance(item, dict) and "action" in item:
                        actions_seen.add(str(item["action"]))

    return {
        "total_tasks": parsed.total_tasks,
        "matmul_tasks_count": parsed.matmul_tasks_count,
        "reduction_tasks_count": parsed.reduction_tasks_count,
        "mean_matmul_duration_s": parsed.mean_matmul_duration_s,
        "mean_reduction_duration_s": parsed.mean_reduction_duration_s,
        "total_compute_s": parsed.total_compute_s,
        "total_transfer_s": parsed.total_transfer_s,
        "total_deserialization_s": parsed.total_deserialization_s,
        "actions_observed": sorted(actions_seen),
        "startstops_types": sorted(startstops_types),
        "per_worker_task_counts": dict(sorted(per_worker.items())),
    }


def _transfer_delta(before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    per_worker: Dict[str, Dict[str, int]] = {}
    tot_in_b = 0
    tot_out_b = 0
    tot_in_c = 0
    tot_out_c = 0
    for addr, af in after.items():
        bf = before.get(addr, {})
        d_in_b = max(0, af.get("incoming_bytes", 0) - bf.get("incoming_bytes", 0))
        d_out_b = max(0, af.get("outgoing_bytes", 0) - bf.get("outgoing_bytes", 0))
        d_in_c = max(0, af.get("incoming_count_total", 0) - bf.get("incoming_count_total", 0))
        d_out_c = max(0, af.get("outgoing_count_total", 0) - bf.get("outgoing_count_total", 0))
        per_worker[addr] = {
            "incoming_bytes_delta": d_in_b,
            "outgoing_bytes_delta": d_out_b,
            "incoming_count_delta": d_in_c,
            "outgoing_count_delta": d_out_c,
        }
        tot_in_b += d_in_b
        tot_out_b += d_out_b
        tot_in_c += d_in_c
        tot_out_c += d_out_c
    return {
        "total_incoming_bytes_delta": tot_in_b,
        "total_outgoing_bytes_delta": tot_out_b,
        "total_incoming_count_delta": tot_in_c,
        "total_outgoing_count_delta": tot_out_c,
        "per_worker": dict(sorted(per_worker.items())),
    }


def run_pilot_p1_p2() -> int:
    print("================================================================================")
    print("ASSIGNMENT 4 — CONTROLLED MULTI-WORKER PILOT (p=1 vs p=2, 1 PHYSICAL HOST)")
    print("================================================================================")
    verify_multiworker_permission(2)

    manifest = load_sha256_manifest(os.path.join(DATA_DIR, "SHA256SUMS.txt"))
    a250, b250 = _load_verified_pair(250, manifest)
    a500, b500 = _load_verified_pair(500, manifest)
    c250_raw, _ = read_and_verify_matrix_bytes(
        os.path.join(DATA_DIR, "validation", "reference", "C_250.csv"),
        "validation/reference/C_250.csv",
        manifest,
    )
    c250_ref = parse_matrix_bytes(c250_raw)
    validator = MatrixValidator(atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    pilot_report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Controlled p=1 vs p=2 functional and telemetry validation pilot (separate from primary CSV campaign)",
        "physical_machine_count": 1,
        "host_available_ram_start_mib": psutil.virtual_memory().available / (1024.0 * 1024.0),
        "configurations": {},
    }

    c500_matrices: Dict[int, np.ndarray] = {}

    for p in (1, 2):
        print(f"\n--- Running Pilot Configuration p={p} ---")
        cm = ClusterManager(worker_count=p, threads_per_worker=1, worker_memory_limit=None)
        wd: Optional[ResourceWatchdog] = None
        worker_pids_to_check: List[int] = []
        try:
            topo = cm.start(timeout_s=CAMPAIGN_STARTUP_TIMEOUT_S)
            validate_cluster_topology(topo, expected_worker_count=p, expected_threads_per_worker=1)
            worker_pids_to_check = list(topo.worker_pids)
            tc = TelemetryCollector(cm.client)
            # Prime TaskStreamPlugin before timed pilot runs
            _ = cm.client.get_task_stream(count=1)

            wd = ResourceWatchdog(
                pids_to_track=topo.all_pids,
                client=cm.client,
                cluster=cm.cluster,
                rss_ceiling_mib=AGGREGATE_RSS_CEILING_MIB,
                host_floor_mib=HOST_AVAILABLE_RAM_FLOOR_MIB,
                sample_interval_s=0.1,
            )
            wd.start()
            time.sleep(0.2)

            engine = DistributedBlockMatmul(cm.client, block_size=250)
            runs_log: List[Dict[str, Any]] = []

            # 1) n=250 reference correctness run
            idx250 = begin_sample_memory_window(wd)
            res250 = engine.multiply(a250, b250, repetition_id=0, timeout_s=120.0, watchdog=wd)
            time.sleep(0.15)
            agg250, drv250, pws250 = extract_sample_memory_metrics(wd, idx250)
            val250 = validator.validate_matrices(res250.c_matrix, c250_ref)
            assert res250.status == "COMPLETED" and val250.passed, f"p={p} n=250 failed: {val250}"
            runs_log.append({
                "n": 250,
                "repetition": 0,
                "is_warmup": True,
                "status": res250.status,
                "time_e2e_s": res250.time_e2e_s,
                "checksum": res250.checksum,
                "validation_passed": val250.passed,
                "max_abs_diff": val250.max_abs_diff,
                "max_rel_diff": val250.max_rel_diff,
                "object_nbytes": res250.object_nbytes,
                "dask_managed_memory_mib": res250.dask_managed_memory_mib,
                "rss_process_sum_peak_mib": agg250,
                "rss_driver_mib": drv250,
                "peak_single_pid_rss_mib": pws250,
            })
            print(f"[p={p} n=250] COMPLETED in {res250.time_e2e_s:.4f}s | val_max_abs={val250.max_abs_diff:.3e} | peak_rss={agg250:.1f} MiB")

            # 2) n=500 (b=250 -> 8 matmul + 4 add tasks): 1 warmup + 2 measured reps
            for rep, is_warmup in [(0, True), (1, False), (2, False)]:
                time.sleep(0.1)
                tf_before = tc.get_worker_transfer_stats()
                t_ts_start = time.time() - 0.01
                idx500 = begin_sample_memory_window(wd)
                res500 = engine.multiply(a500, b500, repetition_id=rep, timeout_s=120.0, watchdog=wd)
                time.sleep(0.15)
                t_ts_stop = time.time()
                agg500, drv500, pws500 = extract_sample_memory_metrics(wd, idx500)
                tf_after = tc.get_worker_transfer_stats()
                ts_events = cm.client.get_task_stream(start=t_ts_start, stop=t_ts_stop)
                ts_summary = _summarize_task_stream(ts_events, tc)
                tf_delta = _transfer_delta(tf_before, tf_after)
                assert res500.status == "COMPLETED" and res500.c_matrix is not None
                assert bool(np.all(np.isfinite(res500.c_matrix)))
                if rep == 1:
                    c500_matrices[p] = res500.c_matrix.copy()
                runs_log.append({
                    "n": 500,
                    "repetition": rep,
                    "is_warmup": is_warmup,
                    "status": res500.status,
                    "time_e2e_s": res500.time_e2e_s,
                    "checksum": res500.checksum,
                    "object_nbytes": res500.object_nbytes,
                    "dask_managed_memory_mib": res500.dask_managed_memory_mib,
                    "phase_breakdown_s": res500.phase_breakdown_s,
                    "rss_process_sum_peak_mib": agg500,
                    "rss_driver_mib": drv500,
                    "peak_single_pid_rss_mib": pws500,
                    "task_stream": ts_summary,
                    "worker_transfer_delta": tf_delta,
                })
                print(
                    f"[p={p} n=500 rep={rep}] {res500.status} {res500.time_e2e_s:.4f}s | "
                    f"peak_rss={agg500:.1f} MiB | workers_used={len(ts_summary['per_worker_task_counts'])} | "
                    f"compute={ts_summary['total_compute_s']:.4f}s transfer={ts_summary['total_transfer_s']:.4f}s "
                    f"actions={ts_summary['actions_observed']}"
                )

            wd.stop()
            wd_clean = wd._thread is None or not wd._thread.is_alive()
            cm.close()
            time.sleep(0.3)
            surviving_pids = [pid for pid in worker_pids_to_check if psutil.pid_exists(pid)]
            print(f"[p={p} CLEANUP] watchdog_clean={wd_clean} surviving_worker_pids={surviving_pids}")

            pilot_report["configurations"][f"p={p}"] = {
                "topology": dataclasses.asdict(topo),
                "unique_os_pid_count": len(topo.all_pids),
                "watchdog_clean_stop": wd_clean,
                "watchdog_breach_detected": wd.breach_detected,
                "surviving_worker_pids_after_close": surviving_pids,
                "runs": runs_log,
            }
        finally:
            if wd is not None:
                wd.stop()
            cm.close()

    # Cross-check p=1 vs p=2 full C_500 matrices element-by-element
    cross_val = validator.validate_matrices(c500_matrices[2], c500_matrices[1])
    pilot_report["p1_vs_p2_c500_cross_validation"] = {
        "passed": cross_val.passed,
        "elements_compared": cross_val.elements_compared,
        "max_abs_diff": cross_val.max_abs_diff,
        "max_rel_diff": cross_val.max_rel_diff,
        "atol": cross_val.atol_used,
        "rtol": cross_val.rtol_used,
    }
    print(
        f"\n[CROSS-CHECK C_500 p=2 vs p=1] passed={cross_val.passed} "
        f"max_abs={cross_val.max_abs_diff:.3e} max_rel={cross_val.max_rel_diff:.3e}"
    )
    assert cross_val.passed, f"C_500 mismatch between p=1 and p=2: {cross_val}"

    os.makedirs(METADATA_DIR, exist_ok=True)
    out_path = os.path.join(METADATA_DIR, "pilot_multiworker_p1_p2.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(pilot_report, f, indent=2)
    print(f"[SAVED] Pilot report -> {out_path}")
    return 0


def run_diagnostic_characterization(worker_counts: Tuple[int, ...], diag_n: int = 1000) -> str:
    """
    Runs a dedicated diagnostic characterization pass strictly outside the primary
    timed campaign (per SPEC.md) to capture per-worker
    task distribution, TaskStream compute/transfer/deserialize durations, driver
    phase breakdowns, managed memory, and worker transfer byte deltas.
    """
    print("\n================================================================================")
    print(f"SEPARATE DIAGNOSTIC CHARACTERIZATION PASS (n={diag_n}, b=250, p in {worker_counts})")
    print("================================================================================")
    manifest = load_sha256_manifest(os.path.join(DATA_DIR, "SHA256SUMS.txt"))
    a_mat, b_mat = _load_verified_pair(diag_n, manifest)

    diag_report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "specification_reference": "SPEC.md (separate from primary timed campaign)",
        "n": diag_n,
        "block_size": CAMPAIGN_BLOCK_SIZE,
        "physical_machine_count": 1,
        "results_by_p": {},
    }

    for p in worker_counts:
        print(f"\n--- Diagnostic pass p={p}, n={diag_n} ---")
        cm = ClusterManager(worker_count=p, threads_per_worker=1, worker_memory_limit=None)
        wd: Optional[ResourceWatchdog] = None
        try:
            topo = cm.start(timeout_s=CAMPAIGN_STARTUP_TIMEOUT_S)
            validate_cluster_topology(topo, expected_worker_count=p, expected_threads_per_worker=1)
            tc = TelemetryCollector(cm.client)
            _ = cm.client.get_task_stream(count=1)
            wd = ResourceWatchdog(
                pids_to_track=topo.all_pids,
                client=cm.client,
                cluster=cm.cluster,
                rss_ceiling_mib=AGGREGATE_RSS_CEILING_MIB,
                host_floor_mib=HOST_AVAILABLE_RAM_FLOOR_MIB,
                sample_interval_s=0.1,
            )
            wd.start()
            time.sleep(0.2)

            engine = DistributedBlockMatmul(cm.client, block_size=CAMPAIGN_BLOCK_SIZE)
            # Warm-up (not recorded in diagnostics)
            w_res = engine.multiply(a_mat, b_mat, repetition_id=0, timeout_s=CAMPAIGN_SAMPLE_TIMEOUT_S, watchdog=wd)
            if w_res.status != "COMPLETED":
                diag_report["results_by_p"][str(p)] = {
                    "status": w_res.status,
                    "error_reason": w_res.error_reason,
                    "topology": dataclasses.asdict(topo),
                }
                print(f"[DIAG p={p}] Aborted during warmup: {w_res.status} ({w_res.error_reason})")
                continue

            time.sleep(0.15)
            tf_before = tc.get_worker_transfer_stats()
            t_ts_start = time.time() - 0.01
            idx = begin_sample_memory_window(wd)
            res = engine.multiply(a_mat, b_mat, repetition_id=1, timeout_s=CAMPAIGN_SAMPLE_TIMEOUT_S, watchdog=wd)
            time.sleep(0.15)
            t_ts_stop = time.time()
            agg_rss, drv_rss, pws_rss = extract_sample_memory_metrics(wd, idx)
            tf_after = tc.get_worker_transfer_stats()
            ts_events = cm.client.get_task_stream(start=t_ts_start, stop=t_ts_stop)
            ts_summary = _summarize_task_stream(ts_events, tc)
            tf_delta = _transfer_delta(tf_before, tf_after)

            diag_report["results_by_p"][str(p)] = {
                "status": res.status,
                "error_reason": res.error_reason,
                "time_e2e_s": res.time_e2e_s,
                "checksum": res.checksum,
                "object_nbytes": res.object_nbytes,
                "dask_managed_memory_mib": res.dask_managed_memory_mib,
                "phase_breakdown_s": res.phase_breakdown_s,
                "worker_memory_stats": res.worker_memory_stats,
                "rss_process_sum_peak_mib": agg_rss,
                "rss_driver_mib": drv_rss,
                "peak_single_pid_rss_mib": pws_rss,
                "topology": dataclasses.asdict(topo),
                "task_stream": ts_summary,
                "worker_transfer_delta": tf_delta,
                "overlap_ratio_compute_over_e2e": (
                    (ts_summary["total_compute_s"] / res.time_e2e_s)
                    if res.time_e2e_s and res.time_e2e_s > 0
                    else None
                ),
            }
            print(
                f"[DIAG p={p}] {res.status} e2e={res.time_e2e_s:.4f}s | "
                f"sum_compute={ts_summary['total_compute_s']:.4f}s | "
                f"sum_transfer={ts_summary['total_transfer_s']:.4f}s | "
                f"workers_active={len(ts_summary['per_worker_task_counts'])}/{p} | "
                f"w2w_bytes={tf_delta['total_incoming_bytes_delta']} | peak_rss={agg_rss:.1f} MiB"
            )
        except Exception as exc:
            diag_report["results_by_p"][str(p)] = {"status": "FAILED", "error_reason": str(exc)}
            print(f"[DIAG p={p}] Exception: {exc}")
        finally:
            if wd is not None:
                wd.stop()
            cm.close()
            time.sleep(0.5)

    out_path = os.path.join(METADATA_DIR, "strong_scaling_diagnostics.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(diag_report, f, indent=2)
    print(f"[SAVED] Diagnostic characterization -> {out_path}")
    return out_path


def run_strong_scaling_campaign(worker_counts: Tuple[int, ...] = STRONG_SCALING_WORKERS) -> int:
    print("================================================================================")
    print("ASSIGNMENT 4 — DEFINITIVE LOCAL MULTI-WORKER STRONG-SCALING CAMPAIGN")
    print(f"Sizes: {STRONG_SCALING_SIZES} | b={CAMPAIGN_BLOCK_SIZE} | p in {worker_counts} | 1 Physical Host")
    print("================================================================================")
    for p in worker_counts:
        verify_multiworker_permission(p)

    manifest_data: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "physical_machine_count": 1,
        "block_size": CAMPAIGN_BLOCK_SIZE,
        "sizes": list(STRONG_SCALING_SIZES),
        "repetitions_per_dimension": CAMPAIGN_REPETITIONS,
        "worker_counts_planned": list(worker_counts),
        "runs": [],
    }

    completed_workers: List[int] = []

    for p in worker_counts:
        print(f"\n>>> Starting Strong-Scaling Campaign Block for p={p} <<<")
        runner = CampaignRunner(
            data_dir=DATA_DIR,
            results_csv=RESULTS_CSV,
            metadata_dir=METADATA_DIR,
            worker_count=p,
            sizes=STRONG_SCALING_SIZES,
            block_size=CAMPAIGN_BLOCK_SIZE,
            repetitions=CAMPAIGN_REPETITIONS,
            campaign_name=f"Strong Scaling Local Multi-Worker Campaign (p={p})",
            sample_prefix=f"ss_p{p}",
        )
        summary = runner.run()
        rel_sidecar = os.path.relpath(summary["sidecar_path"], REPO_ROOT).replace("\\", "/")
        manifest_data["runs"].append({
            "worker_count": p,
            "run_id": summary["run_id"],
            "sidecar_path": rel_sidecar,
            "total_samples": summary["total_samples"],
            "completed_samples": summary["completed_samples"],
            "failed_samples": summary["failed_samples"],
            "session_duration_s": summary["session_duration_s"],
        })
        if summary["completed_samples"] > 0:
            completed_workers.append(p)
        # Brief pause between clusters for OS process/socket reclamation
        time.sleep(1.5)

    manifest_path = os.path.join(METADATA_DIR, "strong_scaling_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\n[SAVED] Strong-scaling manifest -> {manifest_path}")

    # Run separate diagnostic characterization pass outside primary timings
    if completed_workers:
        run_diagnostic_characterization(tuple(completed_workers), diag_n=1000)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assignment 4 Multi-Worker Pilot & Strong-Scaling Runner")
    parser.add_argument("--pilot-p1-p2", action="store_true", help="Run controlled p=1 vs p=2 validation pilot.")
    parser.add_argument("--campaign", action="store_true", help="Run full strong-scaling campaign (p in 1,2,4,8,10).")
    parser.add_argument("--diagnostics-only", action="store_true", help="Run separate task-stream diagnostics only.")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(STRONG_SCALING_WORKERS),
        help="Worker counts p to execute (default: 1 2 4 8 10).",
    )
    args = parser.parse_args(argv)

    if args.pilot_p1_p2:
        return run_pilot_p1_p2()
    if args.campaign:
        return run_strong_scaling_campaign(tuple(args.workers))
    if args.diagnostics_only:
        run_diagnostic_characterization(tuple(args.workers), diag_n=1000)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
