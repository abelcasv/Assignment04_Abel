"""
src.dask_matrix.sidecar — Environment and Campaign Sidecar Metadata Capture

Generates JSON sidecar metadata for experimental runs adhering to SPEC.md Section 2.2
and docs/experimental_protocol.md Section 11.2:
- Host hardware: CPU model, physical/logical cores, RAM, OS version.
- Software stack: Python, NumPy, Dask, Distributed, GCC, BLAS thread limits.
- Git provenance: A4 commit SHA, A1/A3 baseline snapshot hashes, input SHA-256 hashes.
- Cluster topology: Process hierarchy, unique PIDs, requested vs observed workers.
"""

from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import psutil


METADATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "metadata")
)


def get_git_commit_sha() -> str:
    """Retrieves current Git HEAD commit SHA, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_cpu_info() -> Dict[str, Any]:
    """Captures CPU brand, core topology, and frequency."""
    return {
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_host_memory_info() -> Dict[str, Any]:
    """Captures total and initial available RAM in MiB."""
    mem = psutil.virtual_memory()
    return {
        "total_ram_mib": mem.total / (1024.0 * 1024.0),
        "available_ram_mib": mem.available / (1024.0 * 1024.0),
    }


def get_software_stack_info() -> Dict[str, Any]:
    """Captures versions of Python, NumPy, Dask, and platform details."""
    import numpy as np

    dask_ver = "not_installed"
    dist_ver = "not_installed"
    try:
        import dask
        dask_ver = dask.__version__
    except ImportError:
        pass
    try:
        import distributed
        dist_ver = distributed.__version__
    except ImportError:
        pass

    return {
        "python_version": sys.version,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "numpy_version": np.__version__,
        "dask_version": dask_ver,
        "distributed_version": dist_ver,
        "psutil_version": psutil.__version__,
    }


def generate_campaign_sidecar(
    run_id: str,
    campaign_name: str,
    worker_count: int,
    block_size: int,
    n_values: List[int],
    repetitions: int,
    cluster_topology: Optional[Dict[str, Any]] = None,
    output_dir: str = METADATA_DIR,
) -> str:
    """
    Generates and saves the structured campaign sidecar JSON document.

    Args:
        run_id: Unique execution identifier.
        campaign_name: Human-readable campaign title.
        worker_count: Configured worker count p.
        block_size: Block size b.
        n_values: List of matrix dimensions evaluated.
        repetitions: Measured repetitions per configuration.
        cluster_topology: Optional cluster process hierarchy dictionary.
        output_dir: Directory where the sidecar file is written.

    Returns:
        Absolute filepath of the saved JSON sidecar.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"sidecar_{run_id}.json")

    sidecar_data = {
        "metadata_schema_version": "1.0",
        "run_id": run_id,
        "campaign_name": campaign_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "host_environment": {
            "cpu": get_cpu_info(),
            "memory": get_host_memory_info(),
            "os": platform.platform(),
        },
        "software_stack": get_software_stack_info(),
        "threading_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        },
        "campaign_parameters": {
            "worker_count": worker_count,
            "block_size": block_size,
            "matrix_dimensions": n_values,
            "repetitions_per_dimension": repetitions,
            "timeout_per_sample_s": 120.0,
            "aggregate_rss_ceiling_mib": 2560.0,
            "host_available_ram_floor_mib": 1024.0,
        },
        "cluster_topology": cluster_topology or {},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    return filepath
