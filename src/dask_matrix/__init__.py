"""
src.dask_matrix — Core distributed 2D block matrix multiplication package.
"""

from src.dask_matrix.partitioning import (
    BlockSlice,
    compute_grid_dimensions,
    partition_matrix_2d,
    assemble_matrix_2d,
)
from src.dask_matrix.kernels import (
    block_multiply,
    block_add,
    compute_matrix_checksum,
)
from src.dask_matrix.reduction import (
    build_binary_tree_reduction,
)
from src.dask_matrix.cluster_manager import (
    ClusterManager,
    verify_multiworker_permission,
)
from src.dask_matrix.distributed_engine import (
    DistributedBlockMatmul,
)
from src.dask_matrix.watchdog import (
    ResourceWatchdog,
    ProcessMemorySample,
)
from src.dask_matrix.validation import (
    MatrixValidator,
    ValidationResult,
)
from src.dask_matrix.results_logger import (
    RawResultsLogger,
    MeasurementRecord,
)
from src.dask_matrix.input_verifier import (
    InputIntegrityError,
    load_sha256_manifest,
    compute_file_sha256,
    compute_bytes_sha256,
    read_and_verify_matrix_bytes,
    parse_matrix_bytes,
    verify_and_load_matrix,
)

__all__ = [
    "BlockSlice",
    "compute_grid_dimensions",
    "partition_matrix_2d",
    "assemble_matrix_2d",
    "block_multiply",
    "block_add",
    "compute_matrix_checksum",
    "build_binary_tree_reduction",
    "ClusterManager",
    "verify_multiworker_permission",
    "DistributedBlockMatmul",
    "ResourceWatchdog",
    "ProcessMemorySample",
    "MatrixValidator",
    "ValidationResult",
    "RawResultsLogger",
    "MeasurementRecord",
    "InputIntegrityError",
    "load_sha256_manifest",
    "compute_file_sha256",
    "compute_bytes_sha256",
    "read_and_verify_matrix_bytes",
    "parse_matrix_bytes",
    "verify_and_load_matrix",
]
