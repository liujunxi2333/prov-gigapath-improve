"""WSI 嵌入基准与流水线（由 ``wsi_embed_benchmark`` 与 ``main.py`` 调用）。"""

from .coords import (
    compute_tissue_coords_parallel_strips,
    compute_tissue_coords_parallel_strips_gpu,
    compute_tissue_coords_slow,
    compute_tissue_coords_vectorized,
)
from .datasets import BaselineWSITileDataset, StreamingWSIDataset
from .encoders import build_encoders
from .monitor import ResourceMonitor
from .partition import list_tif_paths, partition_two_queues_by_size, stat_file_sizes
from .pipeline_dual_dir import run_tif_directory_dual_gpu
from .pipeline_single import (
    load_slide_list,
    run_baseline_slide,
    run_stream_slide,
)
from .pipeline_v9 import _default_scan_cpu_workers_v9, run_v9_pipeline
from .utils import apply_tf32, set_seed

__all__ = [
    "BaselineWSITileDataset",
    "StreamingWSIDataset",
    "ResourceMonitor",
    "build_encoders",
    "compute_tissue_coords_slow",
    "compute_tissue_coords_vectorized",
    "compute_tissue_coords_parallel_strips",
    "compute_tissue_coords_parallel_strips_gpu",
    "list_tif_paths",
    "partition_two_queues_by_size",
    "stat_file_sizes",
    "run_tif_directory_dual_gpu",
    "load_slide_list",
    "run_baseline_slide",
    "run_stream_slide",
    "run_v9_pipeline",
    "_default_scan_cpu_workers_v9",
    "set_seed",
    "apply_tf32",
]
