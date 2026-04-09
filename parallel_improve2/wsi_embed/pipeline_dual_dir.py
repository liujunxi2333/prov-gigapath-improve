"""
多切片目录：按文件字节数均衡分为两队，GPU0 / GPU1 各按队内顺序串行处理整张切片；
每张切片在单卡上完成 GPU 条带扫描、tile 编码（Baseline 预处理）与 slide 编码。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import gigapath.slide_encoder as gigapath_slide_encoder
import timm

from .coords import compute_tissue_coords_parallel_strips_gpu
from .datasets import BaselineWSITileDataset
from .monitor import ResourceMonitor
from .partition import list_tif_paths, partition_two_queues_by_size, stat_file_sizes
from .utils import apply_tf32, set_seed


def run_one_slide_scan_tiles_slide_on_gpu(
    gpu_id: int,
    slide_path: str,
    tile_weight: str,
    slide_weight: str,
    out_dir: str,
    *,
    seed: int,
    batch_size: int,
    num_workers_data: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    scan_cpu_workers: int,
    scan_unique_on_gpu: bool,
    dataloader_tuned: bool,
) -> Dict[str, Any]:
    """单卡：GPU 扫描 + DataLoader(Baseline 预处理) + tile ViT + slide encoder。"""
    device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)
    os.makedirs(out_dir, exist_ok=True)

    t_scan0 = time.perf_counter()
    valid_coords, _ = compute_tissue_coords_parallel_strips_gpu(
        slide_path=slide_path,
        tile_size=256,
        target_level=target_level,
        bg_threshold=210,
        scan_step=scan_step,
        num_workers=scan_cpu_workers,
        gpu_id=gpu_id,
        unique_on_gpu=scan_unique_on_gpu,
    )
    scan_seconds = time.perf_counter() - t_scan0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not valid_coords:
        rep = {
            "error": "no_tissue",
            "slide_path": slide_path,
            "gpu_id": gpu_id,
            "scan_seconds": scan_seconds,
        }
        with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        return rep

    tile_enc = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=True,
        img_size=224,
        in_chans=3,
        pretrained_cfg_overlay=dict(file=tile_weight),
    ).to(device)
    tile_enc.eval()

    ds = BaselineWSITileDataset(slide_path, valid_coords, 256, target_level)
    dl_kw: Dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_data,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    if num_workers_data > 0:
        if dataloader_tuned:
            dl_kw["persistent_workers"] = True
            dl_kw["prefetch_factor"] = 4
        else:
            dl_kw["prefetch_factor"] = 2
    loader = DataLoader(**dl_kw)

    all_feat: List[torch.Tensor] = []
    all_coord: List[torch.Tensor] = []
    tiles_seen = 0
    t_tile0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for batch_tiles, batch_coords in loader:
            bt = batch_tiles.to(device, non_blocking=True)
            emb = tile_enc(bt)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            all_feat.append(emb.to(torch.float16).cpu())
            all_coord.append(batch_coords)
            tiles_seen += int(batch_tiles.shape[0])
    tile_seconds = time.perf_counter() - t_tile0

    del tile_enc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    feat = torch.cat(all_feat, dim=0)
    coord = torch.cat(all_coord, dim=0)
    n = int(feat.shape[0])
    rng = torch.Generator().manual_seed(seed)
    if n > max_tokens:
        idx = torch.randperm(n, generator=rng)[:max_tokens]
        feat = feat[idx]
        coord = coord[idx]
    used = int(feat.shape[0])

    slide_enc = gigapath_slide_encoder.create_model(
        pretrained=slide_weight,
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
    ).to(device)
    slide_enc.eval()

    t_slide0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        feat_gpu = feat.unsqueeze(0).to(device, dtype=torch.float16, non_blocking=True)
        coord_gpu = coord.unsqueeze(0).to(device, non_blocking=True)
        rep = slide_enc(feat_gpu, coord_gpu)
    slide_seconds = time.perf_counter() - t_slide0

    vec = rep[0].squeeze().float().cpu()
    torch.save(vec, os.path.join(out_dir, "embedding.pt"))

    report = {
        "mode": "dual_dir_per_slide_single_gpu",
        "slide_path": slide_path,
        "gpu_id": gpu_id,
        "scan_method": "parallel_thumb_strips_gpu",
        "scan_seconds": scan_seconds,
        "tile_seconds": tile_seconds,
        "slide_seconds": slide_seconds,
        "tiles_seen": tiles_seen,
        "orig_tokens": n,
        "slide_tokens_used": used,
        "tile_throughput_tiles_per_sec": tiles_seen / max(tile_seconds, 1e-9),
        "batch_size": batch_size,
        "num_workers_data": num_workers_data,
        "scan_step": scan_step,
    }
    with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _process_queue_on_gpu(
    gpu_id: int,
    queue_paths: List[str],
    output_root: str,
    tile_weight: str,
    slide_weight: str,
    seed: int,
    batch_size: int,
    num_workers_data: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    scan_cpu_workers: int,
    scan_unique_on_gpu: bool,
    flat_output: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for slide_path in queue_paths:
        stem = os.path.splitext(os.path.basename(slide_path))[0]
        if flat_output:
            out_dir = os.path.join(output_root, stem)
        else:
            out_dir = os.path.join(output_root, "per_slide", f"gpu{gpu_id}", stem)
        r = run_one_slide_scan_tiles_slide_on_gpu(
            gpu_id,
            slide_path,
            tile_weight,
            slide_weight,
            out_dir,
            seed=seed,
            batch_size=batch_size,
            num_workers_data=num_workers_data,
            scan_step=scan_step,
            target_level=target_level,
            max_tokens=max_tokens,
            scan_cpu_workers=scan_cpu_workers,
            scan_unique_on_gpu=scan_unique_on_gpu,
            dataloader_tuned=True,
        )
        results.append(r)
    return results


def run_tif_directory_dual_gpu(
    tif_dir: str,
    output_root: str,
    tile_weight: str,
    slide_weight: str,
    *,
    seed: int,
    batch_size: int,
    num_workers_data: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    scan_cpu_workers: int,
    scan_unique_on_gpu: bool,
    recursive: bool,
    monitor: bool,
    monitor_interval: float,
    use_tf32: bool,
    flat_output: bool = False,
) -> Dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("tif_dual 需要至少 2 块 GPU")

    paths = list_tif_paths(tif_dir, recursive=recursive)
    if not paths:
        raise ValueError(f"目录下无 .tif/.tiff/.svs: {tif_dir}")

    q0, q1, part_meta = partition_two_queues_by_size(paths)
    sizes = {p: s for p, s in stat_file_sizes(paths)}

    set_seed(seed)
    apply_tf32(use_tf32)

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "partition.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tif_dir": os.path.abspath(tif_dir),
                "flat_output": bool(flat_output),
                "queue_gpu0": q0,
                "queue_gpu1": q1,
                "file_sizes_bytes": sizes,
                **part_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    mon: Optional[ResourceMonitor] = ResourceMonitor(monitor_interval) if monitor else None
    if mon:
        mon.start()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f0 = ex.submit(
            _process_queue_on_gpu,
            0,
            q0,
            output_root,
            tile_weight,
            slide_weight,
            seed,
            batch_size,
            num_workers_data,
            scan_step,
            target_level,
            max_tokens,
            scan_cpu_workers,
            scan_unique_on_gpu,
            flat_output,
        )
        f1 = ex.submit(
            _process_queue_on_gpu,
            1,
            q1,
            output_root,
            tile_weight,
            slide_weight,
            seed,
            batch_size,
            num_workers_data,
            scan_step,
            target_level,
            max_tokens,
            scan_cpu_workers,
            scan_unique_on_gpu,
            flat_output,
        )
        rows0 = f0.result()
        rows1 = f1.result()
    wall_seconds = time.perf_counter() - t0

    if mon:
        mon.stop()
        summ = mon.summary()
        mon.plot(
            os.path.join(output_root, "gpu_curve_dual_dir.png"),
            title=f"dual-dir tif {os.path.basename(tif_dir.rstrip(os.sep))}",
        )
        mon.save_npz(os.path.join(output_root, "monitor_timeseries.npz"))
    else:
        summ = {}

    summary = {
        "mode": "tif_directory_dual_gpu_balanced",
        "wall_seconds_parallel": wall_seconds,
        "n_slides_total": len(paths),
        "results_gpu0": rows0,
        "results_gpu1": rows1,
        "partition": part_meta,
        **summ,
    }
    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
