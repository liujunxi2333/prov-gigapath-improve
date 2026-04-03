"""单张 WSI：GPU 条带扫描 + 双卡各持完整 tile 模型并行编码半区坐标 + slide 在 GPU1。"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

import gigapath.slide_encoder as gigapath_slide_encoder
import timm

from .coords import compute_tissue_coords_parallel_strips_gpu
from .datasets import BaselineWSITileDataset
from .monitor import ResourceMonitor
from .utils import apply_tf32, set_seed


def _default_scan_cpu_workers_v9() -> int:
    nc = os.cpu_count() or 16
    return int(min(48, max(12, nc)))


def _encode_tiles_on_gpu(
    gpu_id: int,
    slide_path: str,
    coords: List[Tuple[int, int]],
    tile_weight: str,
    batch_size: int,
    num_workers: int,
    target_level: int,
    seed: int,
    dataloader_tuned: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    t0 = time.perf_counter()

    tile_enc = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=True,
        img_size=224,
        in_chans=3,
        pretrained_cfg_overlay=dict(file=tile_weight),
    ).to(device)
    tile_enc.eval()

    ds = BaselineWSITileDataset(slide_path, coords, 256, target_level)
    dl_kw: Dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    if num_workers > 0:
        if dataloader_tuned:
            dl_kw["persistent_workers"] = True
            dl_kw["prefetch_factor"] = 4
        else:
            dl_kw["prefetch_factor"] = 2
    loader = DataLoader(**dl_kw)

    feats: List[torch.Tensor] = []
    crds: List[torch.Tensor] = []
    n_tiles = 0
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        for batch_tiles, batch_coords in loader:
            bt = batch_tiles.to(device, non_blocking=True)
            emb = tile_enc(bt)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            feats.append(emb.to(torch.float16).cpu())
            crds.append(batch_coords)
            n_tiles += int(batch_tiles.shape[0])

    elapsed = time.perf_counter() - t0
    if not feats:
        return torch.empty(0, 1536), torch.empty(0, 2), 0, elapsed
    return torch.cat(feats, dim=0), torch.cat(crds, dim=0), n_tiles, elapsed


def run_v9_pipeline(
    slide_path: str,
    tile_weight: str,
    slide_weight: str,
    out_dir: str,
    *,
    seed: int,
    batch_size: int,
    num_workers_per_gpu: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    monitor: bool,
    use_tf32: bool,
    scan_cpu_workers: int,
    scan_gpu_id: int,
    scan_unique_on_gpu: bool,
    monitor_interval: float,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("v9 需要至少 2 块 GPU（CUDA）")

    set_seed(seed)
    apply_tf32(use_tf32)

    mon = ResourceMonitor(monitor_interval) if monitor else None
    if mon:
        mon.start()

    t_scan0 = time.perf_counter()
    valid_coords, _ = compute_tissue_coords_parallel_strips_gpu(
        slide_path=slide_path,
        tile_size=256,
        target_level=target_level,
        bg_threshold=210,
        scan_step=scan_step,
        num_workers=scan_cpu_workers,
        gpu_id=scan_gpu_id,
        unique_on_gpu=scan_unique_on_gpu,
    )
    scan_seconds = time.perf_counter() - t_scan0
    t_end_scan_wall = mon.elapsed_s() if mon else 0.0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not valid_coords:
        out_early: Dict[str, Any] = {
            "error": "no_tissue",
            "scan_seconds": scan_seconds,
            "scan_parallel_cpu_workers": scan_cpu_workers,
            "scan_gpu_id": scan_gpu_id,
            "scan_unique_on_gpu": scan_unique_on_gpu,
            "monitor_from_start": bool(monitor),
        }
        if mon:
            mon.stop()
            phases = [(0.0, t_end_scan_wall, "① 组织扫描 (无有效组织/坐标为空)")]
            out_early["phase_intervals_s"] = [{"t0": a, "t1": b, "name": c} for a, b, c in phases]
            mon.plot(
                os.path.join(out_dir, "gpu_curve_timeline.png"),
                title=f"outv9 GPU 曲线 + 阶段 — {os.path.basename(slide_path)}",
                phase_intervals=phases,
            )
            mon.save_npz(os.path.join(out_dir, "monitor_timeseries.npz"))
            out_early.update(mon.summary())
        return out_early

    n = len(valid_coords)
    mid = n // 2
    coords0 = valid_coords[:mid]
    coords1 = valid_coords[mid:]

    t_tile_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f0 = ex.submit(
            _encode_tiles_on_gpu,
            0,
            slide_path,
            coords0,
            tile_weight,
            batch_size,
            num_workers_per_gpu,
            target_level,
            seed,
            True,
        )
        f1 = ex.submit(
            _encode_tiles_on_gpu,
            1,
            slide_path,
            coords1,
            tile_weight,
            batch_size,
            num_workers_per_gpu,
            target_level,
            seed,
            True,
        )
        feat0, coord0, n0, t_gpu0 = f0.result()
        feat1, coord1, n1, t_gpu1 = f1.result()
    tile_wall_seconds = time.perf_counter() - t_tile_wall0
    t_end_tile_wall = mon.elapsed_s() if mon else 0.0

    feat = torch.cat([feat0, feat1], dim=0)
    coord = torch.cat([coord0, coord1], dim=0)
    tiles_seen = int(n0 + n1)

    rng = torch.Generator().manual_seed(seed)
    orig_tokens = int(feat.shape[0])
    if orig_tokens > max_tokens:
        idx = torch.randperm(orig_tokens, generator=rng)[:max_tokens]
        feat = feat[idx]
        coord = coord[idx]
    used = int(feat.shape[0])

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    slide_device = torch.device("cuda:1")
    slide_enc = gigapath_slide_encoder.create_model(
        pretrained=slide_weight,
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
    ).to(slide_device)
    slide_enc.eval()

    t_slide0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        feat_gpu = feat.unsqueeze(0).to(
            slide_device, dtype=torch.float16, non_blocking=True
        )
        coord_gpu = coord.unsqueeze(0).to(slide_device, non_blocking=True)
        rep = slide_enc(feat_gpu, coord_gpu)
    slide_seconds = time.perf_counter() - t_slide0
    t_end_slide_wall = mon.elapsed_s() if mon else 0.0

    vec = rep[0].squeeze().float().cpu()
    torch.save(vec, os.path.join(out_dir, "embedding.pt"))

    report: Dict[str, Any] = {
        "mode": "v9_hybrid_gpu_scan",
        "vectorized_scan": True,
        "scan_method": "parallel_thumb_strips_gpu",
        "scan_parallel_cpu_workers": scan_cpu_workers,
        "scan_gpu_id": scan_gpu_id,
        "scan_unique_on_gpu": scan_unique_on_gpu,
        "dataloader_tuned": True,
        "use_tf32": use_tf32,
        "monitor_from_start": bool(monitor),
        "monitor_interval_seconds": monitor_interval if monitor else None,
        "gpu_curve_timeline_png": os.path.join(out_dir, "gpu_curve_timeline.png")
        if monitor
        else None,
        "scan_seconds": scan_seconds,
        "tile_wall_seconds": tile_wall_seconds,
        "tile_gpu0_seconds": t_gpu0,
        "tile_gpu1_seconds": t_gpu1,
        "slide_seconds": slide_seconds,
        "tiles_seen": tiles_seen,
        "coords_gpu0": len(coords0),
        "coords_gpu1": len(coords1),
        "orig_tokens": orig_tokens,
        "slide_tokens_used": used,
        "tile_throughput_tiles_per_sec": tiles_seen / max(tile_wall_seconds, 1e-9),
        "batch_size_per_gpu": batch_size,
        "num_workers_per_gpu": num_workers_per_gpu,
        "scan_step": scan_step,
        "tile_parallel": "two_full_models_no_dataparallel",
        "slide_device": str(slide_device),
    }

    if mon:
        phases = [
            (0.0, t_end_scan_wall, "① 组织扫描 (CPU 条带 + GPU mask/unique)"),
            (t_end_scan_wall, t_end_tile_wall, "② 双卡 Tile 编码 (ViT，坐标前后二分)"),
            (t_end_tile_wall, t_end_slide_wall, "③ Slide 编码 (含 LongNet 加载与前向)"),
        ]
        report["phase_intervals_s"] = [{"t0": a, "t1": b, "name": c} for a, b, c in phases]
        mon.stop()
        report.update(mon.summary())
        mon.plot(
            os.path.join(out_dir, "gpu_curve_timeline.png"),
            title=f"outv9 GPU 利用率/显存 + 阶段时间轴 — {os.path.basename(slide_path)}",
            phase_intervals=phases,
        )
        mon.save_npz(os.path.join(out_dir, "monitor_timeseries.npz"))

    with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
