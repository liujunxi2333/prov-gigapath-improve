#!/usr/bin/env python3
"""
outv9：v7/v8 对齐的 tile/slide 流程 + GPU 参与 upfront 缩略图扫描（吞吐优先）

与 v8 的差异
-------------
- v8：upfront 扫描固定用 CPU 多进程（strip read + numpy）
- v9：CPU 多进程负责 read_region/解码 + 传回下采样 mask；GPU 负责 mask->nonzero->坐标映射->tile 对齐->去重

目标
----
尽可能降低 scan_seconds，并显式增加 GPU 扫描计算占用（tile/slide 仍按 v8：扫描后再启动 NVML 监控）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


def _repo_root() -> str:
    """本仓库根目录（scripts/ 的上一级）。可用环境变量 GIGAPATH_IMPROVE_ROOT 覆盖。"""
    return os.path.abspath(
        os.environ.get(
            "GIGAPATH_IMPROVE_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )
    )


_REPO_ROOT = _repo_root()
# parallel_improve2：wsi_embed_benchmark；仓库根：gigapath 包
sys.path.insert(0, os.path.join(_REPO_ROOT, "parallel_improve2"))
sys.path.insert(0, _REPO_ROOT)

from wsi_embed_benchmark import (  # noqa: E402
    BaselineWSITileDataset,
    ResourceMonitor,
    compute_tissue_coords_parallel_strips_gpu,
    apply_tf32,
    set_seed,
)

import gigapath.slide_encoder as gigapath_slide_encoder  # noqa: E402
import timm  # noqa: E402


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


def _default_scan_cpu_workers_v9() -> int:
    # v9 目标是“最大化 CPU 利用率”（合理上限），让 read_region 尽量并行。
    nc = os.cpu_count() or 16
    return int(min(48, max(12, nc)))


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

    # 扫描用到了 GPU（mask/unique 等），避免缓存显存影响后续加载 tile encoder。
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not valid_coords:
        return {
            "error": "no_tissue",
            "scan_seconds": scan_seconds,
            "scan_parallel_cpu_workers": scan_cpu_workers,
            "scan_gpu_id": scan_gpu_id,
            "scan_unique_on_gpu": scan_unique_on_gpu,
        }

    n = len(valid_coords)
    mid = n // 2
    coords0 = valid_coords[:mid]
    coords1 = valid_coords[mid:]

    mon = ResourceMonitor(monitor_interval) if monitor else None
    if mon:
        mon.start()

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
        "monitor_after_scan_only": True,
        "monitor_interval_seconds": monitor_interval if monitor else None,
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
        mon.stop()
        report.update(mon.summary())
        mon.plot(
            os.path.join(out_dir, "gpu_curve.png"),
            title=f"outv9 parallel-scan hybrid GPU {os.path.basename(slide_path)}",
        )
        mon.save_npz(os.path.join(out_dir, "monitor_timeseries.npz"))

    with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(
        description="outv9: tile/slide same as v8, scan uses GPU-accelerated mask+mapping."
    )
    _slide_default = os.environ.get("SLIDE_PATH", "")
    p.add_argument(
        "--slide_path",
        type=str,
        default=_slide_default,
        help="WSI .tif 路径；也可设置环境变量 SLIDE_PATH",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出目录；默认 <仓库根>/runs/<基名>/v9_run",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tile_weight",
        type=str,
        default=os.environ.get("TILE_WEIGHT", os.path.join(_REPO_ROOT, "weights", "pytorch_model.bin")),
        help="ViT tile 权重；或环境变量 TILE_WEIGHT",
    )
    p.add_argument(
        "--slide_weight",
        type=str,
        default=os.environ.get("SLIDE_WEIGHT", os.path.join(_REPO_ROOT, "weights", "slide_encoder.pth")),
        help="Slide encoder 权重；或环境变量 SLIDE_WEIGHT",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers_per_gpu", type=int, default=4)
    p.add_argument("--scan_step", type=int, default=4)
    p.add_argument("--target_level", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument(
        "--scan_cpu_workers",
        type=int,
        default=-1,
        help="-1=自动（v9 偏大，最大化 CPU strip 读取并行）；>=1 固定 strip 数",
    )
    p.add_argument("--scan_gpu_id", type=int, default=0, help="用于扫描 mask->coords 的 GPU id")
    p.add_argument(
        "--scan_unique_on_gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=GPU 上 unique（吞吐/工程目标优先）；0=同样用 torch.unique（保守兜底）",
    )
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--monitor_interval", type=float, default=0.1, help="NVML 采样间隔（秒）")

    args = p.parse_args()

    if not (args.slide_path or "").strip():
        raise SystemExit(
            "请通过 --slide_path 指定 WSI，或设置环境变量 SLIDE_PATH。"
        )

    if args.scan_cpu_workers < 0:
        scan_w = _default_scan_cpu_workers_v9()
        scan_auto = True
    else:
        scan_w = max(1, int(args.scan_cpu_workers))
        scan_auto = False

    stem = os.path.splitext(os.path.basename(args.slide_path))[0]
    out_dir = args.out_dir.strip() or os.path.join(_REPO_ROOT, "runs", stem, "v9_run")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "pipeline": "outv9_hybrid_gpu_scan",
        "pipeline_revision": "v8 tile/slide + GPU-accelerated parallel thumb scan",
        "slide_path": os.path.abspath(args.slide_path),
        "out_dir": os.path.abspath(out_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "scan_cpu_workers_effective": scan_w,
        "scan_cpu_workers_auto": scan_auto,
        "scan_gpu_id": args.scan_gpu_id,
        "scan_unique_on_gpu": bool(args.scan_unique_on_gpu),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    report = run_v9_pipeline(
        slide_path=args.slide_path,
        tile_weight=args.tile_weight,
        slide_weight=args.slide_weight,
        out_dir=out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers_per_gpu=args.num_workers_per_gpu,
        scan_step=args.scan_step,
        target_level=args.target_level,
        max_tokens=args.max_tokens,
        monitor=args.monitor,
        use_tf32=True,
        scan_cpu_workers=scan_w,
        scan_gpu_id=args.scan_gpu_id,
        scan_unique_on_gpu=bool(args.scan_unique_on_gpu),
        monitor_interval=args.monitor_interval,
    )

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["report_keys"] = list(report.keys())
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] outputs under: {out_dir}")


if __name__ == "__main__":
    main()

