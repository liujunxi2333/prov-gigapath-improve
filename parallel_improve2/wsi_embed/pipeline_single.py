"""单张 WSI：baseline / stream 全流程。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .coords import compute_tissue_coords_slow, compute_tissue_coords_vectorized
from .datasets import BaselineWSITileDataset, StreamingWSIDataset
from .encoders import build_encoders
from .monitor import ResourceMonitor
from .utils import apply_tf32, set_seed


def run_baseline_slide(
    slide_path: str,
    tile_weight: str,
    slide_weight: str,
    *,
    seed: int,
    vectorized_scan: bool,
    dataloader_tuned: bool,
    use_tf32: bool,
    batch_size: int,
    num_workers: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    monitor: bool,
    out_dir: str,
    split_models_across_two_gpus: bool = True,
    tile_parallel: str = "single",
) -> Dict[str, Any]:
    set_seed(seed)
    apply_tf32(use_tf32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vectorized_scan:
        valid_coords, t_scan = compute_tissue_coords_vectorized(
            slide_path, 256, target_level, 210, scan_step
        )
    else:
        valid_coords, t_scan = compute_tissue_coords_slow(
            slide_path, 256, target_level, 210, scan_step
        )

    if not valid_coords:
        return {"error": "no_tissue", "scan_seconds": t_scan}

    ds = BaselineWSITileDataset(slide_path, valid_coords, 256, target_level)
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
    tile_enc, slide_enc, tile_device, slide_device, use_split, dp_tile, tile_par = build_encoders(
        tile_weight,
        slide_weight,
        device,
        split_models_across_two_gpus=split_models_across_two_gpus,
        tile_parallel=tile_parallel,
    )

    mon = ResourceMonitor(0.5) if monitor else None
    if mon:
        mon.start()

    all_feat: List[torch.Tensor] = []
    all_coord: List[torch.Tensor] = []
    tiles_seen = 0

    t_tile0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda" if tile_device.type == "cuda" else "cpu",
        dtype=torch.float16 if tile_device.type == "cuda" else torch.float32,
    ):
        for batch_tiles, batch_coords in loader:
            bt = batch_tiles.to(tile_device, non_blocking=True)
            emb = tile_enc(bt)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            all_feat.append(emb.to(torch.float16).cpu())
            all_coord.append(batch_coords)
            tiles_seen += int(batch_tiles.shape[0])
    t_tile = time.perf_counter() - t_tile0

    feat = torch.cat(all_feat, dim=0)
    coord = torch.cat(all_coord, dim=0)
    n = int(feat.shape[0])
    rng = torch.Generator().manual_seed(seed)
    if n > max_tokens:
        idx = torch.randperm(n, generator=rng)[:max_tokens]
        feat = feat[idx]
        coord = coord[idx]
    used = int(feat.shape[0])

    t_slide0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda" if slide_device.type == "cuda" else "cpu",
        dtype=torch.float16 if slide_device.type == "cuda" else torch.float32,
    ):
        feat = feat.unsqueeze(0).to(
            slide_device, dtype=torch.float16 if slide_device.type == "cuda" else feat.dtype, non_blocking=True
        )
        coord = coord.unsqueeze(0).to(slide_device, non_blocking=True)
        rep = slide_enc(feat, coord)
    t_slide = time.perf_counter() - t_slide0

    vec = rep[0].squeeze().float().cpu()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(vec, os.path.join(out_dir, "embedding.pt"))

    if mon:
        mon.stop()
        summ = mon.summary()
        mon.plot(os.path.join(out_dir, "gpu_curve.png"), title=f"baseline {os.path.basename(slide_path)}")
        mon.save_npz(os.path.join(out_dir, "monitor_timeseries.npz"))
    else:
        summ = {}

    tps = tiles_seen / max(t_tile, 1e-9)
    report = {
        "mode": "baseline",
        "vectorized_scan": vectorized_scan,
        "dataloader_tuned": dataloader_tuned,
        "use_tf32": use_tf32,
        "scan_seconds": t_scan,
        "tile_seconds": t_tile,
        "slide_seconds": t_slide,
        "tiles_seen": tiles_seen,
        "orig_tokens": n,
        "slide_tokens_used": used,
        "tile_throughput_tiles_per_sec": tps,
        "split_models_across_two_gpus": use_split,
        "tile_parallel": tile_par,
        "tile_dataparallel": dp_tile,
        "tile_device": str(tile_device),
        "slide_device": str(slide_device),
        **summ,
    }
    with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def run_stream_slide(
    slide_path: str,
    tile_weight: str,
    slide_weight: str,
    *,
    seed: int,
    use_tf32: bool,
    batch_size: int,
    num_workers: int,
    scan_step: int,
    target_level: int,
    max_tokens: int,
    dataloader_tuned: bool,
    monitor: bool,
    out_dir: str,
    coord_buffer_size: int = 512,
    sort_buffer_coords: bool = True,
    split_models_across_two_gpus: bool = True,
    tile_parallel: str = "dataparallel",
) -> Dict[str, Any]:
    set_seed(seed)
    apply_tf32(use_tf32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StreamingWSIDataset(
        slide_path=slide_path,
        tile_size=256,
        target_level=target_level,
        bg_threshold=210,
        scan_step=scan_step,
        coord_buffer_size=coord_buffer_size,
        sort_buffer_coords=sort_buffer_coords,
    )
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
    tile_enc, slide_enc, tile_device, slide_device, use_split, dp_tile, tile_par = build_encoders(
        tile_weight,
        slide_weight,
        device,
        split_models_across_two_gpus=split_models_across_two_gpus,
        tile_parallel=tile_parallel,
    )

    mon = ResourceMonitor(0.5) if monitor else None
    if mon:
        mon.start()

    all_feat: List[torch.Tensor] = []
    all_coord: List[torch.Tensor] = []
    tiles_seen = 0

    t_tile0 = time.perf_counter()
    t_first_batch: Optional[float] = None
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda" if tile_device.type == "cuda" else "cpu",
        dtype=torch.float16 if tile_device.type == "cuda" else torch.float32,
    ):
        for batch_tiles, batch_coords in loader:
            if t_first_batch is None:
                t_first_batch = time.perf_counter()
            bt = batch_tiles.to(tile_device, non_blocking=True)
            emb = tile_enc(bt)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            all_feat.append(emb.to(torch.float16).cpu())
            all_coord.append(batch_coords)
            tiles_seen += int(batch_tiles.shape[0])
    t_tile = time.perf_counter() - t_tile0
    scan_like_seconds = (t_first_batch - t_tile0) if t_first_batch is not None else t_tile

    feat = torch.cat(all_feat, dim=0)
    coord = torch.cat(all_coord, dim=0)
    n = int(feat.shape[0])
    rng = torch.Generator().manual_seed(seed)
    if n > max_tokens:
        idx = torch.randperm(n, generator=rng)[:max_tokens]
        feat = feat[idx]
        coord = coord[idx]
    used = int(feat.shape[0])

    t_slide0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda" if slide_device.type == "cuda" else "cpu",
        dtype=torch.float16 if slide_device.type == "cuda" else torch.float32,
    ):
        feat = feat.unsqueeze(0).to(
            slide_device, dtype=torch.float16 if slide_device.type == "cuda" else feat.dtype, non_blocking=True
        )
        coord = coord.unsqueeze(0).to(slide_device, non_blocking=True)
        rep = slide_enc(feat, coord)
    t_slide = time.perf_counter() - t_slide0

    vec = rep[0].squeeze().float().cpu()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(vec, os.path.join(out_dir, "embedding.pt"))

    if mon:
        mon.stop()
        summ = mon.summary()
        mon.plot(os.path.join(out_dir, "gpu_curve.png"), title=f"stream {os.path.basename(slide_path)}")
        mon.save_npz(os.path.join(out_dir, "monitor_timeseries.npz"))
    else:
        summ = {}

    tps = tiles_seen / max(t_tile, 1e-9)
    report = {
        "mode": "stream",
        "dataloader_tuned": dataloader_tuned,
        "use_tf32": use_tf32,
        "scan_seconds": None,
        "scan_like_seconds": scan_like_seconds,
        "tile_seconds": t_tile,
        "slide_seconds": t_slide,
        "tiles_seen": tiles_seen,
        "orig_tokens": n,
        "slide_tokens_used": used,
        "tile_throughput_tiles_per_sec": tps,
        "split_models_across_two_gpus": use_split,
        "tile_parallel": tile_par,
        "tile_dataparallel": dp_tile,
        "tile_device": str(tile_device),
        "slide_device": str(slide_device),
        **summ,
    }
    with open(os.path.join(out_dir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def load_slide_list(path: str) -> List[str]:
    slides = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                slides.append(line)
    return slides


def main_benchmark() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["compare", "ablation"], required=True)
    p.add_argument("--slide_list", type=str, help="每行一个 .tif 绝对路径")
    p.add_argument("--tile_weight", type=str, default="/public/home/wang/liujx/pytorch_model.bin")
    p.add_argument("--slide_weight", type=str, default="/public/home/wang/liujx/slide_encoder.pth")
    p.add_argument("--output_root", type=str, default="/public/home/wang/liujx/prov-gigapath-main/11111ovarian/parallel_improve2/benchmark_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--scan_step", type=int, default=4)
    p.add_argument("--target_level", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument("--monitor", action="store_true", help="NVML 曲线 + 平均 GPU 利用率")
    args = p.parse_args()

    if not args.slide_list or not os.path.isfile(args.slide_list):
        raise SystemExit("请提供 --slide_list 文本文件（每行一个切片路径）")

    slides = load_slide_list(args.slide_list)
    if not slides:
        raise SystemExit("slide_list 为空")

    os.makedirs(args.output_root, exist_ok=True)

    if args.mode == "compare":
        comp_rows = []
        for i, sp in enumerate(slides):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            base_name = os.path.splitext(os.path.basename(sp))[0]
            root = os.path.join(args.output_root, "compare_baseline_vs_stream", f"{i:03d}_{base_name}")
            rb = run_baseline_slide(
                sp,
                args.tile_weight,
                args.slide_weight,
                seed=args.seed,
                vectorized_scan=True,
                dataloader_tuned=True,
                use_tf32=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scan_step=args.scan_step,
                target_level=args.target_level,
                max_tokens=args.max_tokens,
                monitor=args.monitor,
                out_dir=os.path.join(root, "baseline"),
            )
            if rb.get("error"):
                print(f"[compare] skip (baseline no tissue): {sp} {rb}")
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            rs = run_stream_slide(
                sp,
                args.tile_weight,
                args.slide_weight,
                seed=args.seed,
                use_tf32=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scan_step=args.scan_step,
                target_level=args.target_level,
                max_tokens=args.max_tokens,
                dataloader_tuned=True,
                monitor=args.monitor,
                out_dir=os.path.join(root, "stream"),
            )
            row = {
                "slide": sp,
                "baseline_total_s": (rb.get("scan_seconds") or 0) + rb.get("tile_seconds", 0) + rb.get("slide_seconds", 0),
                "stream_total_s": rs.get("tile_seconds", 0) + rs.get("slide_seconds", 0),
                "baseline_scan_s": rb.get("scan_seconds"),
                "baseline_tile_s": rb.get("tile_seconds"),
                "baseline_slide_s": rb.get("slide_seconds"),
                "baseline_tiles_per_s": rb.get("tile_throughput_tiles_per_sec"),
                "stream_tile_s": rs.get("tile_seconds"),
                "stream_slide_s": rs.get("slide_seconds"),
                "stream_tiles_per_s": rs.get("tile_throughput_tiles_per_sec"),
            }
            comp_rows.append(row)

        summ_path = os.path.join(args.output_root, "compare_baseline_vs_stream", "summary.json")
        os.makedirs(os.path.dirname(summ_path), exist_ok=True)
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "slides": comp_rows}, f, ensure_ascii=False, indent=2)

        if comp_rows:
            with open(os.path.join(args.output_root, "compare_baseline_vs_stream", "summary.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
                w.writeheader()
                w.writerows(comp_rows)

        print(json.dumps(comp_rows, ensure_ascii=False, indent=2))

    elif args.mode == "ablation":
        rows = []
        configs = [
            ("c0_minimal", dict(vectorized_scan=False, dataloader_tuned=False, use_tf32=False, stream=False)),
            ("c1_vector_scan", dict(vectorized_scan=True, dataloader_tuned=False, use_tf32=False, stream=False)),
            ("c2_plus_dataloader", dict(vectorized_scan=True, dataloader_tuned=True, use_tf32=False, stream=False)),
            ("c3_plus_tf32", dict(vectorized_scan=True, dataloader_tuned=True, use_tf32=True, stream=False)),
            ("c4_plus_stream", dict(vectorized_scan=True, dataloader_tuned=True, use_tf32=True, stream=True)),
        ]

        for sp in slides:
            base_name = os.path.splitext(os.path.basename(sp))[0]
            for tag, cfg in configs:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                set_seed(args.seed)
                out = os.path.join(args.output_root, "ablation", base_name, tag)
                if cfg["stream"]:
                    r = run_stream_slide(
                        sp,
                        args.tile_weight,
                        args.slide_weight,
                        seed=args.seed,
                        use_tf32=cfg["use_tf32"],
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        scan_step=args.scan_step,
                        target_level=args.target_level,
                        max_tokens=args.max_tokens,
                        dataloader_tuned=cfg["dataloader_tuned"],
                        monitor=False,
                        out_dir=out,
                    )
                    scan_s = None
                else:
                    r = run_baseline_slide(
                        sp,
                        args.tile_weight,
                        args.slide_weight,
                        seed=args.seed,
                        vectorized_scan=cfg["vectorized_scan"],
                        dataloader_tuned=cfg["dataloader_tuned"],
                        use_tf32=cfg["use_tf32"],
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        scan_step=args.scan_step,
                        target_level=args.target_level,
                        max_tokens=args.max_tokens,
                        monitor=False,
                        out_dir=out,
                    )
                    scan_s = r.get("scan_seconds")

                rows.append(
                    {
                        "slide": sp,
                        "config": tag,
                        "vectorized_scan": cfg["vectorized_scan"],
                        "dataloader_tuned": cfg["dataloader_tuned"],
                        "use_tf32": cfg["use_tf32"],
                        "stream": cfg["stream"],
                        "scan_seconds": scan_s if not cfg["stream"] else "",
                        "tile_seconds": r.get("tile_seconds"),
                        "slide_seconds": r.get("slide_seconds"),
                        "tiles_per_sec": r.get("tile_throughput_tiles_per_sec"),
                        "tiles_seen": r.get("tiles_seen"),
                    }
                )

        ab_dir = os.path.join(args.output_root, "ablation")
        os.makedirs(ab_dir, exist_ok=True)
        csv_path = os.path.join(ab_dir, "ablation_runs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(ab_dir, "ablation_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers, "max_tokens": args.max_tokens}, f, indent=2)
        print(f"Wrote {csv_path}")
