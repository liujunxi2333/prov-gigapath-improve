#!/usr/bin/env python3
"""
单张 WSI：与 hybrid v9 一致——整张切片一次 GPU 扫描后，将坐标列表按前后两半二分，
GPU0 / GPU1 各跑一半 tile（完整 ViT），再在 GPU1 上做 slide encoder。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone


def _setup_paths() -> str:
    root = os.path.abspath(os.path.dirname(__file__))
    p2 = os.path.join(root, "parallel_improve2")
    if p2 not in sys.path:
        sys.path.insert(0, p2)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def main() -> None:
    repo_root = _setup_paths()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from wsi_embed.pipeline_v9 import _default_scan_cpu_workers_v9, run_v9_pipeline

    _slide_default = os.environ.get("SLIDE_PATH", "")
    p = argparse.ArgumentParser(description="单张 WSI：坐标二分 + 双卡 tile（v9）")
    p.add_argument(
        "--slide_path",
        type=str,
        default=_slide_default,
        help="WSI 路径；也可环境变量 SLIDE_PATH",
    )
    p.add_argument("--out_dir", type=str, default="", help="默认 <仓库>/runs/<基名>/v9_run")
    p.add_argument("--seed", type=int, default=42)
    _default_tile = os.environ.get("TILE_WEIGHT", "/public/home/wang/liujx/pytorch_model.bin")
    _default_slide = os.environ.get("SLIDE_WEIGHT", "/public/home/wang/liujx/slide_encoder.pth")
    p.add_argument("--tile_weight", type=str, default=_default_tile)
    p.add_argument("--slide_weight", type=str, default=_default_slide)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers_per_gpu", type=int, default=4)
    p.add_argument("--scan_step", type=int, default=4)
    p.add_argument("--target_level", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument("--scan_cpu_workers", type=int, default=-1, help="-1 为自动条带进程数")
    p.add_argument("--scan_gpu_id", type=int, default=0, help="扫描阶段 mask→coords 使用的 GPU")
    p.add_argument("--scan_unique_on_gpu", type=int, default=1, choices=[0, 1])
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--monitor_interval", type=float, default=0.1)

    args = p.parse_args()
    slide_path = (args.slide_path or "").strip()
    if not slide_path:
        raise SystemExit("请指定 --slide_path 或环境变量 SLIDE_PATH")

    if args.scan_cpu_workers < 0:
        scan_w = _default_scan_cpu_workers_v9()
        scan_auto = True
    else:
        scan_w = max(1, int(args.scan_cpu_workers))
        scan_auto = False

    stem = os.path.splitext(os.path.basename(slide_path))[0]
    out_dir = args.out_dir.strip() or os.path.join(repo_root, "runs", stem, "v9_run")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "entry": "main_single.py",
        "pipeline": "outv9_hybrid_gpu_scan_coord_split",
        "slide_path": os.path.abspath(slide_path),
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
        slide_path=slide_path,
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
