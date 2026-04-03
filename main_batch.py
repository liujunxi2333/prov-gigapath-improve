#!/usr/bin/env python3
"""
批量 WSI：
  embed_dir — 扫描文件夹内 tif，按文件大小均衡分两队列，GPU0/GPU1 并行（每张切片整卡扫描+tile+slide）。
  benchmark — baseline/stream 对比或消融（slide_list）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys


def _setup_paths() -> str:
    root = os.path.abspath(os.path.dirname(__file__))
    p2 = os.path.join(root, "parallel_improve2")
    if p2 not in sys.path:
        sys.path.insert(0, p2)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _cmd_embed_dir(args: argparse.Namespace, repo_root: str) -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from wsi_embed.pipeline_dual_dir import run_tif_directory_dual_gpu
    from wsi_embed.pipeline_v9 import _default_scan_cpu_workers_v9

    tif_dir = (args.tif_dir or "").strip()
    if not tif_dir or not os.path.isdir(tif_dir):
        raise SystemExit("embed_dir 需要有效的 --tif_dir")

    if args.scan_cpu_workers < 0:
        scan_w = _default_scan_cpu_workers_v9()
    else:
        scan_w = max(1, int(args.scan_cpu_workers))

    out_root = args.output_root.strip() or os.path.join(
        repo_root, "runs", "tif_dual_" + os.path.basename(tif_dir.rstrip(os.sep))
    )

    summary = run_tif_directory_dual_gpu(
        tif_dir=tif_dir,
        output_root=out_root,
        tile_weight=args.tile_weight,
        slide_weight=args.slide_weight,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers_data=args.num_workers,
        scan_step=args.scan_step,
        target_level=args.target_level,
        max_tokens=args.max_tokens,
        scan_cpu_workers=scan_w,
        scan_unique_on_gpu=bool(args.scan_unique_on_gpu),
        recursive=bool(args.recursive),
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
        use_tf32=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] outputs under: {out_root}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from wsi_embed.pipeline_single import main_benchmark

    sys.argv = [sys.argv[0]] + args.remainder
    main_benchmark()


def main() -> None:
    repo_root = _setup_paths()
    p = argparse.ArgumentParser(description="批量 WSI / benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("embed_dir", help="tif 目录：按文件大小分两队列，双卡并行处理多张切片")
    pe.add_argument("--tif_dir", type=str, required=True)
    pe.add_argument("--output_root", type=str, default="")
    pe.add_argument("--seed", type=int, default=42)
    _default_tile = os.environ.get("TILE_WEIGHT", "/public/home/wang/liujx/pytorch_model.bin")
    _default_slide = os.environ.get("SLIDE_WEIGHT", "/public/home/wang/liujx/slide_encoder.pth")
    pe.add_argument("--tile_weight", type=str, default=_default_tile)
    pe.add_argument("--slide_weight", type=str, default=_default_slide)
    pe.add_argument("--batch_size", type=int, default=128)
    pe.add_argument("--num_workers", type=int, default=4, help="每张切片 DataLoader worker 数")
    pe.add_argument("--scan_step", type=int, default=4)
    pe.add_argument("--target_level", type=int, default=0)
    pe.add_argument("--max_tokens", type=int, default=12000)
    pe.add_argument("--scan_cpu_workers", type=int, default=-1)
    pe.add_argument("--scan_unique_on_gpu", type=int, default=1, choices=[0, 1])
    pe.add_argument("--recursive", action="store_true")
    pe.add_argument("--monitor", action="store_true")
    pe.add_argument("--monitor_interval", type=float, default=0.1)

    pb = sub.add_parser("benchmark", help="baseline/stream 对比或消融（--mode compare|ablation + slide_list 等）")
    pb.add_argument("remainder", nargs=argparse.REMAINDER, help="例如 --mode compare --slide_list list.txt ...")

    args = p.parse_args()
    if args.command == "embed_dir":
        _cmd_embed_dir(args, repo_root)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    else:
        raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    main()
