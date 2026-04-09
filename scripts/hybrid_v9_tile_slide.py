#!/usr/bin/env python3
"""
兼容入口：逻辑已迁至 ``wsi_embed.pipeline_v9`` 与仓库根 ``main.py v9``。
可直接运行本脚本，或优先使用 ``python main.py v9 ...``。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone


def _repo_root() -> str:
    return os.path.abspath(
        os.environ.get(
            "GIGAPATH_IMPROVE_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )
    )


_REPO_ROOT = _repo_root()
sys.path.insert(0, os.path.join(_REPO_ROOT, "parallel_improve2"))
sys.path.insert(0, _REPO_ROOT)

from wsi_embed.pipeline_v9 import (  # noqa: E402
    _default_scan_cpu_workers_v9,
    run_v9_pipeline,
)


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
        default=os.environ.get("TILE_WEIGHT", "/public/home/wang/liujx/pytorch_model.bin"),
    )
    p.add_argument(
        "--slide_weight",
        type=str,
        default=os.environ.get("SLIDE_WEIGHT", "/public/home/wang/liujx/slide_encoder.pth"),
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
