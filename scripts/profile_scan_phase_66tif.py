#!/usr/bin/env python3
"""
单张切片第一阶段（组织坐标扫描）瓶颈粗测：对比「池等待/I/O+CPU 条带」与「GPU 条带处理 + merge/unique」耗时。

默认与 main_batch / v9 一致：scan_step=4，scan_cpu_workers 默认与 pipeline_v9 相同（12–48 按核数）。
默认切片路径：<repo>/runs/66.tif，可用 --slide 或环境变量 SLIDE_PATH 覆盖。

用法:
  cd /public/home/wang/liujx/prov-gigapath-improve
  export PYTHONPATH=$PWD:$PWD/parallel_improve2
  python scripts/profile_scan_phase_66tif.py --slide /path/to/66.tif

  # 与集群一致时显式指定:
  SCAN_STEP=4 SCAN_CPU_WORKERS=32 python scripts/profile_scan_phase_66tif.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_roots() -> tuple[str, str]:
    here = os.path.abspath(os.path.dirname(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    p2 = os.path.join(repo, "parallel_improve2")
    return repo, p2


def _default_scan_cpu_workers() -> int:
    nc = os.cpu_count() or 16
    return int(min(48, max(12, nc)))


def main() -> None:
    repo, p2 = _repo_roots()
    if p2 not in sys.path:
        sys.path.insert(0, p2)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    default_slide = os.path.join(repo, "runs", "66.tif")
    env_slide = os.environ.get("SLIDE_PATH", "").strip()

    ap = argparse.ArgumentParser(description="profile compute_tissue_coords_parallel_strips_gpu (phase ①)")
    ap.add_argument(
        "--slide",
        type=str,
        default=env_slide or default_slide,
        help=f"切片路径（默认 runs/66.tif 或 SLIDE_PATH），当前默认: {default_slide}",
    )
    ap.add_argument("--scan_step", type=int, default=int(os.environ.get("SCAN_STEP", "4")))
    ap.add_argument(
        "--scan_cpu_workers",
        type=int,
        default=int(os.environ.get("SCAN_CPU_WORKERS", "-1")),
        help="条带进程数；-1 表示按 CPU 核数取 12–48（与 v9 默认一致）",
    )
    ap.add_argument("--gpu_id", type=int, default=int(os.environ.get("SCAN_GPU_ID", "0")))
    ap.add_argument("--target_level", type=int, default=0)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--bg_threshold", type=int, default=210)
    args = ap.parse_args()

    sw = args.scan_cpu_workers
    if sw < 0:
        sw = _default_scan_cpu_workers()

    slide = os.path.abspath(args.slide)
    if not os.path.isfile(slide):
        print(f"[error] 文件不存在: {slide}", file=sys.stderr)
        print("请放置 66.tif 到 runs/ 或使用 --slide / SLIDE_PATH 指定路径。", file=sys.stderr)
        sys.exit(1)

    import torch

    if not torch.cuda.is_available():
        print("[error] 需要 CUDA", file=sys.stderr)
        sys.exit(1)

    from wsi_embed.coords import compute_tissue_coords_parallel_strips_gpu

    coords, elapsed, bd = compute_tissue_coords_parallel_strips_gpu(
        slide_path=slide,
        tile_size=args.tile_size,
        target_level=args.target_level,
        bg_threshold=args.bg_threshold,
        scan_step=args.scan_step,
        num_workers=sw,
        gpu_id=args.gpu_id,
        unique_on_gpu=True,
        return_breakdown=True,
    )

    assert isinstance(bd, dict)

    print("=== 切片与参数 ===")
    print(json.dumps({"slide": slide, "scan_step": args.scan_step, "scan_cpu_workers": sw, "gpu_id": args.gpu_id}, ensure_ascii=False, indent=2))
    print("=== 元数据 / 分解（秒）===")
    keys_order = [
        "vendor",
        "thumb_level",
        "thumb_wh",
        "level0_wh",
        "level0_tile_size",
        "n_strips",
        "n_strip_delivered",
        "n_strips_with_keys",
        "n_key_samples_before_unique",
        "n_unique_tiles",
        "n_coords_out",
        "meta_open_s",
        "pool_phase_wall_s",
        "wait_next_strip_s",
        "gpu_per_strip_sync_s",
        "gpu_merge_unique_s",
        "total_s",
    ]
    slim = {k: bd.get(k) for k in keys_order if k in bd}
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    if "interpretation" in bd:
        print("=== 瓶颈粗判 ===")
        print(json.dumps(bd["interpretation"], ensure_ascii=False, indent=2))
    print("=== 结论摘要 ===")
    pp = float(bd.get("pool_phase_wall_s", 0.0))
    gs = float(bd.get("gpu_per_strip_sync_s", 0.0))
    gm = float(bd.get("gpu_merge_unique_s", 0.0))
    print(
        f"条带阶段墙钟 {pp:.3f}s | GPU 条带(同步累计) {gs:.3f}s | GPU merge/unique {gm:.3f}s | 总 {elapsed:.3f}s | 输出 tile 数 {len(coords)}"
    )
    if pp > 1e-6 and gs + gm < 0.2 * pp:
        print("粗判: 条带阶段主要时间在多进程读盘/解码与等待结果，GPU 去重占比很小。")
    elif gs + gm > 0.35 * max(pp, 1e-6):
        print("粗判: GPU 在条带或 merge 上时间不可忽视（候选点多或卡较慢时更明显）。")
    else:
        print("粗判: I/O/CPU 与 GPU 均有一定占比，可看 fraction_pool_wall_not_gpu_strip 细读。")


if __name__ == "__main__":
    main()
