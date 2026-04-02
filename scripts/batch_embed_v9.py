#!/usr/bin/env python3
"""
批量读取目录下（可选递归）所有 .tif，对每张切片运行与
``compare_test/outv9_compare/verify_outv9_vs_baseline.py`` 中
``_ensure_v9_embedding`` 相同的 v9 流程：仅调用 ``run_v9_pipeline``，
输出 768 维 slide 向量 ``embedding.pt`` 及 ``perf.json`` 等。

不包含：baseline 对比、cosine/L2 等指标计算。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def _repo_root() -> str:
    return os.path.abspath(
        os.environ.get(
            "GIGAPATH_IMPROVE_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )
    )


_REPO_ROOT = _repo_root()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 与同目录 hybrid_v9_tile_slide 一致，便于 ``from hybrid_v9_tile_slide import run_v9_pipeline``
sys.path.insert(0, _SCRIPT_DIR)

from hybrid_v9_tile_slide import run_v9_pipeline  # noqa: E402


def collect_tifs(input_dir: str, *, recursive: bool) -> List[str]:
    """收集 .tif / .tiff（小写匹配）。"""
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(input_dir)
    out: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                low = name.lower()
                if low.endswith(".tif") or low.endswith(".tiff"):
                    out.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(input_dir)):
            low = name.lower()
            if low.endswith(".tif") or low.endswith(".tiff"):
                out.append(os.path.join(input_dir, name))
    out.sort()
    return out


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(
        description="批量目录 .tif → v9 768d embedding（等价 verify 脚本中的 _ensure_v9_embedding 流程，无 baseline 对比）"
    )
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="含一张或多张 .tif 的目录",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=os.path.join(_REPO_ROOT, "runs"),
        help="输出根目录；每张切片写入 <output_root>/<基名>/v9_run/",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归子目录收集 .tif",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="若已存在 embedding.pt 则跳过",
    )
    p.add_argument("--limit", type=int, default=0, help="最多处理 N 张，0 表示不限制")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers_per_gpu", type=int, default=4)
    p.add_argument("--scan_step", type=int, default=4)
    p.add_argument("--target_level", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument(
        "--scan_cpu_workers",
        type=int,
        default=-1,
        help="-1 使用 hybrid_v9 内建自动策略；>=1 固定条带进程数",
    )
    p.add_argument("--scan_gpu_id", type=int, default=0)
    p.add_argument("--scan_unique_on_gpu", type=int, default=1, choices=[0, 1])
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--monitor_interval", type=float, default=0.1)
    p.add_argument(
        "--tile_weight",
        type=str,
        default=os.environ.get("TILE_WEIGHT", os.path.join(_REPO_ROOT, "weights", "pytorch_model.bin")),
    )
    p.add_argument(
        "--slide_weight",
        type=str,
        default=os.environ.get("SLIDE_WEIGHT", os.path.join(_REPO_ROOT, "weights", "slide_encoder.pth")),
    )

    args = p.parse_args()

    if args.scan_cpu_workers < 0:
        from hybrid_v9_tile_slide import _default_scan_cpu_workers_v9

        scan_w = _default_scan_cpu_workers_v9()
    else:
        scan_w = max(1, int(args.scan_cpu_workers))

    slides = collect_tifs(args.input_dir, recursive=args.recursive)
    if args.limit > 0:
        slides = slides[: args.limit]

    if not slides:
        raise SystemExit(f"目录中未找到 .tif/.tiff: {args.input_dir}")

    os.makedirs(args.output_root, exist_ok=True)
    manifest: Dict[str, Any] = {
        "input_dir": os.path.abspath(args.input_dir),
        "output_root": os.path.abspath(args.output_root),
        "recursive": args.recursive,
        "n_slides": len(slides),
        "scan_cpu_workers": scan_w,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }

    for i, slide_path in enumerate(slides):
        stem = os.path.splitext(os.path.basename(slide_path))[0]
        out_dir = os.path.join(args.output_root, stem, "v9_run")
        emb_path = os.path.join(out_dir, "embedding.pt")

        if args.skip_existing and os.path.isfile(emb_path):
            manifest["runs"].append(
                {
                    "slide_path": slide_path,
                    "out_dir": out_dir,
                    "status": "skipped_existing",
                }
            )
            print(f"[{i+1}/{len(slides)}] skip (exists): {stem}")
            continue

        print(f"[{i+1}/{len(slides)}] run: {slide_path}")
        t0 = time.perf_counter()
        try:
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
        except Exception as e:
            manifest["runs"].append(
                {
                    "slide_path": slide_path,
                    "out_dir": out_dir,
                    "status": "error",
                    "error": str(e),
                }
            )
            print(f"  [error] {e}")
            continue

        elapsed = time.perf_counter() - t0
        row = {
            "slide_path": slide_path,
            "out_dir": out_dir,
            "embedding_pt": emb_path,
            "status": "ok" if not report.get("error") else "no_tissue",
            "wall_seconds": elapsed,
            "report": report,
        }
        manifest["runs"].append(row)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    man_path = os.path.join(args.output_root, "batch_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] manifest: {man_path}")


if __name__ == "__main__":
    main()
