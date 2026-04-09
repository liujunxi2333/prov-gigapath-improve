#!/usr/bin/env python3
"""
对比多种「PNG → 白底 RGB + 金字塔 TIFF」相关实现，输出 JSON，便于找最短耗时路径。

阶段说明：
1) prepare：与 ov_processing_gpu 相同语义，得到 l0 / l1（uint8），不计入各 write 的公平对比时可单独记录。
2) write_*：在 **同一组 l0/l1** 上仅测试 **编码写出**（除 e2e 外）。
3) e2e_ov_processing_gpu：完整 process_one_png（处理 + 默认 deflate 写出），作为端到端基线。

可选依赖：pyvips（libvips）；ImageMagick 的 convert（大图为避免二次 IO 默认跳过）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

# 与 ov_processing_gpu 同目录，复用其分块与合成逻辑
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ov_processing_gpu as ov  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def prepare_l0_l1(
    input_path: Path, gpu_chunk: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    img = Image.open(input_path).convert("RGB")
    W, H = img.size
    tw = th = max(256, int(gpu_chunk))
    hist = ov._accumulate_gray_histogram(img, tw, th)
    T = ov._otsu_threshold_from_hist(hist)
    thresh_full = np.zeros((H, W), dtype=np.uint8)
    ov._fill_threshold_map(img, T, tw, th, thresh_full)
    contours, _ = cv2.findContours(
        thresh_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("no contours")
    mask_np = np.ones((H, W, 3), dtype=np.uint8) * 255
    cv2.drawContours(mask_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    out = np.empty((H, W, 3), dtype=np.uint8)
    ov._gpu_combine_region(img, mask_np, out, device, tw, th, 0, H)
    l0 = out
    l1 = cv2.resize(
        l0, (max(1, W // 2), max(1, H // 2)), interpolation=cv2.INTER_AREA
    )
    return l0, l1, W, H


def _write_tifffile_two_level(
    l0: np.ndarray,
    l1: np.ndarray,
    path: Path,
    *,
    compression: Any,
    write_tile: Tuple[int, int],
    predictor: Optional[int],
    description: str,
) -> None:
    """predictor=None：不写 predictor 标签（JPEG 等与 predictor 不兼容）。"""
    wt, ht = write_tile
    path.parent.mkdir(parents=True, exist_ok=True)
    common: Dict[str, Any] = {
        "photometric": "rgb",
        "compression": compression,
        "tile": (ht, wt),
        "planarconfig": "contig",
    }
    if predictor is not None:
        common["predictor"] = predictor
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(
            l0,
            resolution=(10, 1),
            description=description,
            subifds=1,
            **common,
        )
        tw.write(
            l1,
            resolution=(5, 1),
            subfiletype=1,
            **common,
        )


def _try_pyvips_write_pyramid(l0: np.ndarray, path: Path, *, jpeg_q: int) -> Tuple[bool, str]:
    try:
        import pyvips  # type: ignore
    except Exception as e:
        return False, f"pyvips import failed: {e}"
    try:
        v = pyvips.Image.new_from_array(l0)
        v.write_to_file(
            str(path),
            pyramid=True,
            tile=True,
            compression="jpeg",
            Q=jpeg_q,
            bigtiff=True,
        )
        return True, ""
    except Exception as e:
        return False, str(e)


def _run_timed(name: str, fn: Callable[[], None]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    err: Optional[str] = None
    try:
        fn()
        ok = True
    except Exception as e:
        ok = False
        err = str(e)
    dt = time.perf_counter() - t0
    return {"method": name, "seconds": dt, "ok": ok, "error": err}


def main(argv: Optional[List[str]] = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark PNG→TIFF methods (prepare + writers + e2e)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/public/home/wang/share_group_folder_wang/pathology/ov_images/"
            "raw_datasets/ubc_ocean/train_images/66.png"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo / "runs" / "benchmark_png_to_tif",
        help="输出目录（各方法子目录 + benchmark_result.json）",
    )
    parser.add_argument("--gpu-chunk", type=int, default=4096)
    parser.add_argument("--write-tile", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--predictor", type=int, default=2)
    parser.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="若给定，跳过 prepare，直接加载 l0/l1（keys: l0,l1）",
    )
    parser.add_argument("--skip-e2e", action="store_true", help="不跑端到端 process_one_png")
    parser.add_argument("--skip-pyvips", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--jpeg-q", type=int, default=85)
    args = parser.parse_args(argv)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or (work_dir / "benchmark_result.json")

    device = _device()
    write_tile = (int(args.write_tile[0]), int(args.write_tile[1]))
    desc = json.dumps(
        {
            "Software": "benchmark_png_to_tif_methods",
            "note": "two-level SubIFD for tifffile writers; pyvips may emit deeper pyramid",
        }
    )

    report: Dict[str, Any] = {
        "input": str(Path(args.input).resolve()),
        "work_dir": str(work_dir),
        "device": str(device),
        "gpu_chunk": args.gpu_chunk,
        "prepare_seconds": None,
        "methods": [],
        "notes": [],
    }

    l0: Optional[np.ndarray] = None
    l1: Optional[np.ndarray] = None
    W = H = 0

    if args.from_npz is not None:
        z = np.load(args.from_npz)
        l0, l1 = z["l0"], z["l1"]
        report["notes"].append(f"loaded npz {args.from_npz}")
    else:
        t_prep0 = time.perf_counter()
        l0, l1, W, H = prepare_l0_l1(Path(args.input), args.gpu_chunk, device)
        report["prepare_seconds"] = time.perf_counter() - t_prep0
        npz_path = work_dir / "l0_l1_cache.npz"
        np.savez(npz_path, l0=l0, l1=l1)
        report["cache_npz"] = str(npz_path)

    assert l0 is not None and l1 is not None

    # --- write variants (same arrays) ---
    def add_method(rec: Dict[str, Any], out_path: Path) -> None:
        rec["output_path"] = str(out_path)
        report["methods"].append(rec)

    stem = Path(args.input).stem

    add_method(
        _run_timed(
            "write_tifffile_deflate8",
            lambda: _write_tifffile_two_level(
                l0,
                l1,
                work_dir / "write_tifffile_deflate8" / f"{stem}.tif",
                compression=8,
                write_tile=write_tile,
                predictor=args.predictor,
                description=desc,
            ),
        ),
        work_dir / "write_tifffile_deflate8" / f"{stem}.tif",
    )

    add_method(
        _run_timed(
            "write_tifffile_lzw",
            lambda: _write_tifffile_two_level(
                l0,
                l1,
                work_dir / "write_tifffile_lzw" / f"{stem}.tif",
                compression=5,
                write_tile=write_tile,
                predictor=args.predictor,
                description=desc,
            ),
        ),
        work_dir / "write_tifffile_lzw" / f"{stem}.tif",
    )

    add_method(
        _run_timed(
            "write_tifffile_jpeg",
            lambda: _write_tifffile_two_level(
                l0,
                l1,
                work_dir / "write_tifffile_jpeg" / f"{stem}.tif",
                compression=7,
                write_tile=write_tile,
                predictor=None,
                description=desc,
            ),
        ),
        work_dir / "write_tifffile_jpeg" / f"{stem}.tif",
    )

    if not args.skip_pyvips:
        pv_path = work_dir / "write_pyvips_pyramid_jpeg" / f"{stem}.tif"
        pv_path.parent.mkdir(parents=True, exist_ok=True)

        def _pv() -> None:
            ok, err = _try_pyvips_write_pyramid(l0, pv_path, jpeg_q=args.jpeg_q)
            if not ok:
                raise RuntimeError(err or "pyvips failed")

        add_method(_run_timed("write_pyvips_pyramid_jpeg", _pv), pv_path)
        report["notes"].append(
            "pyvips: pyramid=True 可能生成多于 2 层；与 tifffile 两 SubIFD 结构不必完全一致"
        )

    # --- e2e ---
    if not args.skip_e2e:
        e2e_dir = work_dir / "e2e_ov_processing_gpu"
        if e2e_dir.exists():
            shutil.rmtree(e2e_dir)
        e2e_dir.mkdir(parents=True, exist_ok=True)

        def _e2e() -> None:
            ok, msg = ov.process_one_png(
                Path(args.input),
                e2e_dir,
                device=device,
                gpu_chunk=args.gpu_chunk,
                write_tile=write_tile,
                compression=8,
                predictor=args.predictor,
                save_intermediates=False,
            )
            if not ok:
                raise RuntimeError(msg)

        add_method(_run_timed("e2e_ov_processing_gpu", _e2e), e2e_dir / f"{stem}.tif")

    # --- fastest ---
    ok_methods = [m for m in report["methods"] if m.get("ok")]
    if ok_methods:
        best = min(ok_methods, key=lambda x: float(x["seconds"]))
        report["fastest_overall"] = {
            "method": best["method"],
            "seconds": best["seconds"],
            "output_path": best.get("output_path"),
        }
        write_only = [m for m in ok_methods if m["method"].startswith("write_")]
        if write_only:
            bw = min(write_only, key=lambda x: float(x["seconds"]))
            report["fastest_write_only"] = {
                "method": bw["method"],
                "seconds": bw["seconds"],
            }
            est = (report.get("prepare_seconds") or 0.0) + float(bw["seconds"])
            report["estimate_prepare_plus_fastest_write"] = est

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
