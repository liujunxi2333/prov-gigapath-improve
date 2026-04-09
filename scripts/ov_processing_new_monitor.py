#!/usr/bin/env python3
"""
采用基准里最快的 pyvips 写出路径：
1) 复用 ov_processing_gpu 的分块流程生成白底 level0（l0）与 level1（l1）
2) 使用 pyvips 直接写 pyramidal TIFF
3) 用 tifffile 校验 level0/level1 是否存在，并输出 JSON 结果（含耗时）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ov_processing_gpu as ov  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def _prepare_l0_l1(
    input_path: Path, gpu_chunk: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    tw = th = max(256, int(gpu_chunk))
    hist = ov._accumulate_gray_histogram(img, tw, th)
    thr = ov._otsu_threshold_from_hist(hist)
    thresh_full = np.zeros((h, w), dtype=np.uint8)
    ov._fill_threshold_map(img, thr, tw, th, thresh_full)
    contours, _ = cv2.findContours(
        thresh_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("no contours from threshold map")
    mask_np = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.drawContours(mask_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    out = np.empty((h, w, 3), dtype=np.uint8)
    ov._gpu_combine_region(img, mask_np, out, device, tw, th, 0, h)
    l0 = out
    l1 = cv2.resize(l0, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    return l0, l1


def _write_with_pyvips(l0: np.ndarray, output_tif: Path, jpeg_q: int) -> None:
    try:
        import pyvips  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"pyvips unavailable: {e}") from e

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    img = pyvips.Image.new_from_array(l0)
    img.write_to_file(
        str(output_tif),
        pyramid=True,
        tile=True,
        compression="jpeg",
        Q=int(jpeg_q),
        bigtiff=True,
    )


def _inspect_levels(output_tif: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(output_tif), "ok_two_levels": False}
    with tifffile.TiffFile(output_tif) as tif:
        s0 = tif.series[0]
        levels = list(s0.levels) if getattr(s0, "levels", None) else []
        info["num_levels"] = len(levels)
        if levels:
            info["level0_shape"] = list(levels[0].shape)
        if len(levels) > 1:
            info["level1_shape"] = list(levels[1].shape)
        info["ok_two_levels"] = len(levels) >= 2
    return info


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="PNG -> pyvips pyramidal TIFF（含 level0/level1 检测与耗时输出）"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/public/home/wang/share_group_folder_wang/pathology/ov_images/"
            "raw_datasets/ubc_ocean/train_images/66.png"
        ),
    )
    p.add_argument("--output-dir", type=Path, default=repo / "test_single_new")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--gpu-chunk", type=int, default=4096)
    p.add_argument("--jpeg-q", type=int, default=85)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_png = args.input.resolve()
    if not input_png.is_file():
        raise FileNotFoundError(input_png)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{input_png.stem}.tif"
    out_json = args.output_json or (out_dir / f"{input_png.stem}_convert_report.json")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    t_prep0 = time.perf_counter()
    l0, l1 = _prepare_l0_l1(input_png, args.gpu_chunk, device)
    prep_s = time.perf_counter() - t_prep0

    t_write0 = time.perf_counter()
    _write_with_pyvips(l0, out_tif, args.jpeg_q)
    write_s = time.perf_counter() - t_write0
    total_s = time.perf_counter() - t0

    level_info = _inspect_levels(out_tif)
    result: Dict[str, Any] = {
        "input": str(input_png),
        "output_tif": str(out_tif),
        "output_json": str(out_json),
        "device": str(device),
        "gpu_chunk": int(args.gpu_chunk),
        "jpeg_q": int(args.jpeg_q),
        "prepare_seconds": prep_s,
        "write_seconds": write_s,
        "total_seconds": total_s,
        "prepared_level0_shape": list(l0.shape),
        "prepared_level1_shape": list(l1.shape),
        "tif_level_info": level_info,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if level_info.get("ok_two_levels") else 1


if __name__ == "__main__":
    raise SystemExit(main())
