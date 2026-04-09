#!/usr/bin/env python3
"""
卵巢 UBC OCEAN 风格 PNG 预处理：Otsu + 外轮廓 → 白底组织图，并写出金字塔 JPEG-TIFF（pyvips）。

- **Step2/3（GPU，有 CUDA 时）**：分块灰度直方图、分块二值图；Otsu 阈值与 CPU 公式一致（_otsu_threshold_from_hist）。
- **findContours**：CPU（OpenCV）。
- **合成**：GPU 分块 torch.where；无 CUDA 时 tensor 在 CPU。
- **写出**：pyvips 金字塔 + JPEG 压缩 BigTIFF（不再使用 tifffile 手写 SubIFD）。

双卡：多张 PNG 仍按文件列表分到两进程；单张大图由单卡分块处理（若仍 OOM 可减小 --gpu-chunk）。

兼容导出：`_accumulate_gray_histogram`、`_fill_threshold_map`、`_gpu_combine_region` 等仍供其它脚本 import。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# CPU 参考：灰度 / 直方图 / Otsu / 二值图（与历史实现一致）
# ---------------------------------------------------------------------------


def _rgb_u8_to_gray_u8(arr: np.ndarray) -> np.ndarray:
    """arr: (h,w,3) uint8，与 OpenCV RGB→Gray 权重一致。"""
    if not arr.flags.writeable:
        arr = arr.copy()
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clip(0, 255).astype(np.uint8)


def _otsu_threshold_from_hist(hist: np.ndarray) -> float:
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    bin_idx = np.arange(256, dtype=np.float64)
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * bin_idx)
    mu_t = mu[-1]
    valid = omega > 0
    omega_f = np.maximum(omega, 1e-10)
    w_f = total - omega
    valid2 = valid & (w_f > 0)
    sigma_b2 = np.zeros(256, dtype=np.float64)
    mu_b = mu / omega_f
    mu_f = (mu_t - mu) / np.maximum(w_f, 1e-10)
    sigma_b2[valid2] = (omega[valid2] * w_f[valid2] * (mu_b[valid2] - mu_f[valid2]) ** 2)
    return float(np.argmax(sigma_b2))


def _iter_tiles(width: int, height: int, tw: int, th: int) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for y in range(0, height, th):
        y1 = min(y + th, height)
        for x in range(0, width, tw):
            x1 = min(x + tw, width)
            out.append((x, y, x1, y1))
    return out


def _accumulate_gray_histogram(img: Image.Image, tw: int, th: int) -> np.ndarray:
    W, H = img.size
    hist = np.zeros(256, dtype=np.int64)
    for x0, y0, x1, y1 in _iter_tiles(W, H, tw, th):
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        arr = np.asarray(crop, dtype=np.uint8)
        g = _rgb_u8_to_gray_u8(arr)
        hist += np.bincount(g.ravel(), minlength=256)
    return hist


def _fill_threshold_map(
    img: Image.Image, T: float, tw: int, th: int, out: np.ndarray
) -> None:
    """out: (H, W) uint8，写入 0/255。"""
    W, H = img.size
    for x0, y0, x1, y1 in _iter_tiles(W, H, tw, th):
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        arr = np.asarray(crop, dtype=np.uint8)
        g = _rgb_u8_to_gray_u8(arr).astype(np.float32)
        out[y0:y1, x0:x1] = ((g > T) * 255).astype(np.uint8)


def _gpu_combine_region(
    img: Image.Image,
    mask_np: np.ndarray,
    out: np.ndarray,
    device: torch.device,
    tw: int,
    th: int,
    y_start: int,
    y_end: int,
) -> None:
    """将 [y_start, y_end) 行范围内的块在 GPU/CPU 上合成到 out。"""
    W, H = img.size
    y_end = min(y_end, H)
    for y0 in range(y_start, y_end, th):
        y1 = min(y0 + th, y_end)
        for x0 in range(0, W, tw):
            x1 = min(x0 + tw, W)
            crop = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.uint8)
            if not crop.flags.writeable:
                crop = crop.copy()
            m = mask_np[y0:y1, x0:x1]
            fg = (m == 0).all(axis=-1)
            rgb = torch.from_numpy(crop).to(device, non_blocking=True)
            fg_t = torch.from_numpy(fg).to(device, non_blocking=True)
            comb = torch.where(
                fg_t.unsqueeze(-1),
                rgb,
                torch.full_like(rgb, 255, dtype=torch.uint8),
            )
            out[y0:y1, x0:x1] = comb.detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU 分块：灰度直方图 + 二值图 + 合成后 RGB（与历史 V2 语义一致）
# ---------------------------------------------------------------------------


def _rgb_to_gray_u8_gpu(rgb_u8: torch.Tensor) -> torch.Tensor:
    rgb = rgb_u8.to(dtype=torch.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.clamp_(0, 255).to(torch.uint8)


def prepare_l0_l1(
    input_path: Path,
    gpu_chunk: int,
    device: torch.device,
    *,
    force_cpu_prepare: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读 PNG，返回白底组织图 level0 RGB uint8 与 level1（CPU INTER_AREA，供统计形状）。
    有 CUDA 且未 force_cpu_prepare 时 Step2/3 在 GPU 上分块执行。
    """
    if force_cpu_prepare or device.type != "cuda":
        return _prepare_l0_l1_cpu_fallback(input_path, gpu_chunk, device)
    return _prepare_l0_l1_gpu_accel(input_path, gpu_chunk, device)


def _prepare_l0_l1_gpu_accel(
    input_path: Path, gpu_chunk: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    tw = th = max(256, int(gpu_chunk))
    tiles = _iter_tiles(w, h, tw, th)

    hist = torch.zeros(256, device=device, dtype=torch.float64)
    for x0, y0, x1, y1 in tiles:
        crop = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.uint8)
        if not crop.flags.writeable:
            crop = crop.copy()
        t = torch.from_numpy(crop).to(device=device, non_blocking=True)
        gray = _rgb_to_gray_u8_gpu(t)
        hist += torch.bincount(gray.view(-1).to(torch.int64), minlength=256).to(torch.float64)

    thr = float(_otsu_threshold_from_hist(hist.detach().cpu().numpy()))
    thresh_full = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in tiles:
        crop = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.uint8)
        if not crop.flags.writeable:
            crop = crop.copy()
        t = torch.from_numpy(crop).to(device=device, non_blocking=True)
        gray = _rgb_to_gray_u8_gpu(t)
        g = gray.to(dtype=torch.float32)
        bw = ((g > thr) * 255).to(torch.uint8)
        thresh_full[y0:y1, x0:x1] = bw.cpu().numpy()

    contours, _ = cv2.findContours(thresh_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no contours from threshold map")

    mask_np = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.drawContours(mask_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    out = np.empty((h, w, 3), dtype=np.uint8)
    _gpu_combine_region(img, mask_np, out, device, tw, th, 0, h)
    l0 = out
    l1 = cv2.resize(l0, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    return l0, l1


def _prepare_l0_l1_cpu_fallback(
    input_path: Path, gpu_chunk: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    tw = th = max(256, int(gpu_chunk))
    hist = _accumulate_gray_histogram(img, tw, th)
    thr = _otsu_threshold_from_hist(hist)
    thresh_full = np.zeros((h, w), dtype=np.uint8)
    _fill_threshold_map(img, thr, tw, th, thresh_full)
    contours, _ = cv2.findContours(thresh_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no contours from threshold map")
    mask_np = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.drawContours(mask_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    out = np.empty((h, w, 3), dtype=np.uint8)
    _gpu_combine_region(img, mask_np, out, device, tw, th, 0, h)
    l0 = out
    l1 = cv2.resize(l0, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    return l0, l1


def _write_with_pyvips(l0: np.ndarray, output_tif: Path, jpeg_q: int) -> None:
    import pyvips  # type: ignore

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    v = pyvips.Image.new_from_array(l0)
    v.write_to_file(
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


def write_pyramid_jpeg_tif(l0: np.ndarray, output_tif: Path, jpeg_q: int) -> None:
    """由 RGB uint8 level0 写出 pyvips 金字塔 JPEG BigTIFF。"""
    _write_with_pyvips(l0, output_tif, jpeg_q)


def tif_pyramid_level_info(output_tif: Path) -> Dict[str, Any]:
    """读取写出后 TIFF 的金字塔层信息。"""
    return _inspect_levels(output_tif)


# ---------------------------------------------------------------------------
# 批处理入口：每张图
# ---------------------------------------------------------------------------


def process_one_png(
    input_path: Path,
    output_dir: Path,
    device: torch.device,
    gpu_chunk: int,
    write_tile: Tuple[int, int],
    compression: int,
    predictor: int,
    save_intermediates: bool = True,
    jpeg_q: int = 85,
) -> Tuple[bool, str]:
    """
    写出使用 pyvips；write_tile / compression / predictor 保留参数位以兼容旧调用，当前不参与写出。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    tw = th = max(256, int(gpu_chunk))

    try:
        use_gpu_prepare = device.type == "cuda"
        l0, l1 = prepare_l0_l1(
            input_path, gpu_chunk, device, force_cpu_prepare=not use_gpu_prepare
        )

        img = Image.open(input_path).convert("RGB")
        W, H = img.size

        if save_intermediates:
            hist = _accumulate_gray_histogram(img, tw, th)
            T = _otsu_threshold_from_hist(hist)
            thresh_full = np.zeros((H, W), dtype=np.uint8)
            _fill_threshold_map(img, T, tw, th, thresh_full)
            contours, _ = cv2.findContours(
                thresh_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                mask_np = np.ones((H, W, 3), dtype=np.uint8) * 255
                cv2.drawContours(mask_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
                mask_path = output_dir / f"{stem}_mask.png"
                Image.fromarray(mask_np).save(mask_path)
            combined_png = output_dir / f"{stem}_combined.png"
            Image.fromarray(l0).save(combined_png)

        out_tif = output_dir / f"{stem}.tif"
        write_pyramid_jpeg_tif(l0, out_tif, jpeg_q)

        metadata = {
            "Software": "ov_processing_gpu_pyvips",
            "gpu_chunk": tw,
            "jpeg_q": int(jpeg_q),
            "levels_note": "pyvips pyramid; inspect with tifffile",
            "level0_shape": list(l0.shape),
            "level1_shape": list(l1.shape),
        }
        sidecar = output_dir / f"{stem}_metadata.json"
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return True, str(out_tif)
    except Exception as e:
        return False, str(e)


def _worker_entry(
    gpu_id: int,
    files: Sequence[Path],
    output_dir: Path,
    gpu_chunk: int,
    write_tile: Tuple[int, int],
    compression: int,
    predictor: int,
    save_intermediates: bool,
    jpeg_q: int,
) -> int:
    fails = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    for p in files:
        ok, msg = process_one_png(
            p,
            output_dir,
            device=device,
            gpu_chunk=gpu_chunk,
            write_tile=write_tile,
            compression=compression,
            predictor=predictor,
            save_intermediates=save_intermediates,
            jpeg_q=jpeg_q,
        )
        tag = "OK" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"[gpu{gpu_id}] {tag} {p.name}: {msg}", flush=True)
    return fails


def _parallel_worker(
    gpu_id: int,
    chunk: List[Path],
    output_dir: Path,
    gpu_chunk: int,
    write_tile: Tuple[int, int],
    compression: int,
    predictor: int,
    save_intermediates: bool,
    jpeg_q: int,
    fail_q: mp.Queue,
) -> None:
    fails = _worker_entry(
        gpu_id,
        chunk,
        output_dir,
        gpu_chunk,
        write_tile,
        compression,
        predictor,
        save_intermediates,
        jpeg_q,
    )
    fail_q.put(fails)


def _partition_files(files: List[Path], n: int) -> List[List[Path]]:
    if n <= 1:
        return [files]
    chunks: List[List[Path]] = [[] for _ in range(n)]
    for i, f in enumerate(files):
        chunks[i % n].append(f)
    return [c for c in chunks if c]


def collect_inputs(input_path: Optional[Path], input_dir: Optional[Path]) -> List[Path]:
    if input_path is not None:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        return [input_path.resolve()]
    if input_dir is None:
        raise ValueError("请指定 --input 或 --input-dir")
    d = input_dir.resolve()
    if not d.is_dir():
        raise NotADirectoryError(d)
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_out = repo / "test_single"
    default_in = Path(
        "/public/home/wang/share_group_folder_wang/pathology/ov_images/"
        "raw_datasets/ubc_ocean/train_images/66.png"
    )

    p = argparse.ArgumentParser(
        description="分块 GPU：卵巢 PNG → 白底 + 金字塔 TIFF（pyvips JPEG）"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=default_in,
        help="单个 PNG（默认：UBC train_images/66.png）",
    )
    p.add_argument("--input-dir", type=Path, default=None, help="目录内全部 .png（与 --input 二选一）")
    p.add_argument("--output-dir", type=Path, default=default_out, help="输出目录")
    p.add_argument(
        "--gpu-chunk",
        type=int,
        default=4096,
        help="分块边长（正方形像素，默认 4096）",
    )
    p.add_argument(
        "--write-tile",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("H", "W"),
        help="兼容保留（当前 pyvips 写出不使用该参数）",
    )
    p.add_argument(
        "--compression",
        type=int,
        default=8,
        help="兼容保留（当前使用 --jpeg-q）",
    )
    p.add_argument(
        "--predictor",
        type=int,
        default=2,
        help="兼容保留（当前 pyvips 写出不使用该参数）",
    )
    p.add_argument("--jpeg-q", type=int, default=85, help="pyvips JPEG 质量")
    p.add_argument("--no-intermediates", action="store_true", help="不保存 _mask / _combined PNG")
    p.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="并行 GPU 数（默认 min(2, 可见卡数)；多文件时按文件轮转分配）",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="单进程时使用的 GPU 索引（默认 0）",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    files = collect_inputs(
        None if args.input_dir else args.input,
        args.input_dir,
    )
    out = args.output_dir.resolve()
    write_tile = (int(args.write_tile[0]), int(args.write_tile[1]))
    gpu_chunk = int(args.gpu_chunk)
    save_int = not args.no_intermediates
    jpeg_q = int(args.jpeg_q)

    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    want = args.num_gpus if args.num_gpus is not None else min(2, max(1, n_cuda))
    want = max(1, want)

    if n_cuda == 0:
        print("警告：未检测到 CUDA，前处理走 CPU 路径。", file=sys.stderr)
        fails = _worker_entry(
            0,
            files,
            out,
            gpu_chunk,
            write_tile,
            args.compression,
            args.predictor,
            save_int,
            jpeg_q,
        )
        return 1 if fails else 0

    if len(files) < 2 or want < 2 or n_cuda < 2:
        gid = min(args.gpu_id, n_cuda - 1)
        fails = _worker_entry(
            gid,
            files,
            out,
            gpu_chunk,
            write_tile,
            args.compression,
            args.predictor,
            save_int,
            jpeg_q,
        )
        return 1 if fails else 0

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ng = min(want, n_cuda, len(files))
    chunks = _partition_files(list(files), ng)
    procs: List[Process] = []
    fail_q: mp.Queue = mp.Queue()

    for rank, chunk in enumerate(chunks):
        proc = Process(
            target=_parallel_worker,
            args=(
                rank,
                chunk,
                out,
                gpu_chunk,
                write_tile,
                args.compression,
                args.predictor,
                save_int,
                jpeg_q,
                fail_q,
            ),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
    total_fail = 0
    for _ in range(len(procs)):
        total_fail += int(fail_q.get())
    return 1 if total_fail else 0


def main_single_report(argv: Optional[Sequence[str]] = None) -> int:
    """单图 + JSON 报告（兼容原 ov_processing_new_monitorV2 命令行）。"""
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="单图 PNG→TIFF + JSON 报告（GPU 前处理 + pyvips）")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/public/home/wang/share_group_folder_wang/pathology/ov_images/"
            "raw_datasets/ubc_ocean/train_images/66.png"
        ),
    )
    p.add_argument("--output-dir", type=Path, default=repo / "test_single_new_v2")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--gpu-chunk", type=int, default=4096)
    p.add_argument("--jpeg-q", type=int, default=85)
    p.add_argument("--force-cpu-prepare", action="store_true")
    args = p.parse_args(argv)

    input_png = args.input.resolve()
    if not input_png.is_file():
        raise FileNotFoundError(input_png)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{input_png.stem}.tif"
    out_json = args.output_json or (out_dir / f"{input_png.stem}_convert_report_v2.json")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpu_prepare = device.type == "cuda" and not args.force_cpu_prepare

    t0 = time.perf_counter()
    t_prep0 = time.perf_counter()
    l0, l1 = prepare_l0_l1(
        input_png, args.gpu_chunk, device, force_cpu_prepare=not use_gpu_prepare
    )
    prep_s = time.perf_counter() - t_prep0

    t_w0 = time.perf_counter()
    write_pyramid_jpeg_tif(l0, out_tif, args.jpeg_q)
    write_s = time.perf_counter() - t_w0
    total_s = time.perf_counter() - t0

    lvl = tif_pyramid_level_info(out_tif)
    result: Dict[str, Any] = {
        "input": str(input_png),
        "output_tif": str(out_tif),
        "output_json": str(out_json),
        "device": str(device),
        "gpu_prepare_enabled": bool(use_gpu_prepare),
        "gpu_chunk": int(args.gpu_chunk),
        "jpeg_q": int(args.jpeg_q),
        "prepare_seconds": prep_s,
        "write_seconds": write_s,
        "total_seconds": total_s,
        "prepared_level0_shape": list(l0.shape),
        "prepared_level1_shape": list(l1.shape),
        "tif_level_info": lvl,
        "notes": [
            "逻辑位于 scripts/ov_processing_gpu.py：GPU 分块直方图/二值图 + CPU Otsu 与轮廓 + pyvips 写出。",
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if lvl.get("ok_two_levels") else 1


if __name__ == "__main__":
    raise SystemExit(main())
