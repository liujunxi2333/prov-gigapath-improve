"""组织坐标扫描：CPU 慢循环 / 向量化 / 多进程条带 / GPU 加速条带。"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import List, Tuple

import numpy as np
import openslide
import torch


def compute_tissue_coords_slow(
    slide_path: str,
    tile_size: int,
    target_level: int,
    bg_threshold: int,
    scan_step: int,
) -> Tuple[List[Tuple[int, int]], float]:
    t0 = time.perf_counter()
    with openslide.OpenSlide(slide_path) as slide:
        target_downsample = slide.level_downsamples[target_level]
        level0_tile_size = int(tile_size * target_downsample)
        thumb_level = slide.level_count - 1
        thumb_downsample = slide.level_downsamples[thumb_level]
        w_thumb, h_thumb = slide.level_dimensions[thumb_level]
        thumb_img = slide.read_region((0, 0), thumb_level, (w_thumb, h_thumb)).convert("RGB")
        thumb_gray = np.mean(np.array(thumb_img), axis=2)
        tissue_mask = thumb_gray < bg_threshold
        aligned_coords = set()
        step = max(1, int(scan_step))
        for y in range(0, h_thumb, step):
            for x in range(0, w_thumb, step):
                if tissue_mask[y, x]:
                    x0 = int(x * thumb_downsample)
                    y0 = int(y * thumb_downsample)
                    grid_x0 = (x0 // level0_tile_size) * level0_tile_size
                    grid_y0 = (y0 // level0_tile_size) * level0_tile_size
                    if (
                        grid_x0 + level0_tile_size <= slide.dimensions[0]
                        and grid_y0 + level0_tile_size <= slide.dimensions[1]
                    ):
                        aligned_coords.add((grid_x0, grid_y0))
        out = list(aligned_coords)
    return out, time.perf_counter() - t0


def compute_tissue_coords_vectorized(
    slide_path: str,
    tile_size: int,
    target_level: int,
    bg_threshold: int,
    scan_step: int,
) -> Tuple[List[Tuple[int, int]], float]:
    t0 = time.perf_counter()
    with openslide.OpenSlide(slide_path) as slide:
        target_downsample = slide.level_downsamples[target_level]
        level0_tile_size = int(tile_size * target_downsample)
        thumb_level = slide.level_count - 1
        thumb_downsample = slide.level_downsamples[thumb_level]
        w_thumb, h_thumb = slide.level_dimensions[thumb_level]
        thumb_img = slide.read_region((0, 0), thumb_level, (w_thumb, h_thumb)).convert("RGB")
        thumb_gray = np.mean(np.array(thumb_img), axis=2)
        tissue_mask = thumb_gray < bg_threshold
        step = max(1, int(scan_step))
        sampled_mask = tissue_mask[::step, ::step]
        ys, xs = np.nonzero(sampled_mask)
        if ys.size == 0:
            return [], time.perf_counter() - t0
        xs = xs.astype(np.int64) * step
        ys = ys.astype(np.int64) * step
        x0 = (xs * thumb_downsample).astype(np.int64)
        y0 = (ys * thumb_downsample).astype(np.int64)
        grid_x0 = (x0 // level0_tile_size) * level0_tile_size
        grid_y0 = (y0 // level0_tile_size) * level0_tile_size
        valid = (grid_x0 + level0_tile_size <= slide.dimensions[0]) & (
            grid_y0 + level0_tile_size <= slide.dimensions[1]
        )
        if not np.any(valid):
            return [], time.perf_counter() - t0
        coords = np.stack([grid_x0[valid], grid_y0[valid]], axis=1)
        unique_coords = np.unique(coords, axis=0)
        out = [tuple(map(int, p)) for p in unique_coords.tolist()]
    return out, time.perf_counter() - t0


def _scan_one_thumb_strip(
    slide_path: str,
    thumb_level: int,
    y_thumb_off: int,
    h_strip: int,
    w_thumb: int,
    tile_size: int,
    target_level: int,
    bg_threshold: int,
    step: int,
    slide_w0: int,
    slide_h0: int,
) -> np.ndarray:
    with openslide.OpenSlide(slide_path) as slide:
        strip = slide.read_region((0, y_thumb_off), thumb_level, (w_thumb, h_strip)).convert("RGB")
        td = float(slide.level_downsamples[target_level])
        level0_tile_size = int(tile_size * td)
        thumb_d = float(slide.level_downsamples[thumb_level])
        thumb_gray = np.mean(np.array(strip, dtype=np.float32), axis=2)
        tissue_mask = thumb_gray < float(bg_threshold)
        sampled = tissue_mask[::step, ::step]
        ys, xs = np.nonzero(sampled)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        gty = y_thumb_off + ys.astype(np.int64) * step
        gtx = xs.astype(np.int64) * step
        x0 = (gtx * thumb_d).astype(np.int64)
        y0 = (gty * thumb_d).astype(np.int64)
        gx0 = (x0 // level0_tile_size) * level0_tile_size
        gy0 = (y0 // level0_tile_size) * level0_tile_size
        valid = (gx0 + level0_tile_size <= slide_w0) & (gy0 + level0_tile_size <= slide_h0)
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([gx0[valid], gy0[valid]], axis=1)


def compute_tissue_coords_parallel_strips(
    slide_path: str,
    tile_size: int,
    target_level: int,
    bg_threshold: int,
    scan_step: int,
    num_workers: int = 8,
) -> Tuple[List[Tuple[int, int]], float]:
    t0 = time.perf_counter()
    step = max(1, int(scan_step))
    nw = max(1, int(num_workers))

    with openslide.OpenSlide(slide_path) as slide:
        thumb_level = int(slide.level_count - 1)
        w_thumb, h_thumb = slide.level_dimensions[thumb_level]
        slide_w0, slide_h0 = int(slide.dimensions[0]), int(slide.dimensions[1])

    h_thumb = int(h_thumb)
    w_thumb = int(w_thumb)
    n_strips = min(nw, max(1, h_thumb))
    base = h_thumb // n_strips
    rem = h_thumb % n_strips
    y_offs: List[Tuple[int, int]] = []
    y0 = 0
    for i in range(n_strips):
        h_i = base + (1 if i < rem else 0)
        if h_i > 0:
            y_offs.append((y0, h_i))
            y0 += h_i

    ctx = mp.get_context("spawn")
    args_list = [
        (
            slide_path,
            thumb_level,
            ya,
            hb,
            w_thumb,
            tile_size,
            target_level,
            bg_threshold,
            step,
            slide_w0,
            slide_h0,
        )
        for ya, hb in y_offs
    ]
    with ctx.Pool(processes=len(y_offs)) as pool:
        parts = pool.starmap(_scan_one_thumb_strip, args_list)

    nonempty = [p for p in parts if p.shape[0] > 0]
    if not nonempty:
        return [], time.perf_counter() - t0
    coords = np.vstack(nonempty)
    unique_coords = np.unique(coords, axis=0)
    out = [tuple(map(int, p)) for p in unique_coords.tolist()]
    return out, time.perf_counter() - t0


def _read_one_thumb_strip_sampled_mask(
    slide_path: str,
    thumb_level: int,
    y_thumb_off: int,
    h_strip: int,
    w_thumb: int,
    bg_threshold: int,
    step: int,
) -> Tuple[int, np.ndarray]:
    with openslide.OpenSlide(slide_path) as slide:
        strip = slide.read_region((0, y_thumb_off), thumb_level, (w_thumb, h_strip)).convert("L")
        gray = np.array(strip, dtype=np.uint8)
        sampled = gray[::step, ::step]
        mask_u8 = (sampled < np.uint8(bg_threshold)).astype(np.uint8)
        return y_thumb_off, mask_u8


def _read_one_thumb_strip_sampled_mask_star(
    args: Tuple[str, int, int, int, int, int, int],
) -> Tuple[int, np.ndarray]:
    slide_path, thumb_level, y_thumb_off, h_strip, w_thumb, bg_threshold, step = args
    return _read_one_thumb_strip_sampled_mask(
        slide_path=slide_path,
        thumb_level=thumb_level,
        y_thumb_off=y_thumb_off,
        h_strip=h_strip,
        w_thumb=w_thumb,
        bg_threshold=bg_threshold,
        step=step,
    )


def compute_tissue_coords_parallel_strips_gpu(
    slide_path: str,
    tile_size: int,
    target_level: int,
    bg_threshold: int,
    scan_step: int,
    num_workers: int = 16,
    gpu_id: int = 0,
    unique_on_gpu: bool = True,
) -> Tuple[List[Tuple[int, int]], float]:
    if not torch.cuda.is_available():
        raise RuntimeError("compute_tissue_coords_parallel_strips_gpu 需要 CUDA 可用")

    t0 = time.perf_counter()
    step = max(1, int(scan_step))
    nw = max(1, int(num_workers))

    with openslide.OpenSlide(slide_path) as slide:
        thumb_level = int(slide.level_count - 1)
        w_thumb, h_thumb = slide.level_dimensions[thumb_level]
        slide_w0, slide_h0 = int(slide.dimensions[0]), int(slide.dimensions[1])
        td = float(slide.level_downsamples[target_level])
        thumb_d = float(slide.level_downsamples[thumb_level])

    level0_tile_size = int(tile_size * td)
    if level0_tile_size <= 0:
        return [], time.perf_counter() - t0

    max_tx = (slide_w0 - level0_tile_size) // level0_tile_size
    max_ty = (slide_h0 - level0_tile_size) // level0_tile_size
    if max_tx < 0 or max_ty < 0:
        return [], time.perf_counter() - t0

    tile_rows_for_key = int(max_ty + 1)

    h_thumb = int(h_thumb)
    w_thumb = int(w_thumb)
    n_strips = min(nw, max(1, h_thumb))
    base = h_thumb // n_strips
    rem = h_thumb % n_strips
    y_offs: List[Tuple[int, int]] = []
    y0 = 0
    for i in range(n_strips):
        h_i = base + (1 if i < rem else 0)
        if h_i > 0:
            y_offs.append((y0, h_i))
            y0 += h_i

    ctx = mp.get_context("spawn")
    args_list = [
        (
            slide_path,
            thumb_level,
            ya,
            hb,
            w_thumb,
            bg_threshold,
            step,
        )
        for ya, hb in y_offs
    ]

    device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)

    keys_gpu: List[torch.Tensor] = []
    max_tx_t = torch.tensor(max_tx, device=device, dtype=torch.int64)
    max_ty_t = torch.tensor(max_ty, device=device, dtype=torch.int64)
    td_f64 = float(thumb_d)
    level0_tile_size_t = torch.tensor(level0_tile_size, device=device, dtype=torch.int64)
    tile_rows_t = torch.tensor(tile_rows_for_key, device=device, dtype=torch.int64)

    with ctx.Pool(processes=len(y_offs)) as pool:
        for y_thumb_off, mask_u8 in pool.imap_unordered(
            _read_one_thumb_strip_sampled_mask_star, args_list
        ):
            if mask_u8.size == 0:
                continue
            mask_t = torch.from_numpy(mask_u8).to(device=device, non_blocking=True)
            ys, xs = torch.nonzero(mask_t, as_tuple=True)
            if ys.numel() == 0:
                continue

            ys_i = ys.to(torch.int64)
            xs_i = xs.to(torch.int64)
            gty = ys_i * step + int(y_thumb_off)
            gtx = xs_i * step

            x0 = (gtx.to(torch.float64) * td_f64).to(torch.int64)
            y0 = (gty.to(torch.float64) * td_f64).to(torch.int64)

            tx = x0 // level0_tile_size_t
            ty = y0 // level0_tile_size_t

            valid = (tx <= max_tx_t) & (ty <= max_ty_t)
            if not torch.any(valid):
                continue
            tx = tx[valid]
            ty = ty[valid]

            key = tx * tile_rows_t + ty
            keys_gpu.append(key)

    if not keys_gpu:
        return [], time.perf_counter() - t0

    all_keys = torch.cat(keys_gpu, dim=0)
    if unique_on_gpu:
        unique_keys = torch.unique(all_keys)
    else:
        unique_keys = torch.unique(all_keys)

    tx_u = (unique_keys // tile_rows_t).to(torch.int64)
    ty_u = (unique_keys % tile_rows_t).to(torch.int64)

    grid_x0 = (tx_u * level0_tile_size_t).to(torch.int64).detach().cpu().numpy()
    grid_y0 = (ty_u * level0_tile_size_t).to(torch.int64).detach().cpu().numpy()

    out = [(int(x), int(y)) for x, y in zip(grid_x0.tolist(), grid_y0.tolist())]
    return out, time.perf_counter() - t0
