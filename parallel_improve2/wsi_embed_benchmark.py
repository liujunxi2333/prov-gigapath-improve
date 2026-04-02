#!/usr/bin/env python3
"""
WSI -> 768d 嵌入基准与消融实验（同一权重、同一 seed、同一批切片）。

Baseline：先完成组织坐标枚举（非流式），再 DataLoader 读瓦片 + tile encoder + slide encoder。
  - 与 combine_improve_2.AsyncWSIMemoryDataset 一致：全量坐标后再迭代。
  - 向量化扫描：与 single_slide_monitor_66_v3 一致（np.nonzero + unique）。
  - 慢扫描：缩略图上双重 for（与向量化同 scan_step 网格，便于公平对比扫描实现）。

Stream：StreamingWSIDataset（边扫边产出）；双卡时可选 tile DataParallel（batch 维切到 GPU0+1），slide 仍在另一张卡上单次前向。

注：20251203improve_evaluate.py 从预切 PNG 目录读图，与本脚本 WSI 管线不同；本基准以 combine_improve_2 式 WSI 为准。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
import psutil
import torch
import timm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms

import gigapath.slide_encoder as gigapath_slide_encoder

# ---------------------------------------------------------------------------
# 坐标扫描：慢循环 vs 向量化（同 scan_step 网格）
# ---------------------------------------------------------------------------


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
    """
    子进程：读取缩略图一条水平带，向量化得到 level0 网格角点（与 compute_tissue_coords_vectorized 数学一致）。
    必须顶层函数以便 spawn 下可 pickle。
    """
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
    """
    用多进程 CPU 并行加速 upfront 扫描：将最低分辨率缩略图按 **行方向切成多条带**，
    每进程独立 OpenSlide + read_region + numpy，与 ``compute_tissue_coords_vectorized`` 结果一致（再 ``np.unique``）。

    适用：单块大图 ``read_region``/解码很慢时，磁盘与 libvips/libtiff 可能并行服务多条带。
    ``num_workers=1`` 等价于单条全高条带（略异于原函数内存布局，结果应一致）。
    """
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


# ---------------------------------------------------------------------------
# GPU 加速扫描（v9 实验方向）
# ---------------------------------------------------------------------------


def _read_one_thumb_strip_sampled_mask(
    slide_path: str,
    thumb_level: int,
    y_thumb_off: int,
    h_strip: int,
    w_thumb: int,
    bg_threshold: int,
    step: int,
) -> Tuple[int, np.ndarray]:
    """
    子进程：读取缩略图水平带，转灰度后在 CPU 做“稀疏采样 + 阈值”，
    只返回下采样后的 mask；后续映射/去重在 GPU 上完成。

    必须顶层函数以便 spawn 下可 pickle。
    """
    with openslide.OpenSlide(slide_path) as slide:
        strip = slide.read_region((0, y_thumb_off), thumb_level, (w_thumb, h_strip)).convert("L")
        gray = np.array(strip, dtype=np.uint8)
        sampled = gray[::step, ::step]
        # uint8{0,1}，便于 torch.nonzero
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
    """
    用多进程 CPU 并行读取缩略图条带（upfront 扫描），并在 GPU 上完成：
    mask->nonzero->坐标映射->tile 网格对齐->去重。

    与 compute_tissue_coords_parallel_strips 的数学映射保持一致（使用相同的 floor/int cast 逻辑）。
    """
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

    # 有效 tile 的边界：grid_x0 + level0_tile_size <= slide_w0
    max_tx = (slide_w0 - level0_tile_size) // level0_tile_size
    max_ty = (slide_h0 - level0_tile_size) // level0_tile_size
    if max_tx < 0 or max_ty < 0:
        return [], time.perf_counter() - t0

    tile_rows_for_key = int(max_ty + 1)  # key: tx * tile_rows + ty

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

    # 为减少 GPU 内存峰值：按 strip 逐条处理，key 先在 GPU 上去重/合并。
    keys_gpu: List[torch.Tensor] = []
    max_tx_t = torch.tensor(max_tx, device=device, dtype=torch.int64)
    max_ty_t = torch.tensor(max_ty, device=device, dtype=torch.int64)
    td_f64 = float(thumb_d)  # 用于 x0/y0 的 float->int cast
    level0_tile_size_t = torch.tensor(level0_tile_size, device=device, dtype=torch.int64)
    tile_rows_t = torch.tensor(tile_rows_for_key, device=device, dtype=torch.int64)

    with ctx.Pool(processes=len(y_offs)) as pool:
        # imap_unordered 让 CPU 解码结果更快喂到 GPU（最大化吞吐）。
        for y_thumb_off, mask_u8 in pool.imap_unordered(
            _read_one_thumb_strip_sampled_mask_star, args_list
        ):
            if mask_u8.size == 0:
                continue
            mask_t = torch.from_numpy(mask_u8).to(device=device, non_blocking=True)
            # mask_u8: 0/1
            ys, xs = torch.nonzero(mask_t, as_tuple=True)
            if ys.numel() == 0:
                continue

            ys_i = ys.to(torch.int64)
            xs_i = xs.to(torch.int64)
            gty = ys_i * step + int(y_thumb_off)
            gtx = xs_i * step

            # x0 = (gtx * thumb_d).astype(np.int64)  (正数情况下等价于 floor)
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
        # 保守：仍然用 torch.unique，避免 CPU 去重逻辑变慢（v9 目标是吞吐优先）
        unique_keys = torch.unique(all_keys)

    tx_u = (unique_keys // tile_rows_t).to(torch.int64)
    ty_u = (unique_keys % tile_rows_t).to(torch.int64)

    grid_x0 = (tx_u * level0_tile_size_t).to(torch.int64).detach().cpu().numpy()
    grid_y0 = (ty_u * level0_tile_size_t).to(torch.int64).detach().cpu().numpy()

    out = [(int(x), int(y)) for x, y in zip(grid_x0.tolist(), grid_y0.tolist())]
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Baseline Dataset（全坐标后再 __getitem__）
# ---------------------------------------------------------------------------


class BaselineWSITileDataset(Dataset):
    def __init__(
        self,
        slide_path: str,
        valid_coords: List[Tuple[int, int]],
        tile_size: int,
        target_level: int,
    ):
        self.slide_path = slide_path
        self.valid_coords = valid_coords
        self.tile_size = tile_size
        self.target_level = target_level
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.valid_coords)

    def __getitem__(self, idx: int):
        if not hasattr(self, "slide"):
            self.slide = openslide.OpenSlide(self.slide_path)
        x0, y0 = self.valid_coords[idx]
        tile_img = self.slide.read_region(
            (x0, y0), self.target_level, (self.tile_size, self.tile_size)
        ).convert("RGB")
        tile_tensor = self.transform(tile_img)
        td = self.slide.level_downsamples[self.target_level]
        return tile_tensor, torch.tensor([int(x0 // td), int(y0 // td)], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Streaming（与 v4_stream_exp 一致）
# ---------------------------------------------------------------------------


class StreamingWSIDataset(IterableDataset):
    def __init__(
        self,
        slide_path: str,
        tile_size: int = 256,
        target_level: int = 0,
        bg_threshold: int = 210,
        scan_step: int = 4,
        coord_buffer_size: int = 512,
        sort_buffer_coords: bool = True,
    ):
        super().__init__()
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.target_level = target_level
        self.bg_threshold = bg_threshold
        self.scan_step = max(1, int(scan_step))
        self.coord_buffer_size = max(32, int(coord_buffer_size))
        self.sort_buffer_coords = bool(sort_buffer_coords)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        # Precompute thumbnail mask in main process once. With default Linux/fork
        # workers, this is shared copy-on-write and avoids repeated thumbnail decode.
        self._precompute_thumb_mask()

    def _precompute_thumb_mask(self) -> None:
        with openslide.OpenSlide(self.slide_path) as slide:
            self._target_downsample = float(slide.level_downsamples[self.target_level])
            self._level0_tile_size = int(self.tile_size * self._target_downsample)
            self._thumb_level = int(slide.level_count - 1)
            self._thumb_downsample = float(slide.level_downsamples[self._thumb_level])
            w_thumb, h_thumb = slide.level_dimensions[self._thumb_level]
            self._w_thumb = int(w_thumb)
            self._h_thumb = int(h_thumb)
            thumb_img = slide.read_region((0, 0), self._thumb_level, (w_thumb, h_thumb)).convert("RGB")
            thumb_gray = np.mean(np.array(thumb_img), axis=2)
            self._tissue_mask = (thumb_gray < self.bg_threshold)

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers

        slide = openslide.OpenSlide(self.slide_path)
        try:
            td = float(slide.level_downsamples[self.target_level])
            level0_tile_size = int(self._level0_tile_size)
            thumb_downsample = float(self._thumb_downsample)
            w_thumb = int(self._w_thumb)
            h_thumb = int(self._h_thumb)
            tissue_mask = self._tissue_mask

            y_start = worker_id * self.scan_step
            y_stride = self.scan_step * num_workers
            x_stride = self.scan_step
            local_seen = set()
            coord_buffer: List[Tuple[int, int]] = []

            def flush_buffer():
                if not coord_buffer:
                    return
                if self.sort_buffer_coords:
                    # Improve read locality (y-major) to reduce random I/O.
                    coord_buffer.sort(key=lambda p: (p[1], p[0]))
                for gx0, gy0 in coord_buffer:
                    tile_img = slide.read_region(
                        (gx0, gy0), self.target_level, (self.tile_size, self.tile_size)
                    ).convert("RGB")
                    yield self.transform(tile_img), torch.tensor(
                        [int(gx0 // td), int(gy0 // td)], dtype=torch.float32
                    )
                coord_buffer.clear()

            for y in range(y_start, h_thumb, y_stride):
                row = tissue_mask[y]
                xs = np.nonzero(row[::x_stride])[0].astype(np.int64)
                if xs.size == 0:
                    continue
                xs = xs * x_stride
                x0 = (xs * thumb_downsample).astype(np.int64)
                y0 = np.full_like(x0, int(y * thumb_downsample))
                gx0 = (x0 // level0_tile_size) * level0_tile_size
                gy0 = (y0 // level0_tile_size) * level0_tile_size

                valid = (gx0 + level0_tile_size <= slide.dimensions[0]) & (
                    gy0 + level0_tile_size <= slide.dimensions[1]
                )
                if not np.any(valid):
                    continue
                gx0 = gx0[valid]
                gy0 = gy0[valid]

                # Global ownership partition across workers (deterministic):
                # ensure one tile coordinate is yielded by exactly one worker.
                gx = (gx0 // level0_tile_size).astype(np.int64)
                gy = (gy0 // level0_tile_size).astype(np.int64)
                owners = ((gx * 1315423911 + gy * 2654435761) % num_workers) == worker_id
                if not np.any(owners):
                    continue
                gx0 = gx0[owners]
                gy0 = gy0[owners]

                coords = np.stack([gx0, gy0], axis=1)
                coords = np.unique(coords, axis=0)
                for p in coords.tolist():
                    key = (int(p[0]), int(p[1]))
                    if key in local_seen:
                        continue
                    local_seen.add(key)
                    coord_buffer.append(key)
                    if len(coord_buffer) >= self.coord_buffer_size:
                        yield from flush_buffer()
            yield from flush_buffer()
        finally:
            slide.close()


# ---------------------------------------------------------------------------
# 资源监控
# ---------------------------------------------------------------------------


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.timestamps: List[float] = []
        self.gpu_util: List[List[float]] = []
        self.gpu_mem_gb: List[List[float]] = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self.t0 = 0.0
        self._ok = False
        self._handles = []

    def start(self):
        self.t0 = time.time()
        try:
            from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex

            nvmlInit()
            n = int(nvmlDeviceGetCount())
            self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self.gpu_util = [[] for _ in range(n)]
            self.gpu_mem_gb = [[] for _ in range(n)]
            self._ok = True
        except Exception as e:
            print(f"[monitor] NVML unavailable: {e}")
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join()
        if self._ok:
            from pynvml import nvmlShutdown

            nvmlShutdown()

    def _run(self):
        while not self._stop.is_set():
            self.timestamps.append(time.time() - self.t0)
            if self._ok and self._handles:
                from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

                for i, h in enumerate(self._handles):
                    m = nvmlDeviceGetMemoryInfo(h)
                    u = nvmlDeviceGetUtilizationRates(h)
                    self.gpu_mem_gb[i].append(m.used / (1024**3))
                    self.gpu_util[i].append(float(u.gpu))
            time.sleep(self.interval)

    def summary(self) -> Dict[str, Any]:
        if not self.timestamps:
            return {}
        out: Dict[str, Any] = {}
        for i in range(len(self.gpu_util)):
            u = self.gpu_util[i]
            if u:
                out[f"gpu{i}_util_mean"] = float(np.mean(u))
                out[f"gpu{i}_util_max"] = float(np.max(u))
            g = self.gpu_mem_gb[i] if i < len(self.gpu_mem_gb) else []
            if g:
                out[f"gpu{i}_mem_peak_gb"] = float(np.max(g))
        return out

    def plot(self, path: str, title: str):
        if not self.timestamps or not self.gpu_util:
            return
        t = np.array(self.timestamps)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for i, arr in enumerate(self.gpu_util):
            if len(arr) == len(t):
                axes[0].plot(t, arr, label=f"GPU{i} util %")
        axes[0].set_ylabel("GPU util %")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        for i, arr in enumerate(self.gpu_mem_gb):
            if len(arr) == len(t):
                axes[1].plot(t, arr, label=f"GPU{i} mem GB")
        axes[1].set_ylabel("GPU mem GB")
        axes[1].set_xlabel("time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_npz(self, path: str) -> None:
        """保存 GPU 利用率/显存时间序列，便于离线叠加对比图。"""
        if not self.timestamps:
            return
        d: Dict[str, Any] = {"t": np.array(self.timestamps, dtype=np.float64)}
        for i, arr in enumerate(self.gpu_util):
            if arr:
                d[f"gpu{i}_util"] = np.array(arr, dtype=np.float64)
        for i, arr in enumerate(self.gpu_mem_gb):
            if arr:
                d[f"gpu{i}_mem_gb"] = np.array(arr, dtype=np.float64)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **d)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_tf32(enable: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable
    torch.backends.cudnn.benchmark = enable
    if enable:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")


def build_encoders(
    tile_weight: str,
    slide_weight: str,
    device: torch.device,
    split_models_across_two_gpus: bool = True,
    *,
    tile_parallel: str = "single",
):
    """
    tile_parallel:
      - "single": 整块 batch 在单卡上做 tile forward（split 模式下默认 cuda:0）。
      - "dataparallel": 双卡 DataParallel，按 batch 维切分到 GPU0+GPU1（tile 阶段两卡同时算）。
    说明：slide encoder 仍是单次长序列前向，标准实现无法在「数学上」与 tile 并行；
    本方案让 **tile 阶段**尽可能吃满两卡；slide 放在另一张卡上，减少与 tile 权重争用显存。
    """
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    use_split = bool(split_models_across_two_gpus and device.type == "cuda" and n_gpu >= 2)
    tile_parallel = (tile_parallel or "single").strip().lower()
    if tile_parallel not in ("single", "dataparallel"):
        raise ValueError('tile_parallel must be "single" or "dataparallel"')

    tile_device = torch.device("cuda:0") if use_split else device
    slide_device = torch.device("cuda:1") if use_split else device

    tile_enc = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=True,
        img_size=224,
        in_chans=3,
        pretrained_cfg_overlay=dict(file=tile_weight),
    ).to(tile_device)
    tile_enc.eval()

    dp_tile = False
    if use_split and tile_parallel == "dataparallel" and n_gpu >= 2:
        tile_enc = torch.nn.DataParallel(tile_enc, device_ids=[0, 1])
        dp_tile = True
    elif (not use_split) and device.type == "cuda" and n_gpu >= 2:
        tile_enc = torch.nn.DataParallel(tile_enc, device_ids=[0, 1])
        dp_tile = True

    slide_enc = gigapath_slide_encoder.create_model(
        pretrained=slide_weight,
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
    ).to(slide_device)
    slide_enc.eval()
    return tile_enc, slide_enc, tile_device, slide_device, use_split, dp_tile, tile_parallel


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


def main():
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
        # Baseline：向量化扫描 + DataLoader 调参 + TF32（与当前生产默认接近）；Stream：同 seed/权重/token
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
        # 线性累加：minimal -> +vec -> +dl -> +tf32 -> +stream（每张切片一行组结果写入 ablation_runs.csv）
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


if __name__ == "__main__":
    main()
