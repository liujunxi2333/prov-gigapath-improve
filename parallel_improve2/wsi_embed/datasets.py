from __future__ import annotations

from typing import List, Tuple

import numpy as np
import openslide
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms


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
            self._tissue_mask = thumb_gray < self.bg_threshold

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
