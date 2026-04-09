#!/usr/bin/env python3
"""
卵巢切片风格：文件夹内 PNG → 金字塔 JPEG-TIFF。

- 单图处理逻辑在 `ov_processing_gpu.py`（`prepare_l0_l1` + `write_pyramid_jpeg_tif`）；本文件仅提供目录批量与路径规划。
- 提供可导入的函数接口与命令行入口。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ov_processing_gpu as ov  # noqa: E402


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def list_png_files(input_dir: Path, *, recursive: bool = False) -> List[Path]:
    """
    列出目录下的 .png / .PNG 文件。
    - recursive=False：仅当前目录一层。
    - recursive=True：递归子目录，返回路径按字符串排序。
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if recursive:
        found = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    else:
        found = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(found)


def output_tif_path_for_png(
    input_png: Path, input_root: Path, output_tif_dir: Path, *, flat: bool
) -> Path:
    """
    由输入 PNG 决定输出 .tif 路径。
    - flat=True：output_tif_dir / {stem}.tif（重名会覆盖）。
    - flat=False：保持相对 input_root 的子目录结构。
    """
    output_tif_dir = output_tif_dir.resolve()
    if flat:
        return output_tif_dir / f"{input_png.stem}.tif"
    rel = input_png.resolve().relative_to(input_root.resolve())
    return (output_tif_dir / rel).with_suffix(".tif")


@dataclass
class ConvertResult:
    """单张 PNG 转换结果。"""

    input_png: str
    output_tif: str
    ok: bool
    error: Optional[str] = None
    prepare_seconds: float = 0.0
    write_seconds: float = 0.0
    total_seconds: float = 0.0
    tif_level_info: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "input_png": self.input_png,
            "output_tif": self.output_tif,
            "ok": self.ok,
            "prepare_seconds": self.prepare_seconds,
            "write_seconds": self.write_seconds,
            "total_seconds": self.total_seconds,
            "tif_level_info": self.tif_level_info,
        }
        if self.error:
            d["error"] = self.error
        d.update(self.extra)
        return d


def convert_png_to_tif(
    input_png: Path,
    output_tif: Path,
    *,
    gpu_chunk: int = 4096,
    jpeg_q: int = 85,
    device: Optional[torch.device] = None,
    force_cpu_prepare: bool = False,
) -> ConvertResult:
    """
    将单张 PNG 转为金字塔 TIFF（V2 管线）。

    Parameters
    ----------
    input_png : 输入 PNG 路径。
    output_tif : 输出 .tif 路径（父目录会自动创建）。
    gpu_chunk : 分块边长（像素），不小于 256。
    jpeg_q : pyvips JPEG 质量。
    device : 若为 None，则 cuda 可用时用 cuda:0，否则 CPU。
    force_cpu_prepare : True 时 Step2/3 用 CPU（无 CUDA 或调试时用）。

    Returns
    -------
    ConvertResult
    """
    input_png = input_png.resolve()
    output_tif = output_tif.resolve()
    if not input_png.is_file():
        return ConvertResult(
            str(input_png),
            str(output_tif),
            ok=False,
            error=f"not a file: {input_png}",
        )

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpu_prepare = device.type == "cuda" and not force_cpu_prepare

    t0 = time.perf_counter()
    try:
        t_prep0 = time.perf_counter()
        l0, _ = ov.prepare_l0_l1(
            input_png, gpu_chunk, device, force_cpu_prepare=not use_gpu_prepare
        )
        prep_s = time.perf_counter() - t_prep0

        t_w0 = time.perf_counter()
        ov.write_pyramid_jpeg_tif(l0, output_tif, jpeg_q)
        write_s = time.perf_counter() - t_w0
        total_s = time.perf_counter() - t0

        lvl = ov.tif_pyramid_level_info(output_tif)
        ok = bool(lvl.get("ok_two_levels"))
        err = None if ok else "tif missing expected pyramid levels"
        return ConvertResult(
            str(input_png),
            str(output_tif),
            ok=ok,
            error=err,
            prepare_seconds=prep_s,
            write_seconds=write_s,
            total_seconds=total_s,
            tif_level_info=lvl,
            extra={
                "device": str(device),
                "gpu_prepare_enabled": use_gpu_prepare,
                "gpu_chunk": int(gpu_chunk),
                "jpeg_q": int(jpeg_q),
            },
        )
    except Exception as e:
        return ConvertResult(
            str(input_png),
            str(output_tif),
            ok=False,
            error=str(e),
            total_seconds=time.perf_counter() - t0,
            extra={"device": str(device), "gpu_prepare_enabled": use_gpu_prepare},
        )


def convert_png_folder_to_tif(
    input_dir: Path,
    output_tif_dir: Path,
    *,
    recursive: bool = False,
    flat_output: bool = True,
    gpu_chunk: int = 4096,
    jpeg_q: int = 85,
    device: Optional[torch.device] = None,
    force_cpu_prepare: bool = False,
    skip_existing: bool = False,
) -> List[ConvertResult]:
    """
    将 ``input_dir`` 下所有 PNG 转为 TIFF，写入 ``output_tif_dir``。

    Parameters
    ----------
    input_dir : 含 PNG 的目录。
    output_tif_dir : 输出根目录（不存在则创建）。
    recursive : 是否递归子目录。
    flat_output : True 时所有 .tif 直接放在 output_tif_dir 下（仅 stem，可能重名覆盖）；
                  False 时在 output_tif_dir 下镜像相对 input_dir 的子路径。
    skip_existing : True 时若目标 .tif 已存在则跳过（ok=True，error=None，耗时记 0）。

    Returns
    -------
    每张输入对应一个 ConvertResult，顺序与 list_png_files 一致。
    """
    input_dir = input_dir.resolve()
    output_tif_dir = output_tif_dir.resolve()
    output_tif_dir.mkdir(parents=True, exist_ok=True)

    pngs = list_png_files(input_dir, recursive=recursive)
    results: List[ConvertResult] = []
    for png in pngs:
        out_path = output_tif_path_for_png(png, input_dir, output_tif_dir, flat=flat_output)
        if skip_existing and out_path.is_file():
            results.append(
                ConvertResult(
                    str(png),
                    str(out_path),
                    ok=True,
                    extra={"skipped": True, "reason": "output exists"},
                )
            )
            continue
        results.append(
            convert_png_to_tif(
                png,
                out_path,
                gpu_chunk=gpu_chunk,
                jpeg_q=jpeg_q,
                device=device,
                force_cpu_prepare=force_cpu_prepare,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="将文件夹内 PNG 批量转为金字塔 TIFF（V2 GPU 前处理 + pyvips）。"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="含 PNG 的输入目录",
    )
    p.add_argument(
        "--output-tif-dir",
        type=Path,
        required=True,
        help="输出 TIFF 根目录",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录中的 PNG（输出可用 --mirror-dirs 保留结构）",
    )
    p.add_argument(
        "--mirror-dirs",
        action="store_true",
        help="在输出目录下镜像输入相对路径；默认扁平输出为 output_dir/{stem}.tif",
    )
    p.add_argument("--gpu-chunk", type=int, default=4096)
    p.add_argument("--jpeg-q", type=int, default=85)
    p.add_argument("--force-cpu-prepare", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="目标 .tif 已存在则跳过")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="将批量结果写入 JSON（每张 ok/error/耗时）",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    flat = not args.mirror_dirs
    results = convert_png_folder_to_tif(
        args.input_dir,
        args.output_tif_dir,
        recursive=args.recursive,
        flat_output=flat,
        gpu_chunk=args.gpu_chunk,
        jpeg_q=args.jpeg_q,
        force_cpu_prepare=args.force_cpu_prepare,
        skip_existing=args.skip_existing,
    )
    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_tif_dir": str(args.output_tif_dir.resolve()),
        "count": len(results),
        "ok": n_ok,
        "failed": n_fail,
        "items": [r.to_dict() for r in results],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
