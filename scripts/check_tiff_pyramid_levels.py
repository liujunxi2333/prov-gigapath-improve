#!/usr/bin/env python3
"""
检测多分辨率 TIFF 是否包含 level0（全分辨率）与 level1（降采样层，通常为 SubIFD）。

使用 tifffile 读取；若 `series[0].levels` 存在则直接计数，否则根据页 / SubIFD 推断。

退出码：同时存在至少两层 → 0；否则 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import tifffile


def analyze(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    out: Dict[str, Any] = {"path": str(path), "ok_two_levels": False, "levels": []}

    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            out["error"] = "无 TIFF series"
            return out

        s0 = tif.series[0]
        out["series0_shape"] = list(s0.shape) if hasattr(s0, "shape") else None
        out["series0_dtype"] = str(s0.dtype) if hasattr(s0, "dtype") else None

        levels = getattr(s0, "levels", None)
        if levels:
            for i, lev in enumerate(levels):
                entry: Dict[str, Any] = {"index": i, "name": f"level{i}"}
                if hasattr(lev, "shape"):
                    entry["shape"] = list(lev.shape)
                if hasattr(lev, "dtype"):
                    entry["dtype"] = str(lev.dtype)
                if hasattr(lev, "is_pyramid"):
                    entry["is_pyramid"] = bool(lev.is_pyramid)
                out["levels"].append(entry)
            out["ok_two_levels"] = len(levels) >= 2
            return out

        # 无 .levels 时：单页 + SubIFD 常见于金字塔
        pages = list(tif.pages)
        if not pages:
            out["error"] = "无 pages"
            return out

        p0 = pages[0]
        n_sub = int(getattr(p0, "subifds", 0) or 0)
        entry0 = {"index": 0, "name": "level0", "shape": list(p0.shape)}
        out["levels"].append(entry0)
        if n_sub > 0 and hasattr(p0, "pages"):
            try:
                for j in range(min(n_sub, len(p0.pages))):
                    pj = p0.pages[j]
                    out["levels"].append(
                        {"index": j + 1, "name": f"level{j + 1}", "shape": list(pj.shape)}
                    )
            except Exception as e:
                out["subifd_note"] = str(e)
        out["ok_two_levels"] = len(out["levels"]) >= 2
        if not out["ok_two_levels"] and n_sub > 0:
            out["ok_two_levels"] = True
            out["note"] = "根据首页 subifds>0 推断存在降采样层（未展开 shape）"

    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_tif = repo / "test_single" / "66.tif"

    p = argparse.ArgumentParser(description="检测 TIFF 是否含 level0 + level1（金字塔）")
    p.add_argument(
        "tif_path",
        type=Path,
        nargs="?",
        default=default_tif,
        help=f"TIFF 路径（默认 {default_tif}）",
    )
    p.add_argument("--json", action="store_true", help="只输出 JSON")
    args = p.parse_args()

    tif_path = args.tif_path
    if not tif_path.is_file():
        msg = f"文件不存在: {tif_path}"
        if args.json:
            print(json.dumps({"error": msg}, ensure_ascii=False))
        else:
            print(msg, file=sys.stderr)
        return 1

    info = analyze(tif_path)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"文件: {info['path']}")
        if info.get("series0_shape") is not None:
            print(f"series[0] shape: {info['series0_shape']} dtype={info.get('series0_dtype')}")
        for lev in info.get("levels", []):
            print(f"  {lev.get('name', 'level')}: shape={lev.get('shape')}")
        if info.get("note"):
            print(f"说明: {info['note']}")
        if info.get("error"):
            print(f"错误: {info['error']}")
        print(
            "结论: "
            + ("已检测到至少两层（level0 + level1）" if info.get("ok_two_levels") else "未确认两层")
        )

    return 0 if info.get("ok_two_levels") else 1


if __name__ == "__main__":
    raise SystemExit(main())
