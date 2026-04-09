#!/usr/bin/env python3
"""
兼容入口：请优先使用
  - main_single.py  单张 WSI（坐标二分 v9，与 hybrid 一致）
  - main_batch.py   批量目录 embed_dir + benchmark

本脚本将第一个参数转发到上述二者之一，便于旧命令行不改参数位置。
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    root = os.path.abspath(os.path.dirname(__file__))
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()
        if cmd in ("single", "v9", "one", "1"):
            path = os.path.join(root, "main_single.py")
            sys.argv = [path] + sys.argv[2:]
            runpy.run_path(path, run_name="__main__")
            return
        if cmd in ("batch", "multi", "many"):
            path = os.path.join(root, "main_batch.py")
            sys.argv = [path] + sys.argv[2:]
            runpy.run_path(path, run_name="__main__")
            return
        # 旧版 main.py 子命令兼容
        if cmd == "tif_dual":
            path = os.path.join(root, "main_batch.py")
            sys.argv = [path, "embed_dir"] + sys.argv[2:]
            runpy.run_path(path, run_name="__main__")
            return
        if cmd == "benchmark":
            path = os.path.join(root, "main_batch.py")
            sys.argv = [path, "benchmark"] + sys.argv[2:]
            runpy.run_path(path, run_name="__main__")
            return

    print(
        "用法:\n"
        "  python main_single.py --slide_path ...     # 单张 WSI，坐标二分双卡 tile（v9）\n"
        "  python main_batch.py embed_dir --tif_dir ...\n"
        "  python main_batch.py benchmark --mode compare --slide_list ...\n"
        "\n"
        "兼容:\n"
        "  python main.py single --slide_path ...\n"
        "  python main.py v9 --slide_path ...\n"
        "  python main.py batch embed_dir --tif_dir ...\n"
        "  python main.py tif_dual --tif_dir ...\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
