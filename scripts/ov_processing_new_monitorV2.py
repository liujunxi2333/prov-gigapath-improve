#!/usr/bin/env python3
"""
兼容入口：历史 Slurm/脚本仍调用本文件名。

单图 PNG→TIFF + JSON 报告的实现已合并至 `ov_processing_gpu.py`（`main_single_report`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ov_processing_gpu import main_single_report  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_single_report())
