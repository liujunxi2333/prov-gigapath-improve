#!/usr/bin/env python3
"""
兼容入口：历史脚本可能 ``import wsi_embed_benchmark``。
新代码请使用仓库根目录 ``main.py`` 或 ``from wsi_embed import ...``（需将 parallel_improve2 加入 PYTHONPATH）。
"""

from __future__ import annotations

from wsi_embed import *  # noqa: F403
from wsi_embed import __all__ as __wsi_all__

__all__ = list(__wsi_all__)


def main() -> None:
    from wsi_embed.pipeline_single import main_benchmark

    main_benchmark()


if __name__ == "__main__":
    main()
