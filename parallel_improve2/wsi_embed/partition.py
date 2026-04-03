"""按文件大小将目录中的 WSI 切片均衡分到两个队列（不读取图像内容）。"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

_TIF_EXTS = (".tif", ".tiff", ".TIF", ".TIFF")


def list_tif_paths(directory: str, *, recursive: bool = False) -> List[str]:
    """列出目录下所有 .tif/.tiff 路径，按路径排序以保证确定性。"""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"不是目录: {directory}")
    out: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.endswith(_TIF_EXTS):
                    out.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and name.endswith(_TIF_EXTS):
                out.append(path)
    out.sort()
    return out


def stat_file_sizes(paths: Sequence[str]) -> List[Tuple[str, int]]:
    """仅 ``os.stat`` 取文件字节数，不读图像内容。"""
    return [(p, int(os.stat(p).st_size)) for p in paths]


def partition_two_queues_by_size(paths: Sequence[str]) -> Tuple[List[str], List[str], dict]:
    """
    多路划分贪心：按文件大小降序，每次放入当前总负载较小的一队。
    队内顺序为加入顺序（与贪心处理顺序一致）。
    返回 (queue_gpu0, queue_gpu1, meta)。
    """
    items = stat_file_sizes(paths)
    # 大文件优先，路径作 tie-break
    items.sort(key=lambda x: (-x[1], x[0]))
    q0: List[str] = []
    q1: List[str] = []
    s0 = 0
    s1 = 0
    for p, sz in items:
        if s0 <= s1:
            q0.append(p)
            s0 += sz
        else:
            q1.append(p)
            s1 += sz
    meta = {
        "total_bytes_q0": s0,
        "total_bytes_q1": s1,
        "n_slides_q0": len(q0),
        "n_slides_q1": len(q1),
        "balance_ratio": float(min(s0, s1) / max(max(s0, s1), 1)),
    }
    return q0, q1, meta
