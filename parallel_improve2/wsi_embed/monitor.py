"""GitHub 精简包：资源监控为空实现，避免 pynvml/matplotlib 监控依赖；--monitor 时不采集数据。"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.timestamps: List[float] = []
        self.gpu_util: List[List[float]] = []
        self.gpu_mem_gb: List[List[float]] = []
        self.t0 = 0.0

    def start(self) -> None:
        import time

        self.t0 = time.time()

    def stop(self) -> None:
        pass

    def elapsed_s(self) -> float:
        if self.t0 <= 0:
            return 0.0
        import time

        return time.time() - self.t0

    def summary(self) -> Dict[str, Any]:
        return {}

    def plot(
        self,
        path: str,
        title: str,
        *,
        phase_intervals: Optional[Sequence[Tuple[float, float, str]]] = None,
    ) -> None:
        pass

    def save_npz(self, path: str) -> None:
        pass
