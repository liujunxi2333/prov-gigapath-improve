from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


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

    def elapsed_s(self) -> float:
        if self.t0 <= 0:
            return 0.0
        return time.time() - self.t0

    def plot(
        self,
        path: str,
        title: str,
        *,
        phase_intervals: Optional[Sequence[Tuple[float, float, str]]] = None,
    ):
        if not self.timestamps or not self.gpu_util:
            return
        t = np.array(self.timestamps)
        t_max = float(np.max(t)) if t.size else 0.0

        nrows = 3 if phase_intervals else 2
        fig_h = 9.5 if phase_intervals else 7.0
        fig, axes = plt.subplots(nrows, 1, figsize=(12, fig_h), sharex=True)
        if nrows == 2:
            ax_u, ax_m = axes[0], axes[1]
        else:
            ax_u, ax_m, ax_t = axes[0], axes[1], axes[2]

        for i, arr in enumerate(self.gpu_util):
            if len(arr) == len(t):
                ax_u.plot(t, arr, label=f"GPU{i} util %")
        ax_u.set_ylabel("GPU util %")
        ax_u.legend(loc="upper right")
        ax_u.grid(True, alpha=0.3)

        for i, arr in enumerate(self.gpu_mem_gb):
            if len(arr) == len(t):
                ax_m.plot(t, arr, label=f"GPU{i} mem GB")
        ax_m.set_ylabel("GPU mem GB")
        ax_m.legend(loc="upper right")
        ax_m.grid(True, alpha=0.3)

        if phase_intervals:
            colors = ("#8ecfc9", "#ffbe7a", "#fa7f6f", "#beb8dc", "#82b0d2")
            boundary_times: List[float] = []
            for t0_i, t1_i, _name in phase_intervals:
                boundary_times.append(t0_i)
                boundary_times.append(t1_i)
            for bx in boundary_times:
                if 0 < bx < t_max + 1e-6:
                    ax_u.axvline(bx, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)
                    ax_m.axvline(bx, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)

            ax_t.set_ylim(-0.5, len(phase_intervals) - 0.5)
            ax_t.set_yticks(range(len(phase_intervals)))
            ax_t.set_yticklabels([p[2] for p in phase_intervals])
            ax_t.set_xlabel("时间 (s) — 自 NVML 采样开始")
            for yi, (ts, te, _lbl) in enumerate(phase_intervals):
                c = colors[yi % len(colors)]
                w = max(te - ts, 1e-6)
                ax_t.add_patch(
                    Rectangle(
                        (ts, yi - 0.35),
                        w,
                        0.7,
                        facecolor=c,
                        edgecolor="0.3",
                        linewidth=0.8,
                        alpha=0.92,
                    )
                )
                ax_t.text(
                    ts + w * 0.02,
                    yi,
                    f"{w:.1f}s",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="0.15",
                )
            ax_t.set_xlim(0, max(t_max, max((p[1] for p in phase_intervals), default=0) * 1.02))
            ax_t.grid(True, axis="x", alpha=0.3)
        else:
            ax_m.set_xlabel("time (s)")

        fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_npz(self, path: str) -> None:
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
