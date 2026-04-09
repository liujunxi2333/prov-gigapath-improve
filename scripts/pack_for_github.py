#!/usr/bin/env python3
"""
将 prov-gigapath-improveV3 打成适合提交 GitHub 的 zip：
- 排除 .git、runs、slurm .out/.err、__pycache__、大权重等
- 删除 ov_processing_gpu_monitor.py；Slurm 脚本改为直接调用业务 Python
- wsi_embed/monitor.py 替换为无 NVML/无绘图的空实现（接口兼容）
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_buffer_stdio() -> None:
    """避免在非 TTY 下 print 被块缓冲，长时间看不到输出。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _rmtree_robust(path: Path) -> None:
    """
    删除目录树。shutil.rmtree 遇只读文件、部分 NFS 场景可能报 Directory not empty；
    在 Unix 上优先 chmod -R u+rwX 再 rm -rf，与日常清理习惯一致。
    """
    path = Path(path)
    if not path.exists():
        return
    if sys.platform == "win32":
        shutil.rmtree(path, ignore_errors=True)
        return
    subprocess.run(
        ["chmod", "-R", "u+rwX", str(path)],
        check=False,
        capture_output=True,
    )
    r = subprocess.run(["rm", "-rf", str(path)], capture_output=True, text=True)
    if r.returncode != 0 or path.exists():
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(
            f"无法删除暂存目录: {path}" + (f" ({err})" if err else "")
        )

MONITOR_STUB = '''\
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
'''

PACK_README = """# GitHub 精简包说明

本目录由 `scripts/pack_for_github.py` 生成，相对完整仓库已做如下处理：

1. **已排除**：`.git`、`runs/`、Slurm 的 `*.out` / `*.err`、各类 `__pycache__`、`.pt` / `.pth` / `.bin`（保留 `weights/.gitkeep`）、`hpc/gpu_monitor_batch50/`、以及仓库根下常见测试目录 `test_single*` 等运行产物与缓存。
2. **已删除**：`scripts/ov_processing_gpu_monitor.py`（GPU/内存采样包装脚本）。
3. **已修改**：`hpc/submit_ov_new_compare_v1_v2.sh`、`submit_ov_processing_new_monitor.sh`、`submit_benchmark_png_to_tif.sh`、`submit_ov_processing_gpu_monitor.sh` —— 改为直接调用 `python ...`，不再包一层监控进程。
4. **已替换**：`parallel_improve2/wsi_embed/monitor.py` 为**空实现**（`ResourceMonitor` 接口保留，`--monitor` 时不采集、不写曲线）。

**与 OV 转换脚本的关系**：本包**不是**「只有 V2、且全流程 GPU」。仍同时包含 `ov_processing_gpu.py`、`ov_processing_new_monitor.py`（V1）、`ov_processing_new_monitorV2.py`（V2）、`ov_png_folder_to_tif_v2.py` 等。V1/V2/经典管线区别、哪些步骤在 GPU/CPU，见 **`scripts/README_ov_processing_scripts.md`**；仅 V2 批量目录接口见 **`scripts/ov_png_folder_to_tif_v2.md`**。

在完整开发仓库中重新打包请运行：

```bash
python scripts/pack_for_github.py
# 若终端仍长时间无输出，可用无缓冲模式：
# python -u scripts/pack_for_github.py
# 或: PYTHONUNBUFFERED=1 python scripts/pack_for_github.py
```

默认输出 zip 位于本目录的**上一级**（与 `prov-gigapath-improveV3` 同级）。

**说明**：第一步 `rsync` 会扫描并复制整个仓库，目录很大时可能数分钟无新行；脚本已尽量开启行缓冲与 rsync 整体进度（`--info=progress2`）。
"""


def main() -> int:
    import argparse

    _line_buffer_stdio()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="zip 输出路径（默认：仓库上一级 prov-gigapath-improveV3_github_pack.zip）",
    )
    args = ap.parse_args()
    parent = REPO_ROOT.parent
    zip_path = args.output_zip or (parent / "prov-gigapath-improveV3_github_pack.zip")
    stage_name = "prov-gigapath-improveV3"
    stage = parent / f"_{stage_name}_pack_stage" / stage_name

    if stage.parent.exists():
        print("[pack] 清理旧暂存目录...", flush=True)
        _rmtree_robust(stage.parent)

    stage.parent.mkdir(parents=True, exist_ok=True)

    print("[pack] 输出目录:", zip_path, flush=True)
    print("[pack] 暂存目录:", stage, flush=True)

    excludes = [
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.mypy_cache",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=runs",
        "--exclude=hpc/slurm_logs/*.out",
        "--exclude=hpc/slurm_logs/*.err",
        "--exclude=hpc/gpu_monitor_batch50",
        "--exclude=*.pt",
        "--exclude=*.pth",
        "--exclude=*.bin",
        "--exclude=.idea",
        "--exclude=.vscode",
        "--exclude=.cursor",
        "--exclude=*.egg-info",
        "--exclude=.eggs",
        "--exclude=dist",
        "--exclude=build",
        "--exclude=venv",
        "--exclude=.venv",
        # 本地测试输出，体积大且常含只读/异常权限，勿打进 GitHub 包
        "--exclude=test_single",
        "--exclude=test_single_new",
        "--exclude=test_single_new_v2",
    ]
    rsync_tail = excludes + [f"{REPO_ROOT}/", f"{stage}/"]
    # --info=progress2：整体进度；若 rsync 过旧则回退为静默 -a
    cmd_try = ["rsync", "-a", "--info=progress2"] + rsync_tail
    print("[pack] 1/4 rsync 复制仓库（大目录可能需数分钟，请稍候）...", flush=True)
    print("[pack] 执行:", " ".join(cmd_try[:4]), "... ->", stage, flush=True)
    r = subprocess.run(cmd_try)
    if r.returncode != 0:
        print(
            "[pack] 带 progress2 的 rsync 失败，改用 rsync -a ...",
            flush=True,
        )
        subprocess.run(["rsync", "-a"] + rsync_tail, check=True)
    print("[pack] rsync 完成。", flush=True)

    mon_script = stage / "scripts" / "ov_processing_gpu_monitor.py"
    if mon_script.is_file():
        mon_script.unlink()

    (stage / "parallel_improve2" / "wsi_embed" / "monitor.py").write_text(
        MONITOR_STUB, encoding="utf-8"
    )
    (stage / "GITHUB_PACK_README.md").write_text(PACK_README, encoding="utf-8")

    print("[pack] 2/4 写入精简版 Slurm 脚本与 monitor 桩...", flush=True)
    _write_submit_scripts(stage)

    # zip：顶层目录名为 prov-gigapath-improveV3
    if zip_path.exists():
        zip_path.unlink()
    root_for_zip = stage.parent
    print("[pack] 3/4 正在压缩为 zip（文件多时也较慢）...", flush=True)
    file_count = 0
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for folder, _, files in os.walk(root_for_zip / stage_name):
            for fn in files:
                fp = Path(folder) / fn
                arc = fp.relative_to(root_for_zip)
                zf.write(fp, arc.as_posix())
                file_count += 1
                if file_count % 3000 == 0:
                    print(f"[pack]    已加入 {file_count} 个文件...", flush=True)

    print("[pack] 删除暂存目录...", flush=True)
    _rmtree_robust(stage.parent)
    print(f"[pack] 4/4 完成。共 {file_count} 个文件 -> {zip_path}", flush=True)
    return 0


def _write_submit_scripts(stage: Path) -> None:
    hpc = stage / "hpc"
    (hpc / "submit_ov_new_compare_v1_v2.sh").write_text(
        _SUBMIT_V1V2, encoding="utf-8"
    )
    (hpc / "submit_ov_processing_new_monitor.sh").write_text(
        _SUBMIT_NEW_MON, encoding="utf-8"
    )
    (hpc / "submit_benchmark_png_to_tif.sh").write_text(
        _SUBMIT_BENCH, encoding="utf-8"
    )
    (hpc / "submit_ov_processing_gpu_monitor.sh").write_text(
        _SUBMIT_OV_GPU, encoding="utf-8"
    )


_SUBMIT_V1V2 = r'''#!/bin/bash
# 对比 ov_processing_new_monitor.py (V1) 与 ov_processing_new_monitorV2.py (V2)
# 精简包：直接运行，无 gpu_monitor 包装。
#SBATCH --job-name=ov_v1v2_cmp
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/ov_v1v2_cmp_%j.out
#SBATCH --error=hpc/slurm_logs/ov_v1v2_cmp_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_V3_ROOT:-/public/home/wang/liujx/prov-gigapath-improveV3}"
cd "${ROOT}"
mkdir -p "${ROOT}/hpc/slurm_logs"

CONDA_ROOT="${CONDA_ROOT:-/public/home/wang/liujx/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/bin/activate" gigapath
echo "Using Python: $(command -v python)"
python --version
PY_BIN="${PY_BIN:-${CONDA_ROOT}/envs/gigapath/bin/python}"
echo "Using PY_BIN=${PY_BIN}"
echo "Running environment preflight..."
python - <<'PY'
import importlib
mods = ["cv2", "torch", "tifffile", "pyvips", "numpy", "PIL"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("[preflight] Missing/broken modules:")
    for name, err in missing:
        print(f"  - {name}: {err}")
    raise SystemExit(2)
import cv2, torch, tifffile, numpy
print(f"[preflight] cv2={cv2.__version__}")
print(f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[preflight] tifffile={tifffile.__version__}")
print(f"[preflight] numpy={numpy.__version__}")
print("[preflight] all required modules import OK")
PY

export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

INPUT_PNG="${INPUT_PNG:-/public/home/wang/share_group_folder_wang/pathology/ov_images/raw_datasets/ubc_ocean/train_images/66.png}"
GPU_CHUNK="${GPU_CHUNK:-4096}"
JPEG_Q="${JPEG_Q:-85}"

BASE_RUN_DIR="${BASE_RUN_DIR:-${ROOT}/runs/ov_new_compare_${SLURM_JOB_ID:-manual}}"
V1_OUT="${BASE_RUN_DIR}/v1"
V2_OUT="${BASE_RUN_DIR}/v2"
V1_JSON="${V1_OUT}/66_convert_report.json"
V2_JSON="${V2_OUT}/66_convert_report_v2.json"
CMP_JSON="${BASE_RUN_DIR}/v1_v2_compare.json"

mkdir -p "${BASE_RUN_DIR}" "${V1_OUT}" "${V2_OUT}"

echo "ROOT=${ROOT}"
echo "INPUT_PNG=${INPUT_PNG}"
echo "BASE_RUN_DIR=${BASE_RUN_DIR}"
echo "GPU_CHUNK=${GPU_CHUNK} JPEG_Q=${JPEG_Q}"
nvidia-smi -L 2>/dev/null || true

"${PY_BIN}" "${ROOT}/scripts/ov_processing_new_monitor.py" \
  --input "${INPUT_PNG}" \
  --output-dir "${V1_OUT}" \
  --output-json "${V1_JSON}" \
  --gpu-chunk "${GPU_CHUNK}" \
  --jpeg-q "${JPEG_Q}"

"${PY_BIN}" "${ROOT}/scripts/ov_processing_new_monitorV2.py" \
  --input "${INPUT_PNG}" \
  --output-dir "${V2_OUT}" \
  --output-json "${V2_JSON}" \
  --gpu-chunk "${GPU_CHUNK}" \
  --jpeg-q "${JPEG_Q}"

"${PY_BIN}" - <<PY
import json
from pathlib import Path
v1 = Path("${V1_JSON}")
v2 = Path("${V2_JSON}")
out = Path("${CMP_JSON}")
d1 = json.loads(v1.read_text())
d2 = json.loads(v2.read_text())
p1 = float(d1.get("prepare_seconds", 0.0))
p2 = float(d2.get("prepare_seconds", 0.0))
res = {
  "v1_json": str(v1),
  "v2_json": str(v2),
  "v1_prepare_seconds": p1,
  "v2_prepare_seconds": p2,
  "delta_v2_minus_v1_seconds": p2 - p1,
  "faster_prepare": "v2" if p2 < p1 else ("v1" if p1 < p2 else "equal"),
  "v1_total_seconds": float(d1.get("total_seconds", 0.0)),
  "v2_total_seconds": float(d2.get("total_seconds", 0.0)),
  "v1_tif_levels": d1.get("tif_level_info", {}),
  "v2_tif_levels": d2.get("tif_level_info", {}),
}
out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
PY

echo "Done."
echo "V1 JSON: ${V1_JSON}"
echo "V2 JSON: ${V2_JSON}"
echo "COMPARE JSON: ${CMP_JSON}"
'''

_SUBMIT_NEW_MON = r'''#!/bin/bash
# 新版：pyvips 写出路径；精简包无监控包装。
#SBATCH --job-name=ov_new_mon
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/ov_new_monitor_%j.out
#SBATCH --error=hpc/slurm_logs/ov_new_monitor_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_V3_ROOT:-/public/home/wang/liujx/prov-gigapath-improveV3}"
cd "${ROOT}"
mkdir -p "${ROOT}/hpc/slurm_logs"

CONDA_ROOT="${CONDA_ROOT:-/public/home/wang/liujx/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/bin/activate" gigapath
echo "Using Python: $(command -v python)"
python --version

export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

INPUT_PNG="${INPUT_PNG:-/public/home/wang/share_group_folder_wang/pathology/ov_images/raw_datasets/ubc_ocean/train_images/66.png}"
OUT_DIR="${OUT_DIR:-${ROOT}/test_single_new}"
GPU_CHUNK="${GPU_CHUNK:-4096}"
JPEG_Q="${JPEG_Q:-85}"
REPORT_JSON="${REPORT_JSON:-${OUT_DIR}/66_convert_report.json}"

mkdir -p "${OUT_DIR}"

echo "ROOT=${ROOT}"
echo "INPUT_PNG=${INPUT_PNG}"
echo "OUT_DIR=${OUT_DIR}"
echo "GPU_CHUNK=${GPU_CHUNK} JPEG_Q=${JPEG_Q}"
nvidia-smi -L 2>/dev/null || true

python "${ROOT}/scripts/ov_processing_new_monitor.py" \
    --input "${INPUT_PNG}" \
    --output-dir "${OUT_DIR}" \
    --output-json "${REPORT_JSON}" \
    --gpu-chunk "${GPU_CHUNK}" \
    --jpeg-q "${JPEG_Q}"

echo "Done."
echo "TIFF: ${OUT_DIR}/66.tif"
echo "Report JSON: ${REPORT_JSON}"
'''

_SUBMIT_BENCH = r'''#!/bin/bash
# benchmark_png_to_tif_methods.py；精简包无监控包装。
#SBATCH --job-name=bench_png_tif
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/bench_png_tif_%j.out
#SBATCH --error=hpc/slurm_logs/bench_png_tif_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_V3_ROOT:-/public/home/wang/liujx/prov-gigapath-improveV3}"
cd "${ROOT}"
mkdir -p "${ROOT}/hpc/slurm_logs"

export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"

INPUT_PNG="${INPUT_PNG:-/public/home/wang/share_group_folder_wang/pathology/ov_images/raw_datasets/ubc_ocean/train_images/66.png}"
WORK_DIR="${WORK_DIR:-${ROOT}/runs/benchmark_png_to_tif_${SLURM_JOB_ID:-run}}"
GPU_CHUNK="${GPU_CHUNK:-4096}"

CONDA_ROOT="${CONDA_ROOT:-/public/home/wang/liujx/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/bin/activate" gigapath
echo "Using Python: $(command -v python)"
python --version

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

mkdir -p "${WORK_DIR}"

JSON_OUT="${WORK_DIR}/benchmark_result.json"

echo "ROOT=${ROOT}"
echo "INPUT_PNG=${INPUT_PNG}"
echo "WORK_DIR=${WORK_DIR}"
echo "JSON_OUT=${JSON_OUT}"
nvidia-smi -L 2>/dev/null || true

python "${ROOT}/scripts/benchmark_png_to_tif_methods.py" \
    --input "${INPUT_PNG}" \
    --work-dir "${WORK_DIR}" \
    --gpu-chunk "${GPU_CHUNK}" \
    --json-out "${JSON_OUT}"

echo "Done. JSON: ${JSON_OUT}"
'''

_SUBMIT_OV_GPU = r'''#!/bin/bash
# ov_processing_gpu.py；精简包无监控包装。
#SBATCH --job-name=ov_proc_gpu
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/ov_proc_gpu_monitor_%j.out
#SBATCH --error=hpc/slurm_logs/ov_proc_gpu_monitor_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_V3_ROOT:-/public/home/wang/liujx/prov-gigapath-improveV3}"
cd "${ROOT}"
mkdir -p "${ROOT}/hpc/slurm_logs"

export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"

INPUT_PNG="${INPUT_PNG:-/public/home/wang/share_group_folder_wang/pathology/ov_images/raw_datasets/ubc_ocean/train_images/66.png}"
INPUT_DIR="${INPUT_DIR:-}"
OUT_DIR="${OUT_DIR:-${ROOT}/test_single}"
GPU_CHUNK="${GPU_CHUNK:-4096}"

CONDA_ROOT="${CONDA_ROOT:-/public/home/wang/liujx/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/bin/activate" gigapath
echo "Using Python: $(command -v python)"
python --version

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

mkdir -p "${OUT_DIR}"

echo "ROOT=${ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "INPUT_PNG=${INPUT_PNG} INPUT_DIR=${INPUT_DIR} GPU_CHUNK=${GPU_CHUNK}"
nvidia-smi -L 2>/dev/null || true

ARGS=(--output-dir "${OUT_DIR}")
if [[ -n "${INPUT_DIR}" ]]; then
  ARGS+=(--input-dir "${INPUT_DIR}")
else
  ARGS+=(--input "${INPUT_PNG}")
fi

python "${ROOT}/scripts/ov_processing_gpu.py" "${ARGS[@]}" --gpu-chunk "${GPU_CHUNK}" --num-gpus 2

echo "Done. Outputs: ${OUT_DIR}"
'''

if __name__ == "__main__":
    sys.exit(main())
