#!/bin/bash
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
