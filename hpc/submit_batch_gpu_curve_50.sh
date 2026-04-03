#!/bin/bash
# 双卡批量流程（embed_dir）+ NVML 监控，用于观察 GPU0/GPU1 利用率曲线。
# 请先运行：bash hpc/build_finaltif_sample_50.sh
# 再：export TIF_STAGING=/public/home/wang/liujx/prov-gigapath-improve/hpc/gpu_monitor_batch50/tif_staging_50
#     sbatch hpc/submit_batch_gpu_curve_50.sh
#
#SBATCH --job-name=improve_batch_gpu_curve
#SBATCH --output=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/batch_gpu_curve_%j.out
#SBATCH --error=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/batch_gpu_curve_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=200:00:00

set -euo pipefail

module load gcc-toolset/12
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

ROOT="${GIGAPATH_IMPROVE_ROOT:-/public/home/wang/liujx/prov-gigapath-improve}"
export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"

export TILE_WEIGHT="${TILE_WEIGHT:-/public/home/wang/liujx/pytorch_model.bin}"
export SLIDE_WEIGHT="${SLIDE_WEIGHT:-/public/home/wang/liujx/slide_encoder.pth}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 默认指向 build_finaltif_sample_50.sh 生成的软链目录
TIF_STAGING="${TIF_STAGING:-${ROOT}/hpc/gpu_monitor_batch50/tif_staging_50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCAN_CPU_WORKERS="${SCAN_CPU_WORKERS:--1}"
SCAN_UNIQUE_ON_GPU="${SCAN_UNIQUE_ON_GPU:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.1}"

OUT_ROOT="${BATCH_OUT_ROOT:-${ROOT}/runs/batch_gpu_curve_50_${SLURM_JOB_ID:-local}}"

mkdir -p "${ROOT}/runs"

echo "ROOT=${ROOT}"
echo "TIF_STAGING=${TIF_STAGING}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "TILE_WEIGHT=${TILE_WEIGHT}"
echo "SLIDE_WEIGHT=${SLIDE_WEIGHT}"
nvidia-smi -L 2>/dev/null || true

_n_entries="$(find "${TIF_STAGING}" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')"
if [[ ! -d "${TIF_STAGING}" ]] || [[ "${_n_entries}" -lt 1 ]]; then
  echo "无效或空的 TIF_STAGING: ${TIF_STAGING}" >&2
  echo "请先执行: bash ${ROOT}/hpc/build_finaltif_sample_50.sh" >&2
  exit 1
fi

python "${ROOT}/main_batch.py" embed_dir \
  --tif_dir "${TIF_STAGING}" \
  --output_root "${OUT_ROOT}" \
  --tile_weight "${TILE_WEIGHT}" \
  --slide_weight "${SLIDE_WEIGHT}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --scan_cpu_workers "${SCAN_CPU_WORKERS}" \
  --scan_unique_on_gpu "${SCAN_UNIQUE_ON_GPU}" \
  --monitor \
  --monitor_interval "${MONITOR_INTERVAL}"

echo "[done] 汇总: ${OUT_ROOT}/summary.json"
echo "[done] 双卡曲线: ${OUT_ROOT}/gpu_curve_dual_dir.png"
