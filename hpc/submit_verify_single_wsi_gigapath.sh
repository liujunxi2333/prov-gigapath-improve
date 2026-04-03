#!/bin/bash
# 仿照 compare_test/outv9_compare/submit_verify_outv9_vs_baseline.sh：
# 在 HPC 上激活 gigapath 环境，跑通 prov-gigapath-improve 单张 WSI（main_single.py，坐标二分 v9）。
#
# 使用前：
#   1) mkdir -p /public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs
#   2) export SLIDE_PATH=/path/to/one.tif
#   3) sbatch hpc/submit_verify_single_wsi_gigapath.sh
#
#SBATCH --job-name=improve_single_verify
#SBATCH --output=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/verify_single_%j.out
#SBATCH --error=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/verify_single_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=8:00:00

set -euo pipefail

module load gcc-toolset/12
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

ROOT="${GIGAPATH_IMPROVE_ROOT:-/public/home/wang/liujx/prov-gigapath-improve}"
export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"

# 默认权重（可用 TILE_WEIGHT / SLIDE_WEIGHT 覆盖；slide 与 tile 为不同文件）
export TILE_WEIGHT="${TILE_WEIGHT:-/public/home/wang/liujx/pytorch_model.bin}"
export SLIDE_WEIGHT="${SLIDE_WEIGHT:-/public/home/wang/liujx/slide_encoder.pth}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

SLIDE_PATH="${SLIDE_PATH:-}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
SCAN_CPU_WORKERS="${SCAN_CPU_WORKERS:--1}"
SCAN_GPU_ID="${SCAN_GPU_ID:-0}"
SCAN_UNIQUE_ON_GPU="${SCAN_UNIQUE_ON_GPU:-1}"
# 默认开启 NVML 曲线 + 阶段时间轴图 gpu_curve_timeline.png；不需要时 export MONITOR=0
MONITOR="${MONITOR:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.1}"

mkdir -p "${ROOT}/runs"

echo "ROOT=${ROOT}"
echo "SLIDE_PATH=${SLIDE_PATH}"
echo "TILE_WEIGHT=${TILE_WEIGHT}"
echo "SLIDE_WEIGHT=${SLIDE_WEIGHT}"
nvidia-smi -L 2>/dev/null || true

EXTRA=()
if [[ "${MONITOR}" == "1" ]]; then
  EXTRA+=(--monitor --monitor_interval "${MONITOR_INTERVAL}")
fi

if [[ -z "${SLIDE_PATH}" ]]; then
  echo "请设置 SLIDE_PATH 指向一张测试用 .tif，例如：" >&2
  echo "  export SLIDE_PATH=/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif/1101.tif" >&2
  exit 1
fi

python "${ROOT}/main_single.py" \
  --slide_path "${SLIDE_PATH}" \
  --tile_weight "${TILE_WEIGHT}" \
  --slide_weight "${SLIDE_WEIGHT}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers_per_gpu "${NUM_WORKERS_PER_GPU}" \
  --scan_cpu_workers "${SCAN_CPU_WORKERS}" \
  --scan_gpu_id "${SCAN_GPU_ID}" \
  --scan_unique_on_gpu "${SCAN_UNIQUE_ON_GPU}" \
  "${EXTRA[@]}"

echo "[done] main_single.py finished."
