#!/bin/bash
# 集群示例：双卡 L40S / A100 等；请按本机分区名修改 #SBATCH 行
#SBATCH --job-name=outv9_hybrid_gpu_scan
#SBATCH --output=slurm_outv9_hybrid_%j.out
#SBATCH --error=slurm_outv9_hybrid_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=48:00:00

set -euo pipefail

# 仓库根目录（在提交节点上请改为 clone 后的绝对路径）
ROOT="${GIGAPATH_IMPROVE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 若使用 conda：
# module load gcc-toolset/12   # 按需
# source /path/to/miniconda3/bin/activate gigapath

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
MONITOR="${MONITOR:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.1}"

mkdir -p "${ROOT}/runs"

echo "ROOT=${ROOT}"
echo "SLIDE_PATH=${SLIDE_PATH} BATCH_SIZE=${BATCH_SIZE} ..."
nvidia-smi -L 2>/dev/null || true

EXTRA=()
if [[ "${MONITOR}" == "1" ]]; then
  EXTRA+=(--monitor --monitor_interval "${MONITOR_INTERVAL}")
fi

if [[ -z "${SLIDE_PATH}" ]]; then
  echo "请设置环境变量 SLIDE_PATH 指向 .tif 切片" >&2
  exit 1
fi

python "${ROOT}/scripts/hybrid_v9_tile_slide.py" \
  --slide_path "${SLIDE_PATH}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers_per_gpu "${NUM_WORKERS_PER_GPU}" \
  --scan_cpu_workers "${SCAN_CPU_WORKERS}" \
  --scan_gpu_id "${SCAN_GPU_ID}" \
  --scan_unique_on_gpu "${SCAN_UNIQUE_ON_GPU}" \
  "${EXTRA[@]}"

STEM="$(basename "${SLIDE_PATH}" .tif)"
echo "Done. See ${ROOT}/runs/${STEM}/v9_run/"
