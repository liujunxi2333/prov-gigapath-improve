#!/bin/bash
# 阶段① 扫描瓶颈粗测（profile_scan_phase_66tif.py），单卡。
# 默认切片：11111ovarian finaltif 66.tif；可用 export SLIDE_PATH=... 覆盖。
#
# 提交:
#   sbatch /public/home/wang/liujx/prov-gigapath-improve/hpc/submit_profile_scan_phase.sh
#
#SBATCH --job-name=prof_scan_66tif
#SBATCH --output=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/prof_scan_66tif_%j.out
#SBATCH --error=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/prof_scan_66tif_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_ROOT:-/public/home/wang/liujx/prov-gigapath-improve}"
mkdir -p "${ROOT}/hpc/slurm_logs"

echo "[prof_scan] start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local}"

DEFAULT_SLIDE="/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif/66.tif"
SLIDE_PATH="${SLIDE_PATH:-${DEFAULT_SLIDE}}"

if [[ ! -f "${SLIDE_PATH}" ]]; then
  echo "[error] 切片不存在: ${SLIDE_PATH}" >&2
  exit 1
fi

module load gcc-toolset/12
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

export PYTHONPATH="${ROOT}:${ROOT}/parallel_improve2:${PYTHONPATH:-}"
export SCAN_STEP="${SCAN_STEP:-4}"
export SCAN_CPU_WORKERS="${SCAN_CPU_WORKERS:--1}"
export SCAN_GPU_ID="${SCAN_GPU_ID:-0}"

echo "ROOT=${ROOT}"
echo "SLIDE_PATH=${SLIDE_PATH}"
echo "SCAN_STEP=${SCAN_STEP} SCAN_CPU_WORKERS=${SCAN_CPU_WORKERS} SCAN_GPU_ID=${SCAN_GPU_ID}"
nvidia-smi -L 2>/dev/null || true

python "${ROOT}/scripts/profile_scan_phase_66tif.py" --slide "${SLIDE_PATH}"

echo "[prof_scan] done $(date -Is) job=${SLURM_JOB_ID:-local}"
