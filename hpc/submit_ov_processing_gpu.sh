#!/bin/bash
# 参照 prov-gigapath-improveV2/submit_outv9_hybrid.sh 的资源与写法；
# 在 HPC 上跑 ov_processing_gpu（双卡 L40S，多 PNG 时按文件分到两进程）。
#SBATCH --job-name=ov_proc_gpu
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/ov_proc_gpu_%j.out
#SBATCH --error=hpc/slurm_logs/ov_proc_gpu_%j.err
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
# 大图可调小（如 2048）进一步省显存
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

echo "Done. Outputs under ${OUT_DIR}"
