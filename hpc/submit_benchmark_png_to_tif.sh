#!/bin/bash
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
