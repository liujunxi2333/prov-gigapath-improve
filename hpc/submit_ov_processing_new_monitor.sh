#!/bin/bash
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
