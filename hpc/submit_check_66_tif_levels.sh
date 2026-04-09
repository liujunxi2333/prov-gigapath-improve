#!/bin/bash
# 检测 test_single/66.tif 是否包含金字塔 level0 + level1（仅读元数据/结构，较快）。
#SBATCH --job-name=check_66_tif
#SBATCH --chdir=/public/home/wang/liujx/prov-gigapath-improveV3
#SBATCH --output=hpc/slurm_logs/check_66_tif_%j.out
#SBATCH --error=hpc/slurm_logs/check_66_tif_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

ROOT="${GIGAPATH_IMPROVE_V3_ROOT:-/public/home/wang/liujx/prov-gigapath-improveV3}"
cd "${ROOT}"
mkdir -p "${ROOT}/hpc/slurm_logs"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TIF_PATH="${TIF_PATH:-${ROOT}/test_single/66.tif}"

CONDA_ROOT="${CONDA_ROOT:-/public/home/wang/liujx/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/bin/activate" gigapath
echo "Using Python: $(command -v python)"
python --version

echo "ROOT=${ROOT}"
echo "TIF_PATH=${TIF_PATH}"
ls -la "${TIF_PATH}" 2>/dev/null || echo "警告: 文件不存在或无权限"

python "${ROOT}/scripts/check_tiff_pyramid_levels.py" "${TIF_PATH}" --json
rc=$?
echo "exit_code=${rc}"
exit "${rc}"
