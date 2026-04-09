#!/bin/bash
# TCGA-GBM SVS 批量 embed_dir（双卡）：每张切片得到 GigaPath slide 768 维向量，
# 保存为 output_root/<切片基名>/embedding.pt，另含 perf.json。
#
# 提交:
#   sbatch /public/home/wang/liujx/prov-gigapath-improve/hpc/submit_tcga_gbm_svs_embed.sh
#
# 若 svs 在子目录中，可设置环境变量后提交:
#   export TCGA_GBM_RECURSIVE=1
#   sbatch ...
#
#SBATCH --job-name=tcga_gbm_svs
#SBATCH --output=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/tcga_gbm_svs_%j.out
#SBATCH --error=/public/home/wang/liujx/prov-gigapath-improve/hpc/slurm_logs/tcga_gbm_svs_%j.err
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

TIF_DIR="${TCGA_GBM_SVS_DIR:-/public/home/wang/share_group_folder_wang/pathology/病理数据1/低级别胶质瘤/svs}"
OUT_ROOT="${TCGA_GBM_OUT_ROOT:-/public/home/wang/liujx/prov-gigapath-improve/output2}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCAN_CPU_WORKERS="${SCAN_CPU_WORKERS:--1}"
SCAN_UNIQUE_ON_GPU="${SCAN_UNIQUE_ON_GPU:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.1}"

RECURSIVE_ARGS=()
if [[ "${TCGA_GBM_RECURSIVE:-0}" == "1" ]]; then
  RECURSIVE_ARGS=(--recursive)
fi

mkdir -p "${OUT_ROOT}" "${ROOT}/hpc/slurm_logs"

echo "ROOT=${ROOT}"
echo "TIF_DIR=${TIF_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
nvidia-smi -L 2>/dev/null || true

if [[ ! -d "${TIF_DIR}" ]]; then
  echo "目录不存在: ${TIF_DIR}" >&2
  exit 1
fi

python "${ROOT}/main_batch.py" embed_dir \
  --tif_dir "${TIF_DIR}" \
  --output_root "${OUT_ROOT}" \
  --tile_weight "${TILE_WEIGHT}" \
  --slide_weight "${SLIDE_WEIGHT}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --scan_cpu_workers "${SCAN_CPU_WORKERS}" \
  --scan_unique_on_gpu "${SCAN_UNIQUE_ON_GPU}" \
  --flat_output \
  --monitor \
  --monitor_interval "${MONITOR_INTERVAL}" \
  "${RECURSIVE_ARGS[@]}"

echo "[done] summary: ${OUT_ROOT}/summary.json"
echo "[done] 每张 768 维: ${OUT_ROOT}/<切片名>/embedding.pt（torch.load 后为 shape [768]）"
