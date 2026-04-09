#!/bin/bash
# 从 finaltif 按文件名排序取前 50 张，生成 CSV，并在本目录下建 tif_staging_50/ 软链，
# 供 main_batch.py embed_dir + --monitor 画双卡占用率曲线。
#
# 用法（在登录节点或任意可访问 finaltif 的机器上）：
#   bash hpc/build_finaltif_sample_50.sh
# 或自定义：
#   FINALTIF=/path/to/finaltif OUT_DIR=/path/to/out bash hpc/build_finaltif_sample_50.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINALTIF="${FINALTIF:-/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/gpu_monitor_batch50}"
N="${N:-50}"

if [[ ! -d "${FINALTIF}" ]]; then
  echo "目录不存在: ${FINALTIF}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/tif_staging_50"
rm -f "${OUT_DIR}/tif_staging_50"/*

TMP="$(mktemp)"
# 避免 find|sort|head 在 pipefail 下因 SIGPIPE 退出码 141
mapfile -t _all_tifs < <(find "${FINALTIF}" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | sort)
: > "${TMP}"
i=0
for f in "${_all_tifs[@]}"; do
  [[ $i -ge "${N}" ]] && break
  printf '%s\n' "${f}" >> "${TMP}"
  i=$((i + 1))
done
COUNT="$(wc -l < "${TMP}" | tr -d ' ')"
if [[ "${COUNT}" -lt 1 ]]; then
  echo "未在 ${FINALTIF} 找到 tif/tiff" >&2
  rm -f "${TMP}"
  exit 1
fi

CSV="${OUT_DIR}/slides_${N}.csv"
{
  echo "slide_path"
  while IFS= read -r f; do
    printf '%s\n' "${f}"
  done < "${TMP}"
} > "${CSV}"

while IFS= read -r f; do
  base="$(basename "${f}")"
  ln -sfn "${f}" "${OUT_DIR}/tif_staging_50/${base}"
done < "${TMP}"

rm -f "${TMP}"

echo "Wrote ${CSV} (${COUNT} rows + header)"
echo "Symlinks: ${OUT_DIR}/tif_staging_50/"
echo "批量监控提交示例："
echo "  export TIF_STAGING=${OUT_DIR}/tif_staging_50"
echo "  sbatch ${SCRIPT_DIR}/submit_batch_gpu_curve_50.sh"
