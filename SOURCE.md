# 源码同步说明（便于审计）

本目录由以下路径复制/整理而来（日期以打包日为准）：

| 组件 | 源路径 |
|------|--------|
| `gigapath/` | `/public/home/wang/liujx/prov-gigapath-main/gigapath/` |
| `parallel_improve2/wsi_embed_benchmark.py` | `/public/home/wang/liujx/prov-gigapath-main/11111ovarian/parallel_improve2/wsi_embed_benchmark.py` |
| `scripts/hybrid_v9_tile_slide.py` | 基于 `/public/home/wang/liujx/prov-gigapath-main/11111ovarian/compare_test/outv9/hybrid_v9_tile_slide.py` 修改路径与默认参数，便于独立仓库运行 |
| `scripts/batch_embed_v9.py` | 新增：遍历目录内 `.tif`，对每张调用与 `verify_outv9_vs_baseline.py` 中 `_ensure_v9_embedding` 相同的 `run_v9_pipeline`（无 baseline 对比） |

`requirements.txt` 由参考 conda 环境 `gigapath` 的 `pip freeze` 生成，并移除了不可移植的本地 `PySocks @ file://...` 行。
