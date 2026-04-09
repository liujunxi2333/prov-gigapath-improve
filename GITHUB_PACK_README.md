# GitHub 精简包说明

本目录由 `scripts/pack_for_github.py` 生成，相对完整仓库已做如下处理：

1. **已排除**：`.git`、`runs/`、Slurm 的 `*.out` / `*.err`、各类 `__pycache__`、`.pt` / `.pth` / `.bin`（保留 `weights/.gitkeep`）、`hpc/gpu_monitor_batch50/`、以及仓库根下常见测试目录 `test_single*` 等运行产物与缓存。
2. **已删除**：`scripts/ov_processing_gpu_monitor.py`（GPU/内存采样包装脚本）。
3. **已修改**：`hpc/submit_ov_new_compare_v1_v2.sh`、`submit_ov_processing_new_monitor.sh`、`submit_benchmark_png_to_tif.sh`、`submit_ov_processing_gpu_monitor.sh` —— 改为直接调用 `python ...`，不再包一层监控进程。
4. **已替换**：`parallel_improve2/wsi_embed/monitor.py` 为**空实现**（`ResourceMonitor` 接口保留，`--monitor` 时不采集、不写曲线）。

**与 OV 转换脚本的关系**：本包**不是**「只有 V2、且全流程 GPU」。仍同时包含 `ov_processing_gpu.py`、`ov_processing_new_monitor.py`（V1）、`ov_processing_new_monitorV2.py`（V2）、`ov_png_folder_to_tif_v2.py` 等。V1/V2/经典管线区别、哪些步骤在 GPU/CPU，见 **`scripts/README_ov_processing_scripts.md`**；仅 V2 批量目录接口见 **`scripts/ov_png_folder_to_tif_v2.md`**。

在完整开发仓库中重新打包请运行：

```bash
python scripts/pack_for_github.py
# 若终端仍长时间无输出，可用无缓冲模式：
# python -u scripts/pack_for_github.py
# 或: PYTHONUNBUFFERED=1 python scripts/pack_for_github.py
```

默认输出 zip 位于本目录的**上一级**（与 `prov-gigapath-improveV3` 同级）。

**说明**：第一步 `rsync` 会扫描并复制整个仓库，目录很大时可能数分钟无新行；脚本已尽量开启行缓冲与 rsync 整体进度（`--info=progress2`）。
