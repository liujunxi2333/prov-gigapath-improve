# 卵巢 PNG→TIFF 相关脚本说明（V1 / V2 / 精简包）

## 精简 GitHub 包是不是「全是 GPU 上的 V2」？

**不是。** `pack_for_github.py` 打出来的 zip 是**整仓库的代码子集**（去掉日志、部分运行产物、监控包装等），**仍会包含多条入口**：

- **`ov_processing_gpu.py`（主实现）**：有 CUDA 时 **Step2/3** 在 **GPU** 上分块直方图与二值图；**Otsu** 与 **findContours** 在 **CPU**；**合成**在 GPU/CPU tensor；**写出**为 **pyvips** 金字塔 JPEG BigTIFF。导出 **`prepare_l0_l1`**、**`write_pyramid_jpeg_tif`**、**`tif_pyramid_level_info`** 及历史兼容函数 **`_accumulate_gray_histogram`** 等。CLI **`main`**：多文件/`--input-dir` 批处理；**`main_single_report`**：单图 + JSON。
- **`ov_processing_new_monitor.py`（常称 V1）**：前处理走 **CPU**（`ov_processing_gpu` 中的 `_accumulate_gray_histogram` 等），写出 **pyvips**（与主实现的 GPU 前处理路径不同，便于对比）。
- **`ov_processing_new_monitorV2.py`**：薄封装，**直接调用** `ov_processing_gpu.main_single_report`（保留文件名供 Slurm 兼容）。
- **`ov_png_folder_to_tif_v2.py`**：目录批量，内部调用 **`ov.prepare_l0_l1`** + **`ov.write_pyramid_jpeg_tif`**。

精简包内**已删除**的是 **`ov_processing_gpu_monitor.py`**（监控包装），**不是**业务逻辑。

更细的「打包做了什么」见仓库根目录（解压后）的 **`GITHUB_PACK_README.md`**。

---

## 各脚本分工一览

| 脚本 | 典型用途 | 前处理（直方图/Otsu/二值图） | 轮廓 | 合成 | 写 TIFF |
|------|----------|------------------------------|------|------|---------|
| `ov_processing_gpu.py` | **主**：`main` 批处理 / `main_single_report` 单图 JSON | **GPU** 分块（有 CUDA）否则 CPU | **CPU** | **GPU/CPU** 分块 | **pyvips** |
| `ov_processing_new_monitor.py`（V1） | 单图 + JSON（CPU 前处理对照） | **CPU** 分块 | **CPU** | **GPU** | **pyvips** |
| `ov_processing_new_monitorV2.py` | 兼容入口 → `main_single_report` | 同左栏 `ov_processing_gpu` | 同左 | 同左 | **pyvips** |
| `ov_png_folder_to_tif_v2.py` | **整文件夹 PNG → 输出目录 TIFF** | 调用 `prepare_l0_l1` | 同左 | 同左 | **pyvips** |

**说明**：没有任何一条管线是「100% 在 GPU 上跑完」——**OpenCV `findContours`** 与 **Otsu 阈值计算（V2 为在 CPU 上对整图直方图调用与 V1 相同的函数）** 仍在 CPU；V2 只是把最重的**逐块灰度直方图与逐块二值图**放到 GPU。

---

## 只想用「V2 + 批量目录」时看哪份 md？

请直接看：**`scripts/ov_png_folder_to_tif_v2.md`**（对应 `ov_png_folder_to_tif_v2.py` 的命令行与 Python API）。

---

## Slurm 与对比作业

- V1 vs V2 耗时对比：`hpc/submit_ov_new_compare_v1_v2.sh`（精简包内已改为**不**经过 `ov_processing_gpu_monitor.py`）。
- 仅跑 V1 单图：`hpc/submit_ov_processing_new_monitor.sh`。
- 经典 GPU 分块管线：`hpc/submit_ov_processing_gpu_monitor.sh`（文件名仍带 monitor 历史；精简包内为**直接**调用 `ov_processing_gpu.py`）。
