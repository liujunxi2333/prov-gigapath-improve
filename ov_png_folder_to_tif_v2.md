# ov_png_folder_to_tif_v2：文件夹批量 PNG → 金字塔 TIFF（V2）

脚本路径：`scripts/ov_png_folder_to_tif_v2.py`

与仓库内其它 `ov_processing*.py`（V1 / 经典 GPU 分块管线等）的关系见 **`README_ov_processing_scripts.md`**。

卵巢 UBC OCEAN 风格预处理：**仅保留 V2 管线**——分块 GPU 灰度直方图与 Otsu（阈值与二值化与 `ov_processing_gpu` 中 CPU 参考一致）、CPU 上 `findContours`、GPU 分块合成前景、**pyvips** 写出带金字塔的 JPEG 压缩 BigTIFF。

## 依赖

- 同目录下的 `ov_processing_gpu.py`：调用 **`prepare_l0_l1`**、**`write_pyramid_jpeg_tif`**、**`tif_pyramid_level_info`**（单图管线已全部集中在该文件）。
- 运行环境需具备：`torch`、`opencv-python`、`Pillow`、`numpy`、`tifffile`、`pyvips` 等（与仓库其余 OV 脚本一致）。

## 公开 Python API

将 `scripts` 加入 `sys.path` 后导入（或与脚本同目录执行）：

| 函数 | 说明 |
|------|------|
| `list_png_files(input_dir, recursive=False)` | 列出目录下 `.png` / `.PNG`；`recursive=True` 时递归子目录。 |
| `output_tif_path_for_png(input_png, input_root, output_tif_dir, flat=True)` | 由输入 PNG 决定输出 `.tif` 路径。`flat=True` 为 `output_dir/{stem}.tif`（不同子目录同名会覆盖）；`flat=False` 时在输出根下镜像相对 `input_root` 的子路径。 |
| `convert_png_to_tif(input_png, output_tif, **kwargs)` | 单张转换，返回 `ConvertResult`。 |
| `convert_png_folder_to_tif(input_dir, output_tif_dir, **kwargs)` | 批量转换，返回 `List[ConvertResult]`。 |

### `ConvertResult`

字段包括：`ok`、`error`、`input_png`、`output_tif`、`prepare_seconds`、`write_seconds`、`total_seconds`、`tif_level_info`，以及 `extra`（如 `device`、`gpu_prepare_enabled` 等）。调用 `result.to_dict()` 可得到可 JSON 序列化的字典。

### `convert_png_to_tif` 主要参数

- `gpu_chunk`：分块边长（像素），不小于 256，默认 `4096`。
- `jpeg_q`：pyvips JPEG 质量，默认 `85`。
- `device`：默认自动选择 `cuda:0`（若可用）否则 CPU。
- `force_cpu_prepare`：为 `True` 时 Step2/3 走 CPU（调试用或无 CUDA）。

### `convert_png_folder_to_tif` 主要参数

- `recursive`：是否递归子目录中的 PNG。
- `flat_output`：与 CLI 的「扁平 / 镜像目录」对应；`True` 时全部 TIFF 直接落在 `output_tif_dir` 下。
- `skip_existing`：目标 `.tif` 已存在则跳过（`ok=True`，`extra` 中带 `skipped`）。

## 命令行用法

在仓库根目录执行时，请将 `INPUT`、`OUTPUT` 换成实际路径：

```bash
# 仅当前目录一层 PNG → 全部写到 output_tif（扁平：{stem}.tif）
python scripts/ov_png_folder_to_tif_v2.py \
  --input-dir /path/to/png_folder \
  --output-tif-dir /path/to/output_tif
```

```bash
# 递归子目录，并在输出下保留相对子目录结构
python scripts/ov_png_folder_to_tif_v2.py \
  --input-dir /path/to/png_folder \
  --output-tif-dir /path/to/output_tif \
  --recursive --mirror-dirs
```

```bash
# 将批量结果另存为 JSON
python scripts/ov_png_folder_to_tif_v2.py \
  --input-dir /path/to/png_folder \
  --output-tif-dir /path/to/output_tif \
  --summary-json /path/to/batch_summary.json
```

### CLI 选项摘要

| 选项 | 说明 |
|------|------|
| `--input-dir` | 含 PNG 的输入目录（必填） |
| `--output-tif-dir` | 输出 TIFF 根目录（必填） |
| `--recursive` | 递归扫描 PNG |
| `--mirror-dirs` | 输出镜像输入相对路径；默认不开启（扁平输出） |
| `--gpu-chunk` | 分块大小，默认 `4096` |
| `--jpeg-q` | JPEG 质量，默认 `85` |
| `--force-cpu-prepare` | 禁用 GPU 版 Step2/3 |
| `--skip-existing` | 已存在同名输出则跳过 |
| `--summary-json` | 写入汇总 JSON 路径 |

**退出码**：全部成功为 `0`，任一张失败为 `1`。标准输出会打印包含 `items` 的 JSON 摘要。

## Python 调用示例

```python
from pathlib import Path

# 需能导入同目录下的 ov_processing_gpu
from ov_png_folder_to_tif_v2 import convert_png_folder_to_tif, convert_png_to_tif

# 批量
results = convert_png_folder_to_tif(
    Path("/data/pngs"),
    Path("/data/tifs"),
    recursive=False,
    flat_output=True,
)

# 单张
r = convert_png_to_tif(Path("/data/pngs/a.png"), Path("/data/tifs/a.tif"))
assert r.ok
```

## 注意事项

- 扁平输出（默认）时，若递归且不同子目录存在相同文件名，后写会覆盖先写，建议使用 `--mirror-dirs` 或保持非递归单层目录。
- 无有效组织前景轮廓时可能抛出 `no contours from threshold map`，对应条目在批量结果中 `ok=False` 并带 `error` 说明。
