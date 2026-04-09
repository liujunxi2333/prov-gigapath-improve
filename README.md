# GigaPath Improve

本仓库是 https://github.com/prov-gigapath/prov-gigapath 项目的提升版本
## 较原项目对单张切片从图片转换成张量的速度提升了4-5倍
<img width="1600" height="720" alt="speedup_total_bar" src="https://github.com/user-attachments/assets/a6f8020b-f78e-4245-a8f2-62bc8fdcfea6" />

增加了ov_png_folder_to_tif_v2.py脚本，实现了gpu加速对png图片转换成level0+level1层级的tif，具体详见ov_png_folder_to_tif_v2.md
--update 2026.4.9

## 功能概要

### 单张 WSI（`main_single.py` / `run_v9_pipeline`）

1. **upfront 组织坐标扫描（v9）**  
   - CPU：多进程按缩略图 **水平条带**并行 `OpenSlide.read_region`。  
   - GPU：对下采样 mask 做 `nonzero → 映射到 level0 tile 网格 → 去重`（`compute_tissue_coords_parallel_strips_gpu`，见 `parallel_improve2/wsi_embed/coords.py`）。

2. **Tile 编码（坐标二分）**  
   - 整张切片扫描得到 `valid_coords` 后，按 **`mid = n // 2`** 分为前后两半：前半 → `cuda:0`，后半 → `cuda:1`。  
   - 每卡 **独立一份完整 ViT tile 模型**（**不使用** `DataParallel` 的 batch gather）。  
   - 瓦片预处理与 Baseline 一致：`Resize(256) → CenterCrop(224) → ImageNet normalize`（`BaselineWSITileDataset`）。

3. **Slide 编码**  
   - `max_tokens` 子采样后，在 **`cuda:1`** 单次 slide encoder 前向。

4. **可选监控**  
   - `--monitor`：NVML 采样 GPU 利用率与显存（默认在 **扫描结束后** 开始，避免长扫描稀释曲线）。

### 批量目录、双卡均衡队列（`main_batch.py embed_dir`）

- 对目录内 `.tif/.tiff` 仅 **`os.stat` 取文件大小**（不读像素），按 **总字节数贪心** 分成两队，使两队负载尽量接近；**队列 0 → GPU0、队列 1 → GPU1** 并行处理不同切片。  
- **每张切片**在 **单卡**上完成：GPU 条带扫描 → DataLoader（同上预处理）→ tile → slide；输出按 `per_slide/gpu{0,1}/<基名>/` 组织。  
- 汇总与分区信息：`partition.json`、`summary.json`；`--monitor` 时生成 **`gpu_curve_dual_dir.png`** 等。

### 基准对比 / 消融（`main_batch.py benchmark`）

- 原 `wsi_embed_benchmark` 的 baseline vs stream **compare / ablation**（`--slide_list` 文本，每行一张路径）。

## 硬件与显存建议

| 项目 | 建议 |
|------|------|
| **GPU** | **至少 2 张** NVIDIA GPU（CUDA ≥ 11.8 或与 PyTorch 轮子匹配）；单张 v9 脚本中 tile 与 slide 固定使用 `cuda:0` 与 `cuda:1`。 |
| **GPU 显存（单卡）** | 参考实测峰值：tile 阶段约 **~10 GB**（ViT 单卡）；slide 阶段在 **GPU1** 上峰值约 **~30 GB**（与序列长度、batch 有关）。**建议每卡 ≥ 40 GB（如 L40S 48GB、A100 40GB）**；若显存紧张可降低 `--batch_size` 或 `--max_tokens`。 |
| **GPU 显存（扫描）** | 在 `scan_gpu_id`（默认 0）上额外占用少量显存用于 mask/unique；扫描结束后脚本会 `empty_cache()`。 |
| **CPU** | 扫描会启动多进程条带读取；建议 **≥ 16 核**（集群作业示例 `cpus-per-task=32`）。 |
| **内存** | 大 WSI 解码与多 worker 预取：建议 **≥ 64 GB**，集群上 **120 GB** 更稳妥。 |
| **磁盘** | WSI 常为数十 GB；输出 `embedding.pt` 与 `perf.json` 很小。 |

## 软件环境

- **Python 3.10+**（参考环境为 3.10）
- **CUDA** 与 **PyTorch** 版本需匹配（见 `requirements.txt`；若本机 CUDA 不同，请从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 wheel）
- **OpenSlide**：`openslide-python` + 系统级 libopenslide
- **libvips**：`requirements.txt` 中的 **`pyvips`** 用于卵巢 PNG→TIFF 写出（`scripts/ov_processing_gpu.py` 等）；`pip install` 后仍需系统或 conda-forge 提供 **libvips**（见 `requirements.txt` 顶部注释）。
- **可选**：`pynvml` / `nvidia-ml-py`（用于 `--monitor`）

## 安装步骤

```bash
git clone <your-fork-or-url> gigapath-improve
cd gigapath-improve

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 先按本机 CUDA 安装 PyTorch，再：
pip install -r requirements.txt
```

将预训练权重放入 `weights/`（见 `weights/README.txt`），或通过环境变量 / 命令行指定。

本仓库在 **本机 liujx 账号** 上的默认查找路径为（`main_single.py`、`main_batch.py embed_dir`、`scripts/hybrid_v9_tile_slide.py` 及 `hpc/*.sh`、`submit_outv9_hybrid.sh` 已对齐）：

- Tile：`/public/home/wang/liujx/pytorch_model.bin`
- Slide：`/public/home/wang/liujx/slide_encoder.pth`

其他环境请覆盖：

```bash
export TILE_WEIGHT=/path/to/pytorch_model.bin
export SLIDE_WEIGHT=/path/to/slide_encoder.pth
```

## 运行

### 环境变量（建议）

```bash
export GIGAPATH_IMPROVE_ROOT="$(pwd)"
export PYTHONPATH="${GIGAPATH_IMPROVE_ROOT}:${GIGAPATH_IMPROVE_ROOT}/parallel_improve2:${PYTHONPATH}"
```

`main_single.py` / `main_batch.py` 会自动把上述路径加入 `sys.path`，但交互式 `import wsi_embed` 时仍需 `PYTHONPATH` 包含 `parallel_improve2`。

### 单张 WSI（推荐入口）

```bash
python main_single.py \
  --slide_path /path/to/slide.tif \
  --tile_weight weights/pytorch_model.bin \
  --slide_weight weights/slide_encoder.pth \
  --batch_size 128 \
  --num_workers_per_gpu 4 \
  --scan_cpu_workers -1 \
  --scan_gpu_id 0 \
  --monitor
```

或使用兼容路由：

```bash
python main.py single --slide_path /path/to/slide.tif ...
python main.py v9   --slide_path /path/to/slide.tif ...   # 同上
```

等价脚本（仍可用）：

```bash
python scripts/hybrid_v9_tile_slide.py --slide_path /path/to/slide.tif ...
```

默认输出：`<仓库根>/runs/<切片基名>/v9_run/`（`embedding.pt`、`perf.json`、`run_meta.json`；`--monitor` 另有 `gpu_curve.png` 等）。

### 批量：目录双卡均衡 + 监控曲线

适用于「文件夹内多张切片、两卡并行、观察 GPU 占用」：

```bash
python main_batch.py embed_dir \
  --tif_dir /path/to/folder_with_tifs \
  --output_root ./runs/my_batch_run \
  --batch_size 128 \
  --num_workers 4 \
  --monitor \
  --monitor_interval 0.1
```

### 批量：基准 compare / ablation

```bash
python main_batch.py benchmark --mode compare --slide_list /path/slides.txt ...
```

### 兼容入口 `main.py`

- `python main.py tif_dual --tif_dir ...` → 等价于 `main_batch.py embed_dir ...`  
- `python main.py benchmark ...` → 等价于 `main_batch.py benchmark ...`

### 与 `scripts/batch_embed_v9.py` 的区别

`batch_embed_v9.py` 对目录内每张切片 **逐张** 调用 **`run_v9_pipeline`**（单张内部的坐标二分），**不会**按文件大小把不同切片分到 GPU0/1 两队列。若需要「多切片 + 按大小均衡双队列 + 每卡整图流水线」，请使用 **`main_batch.py embed_dir`**。

## SLURM 集群

### 单张（仓库根 `submit_outv9_hybrid.sh`）

```bash
export SLIDE_PATH=/path/to/your.tif
export GIGAPATH_IMPROVE_ROOT=/path/to/gigapath-improve
sbatch submit_outv9_hybrid.sh
```

### 单张 + 指定 conda `gigapath` 环境自检

编辑分区名等后：

```bash
export SLIDE_PATH=/path/to/your.tif
sbatch hpc/submit_verify_single_wsi_gigapath.sh
```

日志：`hpc/slurm_logs/verify_single_<jobid>.out`

### 批量 50 张 + 双卡利用率曲线

在登录节点生成清单与软链目录（默认从 `prov-gigapath-main/.../finaltif` 取排序后前 50 张，可改 `FINALTIF` / `OUT_DIR`）：

```bash
bash hpc/build_finaltif_sample_50.sh
sbatch hpc/submit_batch_gpu_curve_50.sh
```

可选：`export TIF_STAGING=...`、`export BATCH_OUT_ROOT=...`。输出目录内查看 `gpu_curve_dual_dir.png`、`summary.json`。

## 与原仓库的关系

- **`gigapath/`**：来自原工程 `prov-gigapath-main/gigapath`，用于 `import gigapath.slide_encoder`。
- **`parallel_improve2/wsi_embed/`**：由原 `wsi_embed_benchmark.py` **拆分**（坐标扫描、数据集、监控、单张/ v9 / 目录双卡流水线等）；**`wsi_embed_benchmark.py`** 保留为 **`from wsi_embed import *`** 的兼容入口，旧代码 `import wsi_embed_benchmark` 仍可用。

## 上传到 GitHub

1. 不要提交大文件：`weights/*.pt`、`weights/*.pth`、`runs/`、`hpc/gpu_monitor_batch50/` 等已在 `.gitignore`。
2. `requirements.txt` 含完整 pip 冻结；若推送后他人安装失败，请根据其 CUDA 版本单独安装 `torch/torchvision` 后再分段安装依赖。

## 许可证

原 GigaPath 与相关权重版权归各自所有者；本仓库内**自写脚本与说明**可按项目需要单独选择许可证（如 MIT）。若需对外发布，请确认上游模型与代码的许可条款。
