# GigaPath Improve — WSI v9 Hybrid（GPU 加速扫描 + 双卡 Tile + Slide）

本仓库是基于原项目prov-gigapath的升级版，旨在更快的处理大量的病理切片。
本项目相较于原项目baseline，对单张切片从tif处理成768维向量速度快了78%左右
<img width="1600" height="720" alt="speedup_total_bar" src="https://github.com/user-attachments/assets/3b6ea3b6-9729-4a55-b2e7-654799a8289e" />
如图为随机抽取五张切片的提升幅度，均有4-5倍左右的性能提升

## 功能概要

1. **upfront 组织坐标扫描（v9）**  
   - CPU：多进程按缩略图 **水平条带**并行 `OpenSlide.read_region`。  
   - GPU：对下采样 mask 做 `nonzero → 映射到 level0 tile 网格 → 去重`（`compute_tissue_coords_parallel_strips_gpu`，见 `parallel_improve2/wsi_embed_benchmark.py`）。

2. **Tile 编码**  
   - 坐标二分：前半给 `cuda:0`、后半给 `cuda:1`。  
   - 每卡 **独立一份完整 ViT tile 模型**（**不使用** `DataParallel` 的 batch gather）。

3. **Slide 编码**  
   - `max_tokens` 子采样后，在 **`cuda:1`** 单次 slide encoder 前向。

4. **可选监控**  
   - `--monitor`：NVML 采样 GPU 利用率与显存（默认在 **扫描结束后** 开始，避免长扫描稀释曲线）。

## 目录结构

```
gigapath-improve/
├── README.md                 # 本说明
├── requirements.txt          # 从参考 conda 环境 pip freeze（见下方说明）
├── .gitignore
├── gigapath/                 # GigaPath 官方/项目内 slide_encoder 及 torchscale 依赖（Python 包）
├── parallel_improve2/
│   └── wsi_embed_benchmark.py  # 基准脚本：含 v9 GPU 扫描、BaselineWSITileDataset、ResourceMonitor 等
├── scripts/
│   ├── hybrid_v9_tile_slide.py # 单张切片 v9 入口
│   └── batch_embed_v9.py       # 批量目录内 .tif → 各 768d embedding（流程同 verify 中 _ensure_v9_embedding，无 baseline 对比）
├── weights/
│   ├── README.txt            # 权重文件放置说明
│   └── .gitkeep
├── submit_outv9_hybrid.sh    # SLURM 提交示例（需改分区/路径）
└── runs/                     # 默认输出目录（运行后生成，gitignore）
```

## 硬件与显存建议

| 项目 | 建议 |
|------|------|
| **GPU** | **至少 2 张** NVIDIA GPU（CUDA ≥ 11.8 或与 PyTorch 轮子匹配）；脚本中 tile 与 slide 固定使用 `cuda:0` 与 `cuda:1`。 |
| **GPU 显存（单卡）** | 参考实测峰值：tile 阶段约 **~10 GB**（ViT 单卡）；slide 阶段在 **GPU1** 上峰值约 **~30 GB**（与序列长度、batch 有关）。**建议每卡 ≥ 40 GB（如 L40S 48GB、A100 40GB）**；若显存紧张可降低 `--batch_size` 或 `--max_tokens`。 |
| **GPU 显存（扫描）** | 在 `scan_gpu_id`（默认 0）上额外占用少量显存用于 mask/unique；扫描结束后脚本会 `empty_cache()`。 |
| **CPU** | 扫描会启动多进程条带读取；建议 **≥ 16 核**（集群作业示例 `cpus-per-task=32`）。 |
| **内存** | 大 WSI 解码与多 worker 预取：建议 **≥ 64 GB**，集群上 **120 GB** 更稳妥。 |
| **磁盘** | WSI 常为数十 GB；输出 `embedding.pt` 与 `perf.json` 很小。 |

## 软件环境

- **Python 3.10+**（参考环境为 3.10）
- **CUDA** 与 **PyTorch** 版本需匹配（见 `requirements.txt` 中的 `torch==2.10.0` 等；若本机 CUDA 不同，请从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 wheel，再按需安装其余依赖）
- **OpenSlide**：`openslide-python` + 系统级 libopenslide（Linux 常见包名 `openslide-tools` / `libopenslide-dev`）
- **可选**：`pynvml` / `nvidia-ml-py`（用于 `--monitor`）
本项目需要两个预训练的模型，获取方式参见
https://huggingface.co/prov-gigapath/prov-gigapath
## 安装步骤

```bash
git clone <your-fork-or-url> gigapath-improve
cd gigapath-improve

# 建议新建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装 PyTorch（请按你的 CUDA 版本从官网选择一条命令），再安装其余依赖：
pip install -r requirements.txt
```

将预训练权重放入 `weights/`（见 `weights/README.txt`），或自定义路径：

```bash
export TILE_WEIGHT=/path/to/pytorch_model.bin
export SLIDE_WEIGHT=/path/to/slide_encoder.pth
```

## 运行

**必须**提供切片路径（或环境变量 `SLIDE_PATH`）：

```bash
export GIGAPATH_IMPROVE_ROOT="$(pwd)"   # 可选，默认从脚本位置推断
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python scripts/hybrid_v9_tile_slide.py \
  --slide_path /path/to/slide.tif \
  --tile_weight weights/pytorch_model.bin \
  --slide_weight weights/slide_encoder.pth \
  --batch_size 128 \
  --num_workers_per_gpu 4 \
  --scan_cpu_workers 16 \
  --scan_gpu_id 0 \
  --monitor
```

默认输出目录：`<仓库根>/runs/<切片基名>/v9_run/`，包含 `embedding.pt`、`perf.json`、`run_meta.json`；若 `--monitor` 另有 `gpu_curve.png`、`monitor_timeseries.npz`。

### 批量目录（与 `verify_outv9_vs_baseline.py` 中 v9 部分一致）

原脚本 `11111ovarian/compare_test/outv9_compare/verify_outv9_vs_baseline.py` 里，对每张切片实际只做 **`run_v9_pipeline(...)`**（即 `_ensure_v9_embedding`）。本仓库用 `scripts/batch_embed_v9.py` 实现**同一调用链**，但**不包含**与 baseline 的向量对比、cosine、加速比报表等。

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python scripts/batch_embed_v9.py \
  --input_dir /path/to/folder_with_tifs \
  --output_root ./runs \
  --recursive \
  --skip_existing \
  --tile_weight weights/pytorch_model.bin \
  --slide_weight weights/slide_encoder.pth
```

- 每张切片输出：`<output_root>/<基名>/v9_run/embedding.pt`（768 维）、`perf.json` 等，与单张 `hybrid_v9_tile_slide.py` 一致。
- 批量汇总：`--output_root/batch_manifest.json`（每张的状态与 `report` 摘要）。

## SLURM 集群

编辑 `submit_outv9_hybrid.sh` 中的 `#SBATCH` 分区、`module`、`conda` 路径，然后：

```bash
export SLIDE_PATH=/path/to/your.tif
export GIGAPATH_IMPROVE_ROOT=/path/to/gigapath-improve
sbatch submit_outv9_hybrid.sh
```

## 与原仓库的关系

- **`gigapath/`**：来自原工程 `prov-gigapath-main/gigapath`，用于 `import gigapath.slide_encoder`。
- **`parallel_improve2/wsi_embed_benchmark.py`**：来自原工程 `11111ovarian/parallel_improve2/`，在原有 baseline/stream 扫描等基础上，**增加了 v9 的 GPU 扫描实现**（`compute_tissue_coords_parallel_strips_gpu` 等）。

## 上传到 GitHub

1. 不要提交大文件：`weights/*.pt`、`weights/*.pth`、`runs/` 已在 `.gitignore`。
2. `requirements.txt` 含完整 pip 冻结；若推送后他人安装失败，请根据其 CUDA 版本单独安装 `torch/torchvision` 后再 `pip install -r requirements.txt --no-deps` 或分段安装。

## 许可证

原 GigaPath 与相关权重版权归各自所有者；本仓库内**自写脚本与说明**可按项目需要单独选择许可证（如 MIT）。若需对外发布，请确认上游模型与代码的许可条款。
