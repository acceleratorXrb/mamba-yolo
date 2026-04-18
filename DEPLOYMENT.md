# 项目部署与完整复现说明

本说明只保留一条目标：

- 你从 GitHub 克隆本项目后，严格按本文档一步一步执行，能够**从零完整恢复环境、下载公开数据、处理数据、训练模型、评测模型、生成可视化结果**

为了保证这一点，本文档只使用**公开可下载**、**仓库内已有脚本可处理**的流程。

因此：

- 本文档支持两条数据线完整恢复：
  - `small UAVDT`
  - `official full UAVDT benchmark`
- **不依赖** 你当前机器上的任何已有权重、缓存、环境或 `output_dir`

如果你严格按本文档执行，最终会得到：

- 可运行的 Python/CUDA 环境
- 可运行的 `selective_scan` 扩展
- 可处理完成的 `small UAVDT` 数据
- 处理完成的 `full UAVDT` 数据
- 能正常训练的时序版 `Mamba-YOLO`
- 能正常输出评测指标的 `UAVDT` 正式评测结果
- 能正常导出预测可视化图

## 1. 上传到 GitHub 时应该保留什么

建议上传到 GitHub 的内容只包括源码、配置和说明文件，不要上传数据集、环境和训练产物。

建议保留：

- `README.md`
- `DEPLOYMENT.md`
- `configs/`
- `scripts/`
- `notes/`
- `references/`，如果你希望把论文/代码调研记录一并保留
- `official-mamba-yolo/` 源码本体
  - 保留 `ultralytics/`、`selective_scan/csrc/`、`selective_scan/setup.py`、`mbyolo_train.py`、`pyproject.toml`
  - 不保留编译后的 `.so`、`build/`、以及它原来的嵌套 `.git/`
- 根目录 `.gitignore`

不要上传：

- `.conda/`
- `.python_pkgs/`
- `data/raw/`
- `data/external/`
- `data/processed/`
- `output_dir/`
- `official-mamba-yolo/output_dir/`
- 所有权重文件，例如 `*.pt`、`*.pth`
- 编译产物，例如 `official-mamba-yolo/selective_scan/*.so`
- 各类压缩包，例如 `*.zip`

## 2. 推荐的仓库结构

清理后的仓库建议大致保持为：

```text
mamba-yolo3/
├── README.md
├── DEPLOYMENT.md
├── .gitignore
├── configs/
├── notes/
├── references/
├── scripts/
└── official-mamba-yolo/
```

## 3. 第一次上传前的清理步骤

在项目根目录执行以下命令，先把不应该上传的内容删掉：

```bash
cd /path/to/mamba-yolo3

rm -rf .conda .python_pkgs
rm -rf data/raw data/external data/processed
rm -rf output_dir official-mamba-yolo/output_dir
rm -rf official-mamba-yolo/selective_scan/build
rm -rf official-mamba-yolo/selective_scan/*.egg-info
rm -f official-mamba-yolo/selective_scan/*.so
rm -rf official-mamba-yolo/.git
rm -f *.pt *.pth official-mamba-yolo/*.pt
rm -f nonfinite_forward_*.json official-mamba-yolo/nonfinite_forward_*.json
```

如果当前目录还没有初始化 Git，可以继续执行：

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

## 4. 从 GitHub 克隆后恢复项目

### 4.1 克隆仓库

```bash
git clone <你的 GitHub 仓库地址>
cd mamba-yolo3
export REPO_ROOT="$(pwd)"
```

### 4.2 创建并激活环境

当前项目验证过的核心环境为：

- Python `3.11`
- PyTorch `2.3.0`
- CUDA `12.1` 运行时
- 已安装 CUDA Toolkit，并且 `nvcc` 可用

创建环境：

```bash
conda create -n mambayolo python=3.11 -y
conda activate mambayolo
```

安装 PyTorch：

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

安装通用依赖：

```bash
pip install -U pip wheel setuptools
pip install matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop
pip install timm einops packaging ninja pytest
```

在编译 `selective_scan` 之前，先确认 `nvcc` 存在：

```bash
which nvcc
nvcc -V
```

如果这里没有输出，说明当前机器缺少 CUDA Toolkit，后续 `selective_scan` 编译会失败。

安装 `selective_scan` CUDA 扩展与项目本体：

```bash
cd "$REPO_ROOT/official-mamba-yolo/selective_scan"
pip install .
cd "$REPO_ROOT/official-mamba-yolo"
pip install -v -e .
cd "$REPO_ROOT"
```

### 4.3 环境检查

```bash
python - <<'PY'
import torch
import ultralytics
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("ultralytics:", ultralytics.__version__)
PY
```

## 5. 下载并准备数据集

本文档分别给出：

- `small UAVDT` 的公开下载与恢复方式
- `full UAVDT benchmark` 的官方公开下载与恢复方式

### 5.1 下载并准备 small UAVDT

`small UAVDT` 使用公开 `UAVDT.zip` 压缩包恢复。

当前可用公开地址：

- Zenodo 页面：<https://zenodo.org/records/14575517>
- 其中包含文件 `UAVDT.zip`，页面显示大小约 `4.0 GB`
- DOI：`10.5281/zenodo.14575517`

下载压缩包：

```bash
mkdir -p "$REPO_ROOT/data/raw"

wget -O "$REPO_ROOT/data/raw/UAVDT.zip" \
  "https://zenodo.org/records/14575517/files/UAVDT.zip?download=1"
```

也可以使用 `curl`：

```bash
mkdir -p "$REPO_ROOT/data/raw"

curl -L "https://zenodo.org/records/14575517/files/UAVDT.zip?download=1" \
  -o "$REPO_ROOT/data/raw/UAVDT.zip"
```

解压并检查：

```bash
python "$REPO_ROOT/scripts/prepare_uavdt.py" \
  --zip-path "$REPO_ROOT/data/raw/UAVDT.zip" \
  --output-dir "$REPO_ROOT/data/processed" \
  --force-extract
```

### 5.2 下载并准备官方 full UAVDT

这一步完全依赖仓库内脚本和官方公开文件。

#### 5.2.1 下载官方文件

```bash
mkdir -p "$REPO_ROOT/data/raw/uavdt_full"

python "$REPO_ROOT/scripts/download_uavdt_official.py" \
  --output-dir "$REPO_ROOT/data/raw/uavdt_full"
```

#### 5.2.2 解压官方压缩包

```bash
mkdir -p "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/UAV-benchmark-M.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/UAV-benchmark-MOTD_v1.0.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/M_attr.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"
```

#### 5.2.3 整理为 YOLO 格式

```bash
python "$REPO_ROOT/scripts/prepare_uavdt_full.py" \
  --image-root "$REPO_ROOT/data/external/uavdt_full/UAV-benchmark-M" \
  --gt-root "$REPO_ROOT/data/external/uavdt_full/UAV-benchmark-MOTD_v1.0/GT" \
  --attr-root "$REPO_ROOT/data/external/uavdt_full/M_attr" \
  --output-root "$REPO_ROOT/data/processed/UAVDT_full" \
  --yaml-path "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full.yaml"
```

#### 5.2.4 生成训练用 benchmark YAML

```bash
cat > "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" <<EOF
path: $REPO_ROOT/data/processed/UAVDT_full
train: train_full/images
val: test/images
test: test/images
names:
  0: car
  1: truck
  2: bus
EOF
```

### 5.3 修正数据集 YAML 的绝对路径

为了避免路径写死，执行一次：

```bash
python - <<'PY'
from pathlib import Path
repo = Path.cwd()

replacements = {
    repo / "official-mamba-yolo/ultralytics/cfg/datasets/UAVDT.yaml":
        f"""# Ultralytics YOLO, AGPL-3.0 license
# UAVDT dataset in YOLO detection layout.
path: {repo / 'data/processed/UAVDT'}
train: train/images
val: val/images
test: test/images
names:
  0: car
  1: truck
  2: bus
""",
    repo / "official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full.yaml":
        f"""path: {repo / 'data/processed/UAVDT_full'}
train: train/images
val: val/images
test: test/images
names:
  0: car
  1: truck
  2: bus
""",
}

for path, text in replacements.items():
    path.write_text(text)
    print("patched", path)
PY
```

### 5.4 检查数据整理结果

检查 `small UAVDT`：

```bash
python "$REPO_ROOT/scripts/prepare_uavdt.py" \
  --zip-path "$REPO_ROOT/data/raw/UAVDT.zip" \
  --output-dir "$REPO_ROOT/data/processed"
```

检查 `full UAVDT`：

```bash
cat "$REPO_ROOT/data/processed/UAVDT_full/metadata/summary.txt"
```

如果一切正常：

- `small UAVDT` 应该能通过解压校验
- `full UAVDT` 应该能看到 `train / val / test / train_full` 的图像数和目标数统计

## 6. 训练命令

这里分别提供：

- `small UAVDT` 的训练命令
- `full UAVDT benchmark` 的训练命令

### 6.1 small UAVDT 时序训练

```bash
python "$REPO_ROOT/official-mamba-yolo/mbyolo_train.py" \
  --task train \
  --data ultralytics/cfg/datasets/UAVDT.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --cfg "$REPO_ROOT/configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml" \
  --project "$REPO_ROOT/output_dir/uavdt_small" \
  --name mambayolo_uavdt_temporal_dev
```

### 6.2 full UAVDT smoke test

```bash
python "$REPO_ROOT/official-mamba-yolo/mbyolo_train.py" \
  --task train \
  --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --cfg "$REPO_ROOT/configs/train/uavdt_temporal_smoke.yaml" \
  --project "$REPO_ROOT/output_dir/temporal_smoke" \
  --name smoke_check_full
```

### 6.3 full UAVDT 时序训练

```bash
python "$REPO_ROOT/official-mamba-yolo/mbyolo_train.py" \
  --task train \
  --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --cfg "$REPO_ROOT/configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml" \
  --project "$REPO_ROOT/output_dir/uavdt_full_benchmark" \
  --name mambayolo_uavdt_full_benchmark_temporal_dev
```

## 7. 评测命令

### 7.1 small UAVDT 正式评测

```bash
python "$REPO_ROOT/scripts/evaluate_uavdt.py" \
  --weights "$REPO_ROOT/output_dir/uavdt_small/mambayolo_uavdt_temporal_dev/weights/best.pt" \
  --data "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT.yaml" \
  --split val \
  --device 0 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --output-dir "$REPO_ROOT/output_dir/uavdt_eval/uavdt_small_val"
```

### 7.2 用 full smoke 权重验证评测链路

```bash
python "$REPO_ROOT/scripts/evaluate_uavdt.py" \
  --weights "$REPO_ROOT/output_dir/temporal_smoke/smoke_check_full/weights/best.pt" \
  --data "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" \
  --split test \
  --device 0 \
  --imgsz 640 \
  --batch 4 \
  --workers 0 \
  --output-dir "$REPO_ROOT/output_dir/uavdt_eval/smoke_check_full"
```

### 7.3 对 full UAVDT 完整训练结果做正式评测

```bash
python "$REPO_ROOT/scripts/evaluate_uavdt.py" \
  --weights "$REPO_ROOT/output_dir/uavdt_full_benchmark/mambayolo_uavdt_full_benchmark_temporal_dev/weights/best.pt" \
  --data "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" \
  --split test \
  --device 0 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --output-dir "$REPO_ROOT/output_dir/uavdt_eval/uavdt_full_test"
```

## 8. 可视化命令

为了保证文档中的可视化也可复现，这里只使用你按本文档自己训练得到的权重。

### 8.1 small UAVDT 可视化

先任选一张 `small UAVDT` 的验证图像：

```bash
SMALL_TEST_IMAGE=$(find "$REPO_ROOT/data/processed/UAVDT/val/images" -type f | sort | head -n 1)
export SMALL_TEST_IMAGE
echo "$SMALL_TEST_IMAGE"
```

执行 small `UAVDT` 预测可视化：

```bash
python - <<'PY'
from pathlib import Path
import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from scripts.evaluate_uavdt import prepare_temporal_tensor, resolve_temporal_frame_paths, CLASS_NAMES

repo = Path.cwd()
weights = repo / 'output_dir/uavdt_small/mambayolo_uavdt_temporal_dev/weights/best.pt'
image_path = Path(os.environ['SMALL_TEST_IMAGE'])
out_path = repo / 'output_dir/visualizations/small_uavdt_demo_pred.jpg'

device = torch.device('cuda:0')
model = YOLO(str(weights))
raw_model = model.model.to(device)
raw_model.eval()
imgsz = 640

clip_len = int(getattr(raw_model, 'args', {}).get('temporal_clip_length', 3))
stride = int(getattr(raw_model, 'args', {}).get('temporal_stride', 1))

im0 = cv2.imread(str(image_path))
clip_paths = resolve_temporal_frame_paths(image_path, temporal_clip_length=clip_len, temporal_stride=stride)
current = prepare_temporal_tensor(image_path, imgsz, device)
temporal = torch.stack([prepare_temporal_tensor(p, imgsz, device) for p in clip_paths], 0).unsqueeze(0)
temporal_valid = torch.tensor([[1.0 if (p != image_path or idx == clip_len // 2) else 0.0 for idx, p in enumerate(clip_paths)]], device=device, dtype=torch.float32)

with torch.no_grad():
    preds = raw_model(current.unsqueeze(0), temporal_imgs=temporal, temporal_valid=temporal_valid)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, multi_label=True, agnostic=False, max_det=300)

canvas = im0.copy()
pred = preds[0]
if pred is not None and len(pred):
    pred = pred.clone()
    ops.scale_boxes((imgsz, imgsz), pred[:, :4], canvas.shape[:2])
    for box in pred.detach().cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        cls = int(cls)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(canvas, f'{CLASS_NAMES.get(cls, cls)} {score:.2f}', (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), canvas)
print(out_path)
PY
```

### 8.2 full UAVDT 可视化

先任选一张 full `UAVDT` 的测试图像：

```bash
TEST_IMAGE=$(find "$REPO_ROOT/data/processed/UAVDT_full/test/images" -type f | sort | head -n 1)
export TEST_IMAGE
echo "$TEST_IMAGE"
```

再执行 full `UAVDT` 预测可视化：

```bash
python - <<'PY'
from pathlib import Path
import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from scripts.evaluate_uavdt import prepare_temporal_tensor, resolve_temporal_frame_paths, CLASS_NAMES

repo = Path.cwd()
weights = repo / 'output_dir/uavdt_full_benchmark/mambayolo_uavdt_full_benchmark_temporal_dev/weights/best.pt'
image_path = Path(os.environ['TEST_IMAGE'])
out_path = repo / 'output_dir/visualizations/full_uavdt_demo_pred.jpg'

device = torch.device('cuda:0')
model = YOLO(str(weights))
raw_model = model.model.to(device)
raw_model.eval()
imgsz = 640

clip_len = int(getattr(raw_model, 'args', {}).get('temporal_clip_length', 3))
stride = int(getattr(raw_model, 'args', {}).get('temporal_stride', 1))

im0 = cv2.imread(str(image_path))
clip_paths = resolve_temporal_frame_paths(image_path, temporal_clip_length=clip_len, temporal_stride=stride)
current = prepare_temporal_tensor(image_path, imgsz, device)
temporal = torch.stack([prepare_temporal_tensor(p, imgsz, device) for p in clip_paths], 0).unsqueeze(0)
temporal_valid = torch.tensor([[1.0 if (p != image_path or idx == clip_len // 2) else 0.0 for idx, p in enumerate(clip_paths)]], device=device, dtype=torch.float32)

with torch.no_grad():
    preds = raw_model(current.unsqueeze(0), temporal_imgs=temporal, temporal_valid=temporal_valid)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, multi_label=True, agnostic=False, max_det=300)

canvas = im0.copy()
pred = preds[0]
if pred is not None and len(pred):
    pred = pred.clone()
    ops.scale_boxes((imgsz, imgsz), pred[:, :4], canvas.shape[:2])
    for box in pred.detach().cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        cls = int(cls)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(canvas, f'{CLASS_NAMES.get(cls, cls)} {score:.2f}', (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), canvas)
print(out_path)
PY
```

## 9. 最小可运行检查清单

如果你想确认“项目现在已经被完整恢复”，按下面顺序检查：

### 9.1 检查 CUDA

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

### 9.2 检查命令行入口

```bash
python "$REPO_ROOT/official-mamba-yolo/mbyolo_train.py" --help
python "$REPO_ROOT/scripts/evaluate_uavdt.py" --help
```

### 9.3 检查 small UAVDT 数据是否可用

```bash
python "$REPO_ROOT/scripts/prepare_uavdt.py" \
  --zip-path "$REPO_ROOT/data/raw/UAVDT.zip" \
  --output-dir "$REPO_ROOT/data/processed"
```

### 9.4 跑通 full UAVDT smoke test

```bash
python "$REPO_ROOT/official-mamba-yolo/mbyolo_train.py" \
  --task train \
  --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --cfg "$REPO_ROOT/configs/train/uavdt_temporal_smoke.yaml" \
  --project "$REPO_ROOT/output_dir/temporal_smoke" \
  --name smoke_check_full
```

### 9.5 跑通 full UAVDT 正式评测

```bash
python "$REPO_ROOT/scripts/evaluate_uavdt.py" \
  --weights "$REPO_ROOT/output_dir/temporal_smoke/smoke_check_full/weights/best.pt" \
  --data "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" \
  --split test \
  --device 0 \
  --imgsz 640 \
  --batch 4 \
  --workers 0 \
  --output-dir "$REPO_ROOT/output_dir/uavdt_eval/smoke_check_full"
```

### 9.6 跑通 small UAVDT 可视化

先执行：

```bash
SMALL_TEST_IMAGE=$(find "$REPO_ROOT/data/processed/UAVDT/val/images" -type f | sort | head -n 1)
export SMALL_TEST_IMAGE
```

然后执行第 `8.1` 节中的脚本。

### 9.7 跑通 full UAVDT 可视化

先执行：

```bash
TEST_IMAGE=$(find "$REPO_ROOT/data/processed/UAVDT_full/test/images" -type f | sort | head -n 1)
export TEST_IMAGE
```

然后执行第 `8.2` 节中的脚本。

## 10. 说明

- 本文档保证的是“按步骤从零重新跑通项目”，不是“恢复你当前机器上的历史权重和历史输出目录”
- 只要 GitHub 上保留了本文档第 1 节要求的源码内容，并且外部官方数据下载链接可用，就可以按本文档重新训练和复现实验链路
- 本文档现在把 `small UAVDT` 也纳入完整复现流程，因为已经补充了公开下载地址
