# DEPLOYMENT

## 1. GitHub 需要上传的内容

```bash
README.md
DEPLOYMENT.md
.gitignore
configs/
docs/
scripts/
official-mamba-yolo/
```

作用：保留源码、配置、脚本和文档；不上传数据、环境、训练结果和权重。

## 2. GitHub 不需要上传的内容

```bash
.conda/
.python_pkgs/
data/
output_dir/
official-mamba-yolo/output_dir/
*.pt
*.pth
*.onnx
*.engine
*.zip
official-mamba-yolo/selective_scan/build/
official-mamba-yolo/selective_scan/*.egg-info/
official-mamba-yolo/selective_scan/*.so
official-mamba-yolo/.git/
```

作用：避免把环境、数据集、训练产物和编译产物推到仓库。

## 3. 上传前清理

```bash
cd /home/easyai/桌面/mamba-yolo3

rm -rf .conda .python_pkgs
rm -rf data output_dir official-mamba-yolo/output_dir
rm -rf official-mamba-yolo/selective_scan/build
rm -rf official-mamba-yolo/selective_scan/*.egg-info
rm -f official-mamba-yolo/selective_scan/*.so
rm -rf official-mamba-yolo/.git
find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.onnx" -o -name "*.engine" -o -name "*.zip" \) -delete
```

作用：把本地大文件和不可复现内容清掉，只保留源码仓库。

## 4. 克隆项目

```bash
git clone <你的仓库地址>
cd mamba-yolo3
export REPO_ROOT="$(pwd)"
```

作用：拉取项目并固定根目录环境变量，后续命令统一使用。

## 5. 创建环境

```bash
conda create -n mambayolo python=3.11 -y
conda activate mambayolo
```

作用：创建独立 Python 环境。

## 6. 安装 PyTorch

```bash
pip install -U pip wheel setuptools
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

作用：安装当前项目验证过的 PyTorch/CUDA 组合。

## 7. 安装通用依赖

```bash
pip install matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop
pip install timm einops packaging ninja pytest pycocotools
```

作用：安装训练、评测、可视化和 COCO 风格指标所需依赖。

## 8. 检查 CUDA Toolkit

```bash
which nvcc
nvcc -V
```

作用：确认机器安装了 CUDA Toolkit；如果没有 `nvcc`，`selective_scan` 无法编译。

## 9. 编译并安装项目

```bash
cd "$REPO_ROOT/official-mamba-yolo/selective_scan"
pip install .

cd "$REPO_ROOT/official-mamba-yolo"
pip install -v -e .

cd "$REPO_ROOT"
```

作用：编译 `selective_scan` CUDA 扩展，并以可编辑模式安装项目本体。

## 10. 环境自检

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

作用：确认 PyTorch、CUDA 和项目安装成功。

## 11. 下载官方 full UAVDT

```bash
mkdir -p "$REPO_ROOT/data/raw/uavdt_full"

python "$REPO_ROOT/scripts/download_uavdt_official.py" \
  --output-dir "$REPO_ROOT/data/raw/uavdt_full"
```

作用：下载官方 `UAV-benchmark-M.zip`、`UAV-benchmark-MOTD_v1.0.zip`、`M_attr.zip`。

## 12. 解压官方 full UAVDT

```bash
mkdir -p "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/UAV-benchmark-M.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/UAV-benchmark-MOTD_v1.0.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"

unzip -o "$REPO_ROOT/data/raw/uavdt_full/M_attr.zip" \
  -d "$REPO_ROOT/data/external/uavdt_full"
```

作用：把官方 full UAVDT 图片、标注和属性文件解压到统一目录。

## 13. 处理 full UAVDT 为 YOLO 格式

```bash
python "$REPO_ROOT/scripts/prepare_uavdt_full.py" \
  --image-root "$REPO_ROOT/data/external/uavdt_full/UAV-benchmark-M" \
  --gt-root "$REPO_ROOT/data/external/uavdt_full/UAV-benchmark-MOTD_v1.0/GT" \
  --attr-root "$REPO_ROOT/data/external/uavdt_full/M_attr" \
  --output-root "$REPO_ROOT/data/processed/UAVDT_full" \
  --yaml-path "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full.yaml"
```

作用：把官方数据转换成当前项目训练所需的 YOLO 检测格式。

## 14. 生成 full UAVDT benchmark 数据集 YAML

```bash
cat > "$REPO_ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" <<EOF
path: $REPO_ROOT/data/processed/UAVDT_full
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: car
  1: truck
  2: bus
EOF
```

作用：生成 full UAVDT 正式训练/验证/测试入口配置。

## 15. 下载 VisDrone-VID

```bash
python "$REPO_ROOT/scripts/download_visdrone_vid.py" \
  --items train val toolkit \
  --extract
```

作用：下载并解压 `VisDrone2019-VID-train`、`VisDrone2019-VID-val` 和官方 `VID toolkit`。

## 16. 处理 VisDrone-VID 为训练格式

```bash
python "$REPO_ROOT/scripts/prepare_visdrone_vid.py"
```

作用：把 `VisDrone-VID` 转成当前项目可直接训练的连续帧 YOLO 格式，并写出 `VisDroneVID.yaml`。

## 17. 启动 VisDrone-VID 官方原始单帧 baseline

```bash
cd "$REPO_ROOT"
bash scripts/run_visdrone_vid_singleframe_baseline.sh
```

作用：启动始终保留的官方原始单帧 `Mamba-YOLO-T` 基线训练。

## 18. 启动 VisDrone-VID 时序开发训练

```bash
cd "$REPO_ROOT"

python official-mamba-yolo/mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --cfg configs/train/visdrone_vid_temporal_dev.yaml \
  --project output_dir/visdrone_vid \
  --name mambayolo_visdrone_vid_temporal_dev \
  --device 0 \
  --batch_size 6 \
  --workers 4 \
  --epochs 30 \
  --optimizer AdamW \
  --no-amp \
  --val
```

作用：启动当前项目的 `VisDrone-VID` 时序开发训练，训练期自动输出 `AP / AP50 / AP75`。

## 19. 启动 full UAVDT 时序开发训练

```bash
cd "$REPO_ROOT"

python official-mamba-yolo/mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --cfg configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml \
  --project output_dir/uavdt_full_benchmark \
  --name mambayolo_uavdt_full_benchmark_temporal_dev \
  --device 0 \
  --batch_size 4 \
  --workers 8 \
  --epochs 50 \
  --optimizer AdamW \
  --no-amp \
  --val
```

作用：启动 full UAVDT 时序开发训练，训练期每 2 个 epoch 自动验证一次。

## 20. 启动 full UAVDT 时序正式 benchmark

```bash
cd "$REPO_ROOT"

python official-mamba-yolo/mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --cfg configs/train/uavdt_full_benchmark_temporal_strict_img640_adamw.yaml \
  --project output_dir/uavdt_full_benchmark \
  --name mambayolo_uavdt_full_benchmark_temporal_strict \
  --device 0 \
  --batch_size 4 \
  --workers 8 \
  --epochs 100 \
  --optimizer AdamW \
  --no-amp \
  --val
```

作用：启动 full UAVDT 时序正式 benchmark 训练。

## 21. 评测 VisDrone-VID

```bash
cd "$REPO_ROOT"

python scripts/evaluate_visdrone_vid.py \
  --weights output_dir/visdrone_vid/mambayolo_visdrone_vid_temporal_dev/weights/best.pt \
  --data official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml \
  --split val \
  --device 0 \
  --imgsz 640 \
  --batch 6 \
  --output-dir output_dir/visdrone_vid_eval/temporal_dev
```

作用：对 `VisDrone-VID` 计算 `AP / AP50 / AP75`、分类别指标，并保存评测 JSON。

## 22. 导出 full UAVDT 官方 DET 格式结果

```bash
cd "$REPO_ROOT"

python scripts/evaluate_uavdt_official_det.py \
  --weights output_dir/uavdt_full_benchmark/mambayolo_uavdt_full_benchmark_temporal_strict/weights/best.pt \
  --data official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --split test \
  --device 0 \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --detector-name det_MAMBA_YOLO \
  --output-dir output_dir/uavdt_official_det/full_export
```

作用：把 full UAVDT 测试集预测结果导出成官方 `DET toolkit` 需要的 `RES_DET/<detector>/<sequence>.txt` 格式。

## 23. 运行 full UAVDT 官方 DET toolkit（需要 MATLAB 或 Octave）

```bash
cd "$REPO_ROOT"

python scripts/evaluate_uavdt_official_det.py \
  --weights output_dir/uavdt_full_benchmark/mambayolo_uavdt_full_benchmark_temporal_strict/weights/best.pt \
  --data official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
  --split test \
  --device 0 \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --detector-name det_MAMBA_YOLO \
  --output-dir output_dir/uavdt_official_det/full_export \
  --run-toolkit
```

作用：在本机具备 `MATLAB` 或 `Octave` 时，直接调用官方 `UAVDT DET toolkit` 计算正式结果。

## 24. 训练结果文件位置

```bash
output_dir/visdrone_vid/<实验名>/results.csv
output_dir/visdrone_vid/<实验名>/weights/best.pt
output_dir/visdrone_vid/<实验名>/visdrone_vid_val_per_class_metrics.json

output_dir/uavdt_full_benchmark/<实验名>/results.csv
output_dir/uavdt_full_benchmark/<实验名>/weights/best.pt
output_dir/uavdt_official_det/<实验名>/
```

作用：查看训练曲线、最佳权重、分类别指标和官方导出结果。

## 25. 快速检查当前仓库是否恢复成功

```bash
python - <<'PY'
from pathlib import Path
targets = [
    "official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml",
    "official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml",
    "configs/train/visdrone_vid_temporal_dev.yaml",
    "configs/train/visdrone_vid_singleframe_dev.yaml",
    "configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml",
    "official-mamba-yolo/mbyolo_train.py",
]
for t in targets:
    p = Path(t)
    print(t, "OK" if p.exists() else "MISSING")
PY
```

作用：快速确认关键配置、训练入口和数据集 YAML 已经齐全。
