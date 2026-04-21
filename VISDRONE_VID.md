# VisDrone-VID 流程

本项目内使用普通 `VisDrone-VID` 训练/验证流程，不接挑战赛提交流程。

## 1. 下载并解压

```bash
.conda/mambayolo/bin/python scripts/download_visdrone_vid.py \
  --items train val toolkit \
  --extract
```

如果你还想把 `test-dev` 也一并拉下来：

```bash
.conda/mambayolo/bin/python scripts/download_visdrone_vid.py \
  --items train val test-dev toolkit \
  --extract
```

下载后默认目录是：

- `data/external/visdrone_vid/VisDrone2019-VID-train`
- `data/external/visdrone_vid/VisDrone2019-VID-val`
- `data/external/visdrone_vid/VisDrone2018-VID-toolkit`

## 2. 整理成当前项目可直接训练的格式

```bash
.conda/mambayolo/bin/python scripts/prepare_visdrone_vid.py
```

输出目录：

- `data/processed/VisDroneVID/train/images`
- `data/processed/VisDroneVID/train/labels`
- `data/processed/VisDroneVID/val/images`
- `data/processed/VisDroneVID/val/labels`

数据集配置会写到：

- `official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml`

整理后的文件名是时序友好的格式，例如：

- `uav0000013_00000_v_img0000123.jpg`

这样当前 `YOLOTemporalDataset` 就能直接取 `t-1 / t / t+1`。

## 3. Smoke 测试

```bash
cd official-mamba-yolo
../.conda/mambayolo/bin/python mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --cfg configs/train/visdrone_vid_temporal_smoke.yaml \
  --project output_dir/visdrone_vid \
  --name mambayolo_visdrone_vid_temporal_smoke \
  --device 0
```

## 4. 开发训练

```bash
cd official-mamba-yolo
../.conda/mambayolo/bin/python mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --cfg configs/train/visdrone_vid_temporal_dev.yaml \
  --project output_dir/visdrone_vid \
  --name mambayolo_visdrone_vid_temporal_dev \
  --device 0
```

## 5. 正式训练

```bash
cd official-mamba-yolo
../.conda/mambayolo/bin/python mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --cfg configs/train/visdrone_vid_temporal_benchmark.yaml \
  --project output_dir/visdrone_vid \
  --name mambayolo_visdrone_vid_temporal_benchmark \
  --device 0
```

## 6. 验证

```bash
cd official-mamba-yolo
../.conda/mambayolo/bin/python mbyolo_train.py \
  --task val \
  --config ../output_dir/visdrone_vid/mambayolo_visdrone_vid_temporal_benchmark/weights/best.pt \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --device 0 \
  --batch_size 4 \
  --imgsz 640
```

## 7. 论文对比指标评测

如果你要按论文常用的 COCO 风格指标评测 `VisDrone-VID`，直接运行：

```bash
.conda/mambayolo/bin/python scripts/evaluate_visdrone_vid.py \
  --weights output_dir/visdrone_vid/mambayolo_visdrone_vid_temporal_benchmark/weights/best.pt \
  --data official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml \
  --split val \
  --device 0 \
  --imgsz 640 \
  --batch 4 \
  --output-dir output_dir/visdrone_vid_eval
```

这条命令会输出并落盘：

- `AP`
- `AP50`
- `AP75`
- `APs`
- `APm`
- `APl`
- `AR1`
- `AR10`
- `AR100`

结果文件默认在：

- `output_dir/visdrone_vid_eval/visdrone_vid_val_coco_metrics.json`

## 8. 训练阶段默认输出指标

当前 `VisDrone-VID` 配置默认开启了：

- `visdrone_vid_coco_metrics: true`

这意味着训练时每次验证除了常规的：

- `Precision`
- `Recall`
- `mAP50`
- `mAP50-95`

还会额外计算并打印：

- `AP`
- `AP50`
- `AP75`

这一套指标已经在本地直接 `val()` 实测通过，输出示例为：

- `VisDrone-VID COCO metrics: AP=0.0479, AP50=0.1124, AP75=0.0363`

## 9. 说明

- 当前流程默认以 `train` 训练、`val` 验证。
- `test-dev` 在这个项目里是可选的，不作为主验证集。
- `prepare_visdrone_vid.py` 会保留原始图片，训练目录里使用软链接，不复制大文件。
