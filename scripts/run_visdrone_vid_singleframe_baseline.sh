#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "$ROOT/.conda/mambayolo/bin/python" official-mamba-yolo/mbyolo_train.py \
  --task train \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --data ultralytics/cfg/datasets/VisDroneVID.yaml \
  --cfg configs/train/visdrone_vid_singleframe_dev.yaml \
  --project output_dir/visdrone_vid \
  --name mambayolo_visdrone_vid_singleframe_baseline \
  --device 0 \
  --batch_size 16 \
  --workers 4 \
  --epochs 30 \
  --optimizer AdamW \
  --no-amp \
  --val
