#!/usr/bin/env python3
"""YOLOFT smoke test: 5 epochs, val every epoch, COCO output"""
import sys
sys.path.insert(0, '/home/easyai/桌面/mamba-yolo3/third_party/YOLOFT')

from ultralytics.models import YOLOFT

MODEL_YAML = '/home/easyai/桌面/mamba-yolo3/third_party/YOLOFT/config/yoloft/yoloft-S.yaml'
PRETRAIN_MODEL = '/home/easyai/桌面/mamba-yolo3/third_party/YOLOFT/yolov8s.pt'
DATA_YAML = '/home/easyai/桌面/mamba-yolo3/third_party/YOLOFT/config/visdrone2019VID_local_10cls.yaml'
TRAIN_YAML = '/home/easyai/桌面/mamba-yolo3/third_party/YOLOFT/config/train/yoloft_smoke_test.yaml'

model = YOLOFT(MODEL_YAML).load(PRETRAIN_MODEL)
model.train(
    data=DATA_YAML,
    cfg=TRAIN_YAML,
    device=[0],
)
