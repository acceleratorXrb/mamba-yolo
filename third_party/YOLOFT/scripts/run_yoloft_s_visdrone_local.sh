#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLOFT_ROOT="$ROOT/third_party/YOLOFT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-yolov8s.pt}"
DATA_YAML="$YOLOFT_ROOT/config/visdrone2019VID_local_10cls.yaml"
TRAIN_YAML="$YOLOFT_ROOT/config/train/orige_stream_visdrone_local.yaml"
MODEL_YAML="$YOLOFT_ROOT/config/yoloft/yoloft-S.yaml"

"$PYTHON_BIN" "$YOLOFT_ROOT/tools/prepare_visdronevid_local.py"

cd "$YOLOFT_ROOT"
"$PYTHON_BIN" - <<PY
from ultralytics.models import YOLOFT

model = YOLOFT(r"$MODEL_YAML").load(r"$PRETRAIN_MODEL")
model.train(
    data=r"$DATA_YAML",
    cfg=r"$TRAIN_YAML",
    device=[$DEVICE],
)
PY
