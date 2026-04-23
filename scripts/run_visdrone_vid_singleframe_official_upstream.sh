#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="$ROOT/mamba-yolo-official"
ENV_DIR="$ROOT/.conda/mambayolo"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
DATA_CFG="${DATA_CFG:-$ROOT/configs/datasets/VisDroneVID_local.yaml}"
TRAIN_CFG="${TRAIN_CFG:-$ROOT/configs/train/visdrone_vid_singleframe_official_upstream.yaml}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT/output_dir/visdrone_vid_official}"
RUN_NAME="${RUN_NAME:-mambayolo_visdrone_vid_singleframe_official_upstream}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
VAL_BATCH="${VAL_BATCH:-6}"

if [[ ! -d "$UPSTREAM_ROOT" ]]; then
  echo "错误: 未找到官方原始仓库目录: $UPSTREAM_ROOT" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "错误: 未找到 Python 环境: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
"$PYTHON_BIN" - <<PY
from ultralytics import YOLO

model = YOLO(r"$UPSTREAM_ROOT/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml")
model.train(
    data=r"$DATA_CFG",
    cfg=r"$TRAIN_CFG",
    project=r"$PROJECT_DIR",
    name=r"$RUN_NAME",
    device="$DEVICE",
)
PY

BEST_PATH="$(
  find "$PROJECT_DIR" -path '*/weights/best.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}'
)"

if [[ -z "${BEST_PATH:-}" || ! -f "$BEST_PATH" ]]; then
  echo "[WARN] 未找到官方原始单帧训练生成的 best.pt，跳过自动评测。" >&2
  exit 0
fi

RUN_DIR="$(basename "$(dirname "$(dirname "$BEST_PATH")")")"
echo "[INFO] official upstream best.pt: $BEST_PATH"

PYTHONPATH="$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/evaluate_visdrone_vid.py" \
  --weights "$BEST_PATH" \
  --data "$DATA_CFG" \
  --split val \
  --device "$DEVICE" \
  --imgsz "$IMGSZ" \
  --batch "$VAL_BATCH" \
  --output-dir "$ROOT/output_dir/visdrone_vid_eval/$RUN_DIR"
