#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/.conda/mambayolo"
PYTHON_BIN="$ENV_DIR/bin/python"
PIP_BIN="$ENV_DIR/bin/pip"

detect_torch_index() {
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "错误: 未找到 nvcc，无法判断 CUDA Toolkit 版本。" >&2
    exit 1
  fi

  local nvcc_release
  nvcc_release="$(nvcc -V | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
  case "$nvcc_release" in
    11.8)
      echo "cu118"
      ;;
    12.1)
      echo "cu121"
      ;;
    *)
      echo "错误: 当前仅支持 nvcc 11.8 或 12.1，检测到 $nvcc_release" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<EOF
用法:
  bash scripts/deploy_project.sh <command>

命令:
  setup-env                  创建本地环境并安装依赖
  prepare-uavdt              下载并处理 full UAVDT
  prepare-visdrone           下载并处理 VisDrone-VID
  prepare-yoloft             准备 YOLOFT-S 本地对照数据与依赖
  prepare-data               同时处理 UAVDT 和 VisDrone-VID
  train-visdrone-singleframe 启动 VisDrone-VID 官方原始单帧 baseline
  train-visdrone-temporal    启动 VisDrone-VID 时序开发训练
  train-uavdt-temporal       启动 full UAVDT 时序开发训练
  train-yoloft-s             启动 YOLOFT-S 的本地 VisDrone-VID 对照训练
  eval-visdrone              评测 VisDrone-VID 当前 best.pt
  eval-visdrone-official     导出 VisDrone-VID 官方格式结果，并在可用时调用官方 toolkit
  export-uavdt-det           导出 full UAVDT 官方 DET 格式结果
  all                        一次准备主工程与 YOLOFT-S 的训练前依赖

示例:
  bash scripts/deploy_project.sh all
  bash scripts/deploy_project.sh train-visdrone-temporal
EOF
}

warn() {
  echo "[WARN] $*" >&2
}

status_line() {
  local label="$1"
  local status="$2"
  local detail="${3:-}"
  printf '[STATUS] %-22s %-8s %s\n' "$label" "$status" "$detail"
}

latest_main_visdrone_best() {
  find "$ROOT/output_dir/visdrone_vid" -path '*/weights/best.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}'
}

latest_yoloft_best() {
  find "$ROOT/third_party/YOLOFT/runs/detect" -path '*/weights/best.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}'
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "错误: 未找到 conda，请先安装 conda。" >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
}

setup_env() {
  require_conda
  local torch_index
  torch_index="$(detect_torch_index)"

  if [[ ! -x "$PYTHON_BIN" ]]; then
    conda create -p "$ENV_DIR" python=3.11 -y
  fi
  conda activate "$ENV_DIR"

  "$PIP_BIN" install -U pip wheel setuptools
  "$PIP_BIN" install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/${torch_index}"
  "$PIP_BIN" install matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop
  "$PIP_BIN" install timm einops packaging ninja pytest pycocotools

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "错误: 未找到 nvcc，无法编译 selective_scan。" >&2
    exit 1
  fi

  cd "$ROOT/official-mamba-yolo/selective_scan"
  "$PIP_BIN" install --no-build-isolation .

  cd "$ROOT/official-mamba-yolo"
  "$PIP_BIN" install -v -e .

  cd "$ROOT"
  "$PYTHON_BIN" - <<'PY'
import torch
import ultralytics
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("ultralytics:", ultralytics.__version__)
PY
}

write_uavdt_benchmark_yaml() {
  mkdir -p "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets"
  cat > "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" <<EOF
path: $ROOT/data/processed/UAVDT_full
train: train_full/images
val: test/images
test: test/images
names:
  0: car
  1: truck
  2: bus
EOF
}

has_uavdt_archives() {
  local dir="$ROOT/data/raw/uavdt_full"
  [[ -f "$dir/UAV-benchmark-M.zip" ]] && \
  [[ -f "$dir/UAV-benchmark-MOTD_v1.0.zip" ]] && \
  [[ -f "$dir/M_attr.zip" ]]
}

prepare_uavdt() {
  mkdir -p "$ROOT/data/raw/uavdt_full"
  if has_uavdt_archives; then
    echo "skip UAVDT download: required archives already exist in data/raw/uavdt_full"
  else
    "$PYTHON_BIN" "$ROOT/scripts/download_uavdt_official.py" \
      --output-dir "$ROOT/data/raw/uavdt_full"
  fi

  mkdir -p "$ROOT/data/external/uavdt_full"
  unzip -o "$ROOT/data/raw/uavdt_full/UAV-benchmark-M.zip" -d "$ROOT/data/external/uavdt_full"
  unzip -o "$ROOT/data/raw/uavdt_full/UAV-benchmark-MOTD_v1.0.zip" -d "$ROOT/data/external/uavdt_full"
  unzip -o "$ROOT/data/raw/uavdt_full/M_attr.zip" -d "$ROOT/data/external/uavdt_full"

  "$PYTHON_BIN" "$ROOT/scripts/prepare_uavdt_full.py" \
    --image-root "$ROOT/data/external/uavdt_full/UAV-benchmark-M" \
    --gt-root "$ROOT/data/external/uavdt_full/UAV-benchmark-MOTD_v1.0/GT" \
    --attr-root "$ROOT/data/external/uavdt_full/M_attr" \
    --output-root "$ROOT/data/processed/UAVDT_full" \
    --yaml-path "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full.yaml"

  write_uavdt_benchmark_yaml
}

prepare_visdrone() {
  "$PYTHON_BIN" "$ROOT/scripts/download_visdrone_vid.py" \
    --items train val test-dev toolkit \
    --output-dir "$ROOT/data/external/visdrone_vid" \
    --extract \
    --source auto

  "$PYTHON_BIN" "$ROOT/scripts/prepare_visdrone_vid.py" \
    --raw-root "$ROOT/data/external/visdrone_vid" \
    --output-root "$ROOT/data/processed/VisDroneVID" \
    --yaml-path "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml"
}

prepare_yoloft() {
  cd "$ROOT"
  "$PIP_BIN" install 'setuptools<81'
  "$PIP_BIN" install -r "$ROOT/third_party/YOLOFT/requirements.txt"
  "$PYTHON_BIN" "$ROOT/third_party/YOLOFT/tools/prepare_visdronevid_local.py"
}

postflight_summary() {
  local visdrone_yaml="$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml"
  local visdrone_test_dir="$ROOT/data/processed/VisDroneVID/test/images"
  local visdrone_toolkit="$ROOT/data/external/visdrone_vid/VisDrone2018-VID-toolkit"
  local yoloft_data="$ROOT/third_party/YOLOFT/local_data/visdrone2019VID_10cls/train.txt"
  local uavdt_yaml="$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml"
  local uavdt_test="$ROOT/data/processed/UAVDT_full/test/images"

  echo
  echo "=== 部署摘要 ==="
  if [[ -x "$PYTHON_BIN" ]]; then
    status_line "环境" "OK" "$ENV_DIR"
  else
    status_line "环境" "MISSING" "$ENV_DIR"
  fi

  if [[ -d "$visdrone_toolkit" ]]; then
    status_line "VisDrone toolkit" "OK" "$visdrone_toolkit"
  else
    status_line "VisDrone toolkit" "MISSING" "$visdrone_toolkit"
  fi

  if [[ -f "$visdrone_yaml" && -d "$visdrone_test_dir" ]]; then
    status_line "VisDrone test-dev" "OK" "$visdrone_test_dir"
  else
    status_line "VisDrone test-dev" "MISSING" "$visdrone_test_dir"
  fi

  if [[ -f "$yoloft_data" ]]; then
    status_line "YOLOFT local data" "OK" "$yoloft_data"
  else
    status_line "YOLOFT local data" "MISSING" "$yoloft_data"
  fi

  if [[ -f "$uavdt_yaml" && -d "$uavdt_test" ]]; then
    status_line "UAVDT full" "OK" "$uavdt_test"
  else
    status_line "UAVDT full" "PARTIAL" "如需补齐可执行: bash scripts/deploy_project.sh prepare-uavdt"
  fi

  if command -v octave >/dev/null 2>&1 || command -v matlab >/dev/null 2>&1; then
    status_line "官方VID toolkit运行时" "OK" "检测到 octave/matlab"
  else
    status_line "官方VID toolkit运行时" "MISSING" "仅可导出官方格式结果，不能本机直接跑 toolkit"
  fi
  echo
}

train_visdrone_singleframe() {
  cd "$ROOT"
  "$PYTHON_BIN" official-mamba-yolo/mbyolo_train.py \
    --task train \
    --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
    --data ultralytics/cfg/datasets/VisDroneVID.yaml \
    --cfg configs/train/visdrone_vid_singleframe_dev.yaml \
    --project output_dir/visdrone_vid \
    --name mambayolo_visdrone_vid_singleframe_baseline \
    --device 0
  local best_path
  best_path="$(latest_main_visdrone_best)"
  post_eval_main_visdrone "$best_path"
}

train_visdrone_temporal() {
  cd "$ROOT"
  "$PYTHON_BIN" official-mamba-yolo/mbyolo_train.py \
    --task train \
    --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
    --data ultralytics/cfg/datasets/VisDroneVID.yaml \
    --cfg configs/train/visdrone_vid_temporal_dev.yaml \
    --project output_dir/visdrone_vid \
    --name mambayolo_visdrone_vid_temporal_dev \
    --device 0
  local best_path
  best_path="$(latest_main_visdrone_best)"
  post_eval_main_visdrone "$best_path"
}

train_uavdt_temporal() {
  cd "$ROOT"
  exec "$PYTHON_BIN" official-mamba-yolo/mbyolo_train.py \
    --task train \
    --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
    --data ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml \
    --cfg configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml \
    --project output_dir/uavdt_full_benchmark \
    --name mambayolo_uavdt_full_benchmark_temporal_dev \
    --device 0
}

train_yoloft_s() {
  cd "$ROOT"
  env PYTHON_BIN="$PYTHON_BIN" DEVICE=0 PRETRAIN_MODEL=yolov8s.pt \
    bash "$ROOT/third_party/YOLOFT/scripts/run_yoloft_s_visdrone_local.sh"
  local best_path
  best_path="$(latest_yoloft_best)"
  post_eval_yoloft_visdrone "$best_path"
}

eval_visdrone() {
  cd "$ROOT"
  local weights_path="${1:-$(latest_main_visdrone_best)}"
  local run_name
  run_name="$(basename "$(dirname "$(dirname "$weights_path")")")"
  exec "$PYTHON_BIN" scripts/evaluate_visdrone_vid.py \
    --weights "$weights_path" \
    --data "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml" \
    --split val \
    --device 0 \
    --imgsz 640 \
    --batch 6 \
    --output-dir "$ROOT/output_dir/visdrone_vid_eval/$run_name"
}

eval_visdrone_official() {
  cd "$ROOT"
  local weights_path="${1:-$(latest_main_visdrone_best)}"
  local run_name
  run_name="$(basename "$(dirname "$(dirname "$weights_path")")")"
  local extra_args=()
  if command -v octave >/dev/null 2>&1 || command -v matlab >/dev/null 2>&1; then
    extra_args+=(--run-toolkit)
  else
    warn "未检测到 octave/matlab，将仅导出 VisDrone-VID 官方格式结果，不执行官方 toolkit。"
  fi

  exec "$PYTHON_BIN" scripts/evaluate_visdrone_vid_official.py \
    --weights "$weights_path" \
    --data "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml" \
    --split test \
    --device 0 \
    --imgsz 640 \
    --batch 6 \
    --raw-root "$ROOT/data/external/visdrone_vid" \
    --output-dir "$ROOT/output_dir/visdrone_vid_official/$run_name" \
    "${extra_args[@]}"
}

post_eval_main_visdrone() {
  local weights_path="${1:-}"
  if [[ -z "$weights_path" || ! -f "$weights_path" ]]; then
    status_line "VisDrone本地评测" "SKIP" "未找到本次训练生成的 best.pt"
    status_line "VisDrone官方评测" "SKIP" "未找到本次训练生成的 best.pt"
    return 0
  fi

  echo
  echo "=== 训练后自动评测 ==="
  status_line "VisDrone权重" "INFO" "$weights_path"
  if "$PYTHON_BIN" "$ROOT/scripts/evaluate_visdrone_vid.py" \
    --weights "$weights_path" \
    --data "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml" \
    --split val \
    --device 0 \
    --imgsz 640 \
    --batch 6 \
    --output-dir "$ROOT/output_dir/visdrone_vid_eval/$(basename "$(dirname "$(dirname "$weights_path")")")"; then
    status_line "VisDrone本地评测" "OK" "已完成"
  else
    status_line "VisDrone本地评测" "FAIL" "请检查上面的错误输出"
  fi

  local official_args=()
  if command -v octave >/dev/null 2>&1 || command -v matlab >/dev/null 2>&1; then
    official_args+=(--run-toolkit)
  fi
  if "$PYTHON_BIN" "$ROOT/scripts/evaluate_visdrone_vid_official.py" \
    --weights "$weights_path" \
    --data "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml" \
    --split test \
    --device 0 \
    --imgsz 640 \
    --batch 6 \
    --raw-root "$ROOT/data/external/visdrone_vid" \
    --output-dir "$ROOT/output_dir/visdrone_vid_official/$(basename "$(dirname "$(dirname "$weights_path")")")" \
    "${official_args[@]}"; then
    if [[ ${#official_args[@]} -gt 0 ]]; then
      status_line "VisDrone官方评测" "OK" "已导出并调用官方 toolkit"
    else
      status_line "VisDrone官方评测" "OK" "已导出官方格式结果；未检测到 octave/matlab，跳过 toolkit"
    fi
  else
    status_line "VisDrone官方评测" "FAIL" "请检查上面的错误输出"
  fi
}

post_eval_yoloft_visdrone() {
  local weights_path="${1:-}"
  if [[ -z "$weights_path" || ! -f "$weights_path" ]]; then
    status_line "YOLOFT本地评测" "SKIP" "未找到本次训练生成的 best.pt"
    status_line "YOLOFT官方评测" "SKIP" "未找到本次训练生成的 best.pt"
    return 0
  fi

  echo
  echo "=== YOLOFT 训练后自动评测 ==="
  status_line "YOLOFT权重" "INFO" "$weights_path"
  if (
    cd "$ROOT/third_party/YOLOFT" && \
    "$PYTHON_BIN" - <<PY
from ultralytics.models import YOLOFT
model = YOLOFT(r"$weights_path")
model.val(data=r"$ROOT/third_party/YOLOFT/config/visdrone2019VID_local_10cls.yaml",
          cfg=r"$ROOT/third_party/YOLOFT/config/train/orige_stream_visdrone_local.yaml",
          device=[0])
PY
  ); then
    status_line "YOLOFT本地评测" "OK" "已完成"
  else
    status_line "YOLOFT本地评测" "FAIL" "请检查上面的错误输出"
  fi
  status_line "YOLOFT官方评测" "SKIP" "当前仓库尚未接入 YOLOFT -> VisDrone 官方 toolkit 导出链"
}

export_uavdt_det() {
  cd "$ROOT"
  exec "$PYTHON_BIN" scripts/evaluate_uavdt_official_det.py \
    --weights "$ROOT/output_dir/uavdt_full_benchmark/mambayolo_uavdt_full_benchmark_temporal_dev/weights/best.pt" \
    --data "$ROOT/official-mamba-yolo/ultralytics/cfg/datasets/UAVDT_full_benchmark.yaml" \
    --split test \
    --device 0 \
    --imgsz 640 \
    --batch 4 \
    --workers 4 \
    --detector-name det_MAMBA_YOLO \
    --output-dir "$ROOT/output_dir/uavdt_official_det/full_export"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    setup-env)
      setup_env
      ;;
    prepare-uavdt)
      setup_env
      prepare_uavdt
      ;;
    prepare-visdrone)
      setup_env
      prepare_visdrone
      ;;
    prepare-yoloft)
      setup_env
      prepare_yoloft
      ;;
    prepare-data)
      setup_env
      prepare_uavdt
      prepare_visdrone
      ;;
    train-visdrone-singleframe)
      train_visdrone_singleframe
      ;;
    train-visdrone-temporal)
      train_visdrone_temporal
      ;;
    train-uavdt-temporal)
      train_uavdt_temporal
      ;;
    train-yoloft-s)
      train_yoloft_s
      ;;
    eval-visdrone)
      eval_visdrone
      ;;
    eval-visdrone-official)
      eval_visdrone_official
      ;;
    export-uavdt-det)
      export_uavdt_det
      ;;
    all)
      setup_env
      if ! prepare_uavdt; then
        warn "UAVDT 自动准备失败，已跳过 UAVDT，继续准备 VisDrone-VID 和 YOLOFT-S。"
        warn "后续如需补齐 UAVDT，请单独执行: bash scripts/deploy_project.sh prepare-uavdt"
      fi
      prepare_visdrone
      prepare_yoloft
      postflight_summary
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "未知命令: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
