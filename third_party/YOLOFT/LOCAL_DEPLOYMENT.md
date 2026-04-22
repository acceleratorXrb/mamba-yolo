# YOLOFT Local Deployment

## Scope

This directory keeps a local, isolated YOLOFT deployment for comparison against the
main Mamba-YOLO project. The adaptation here targets the already processed
`VisDroneVID` dataset from the main repository and does not modify the upstream
YOLOFT source files beyond adding local helper scripts and configs.

## Files

- `tools/prepare_visdronevid_local.py`
  - Builds a YOLOFT-friendly local data root from `data/processed/VisDroneVID`
  - Creates symlinks instead of copying images and labels
  - Writes `config/visdrone2019VID_local_10cls.yaml`

- `config/train/orige_stream_visdrone_local.yaml`
  - Local training hyperparameters for this repository

- `scripts/run_yoloft_s_visdrone_local.sh`
  - Minimal launcher for `YOLOFT-S`

## Important boundary

The upstream `YOLOFT` repo reports official `VisDrone2019 VID(test-dev)` numbers,
but this local adapter currently points at the repository's processed 10-class
`train/val` copy of VisDrone-VID. That is good enough for local comparison and
deployment verification, but it is not the exact official `test-dev` benchmark
pipeline yet.

## Local launch

```bash
bash third_party/YOLOFT/scripts/run_yoloft_s_visdrone_local.sh
```

## Optional overrides

```bash
PYTHON_BIN=/path/to/python DEVICE=0 PRETRAIN_MODEL=yolov8s.pt \
  bash third_party/YOLOFT/scripts/run_yoloft_s_visdrone_local.sh
```
