from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

from evaluate_uavdt import (
    load_dataset_records,
    run_predictions,
    run_temporal_predictions,
)
from ultralytics import YOLO


OFFICIAL_TEST_SEQUENCES = [
    "M0203",
    "M0205",
    "M0208",
    "M0209",
    "M0403",
    "M0601",
    "M0602",
    "M0606",
    "M0701",
    "M0801",
    "M0802",
    "M1001",
    "M1004",
    "M1007",
    "M1009",
    "M1101",
    "M1301",
    "M1302",
    "M1303",
    "M1401",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predictions in official UAVDT DET format and optionally run toolkit.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights (.pt).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/configs/datasets/UAVDT_full_benchmark.yaml"),
        help="Dataset yaml path.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--batch", type=int, default=8, help="Prediction batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Validation dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/uavdt_full/UAV-benchmark-MOTD_v1.0"),
        help="Official UAVDT toolkit root.",
    )
    parser.add_argument(
        "--detector-name",
        type=str,
        default="det_MAMBA_YOLO",
        help="Folder name to create under RES_DET.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/output_dir/uavdt_official_det"),
        help="Directory to save exported predictions and metadata.",
    )
    parser.add_argument(
        "--run-toolkit",
        action="store_true",
        help="Run official MATLAB/Octave DET toolkit after export if available.",
    )
    return parser.parse_args()


def frame_index_from_image(path: Path) -> int:
    stem = path.stem
    if "_img" not in stem:
        raise ValueError(f"Unexpected full UAVDT image name: {path.name}")
    return int(stem.split("_img")[-1])


def sequence_from_image(path: Path) -> str:
    stem = path.stem
    if "_img" not in stem:
        raise ValueError(f"Unexpected full UAVDT image name: {path.name}")
    return stem.split("_img")[0]


def export_predictions(pred_records, image_paths: list[Path], detector_dir: Path) -> dict:
    grouped: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    image_seq_frame = {}
    for p in image_paths:
        value = (sequence_from_image(p), frame_index_from_image(p))
        image_seq_frame[str(p)] = value
        image_seq_frame[p.stem] = value

    for record in pred_records:
        seq, frame_idx = image_seq_frame[record.image_id]
        x1, y1, x2, y2 = [float(v) for v in record.xyxy.tolist()]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        grouped[seq].append((frame_idx, [frame_idx, -1, x1, y1, w, h, float(record.score), 1, -1]))

    detector_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seq in OFFICIAL_TEST_SEQUENCES:
        rows = sorted(grouped.get(seq, []), key=lambda x: (x[0], -x[1][6]))
        out_path = detector_dir / f"{seq}.txt"
        with out_path.open("w", encoding="utf-8") as handle:
            for _, row in rows:
                handle.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n")
        summary[seq] = len(rows)
    return summary


def detect_runner() -> tuple[list[str] | None, str | None]:
    matlab = shutil.which("matlab")
    if matlab:
        return [matlab, "-batch"], "matlab"
    octave = shutil.which("octave") or shutil.which("octave-cli")
    if octave:
        return [octave, "--quiet", "--eval"], "octave"
    return None, None


def run_official_toolkit(toolkit_root: Path, detector_name: str, output_dir: Path) -> dict:
    runner, runner_name = detect_runner()
    if runner is None:
        raise FileNotFoundError("Neither matlab nor octave/octave-cli is available in PATH.")

    det_eva_dir = toolkit_root / "det_EVA"
    det_eva_dir.mkdir(exist_ok=True)
    matlab_cmd = (
        f"cd('{toolkit_root.as_posix()}'); "
        f"detection='{detector_name}'; "
        f"CalculateDetectionPR_overall(detection); "
        f"CalculateDetectionPR_seq(detection); "
        f"for obj_attr = 1:3, CalculateDetectionPR_obj(detection, obj_attr); end;"
    )
    command = runner + [matlab_cmd]
    completed = subprocess.run(command, cwd=str(toolkit_root), capture_output=True, text=True, check=True)
    result = {
        "runner": runner_name,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "det_eva_dir": str(det_eva_dir),
    }
    (output_dir / "official_toolkit_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "official_toolkit_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (output_dir / "official_toolkit_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    target_device = torch.device("cpu" if str(args.device).lower() == "cpu" else f"cuda:{args.device}")
    model.model.to(target_device)
    _, image_paths = load_dataset_records(args.data, args.split)
    raw_model = getattr(model, "model", None)

    if raw_model is not None and getattr(raw_model, "temporal", False):
        raw_args = getattr(raw_model, "args", {})
        temporal_stride = int(raw_args.get("temporal_stride", 1)) if isinstance(raw_args, dict) else 1
        temporal_clip_length = int(raw_args.get("temporal_clip_length", 3)) if isinstance(raw_args, dict) else 3
        pred_records = run_temporal_predictions(
            model=model,
            image_paths=image_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
            temporal_stride=temporal_stride,
            temporal_clip_length=temporal_clip_length,
        )
    else:
        pred_records = run_predictions(
            model=model,
            image_paths=image_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
        )

    detector_dir = args.toolkit_root / "RES_DET" / args.detector_name
    export_summary = export_predictions(pred_records, image_paths, detector_dir)
    metadata = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "toolkit_root": str(args.toolkit_root.resolve()),
        "detector_name": args.detector_name,
        "detector_dir": str(detector_dir.resolve()),
        "exported_detection_counts": export_summary,
    }
    (args.output_dir / "official_det_export_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.run_toolkit:
        toolkit_result = run_official_toolkit(args.toolkit_root, args.detector_name, args.output_dir)
        metadata["toolkit_run"] = toolkit_result

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
