from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import yaml_load

from evaluate_visdrone_vid import PredictionRecord, load_visdrone_gt, run_predictions


CLASS_ID_TO_TOOLKIT = {i: i + 1 for i in range(10)}
STEM_PATTERN = re.compile(r"^(?P<video>.+)_img(?P<frame>\d+)$")
RAW_SUBSET_DIR = {
    "val": "VisDrone2019-VID-val",
    "test": "VisDrone2019-VID-test-dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/evaluate VisDrone-VID results with official toolkit format.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights (.pt).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml"),
        help="Dataset yaml path.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Use val or test-dev split.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--batch", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/visdrone_vid"),
        help="Directory containing raw extracted VisDrone2019-VID subsets and toolkit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/output_dir/visdrone_vid_official"),
        help="Directory to save official-format txt files and reports.",
    )
    parser.add_argument("--run-toolkit", action="store_true", help="Run official toolkit with matlab/octave if available.")
    return parser.parse_args()


def split_prediction_by_sequence(predictions: list[PredictionRecord], image_paths: list[Path], image_id_map: dict[str, int]):
    image_meta = {}
    for image_path in image_paths:
        match = STEM_PATTERN.match(image_path.stem)
        if match is None:
            raise ValueError(f"Unexpected VisDrone frame name: {image_path.name}")
        image_meta[image_id_map[image_path.stem]] = (match.group("video"), int(match.group("frame")))

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pred in predictions:
        video_name, frame_idx = image_meta[pred.image_id]
        x1, y1, x2, y2 = [float(v) for v in pred.xyxy]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        row = (
            frame_idx,
            f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{pred.score:.6f},{CLASS_ID_TO_TOOLKIT[pred.cls]},-1,-1",
        )
        grouped[video_name].append(row)
    return grouped


def write_official_txt(grouped_predictions: dict[str, list[tuple[int, str]]], image_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_names = sorted({STEM_PATTERN.match(path.stem).group("video") for path in image_paths})
    created = []
    for sequence_name in sequence_names:
        lines = [row for _, row in sorted(grouped_predictions.get(sequence_name, []), key=lambda item: (item[0], item[1]))]
        txt_path = output_dir / f"{sequence_name}.txt"
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        created.append(txt_path)
    return created


def find_matlab_runner() -> list[str] | None:
    octave = shutil.which("octave")
    if octave:
        return [octave, "--quiet", "--no-gui"]
    matlab = shutil.which("matlab")
    if matlab:
        return [matlab, "-batch"]
    return None


def run_official_toolkit(toolkit_root: Path, dataset_path: Path, result_path: Path, summary_path: Path) -> dict:
    runner = find_matlab_runner()
    if runner is None:
        raise RuntimeError("Neither octave nor matlab is available.")

    eval_script = f"""
warning('off','all');
addpath('{toolkit_root.as_posix()}');
isSeqDisplay = false;
datasetPath = '{dataset_path.as_posix()}';
resPath = '{result_path.as_posix()}';
gtPath = fullfile(datasetPath, 'annotations');
seqPath = fullfile(datasetPath, 'sequences');
nameSeqs = findSeqList(gtPath);
numSeqs = length(nameSeqs);
[allgt, alldet] = saveAnnoRes(gtPath, resPath, seqPath, numSeqs, nameSeqs);
[AP, AR, AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500] = calcAccuracy(numSeqs, allgt, alldet);
fid = fopen('{summary_path.as_posix()}', 'w');
fprintf(fid, '{{"AP": %.6f, "AP50": %.6f, "AP75": %.6f, "AR1": %.6f, "AR10": %.6f, "AR100": %.6f, "AR500": %.6f}}', AP_all/100, AP_50/100, AP_75/100, AR_1/100, AR_10/100, AR_100/100, AR_500/100);
fclose(fid);
"""
    if "octave" in runner[0]:
        with tempfile.NamedTemporaryFile("w", suffix=".m", delete=False) as handle:
            handle.write(eval_script)
            temp_script = Path(handle.name)
        try:
            completed = subprocess.run(runner + [str(temp_script)], check=True, capture_output=True, text=True)
        finally:
            temp_script.unlink(missing_ok=True)
    else:
        completed = subprocess.run(runner + [eval_script], check=True, capture_output=True, text=True)

    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"metrics": metrics, "stdout": completed.stdout, "stderr": completed.stderr}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subset_dir = RAW_SUBSET_DIR[args.split]
    raw_subset_root = args.raw_root / subset_dir
    toolkit_root = args.raw_root / "VisDrone2018-VID-toolkit"
    if not raw_subset_root.exists():
        raise FileNotFoundError(f"Missing raw VisDrone subset: {raw_subset_root}")
    if not toolkit_root.exists():
        raise FileNotFoundError(f"Missing official toolkit: {toolkit_root}")

    coco_gt_dict, image_paths, image_id_map = load_visdrone_gt(args.data, args.split)
    model = YOLO(str(args.weights))
    predictions = run_predictions(
        model=model,
        image_paths=image_paths,
        image_id_map=image_id_map,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
    )

    official_dir = args.output_dir / f"{args.split}_official_txt"
    grouped_predictions = split_prediction_by_sequence(predictions, image_paths, image_id_map)
    txt_files = write_official_txt(grouped_predictions, image_paths, official_dir)

    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "raw_subset_root": str(raw_subset_root.resolve()),
        "toolkit_root": str(toolkit_root.resolve()),
        "official_txt_dir": str(official_dir.resolve()),
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
        "num_sequence_files": len(txt_files),
    }

    if args.run_toolkit:
        toolkit_summary_path = args.output_dir / f"{args.split}_official_metrics.json"
        toolkit_result = run_official_toolkit(toolkit_root, raw_subset_root, official_dir, toolkit_summary_path)
        summary["official_metrics"] = toolkit_result["metrics"]
        (args.output_dir / f"{args.split}_official_toolkit_stdout.txt").write_text(
            toolkit_result["stdout"] + ("\nSTDERR:\n" + toolkit_result["stderr"] if toolkit_result["stderr"] else ""),
            encoding="utf-8",
        )

    summary_path = args.output_dir / f"{args.split}_official_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
