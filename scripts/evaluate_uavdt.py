from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils import yaml_load


CLASS_NAMES = {0: "car", 1: "truck", 2: "bus"}
COLLAPSED_CLASS_NAMES = {0: "vehicle"}
VALID_CLASS_IDS = set(CLASS_NAMES)
COCO_AREA_BINS = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, math.inf),
}
OFFICIAL_TEST_SEQUENCES_WITH_IGNORE = {
    "M0203",
    "M0205",
    "M0208",
    "M0403",
    "M0601",
    "M0602",
    "M0606",
    "M0701",
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
}
FRAME_NAME_PATTERN = re.compile(r"^(?P<prefix>.+_img)(?P<frame>\d+)(?P<suffix>\.[^.]+)$")


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    cls: int
    score: float
    xyxy: np.ndarray
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mamba-YOLO on UAVDT with official and common VOD metrics.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights (.pt).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/configs/datasets/UAVDT_full.yaml"),
        help="Dataset yaml path.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--batch", type=int, default=8, help="Prediction batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Validation dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation predictions.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for model prediction.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/output_dir/uavdt_eval"),
        help="Directory to save reports and plots.",
    )
    return parser.parse_args()


def yolo_label_to_xyxy(parts: list[str], width: int, height: int) -> tuple[int, np.ndarray]:
    cls = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:5])
    bw = w * width
    bh = h * height
    x1 = max(0.0, xc * width - bw / 2.0)
    y1 = max(0.0, yc * height - bh / 2.0)
    x2 = min(float(width), xc * width + bw / 2.0)
    y2 = min(float(height), yc * height + bh / 2.0)
    return cls, np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area(xyxy: np.ndarray) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def xywh_to_xyxy(left: float, top: float, width: float, height: float) -> np.ndarray:
    x1 = max(0.0, left)
    y1 = max(0.0, top)
    x2 = max(x1, left + width)
    y2 = max(y1, top + height)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def compute_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes2) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = box_area(box1)
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-9)
    return inter / union


def is_box_fully_inside(inner_xyxy: np.ndarray, outer_xyxy: np.ndarray) -> bool:
    return (
        inner_xyxy[0] > outer_xyxy[0]
        and inner_xyxy[1] > outer_xyxy[1]
        and inner_xyxy[2] < outer_xyxy[2]
        and inner_xyxy[3] < outer_xyxy[3]
    )


def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def precision_recall_curve(tp_flags: np.ndarray, fp_flags: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return recalls, precisions


def compute_ap_for_class(
    gt_by_image: dict[str, list[np.ndarray]],
    preds: list[BoxRecord],
    iou_thr: float,
) -> dict[str, object]:
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return {"ap": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    preds_sorted = sorted(preds, key=lambda x: x.score, reverse=True)
    used = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()}
    tp_flags = np.zeros(len(preds_sorted), dtype=np.float32)
    fp_flags = np.zeros(len(preds_sorted), dtype=np.float32)
    scores = np.zeros(len(preds_sorted), dtype=np.float32)

    for i, pred in enumerate(preds_sorted):
        scores[i] = pred.score
        gt_boxes = gt_by_image.get(pred.image_id, [])
        if not gt_boxes:
            fp_flags[i] = 1.0
            continue

        gt_array = np.stack(gt_boxes, axis=0)
        ious = compute_iou(pred.xyxy, gt_array)
        match_idx = int(np.argmax(ious)) if len(ious) else -1
        if match_idx >= 0 and ious[match_idx] >= iou_thr and not used[pred.image_id][match_idx]:
            used[pred.image_id][match_idx] = True
            tp_flags[i] = 1.0
        else:
            fp_flags[i] = 1.0

    recalls, precisions = precision_recall_curve(tp_flags, fp_flags, n_gt)
    ap = voc_ap(recalls, precisions)
    f1_curve = 2.0 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    best_idx = int(np.argmax(f1_curve)) if len(f1_curve) else 0
    return {
        "ap": ap,
        "precision": float(precisions[best_idx]) if len(precisions) else 0.0,
        "recall": float(recalls[best_idx]) if len(recalls) else 0.0,
        "f1": float(f1_curve[best_idx]) if len(f1_curve) else 0.0,
        "scores": scores,
        "recalls": recalls,
        "precisions": precisions,
    }


def filter_records_by_area(records: Iterable[BoxRecord], area_min: float, area_max: float) -> list[BoxRecord]:
    return [record for record in records if area_min <= record.area < area_max]


def group_ground_truth(records: Iterable[BoxRecord], collapse_classes: bool = False) -> dict[int, dict[str, list[np.ndarray]]]:
    grouped: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        cls = 0 if collapse_classes else record.cls
        grouped[cls][record.image_id].append(record.xyxy)
    return grouped


def group_predictions(records: Iterable[BoxRecord], collapse_classes: bool = False) -> dict[int, list[BoxRecord]]:
    grouped: dict[int, list[BoxRecord]] = defaultdict(list)
    for record in records:
        cls = 0 if collapse_classes else record.cls
        grouped[cls].append(
            BoxRecord(
                image_id=record.image_id,
                cls=cls,
                score=record.score,
                xyxy=record.xyxy,
                area=record.area,
            )
        )
    return grouped


def load_dataset_records(data_yaml: Path, split: str) -> tuple[list[BoxRecord], list[Path]]:
    data_cfg = yaml_load(str(data_yaml))
    dataset_root = Path(data_cfg["path"])
    image_dir = dataset_root / data_cfg[split]
    label_dir = image_dir.parent / "labels"
    image_paths = sorted(image_dir.glob("*.jpg"))
    gt_records: list[BoxRecord] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cls, xyxy = yolo_label_to_xyxy(line.split(), width, height)
            gt_records.append(
                BoxRecord(
                    image_id=image_path.stem,
                    cls=cls,
                    score=1.0,
                    xyxy=xyxy,
                    area=box_area(xyxy),
                )
            )
    return gt_records, image_paths


def resolve_official_gt_root(data_yaml: Path) -> Path:
    workspace_root = Path("/home/easyai/桌面/mamba-yolo3")
    return workspace_root / "data" / "external" / "uavdt_full" / "UAV-benchmark-MOTD_v1.0" / "GT"


def load_ignore_regions(split: str, image_paths: list[Path], official_gt_root: Path) -> dict[str, list[np.ndarray]]:
    if split != "test":
        return {}
    if not official_gt_root.exists():
        return {}

    image_ids = {path.stem for path in image_paths}
    ignore_by_image: dict[str, list[np.ndarray]] = defaultdict(list)

    for sequence in OFFICIAL_TEST_SEQUENCES_WITH_IGNORE:
        ignore_path = official_gt_root / f"{sequence}_gt_ignore.txt"
        if not ignore_path.exists():
            continue
        for raw_line in ignore_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            frame_id, _, left, top, width, height, *_ = line.split(",")
            image_id = f"{sequence}_img{int(frame_id):06d}"
            if image_id not in image_ids:
                continue
            ignore_by_image[image_id].append(
                xywh_to_xyxy(float(left), float(top), float(width), float(height))
            )
    return ignore_by_image


def filter_predictions_in_ignore_regions(
    pred_records: list[BoxRecord],
    ignore_by_image: dict[str, list[np.ndarray]],
) -> tuple[list[BoxRecord], int]:
    if not ignore_by_image:
        return pred_records, 0

    filtered_records: list[BoxRecord] = []
    removed = 0
    for record in pred_records:
        ignore_boxes = ignore_by_image.get(record.image_id, [])
        if any(is_box_fully_inside(record.xyxy, ignore_xyxy) for ignore_xyxy in ignore_boxes):
            removed += 1
            continue
        filtered_records.append(record)
    return filtered_records, removed


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_temporal_frame_paths(image_path: Path, temporal_clip_length: int = 3, temporal_stride: int = 1) -> list[Path]:
    match = FRAME_NAME_PATTERN.match(image_path.name)
    if temporal_clip_length % 2 == 0:
        raise ValueError("temporal_clip_length must be odd.")
    if match is None:
        return [image_path] * temporal_clip_length
    frame_id = int(match.group("frame"))
    frame_width = len(match.group("frame"))
    offsets = [(idx - temporal_clip_length // 2) * temporal_stride for idx in range(temporal_clip_length)]
    resolved = []
    for offset in offsets:
        target_frame = frame_id + offset
        if target_frame < 1:
            resolved.append(image_path)
            continue
        target_name = f"{match.group('prefix')}{target_frame:0{frame_width}d}{match.group('suffix')}"
        target_path = image_path.with_name(target_name)
        resolved.append(target_path if target_path.exists() else image_path)
    return resolved


def prepare_temporal_tensor(image_path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = LetterBox(new_shape=(imgsz, imgsz), scaleup=False)(image=image)
    image = image.transpose(2, 0, 1)[::-1]
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).to(device=device, non_blocking=True).float() / 255.0


def run_temporal_predictions(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
    temporal_stride: int = 1,
    temporal_clip_length: int = 3,
) -> list[BoxRecord]:
    raw_model = model.model
    raw_model.eval()
    model_device = next(raw_model.parameters()).device
    predictions: list[BoxRecord] = []

    for chunk in batched(image_paths, batch):
        current_tensors = []
        temporal_tensors = []
        temporal_valid = []
        for image_path in chunk:
            clip_paths = resolve_temporal_frame_paths(
                image_path, temporal_clip_length=temporal_clip_length, temporal_stride=temporal_stride
            )
            current_tensors.append(prepare_temporal_tensor(image_path, imgsz, model_device))
            temporal_tensors.append(torch.stack([prepare_temporal_tensor(path, imgsz, model_device) for path in clip_paths], 0))
            temporal_valid.append(
                [
                    1.0 if path != image_path or clip_idx == temporal_clip_length // 2 else 0.0
                    for clip_idx, path in enumerate(clip_paths)
                ]
            )

        current_batch = torch.stack(current_tensors, 0)
        temporal_batch = torch.stack(temporal_tensors, 0)
        temporal_valid_batch = torch.tensor(temporal_valid, device=model_device, dtype=torch.float32)

        with torch.no_grad():
            preds = raw_model(current_batch, temporal_imgs=temporal_batch, temporal_valid=temporal_valid_batch)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = ops.non_max_suppression(
                preds,
                conf_thres=conf,
                iou_thres=iou,
                multi_label=True,
                agnostic=False,
                max_det=300,
            )

        for image_path, pred in zip(chunk, preds):
            if pred is None or len(pred) == 0:
                continue
            pred = pred.detach().cpu().numpy()
            for x1, y1, x2, y2, score, cls in pred:
                cls = int(cls)
                if cls not in VALID_CLASS_IDS:
                    continue
                xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
                predictions.append(
                    BoxRecord(
                        image_id=image_path.stem,
                        cls=cls,
                        score=float(score),
                        xyxy=xyxy,
                        area=box_area(xyxy),
                    )
                )
    return predictions


def run_predictions(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
) -> list[BoxRecord]:
    raw_model = getattr(model, "model", None)
    if raw_model is not None and getattr(raw_model, "temporal", False):
        raw_args = getattr(raw_model, "args", {}) or {}
        temporal_stride = int(raw_args.get("temporal_stride", 1)) if isinstance(raw_args, dict) else 1
        temporal_clip_length = int(raw_args.get("temporal_clip_length", 3)) if isinstance(raw_args, dict) else 3
        return run_temporal_predictions(
            model=model,
            image_paths=image_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            temporal_stride=temporal_stride,
            temporal_clip_length=temporal_clip_length,
        )
    predictions: list[BoxRecord] = []
    for chunk in batched(image_paths, batch):
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            stream=False,
            verbose=False,
            save=False,
        )
        for image_path, result in zip(chunk, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            for pred_box, score, cls in zip(xyxy, confs, clses):
                if int(cls) not in VALID_CLASS_IDS:
                    continue
                predictions.append(
                    BoxRecord(
                        image_id=image_path.stem,
                        cls=int(cls),
                        score=float(score),
                        xyxy=pred_box.astype(np.float32),
                        area=box_area(pred_box),
                    )
                )
    return predictions


def summarize_area_metrics(gt_records: list[BoxRecord], pred_records: list[BoxRecord], iou_thr: float) -> dict[str, float]:
    summary = {}
    for name, (area_min, area_max) in COCO_AREA_BINS.items():
        gt_subset = filter_records_by_area(gt_records, area_min, area_max)
        pred_subset = filter_records_by_area(pred_records, area_min, area_max)
        gt_grouped = group_ground_truth(gt_subset, collapse_classes=True)
        pred_grouped = group_predictions(pred_subset, collapse_classes=True)
        result = compute_ap_for_class(gt_grouped.get(0, {}), pred_grouped.get(0, []), iou_thr=iou_thr)
        summary[f"uavdt_ap70_vehicle_{name}"] = result["ap"]
    return summary


def save_pr_plot(recalls: np.ndarray, precisions: np.ndarray, ap: float, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, linewidth=2, label=f"AP@0.7 = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("UAVDT Vehicle PR Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_json_safe(data: object) -> object:
    if isinstance(data, dict):
        return {str(k): make_json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_json_safe(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.floating, np.integer)):
        return data.item()
    return data


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_records, image_paths = load_dataset_records(args.data, args.split)
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {args.data}")

    model = YOLO(str(args.weights))
    ul_metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=args.workers,
        verbose=False,
        plots=False,
        save_json=False,
    )

    pred_records = run_predictions(
        model=model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
    )
    official_gt_root = resolve_official_gt_root(args.data)
    ignore_by_image = load_ignore_regions(args.split, image_paths, official_gt_root)
    pred_records, ignored_prediction_count = filter_predictions_in_ignore_regions(pred_records, ignore_by_image)

    gt_collapsed = group_ground_truth(gt_records, collapse_classes=True)
    pred_collapsed = group_predictions(pred_records, collapse_classes=True)
    official_result = compute_ap_for_class(gt_collapsed.get(0, {}), pred_collapsed.get(0, []), iou_thr=0.7)
    ap50_vehicle = compute_ap_for_class(gt_collapsed.get(0, {}), pred_collapsed.get(0, []), iou_thr=0.5)

    per_class_results = {}
    gt_per_class = group_ground_truth(gt_records, collapse_classes=False)
    pred_per_class = group_predictions(pred_records, collapse_classes=False)
    for cls_id, cls_name in CLASS_NAMES.items():
        result = compute_ap_for_class(gt_per_class.get(cls_id, {}), pred_per_class.get(cls_id, []), iou_thr=0.7)
        per_class_results[cls_name] = {
            "ap70": result["ap"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "gt_count": sum(len(v) for v in gt_per_class.get(cls_id, {}).values()),
        }

    area_metrics = summarize_area_metrics(gt_records, pred_records, iou_thr=0.7)
    total_ms = (
        ul_metrics.speed["preprocess"] + ul_metrics.speed["inference"] + ul_metrics.speed["postprocess"]
    )
    speed_report = {
        "preprocess_ms_per_image": ul_metrics.speed["preprocess"],
        "inference_ms_per_image": ul_metrics.speed["inference"],
        "postprocess_ms_per_image": ul_metrics.speed["postprocess"],
        "pipeline_ms_per_image": total_ms,
        "pipeline_fps": 1000.0 / total_ms if total_ms > 0 else float("nan"),
        "inference_fps": 1000.0 / ul_metrics.speed["inference"] if ul_metrics.speed["inference"] > 0 else float("nan"),
    }

    save_pr_plot(
        recalls=np.asarray(official_result["recalls"]),
        precisions=np.asarray(official_result["precisions"]),
        ap=float(official_result["ap"]),
        output_path=args.output_dir / f"uavdt_{args.split}_vehicle_pr_curve.png",
    )

    class_counts = Counter(record.cls for record in gt_records)
    report = {
        "dataset": {
            "data_yaml": str(args.data),
            "split": args.split,
            "num_images": len(image_paths),
            "num_ground_truth_boxes": len(gt_records),
            "official_gt_root": str(official_gt_root),
            "num_images_with_ignore_regions": len(ignore_by_image),
            "num_filtered_predictions_in_ignore_regions": ignored_prediction_count,
            "ground_truth_class_counts": {CLASS_NAMES[k]: class_counts.get(k, 0) for k in sorted(CLASS_NAMES)},
        },
        "ultralytics_detection_metrics": {
            **ul_metrics.results_dict,
            "speed": dict(ul_metrics.speed),
        },
        "uavdt_official_detection_metrics": {
            "uavdt_ap70_vehicle": official_result["ap"],
            "uavdt_ap50_vehicle": ap50_vehicle["ap"],
            "uavdt_vehicle_precision_at_best_f1": official_result["precision"],
            "uavdt_vehicle_recall_at_best_f1": official_result["recall"],
            "uavdt_vehicle_f1_at_best_f1": official_result["f1"],
        },
        "uavdt_per_class_ap70": per_class_results,
        "uavdt_area_metrics": area_metrics,
        "speed_metrics": speed_report,
        "notes": [
            "UAVDT official DET benchmark uses AP from the Precision-Recall curve with IoU hit threshold 0.7.",
            "Official ignore-region filtering from *_gt_ignore.txt is applied before computing the custom UAVDT DET metrics.",
            "Ultralytics metrics are retained for mAP50-95, mAP50, precision, and recall.",
            "Vehicle AP collapses car/truck/bus into one 'vehicle' class to match the official UAVDT DET setting.",
            "Area metrics follow COCO-style area bins (<32^2, 32^2-96^2, >96^2 pixels).",
            "Official attribute-wise DET metrics and MOT metrics require raw UAVDT attribute or track annotations, which are not present in this processed detection package.",
        ],
    }

    report_path = args.output_dir / f"uavdt_{args.split}_metrics.json"
    report_path.write_text(json.dumps(make_json_safe(report), indent=2, ensure_ascii=False))

    summary_lines = [
        f"split={args.split}",
        f"images={len(image_paths)}",
        f"Ultralytics mAP50-95(B)={ul_metrics.results_dict['metrics/mAP50-95(B)']:.4f}",
        f"Ultralytics mAP50(B)={ul_metrics.results_dict['metrics/mAP50(B)']:.4f}",
        f"UAVDT AP@0.7(vehicle)={official_result['ap']:.4f}",
        f"pipeline_fps={speed_report['pipeline_fps']:.2f}",
        f"report={report_path}",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
