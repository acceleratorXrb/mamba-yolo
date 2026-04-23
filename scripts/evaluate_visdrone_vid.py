from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops, yaml_load


CLASS_NAMES = {
    0: "pedestrian",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
VALID_CLASS_IDS = set(CLASS_NAMES)
FRAME_NAME_PATTERN = re.compile(r"^(?P<prefix>.+_img)(?P<frame>\d+)(?P<suffix>\.[^.]+)$")


@dataclass(frozen=True)
class PredictionRecord:
    image_id: int
    cls: int
    score: float
    xyxy: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisDrone-VID with COCO-style metrics.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights (.pt).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/configs/datasets/VisDroneVID_local.yaml"),
        help="Dataset yaml path.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--batch", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/output_dir/visdrone_vid_eval"),
        help="Directory to save reports.",
    )
    return parser.parse_args()


def xyxy_to_xywh(xyxy: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def yolo_label_to_coco(parts: list[str], width: int, height: int) -> tuple[int, list[float], float]:
    cls = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:5])
    bw = w * width
    bh = h * height
    x1 = max(0.0, xc * width - bw / 2.0)
    y1 = max(0.0, yc * height - bh / 2.0)
    bw = min(float(width) - x1, bw)
    bh = min(float(height) - y1, bh)
    return cls, [x1, y1, bw, bh], max(0.0, bw * bh)


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def resolve_temporal_frame_paths(image_path: Path, temporal_clip_length: int = 3, temporal_stride: int = 1) -> list[Path]:
    if temporal_clip_length % 2 == 0:
        raise ValueError("temporal_clip_length must be odd.")
    match = FRAME_NAME_PATTERN.match(image_path.name)
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


def load_visdrone_gt(data_yaml: Path, split: str) -> tuple[dict, list[Path], dict[str, int]]:
    data_cfg = yaml_load(str(data_yaml))
    dataset_root = Path(data_cfg["path"])
    image_dir = dataset_root / data_cfg[split]
    label_dir = image_dir.parent / "labels"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images = []
    annotations = []
    image_id_map: dict[str, int] = {}
    ann_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        image_id_map[image_path.stem] = image_id
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cls, bbox, area = yolo_label_to_coco(line.split(), width, height)
            if cls not in VALID_CLASS_IDS or area <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(v, 3) for v in bbox],
                    "area": round(area, 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": class_id + 1, "name": name} for class_id, name in CLASS_NAMES.items()]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, image_paths, image_id_map


def run_predictions(
    model: YOLO,
    image_paths: list[Path],
    image_id_map: dict[str, int],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
) -> list[PredictionRecord]:
    raw_model = getattr(model, "model", None)
    if raw_model is not None and getattr(raw_model, "temporal", False):
        return run_temporal_predictions(model, image_paths, image_id_map, imgsz, conf, iou, batch, device)

    predictions: list[PredictionRecord] = []
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
                    PredictionRecord(
                        image_id=image_id_map[image_path.stem],
                        cls=int(cls),
                        score=float(score),
                        xyxy=pred_box.astype(np.float32),
                    )
                )
    return predictions


def run_temporal_predictions(
    model: YOLO,
    image_paths: list[Path],
    image_id_map: dict[str, int],
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    device: str,
) -> list[PredictionRecord]:
    raw_model = model.model
    target_device = torch.device(device if device == "cpu" else f"cuda:{device}")
    raw_model.to(target_device)
    raw_model.eval()
    model_device = next(raw_model.parameters()).device
    raw_args = getattr(raw_model, "args", {}) or {}
    temporal_stride = int(raw_args.get("temporal_stride", 1)) if isinstance(raw_args, dict) else 1
    temporal_clip_length = int(raw_args.get("temporal_clip_length", 3)) if isinstance(raw_args, dict) else 3
    predictions: list[PredictionRecord] = []

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
                predictions.append(
                    PredictionRecord(
                        image_id=image_id_map[image_path.stem],
                        cls=cls,
                        score=float(score),
                        xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    )
                )
    return predictions


def save_coco_gt_json(coco_gt: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8")


def save_predictions_json(predictions: list[PredictionRecord], output_path: Path) -> list[dict]:
    payload = [
        {
            "image_id": pred.image_id,
            "category_id": pred.cls + 1,
            "bbox": [round(v, 3) for v in xyxy_to_xywh(pred.xyxy)],
            "score": round(pred.score, 5),
        }
        for pred in predictions
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def coco_stats_dict(coco_eval: COCOeval) -> dict[str, float]:
    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "ARs": float(stats[9]),
        "ARm": float(stats[10]),
        "ARl": float(stats[11]),
    }


def summarize_per_class(coco_eval: COCOeval, coco_gt: COCO) -> list[dict]:
    precision = coco_eval.eval["precision"]  # [TxRxKxAxM]
    category_ids = coco_gt.getCatIds()
    area_all_idx = 0
    max_det_idx = 2
    rows = []
    for class_idx, category_id in enumerate(category_ids):
        name = coco_gt.loadCats([category_id])[0]["name"]
        ap = precision[:, :, class_idx, area_all_idx, max_det_idx]
        ap = ap[ap > -1]
        ap50 = precision[0, :, class_idx, area_all_idx, max_det_idx]
        ap50 = ap50[ap50 > -1]
        ap75 = precision[5, :, class_idx, area_all_idx, max_det_idx]
        ap75 = ap75[ap75 > -1]
        rows.append(
            {
                "class_id": int(category_id - 1),
                "class_name": name,
                "AP": float(ap.mean()) if ap.size else float("nan"),
                "AP50": float(ap50.mean()) if ap50.size else float("nan"),
                "AP75": float(ap75.mean()) if ap75.size else float("nan"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco_gt_dict, image_paths, image_id_map = load_visdrone_gt(args.data, args.split)
    gt_json = args.output_dir / f"visdrone_vid_{args.split}_gt_coco.json"
    pred_json = args.output_dir / f"visdrone_vid_{args.split}_predictions.json"
    save_coco_gt_json(coco_gt_dict, gt_json)

    model = YOLO(str(args.weights))
    target_device = torch.device("cpu" if str(args.device).lower() == "cpu" else f"cuda:{args.device}")
    model.model.to(target_device)
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
    pred_payload = save_predictions_json(predictions, pred_json)

    coco_gt = COCO(str(gt_json))
    if pred_payload:
        coco_dt = coco_gt.loadRes(str(pred_json))
    else:
        coco_dt = coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(image_id_map.values())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "metrics": coco_stats_dict(coco_eval),
        "per_class": summarize_per_class(coco_eval, coco_gt),
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
    }
    summary_path = args.output_dir / f"visdrone_vid_{args.split}_coco_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
