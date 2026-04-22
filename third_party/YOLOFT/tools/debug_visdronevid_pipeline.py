#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
import torch

YOLOFT_ROOT = Path(__file__).resolve().parents[1]
if str(YOLOFT_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOFT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_movedet_dataset
from ultralytics.utils import yaml_load


def print_section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def read_split_file(root: Path, split_name: str) -> list[str]:
    path = root / split_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty split file: {path}")
    return lines


def resolve_image_path(data: dict, entry: str) -> Path:
    root = Path(data["path"])
    images_dir = data["images_dir"]
    if entry.startswith("./"):
        return root / images_dir / entry[2:]
    return Path(entry)


def resolve_label_path(data: dict, image_path: Path) -> Path:
    image_root = f"/{data['images_dir'].strip('/')}/"
    label_root = f"/{data['labels_dir'].strip('/')}/"
    as_posix = image_path.as_posix()
    if image_root not in as_posix:
        raise RuntimeError(f"Image path does not include images_dir marker: {image_path}")
    return Path(as_posix.replace(image_root, label_root, 1)).with_suffix(".txt")


def inspect_split(data: dict, split_key: str) -> None:
    split_file = data[split_key]
    entries = read_split_file(Path(data["path"]), split_file)
    print(f"{split_key}: split_file={split_file}")
    print(f"{split_key}: entries={len(entries)}")
    print(f"{split_key}: first_entry={entries[0]}")
    print(f"{split_key}: last_entry={entries[-1]}")

    missing_images = 0
    missing_labels = 0
    class_counter: Counter[int] = Counter()
    empty_labels = 0
    malformed_labels = 0
    invalid_classes = 0
    invalid_bbox = 0
    widths: list[float] = []
    heights: list[float] = []
    video_counter: Counter[str] = Counter()
    duplicate_entries = len(entries) - len(set(entries))
    frame_pattern = re.compile(r"_img(\d+)$")
    video_frames: dict[str, list[int]] = {}
    bad_video_prefix = 0

    for entry in entries:
        image_path = resolve_image_path(data, entry)
        label_path = resolve_label_path(data, image_path)
        video_counter[image_path.parent.name] += 1
        if not image_path.stem.startswith(image_path.parent.name):
            bad_video_prefix += 1
        match = frame_pattern.search(image_path.stem)
        if match:
            video_frames.setdefault(image_path.parent.name, []).append(int(match.group(1)))
        if not image_path.exists():
            missing_images += 1
            continue
        if not label_path.exists():
            missing_labels += 1
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                malformed_labels += 1
                continue
            cls_id = int(float(parts[0]))
            if cls_id not in data["names"]:
                invalid_classes += 1
            class_counter[cls_id] += 1
            x, y, w, h = map(float, parts[1:])
            widths.append(w)
            heights.append(h)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                invalid_bbox += 1

    non_monotonic_videos = 0
    frame_gap_videos = 0
    for frames in video_frames.values():
        if frames != sorted(frames):
            non_monotonic_videos += 1
        ordered = sorted(frames)
        if len(ordered) > 1 and any(b <= a for a, b in zip(ordered, ordered[1:])):
            frame_gap_videos += 1

    print(f"{split_key}: checked_entries={len(entries)}")
    print(
        f"{split_key}: missing_images={missing_images}, missing_labels={missing_labels}, empty_labels={empty_labels}, "
        f"malformed_labels={malformed_labels}, invalid_classes={invalid_classes}, invalid_bbox={invalid_bbox}, duplicates={duplicate_entries}"
    )
    print(
        f"{split_key}: videos={len(video_counter)} top5={video_counter.most_common(5)}, "
        f"bad_video_prefix={bad_video_prefix}, non_monotonic_videos={non_monotonic_videos}, repeated_frame_videos={frame_gap_videos}"
    )
    print(f"{split_key}: class_hist={dict(sorted(class_counter.items()))}")
    if widths and heights:
        print(
            f"{split_key}: bbox_w[min/mean/max]={min(widths):.4f}/{float(np.mean(widths)):.4f}/{max(widths):.4f}, "
            f"bbox_h[min/mean/max]={min(heights):.4f}/{float(np.mean(heights)):.4f}/{max(heights):.4f}"
        )


def inspect_dataset(data: dict, cfg, split_key: str, batch: int, workers: int) -> None:
    print_section(f"dataset::{split_key}")
    img_path = Path(data["path"]) / data[split_key]
    dataset = build_movedet_dataset(cfg, img_path, batch, data, mode="train" if split_key == "train" else "val", rank=-1)
    print(f"class={dataset.__class__.__name__}")
    print(f"len={len(dataset)}")
    print(f"ni={dataset.ni}")
    print(f"match_number={getattr(dataset, 'match_number', 'n/a')}, interval={getattr(dataset, 'interval', 'n/a')}")
    if hasattr(dataset, "rho"):
        print(f"rho={dataset.rho}")
    if hasattr(dataset, "sub_video_splits"):
        sub_lengths = [len(x) for x in dataset.sub_video_splits[:10]]
        print(f"sub_video_count={len(dataset.sub_video_splits)}, first10_lengths={sub_lengths}")
    if hasattr(dataset, "per_gpu_total_frames"):
        print(f"per_gpu_total_frames={dataset.per_gpu_total_frames}")

    sample = dataset[0]
    print(f"sample_keys={sorted(sample.keys())}")
    print(f"sample_img_shape={tuple(sample['img'].shape)}")
    print(f"sample_cls_shape={tuple(sample['cls'].shape)}, sample_bbox_shape={tuple(sample['bboxes'].shape)}")
    print(f"sample_img_meta={sample.get('img_metas')}")
    print(f"sample_index={sample.get('index')}")

    print_section(f"dataloader::{split_key}")
    dataloader = build_dataloader(dataset, batch=batch, workers=workers, shuffle=False, rank=-1)
    batch_data = next(iter(dataloader))
    print(f"batch_keys={sorted(batch_data.keys())}")
    print(f"batch_img_backbone_shape={tuple(batch_data['img']['backbone'].shape)}")
    print(f"batch_idx_shape={tuple(batch_data['batch_idx'].shape)}")
    print(f"batch_cls_shape={tuple(batch_data['cls'].shape)}, batch_bbox_shape={tuple(batch_data['bboxes'].shape)}")
    print(f"batch_img_metas_count={len(batch_data['img']['img_metas'])}")
    print(f"batch_first_meta={batch_data['img']['img_metas'][0]}")
    unique_cls = sorted(torch.unique(batch_data["cls"].to(torch.int64)).tolist()) if batch_data["cls"].numel() else []
    print(f"batch_unique_classes={unique_cls}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug local VisDrone-VID -> YOLOFT data pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=YOLOFT_ROOT / "config" / "visdrone2019VID_local_10cls.yaml",
        help="YOLOFT local dataset yaml.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=YOLOFT_ROOT / "config" / "train" / "orige_stream_visdrone_local.yaml",
        help="YOLOFT local training cfg.",
    )
    parser.add_argument("--batch", type=int, default=4, help="Debug dataloader batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Debug dataloader workers.")
    args = parser.parse_args()

    data = yaml_load(args.data)
    cfg = get_cfg(args.cfg)
    cfg.imgsz = 640 if cfg.imgsz is None else cfg.imgsz
    cfg.batch = args.batch
    cfg.workers = args.workers
    cfg.classes = list(range(len(data["names"])))
    cfg.cache = False
    cfg.rect = False
    cfg.task = "detect"

    print_section("config")
    print(f"data_yaml={args.data.resolve()}")
    print(f"train_cfg={args.cfg.resolve()}")
    print(f"dataset_root={data['path']}")
    print(f"datasetname={data['datasetname']}")
    print(f"names={data['names']}")
    print(f"split_length={data['split_length']}, match_number={data['match_number']}, interval={data['interval']}, rho={data['rho']}")
    print(f"debug_batch={args.batch}, debug_workers={args.workers}, imgsz={cfg.imgsz}")

    print_section("raw_splits")
    inspect_split(data, "train")
    inspect_split(data, "test")

    inspect_dataset(data, cfg, "train", args.batch, args.workers)
    inspect_dataset(data, cfg, "val", args.batch, args.workers)

    print_section("result")
    print("pipeline_debug=OK")


if __name__ == "__main__":
    main()
