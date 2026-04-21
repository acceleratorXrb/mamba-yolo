#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image


VALID_CATEGORY_IDS = set(range(1, 11))
CATEGORY_TO_CLASS = {cid: cid - 1 for cid in range(1, 11)}
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


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def parse_annotation_file(annotation_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    frame_to_boxes: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for raw_line in annotation_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 10:
            raise ValueError(f"Unexpected annotation format in {annotation_path}: {raw_line}")
        frame_id = int(parts[0])
        left = float(parts[2])
        top = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        score = int(float(parts[6]))
        category_id = int(float(parts[7]))

        if score != 1 or category_id not in VALID_CATEGORY_IDS:
            continue
        if width <= 0 or height <= 0:
            continue

        frame_to_boxes[frame_id].append((CATEGORY_TO_CLASS[category_id], left, top, width, height))
    return frame_to_boxes


def convert_boxes_to_yolo(
    boxes: list[tuple[int, float, float, float, float]], image_width: int, image_height: int
) -> list[str]:
    lines: list[str] = []
    for cls, left, top, width, height in boxes:
        x1 = max(0.0, left)
        y1 = max(0.0, top)
        x2 = min(float(image_width), left + width)
        y2 = min(float(image_height), top + height)
        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / image_width
        cy = ((y1 + y2) / 2.0) / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def process_subset(subset_root: Path, split_name: str, output_root: Path) -> tuple[int, int, int]:
    sequences_root = subset_root / "sequences"
    annotations_root = subset_root / "annotations"
    if not sequences_root.is_dir():
        raise FileNotFoundError(f"Missing sequences dir: {sequences_root}")
    has_annotations = annotations_root.is_dir()
    if split_name in {"train", "val"} and not has_annotations:
        raise FileNotFoundError(f"Missing annotations dir: {annotations_root}")

    out_images = output_root / split_name / "images"
    out_labels = output_root / split_name / "labels"
    metadata_dir = output_root / "metadata"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sequence_names = []
    image_count = 0
    object_count = 0

    for sequence_dir in sorted(sequences_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        sequence_name = sequence_dir.name
        sequence_names.append(sequence_name)
        annotation_path = annotations_root / f"{sequence_name}.txt"
        if has_annotations and not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        boxes_by_frame = parse_annotation_file(annotation_path) if annotation_path.exists() else {}
        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {sequence_dir}")

        for frame_path in frame_paths:
            try:
                frame_id = int(frame_path.stem)
            except ValueError as exc:
                raise ValueError(f"Frame name must be numeric in {sequence_dir}, got {frame_path.name}") from exc

            canonical_name = f"{sequence_name}_img{frame_id:07d}"
            safe_symlink(frame_path, out_images / f"{canonical_name}.jpg")

            with Image.open(frame_path) as image:
                width, height = image.size
            lines = convert_boxes_to_yolo(boxes_by_frame.get(frame_id, []), width, height)
            write_lines(out_labels / f"{canonical_name}.txt", lines)
            image_count += 1
            object_count += len(lines)

    (metadata_dir / f"{split_name}_sequences.txt").write_text("\n".join(sequence_names) + "\n")
    return len(sequence_names), image_count, object_count


def write_dataset_yaml(output_root: Path, yaml_path: Path) -> None:
    lines = [
        f"path: {output_root.resolve()}",
        "train: train/images",
        "val: val/images",
        "names:",
    ]
    test_images_dir = output_root / "test" / "images"
    if test_images_dir.exists():
        lines.insert(3, "test: test/images")
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VisDrone-VID in YOLO temporal-friendly format.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/visdrone_vid"),
        help="Directory containing extracted VisDrone2019-VID-* subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/processed/VisDroneVID"),
        help="Directory to save YOLO-format processed data.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/cfg/datasets/VisDroneVID.yaml"),
        help="Path to write the dataset YAML.",
    )
    args = parser.parse_args()

    subsets = {
        "train": args.raw_root / "VisDrone2019-VID-train",
        "val": args.raw_root / "VisDrone2019-VID-val",
    }
    optional_test_dir = args.raw_root / "VisDrone2019-VID-test-dev"
    if optional_test_dir.exists():
        subsets["test"] = optional_test_dir

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    for split_name, subset_root in subsets.items():
        seq_count, image_count, object_count = process_subset(subset_root, split_name, args.output_root)
        line = f"{split_name}: sequences={seq_count}, images={image_count}, objects={object_count}"
        summary_lines.append(line)
        print(line)

    metadata_dir = args.output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    write_dataset_yaml(args.output_root, args.yaml_path)
    print(f"yaml: {args.yaml_path}")


if __name__ == "__main__":
    main()
