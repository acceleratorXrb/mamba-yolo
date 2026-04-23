#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 540
VALID_CATEGORY_IDS = {1, 2, 3}
CATEGORY_TO_CLASS = {1: 0, 2: 1, 3: 2}


@dataclass(frozen=True)
class SequenceSplit:
    train: list[str]
    val: list[str]
    test: list[str]
    train_full: list[str]


def normalize_sequence_name(path: Path) -> str:
    return path.name.split("_")[0].replace(" ", "")


def load_official_split(attr_root: Path) -> tuple[list[str], list[str]]:
    train = sorted(normalize_sequence_name(path) for path in (attr_root / "train").glob("*_attr.txt"))
    test = sorted(normalize_sequence_name(path) for path in (attr_root / "test").glob("*_attr.txt"))
    if not train or not test:
        raise FileNotFoundError(f"Could not infer train/test split from {attr_root}")
    return train, test


def build_dev_split(train_sequences: list[str], test_sequences: list[str]) -> SequenceSplit:
    val = [seq for idx, seq in enumerate(sorted(train_sequences)) if idx % 5 == 4]
    train = [seq for seq in sorted(train_sequences) if seq not in set(val)]
    return SequenceSplit(train=train, val=val, test=sorted(test_sequences), train_full=sorted(train_sequences))


def parse_gt_file(gt_path: Path) -> dict[int, list[str]]:
    frame_to_labels: dict[int, list[str]] = defaultdict(list)
    for raw_line in gt_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        frame_id, _, left, top, width, height, _, _, category = raw_line.split(",")
        category_id = int(category)
        if category_id not in VALID_CATEGORY_IDS:
            continue

        x1 = max(0.0, float(left))
        y1 = max(0.0, float(top))
        x2 = min(float(IMAGE_WIDTH), x1 + float(width))
        y2 = min(float(IMAGE_HEIGHT), y1 + float(height))
        if x2 <= x1 or y2 <= y1:
            continue

        cls = CATEGORY_TO_CLASS[category_id]
        cx = ((x1 + x2) / 2.0) / IMAGE_WIDTH
        cy = ((y1 + y2) / 2.0) / IMAGE_HEIGHT
        w = (x2 - x1) / IMAGE_WIDTH
        h = (y2 - y1) / IMAGE_HEIGHT
        frame_to_labels[int(frame_id)].append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return frame_to_labels


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def populate_split(
    split_name: str,
    sequences: list[str],
    image_root: Path,
    gt_root: Path,
    output_root: Path,
) -> tuple[int, int]:
    image_dir = output_root / split_name / "images"
    label_dir = output_root / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0

    for sequence in sequences:
        sequence_dir = image_root / sequence
        gt_file = gt_root / f"{sequence}_gt_whole.txt"
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"Missing sequence directory: {sequence_dir}")
        if not gt_file.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_file}")

        labels_by_frame = parse_gt_file(gt_file)
        images = sorted(sequence_dir.glob("img*.jpg"))
        if not images:
            raise FileNotFoundError(f"No images found in {sequence_dir}")

        for image_path in images:
            frame_id = int(image_path.stem.replace("img", ""))
            basename = f"{sequence}_{image_path.stem}"
            safe_symlink(image_path, image_dir / f"{basename}.jpg")
            frame_labels = labels_by_frame.get(frame_id, [])
            write_lines(label_dir / f"{basename}.txt", frame_labels)
            image_count += 1
            object_count += len(frame_labels)

    return image_count, object_count


def write_split_manifest(output_root: Path, split_name: str, sequences: list[str]) -> None:
    manifest_dir = output_root / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{split_name}_sequences.txt").write_text("\n".join(sequences) + "\n")


def write_dataset_yaml(output_root: Path, yaml_path: Path) -> None:
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: bus",
                "",
            ]
        )
    )


def write_summary(output_root: Path, counts: dict[str, tuple[int, int]]) -> None:
    lines = []
    for split_name, (image_count, object_count) in counts.items():
        lines.append(f"{split_name}: images={image_count}, objects={object_count}")
    (output_root / "metadata" / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official UAVDT DET benchmark in YOLO format.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/uavdt_full/UAV-benchmark-M"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/uavdt_full/UAV-benchmark-MOTD_v1.0/GT"),
    )
    parser.add_argument(
        "--attr-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/uavdt_full/M_attr"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/processed/UAVDT_full"),
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/configs/datasets/UAVDT_full.yaml"),
    )
    args = parser.parse_args()

    official_train, official_test = load_official_split(args.attr_root)
    split = build_dev_split(official_train, official_test)
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, tuple[int, int]] = {}
    for split_name, sequences in {
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "train_full": split.train_full,
    }.items():
        write_split_manifest(args.output_root, split_name, sequences)
        counts[split_name] = populate_split(split_name, sequences, args.image_root, args.gt_root, args.output_root)
        print(f"{split_name}: sequences={len(sequences)}, images={counts[split_name][0]}, objects={counts[split_name][1]}")

    write_summary(args.output_root, counts)
    write_dataset_yaml(args.output_root, args.yaml_path)
    print(f"yaml: {args.yaml_path}")


if __name__ == "__main__":
    main()
