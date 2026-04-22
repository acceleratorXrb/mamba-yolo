#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


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
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_split(source_root: Path, local_root: Path, split_name: str, target_split: str) -> int:
    source_images = source_root / split_name / "images"
    source_labels = source_root / split_name / "labels"
    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing source images dir: {source_images}")
    if not source_labels.is_dir():
        raise FileNotFoundError(f"Missing source labels dir: {source_labels}")

    target_images = local_root / "images" / target_split
    target_labels = local_root / "yolo" / target_split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []
    count = 0
    for image_path in sorted(source_images.glob("*.jpg")):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        # YOLOFT groups frames by the parent directory name. If all images are
        # flattened into images/train or images/test, the whole split becomes one
        # fake "video", which breaks stream sampling. Recreate a per-video folder
        # from the VisDrone filename prefix before `_img`.
        video_name = image_path.stem.split("_img")[0]
        rel_image = Path(target_split) / video_name / image_path.name
        rel_label = Path(target_split) / video_name / label_path.name
        safe_symlink(image_path, target_images / video_name / image_path.name)
        safe_symlink(label_path, target_labels / video_name / label_path.name)
        # YOLOFT only rewrites txt entries when they start with "./". In that
        # code path it prepends `path/images_dir`, so entries here must use
        # "./<split>/<name>" instead of plain relative paths.
        entries.append(f"./{rel_image.as_posix()}")
        count += 1

    split_txt = local_root / ("train.txt" if target_split == "train" else "test.txt")
    write_lines(split_txt, entries)
    return count


def write_dataset_yaml(local_root: Path, yaml_path: Path) -> None:
    lines = [
        "# Auto-generated local VisDrone-VID adapter for YOLOFT",
        "datasetname: MOVEHomoDETDataset_stream",
        f"path: {local_root.resolve()}",
        "train: train.txt",
        "val: test.txt",
        "test: test.txt",
        "labels_dir: yolo/",
        "images_dir: images/",
        "val_reimgsz: True",
        "split_length: [8, 50]",
        "match_number: 1",
        "interval: 1",
        # YOLOFT's stream dataset implementation accesses rho unconditionally.
        # Keep the author's VisDrone default so local configs behave the same.
        "rho: 4",
        "save_json: False",
        "names:",
    ]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Prepare local VisDrone-VID layout for YOLOFT.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root / "data" / "processed" / "VisDroneVID",
        help="Processed VisDroneVID root from the main project.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=root / "third_party" / "YOLOFT" / "local_data" / "visdrone2019VID_10cls",
        help="Local YOLOFT-formatted data root.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=root / "third_party" / "YOLOFT" / "config" / "visdrone2019VID_local_10cls.yaml",
        help="Where to write the local YOLOFT dataset yaml.",
    )
    args = parser.parse_args()

    args.local_root.mkdir(parents=True, exist_ok=True)
    train_count = prepare_split(args.source_root, args.local_root, "train", "train")
    val_count = prepare_split(args.source_root, args.local_root, "val", "test")
    write_dataset_yaml(args.local_root, args.yaml_path)

    summary = [
        f"source_root: {args.source_root.resolve()}",
        f"local_root: {args.local_root.resolve()}",
        f"yaml: {args.yaml_path.resolve()}",
        f"train_images: {train_count}",
        f"test_images: {val_count}",
    ]
    write_lines(args.local_root / "summary.txt", summary)
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
