from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate the processed UAVDT dataset zip.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/raw/UAVDT.zip"),
        help="Path to the downloaded UAVDT zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/processed"),
        help="Directory to extract the UAVDT dataset into.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the zip even if the target dataset directory already exists.",
    )
    return parser.parse_args()


def extract_dataset(zip_path: Path, output_dir: Path, force_extract: bool) -> Path:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")

    dataset_root = output_dir / "UAVDT"
    if not dataset_root.is_dir() or force_extract:
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Expected dataset at: {dataset_root}")
    return dataset_root


def count_files(directory: Path, suffix: str) -> int:
    return sum(1 for _ in directory.glob(f"*{suffix}"))


def validate_dataset(dataset_root: Path) -> None:
    expected = {
        "train": (1266, 1266),
        "val": (271, 271),
        "test": (272, 272),
    }
    for split, (expected_images, expected_labels) in expected.items():
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        image_count = count_files(image_dir, ".jpg")
        label_count = count_files(label_dir, ".txt")
        if image_count != expected_images or label_count != expected_labels:
            raise ValueError(
                f"{split} split mismatch: images={image_count}, labels={label_count}, "
                f"expected images={expected_images}, labels={expected_labels}"
            )


def main() -> None:
    args = parse_args()
    dataset_root = extract_dataset(args.zip_path, args.output_dir, args.force_extract)
    validate_dataset(dataset_root)
    print(f"UAVDT prepared at: {dataset_root}")


if __name__ == "__main__":
    main()
