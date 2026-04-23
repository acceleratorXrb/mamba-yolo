from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMG_EXT = ".jpg"
LABEL_EXT = ".txt"


@dataclass(frozen=True)
class ParsedName:
    sequence: str
    frame: int


def normalize_token(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "seq"


def parse_stem(stem: str) -> ParsedName:
    m = re.match(r"^(?P<prefix>.+?)_n(?P<frame>\d+)$", stem)
    if m:
        return ParsedName(normalize_token(m.group("prefix")), int(m.group("frame")))

    m = re.match(r"^(?P<prefix>.+?720p)(?P<frame>\d+)$", stem)
    if m:
        return ParsedName(normalize_token(m.group("prefix")), int(m.group("frame")))

    # Patterns like dji_399_150m00801_1 or dhi_05500501_5 where the last suffix is a stream/view id
    m = re.match(r"^(?P<prefix>.*?)(?P<frame>\d{3,})_(?P<suffix>\d+)$", stem)
    if m and any(tag in m.group("prefix").lower() for tag in ("dji", "dhi", "150m")):
        sequence = f"{normalize_token(m.group('prefix'))}_v{m.group('suffix')}"
        return ParsedName(sequence, int(m.group("frame")))

    m = re.match(r"^(?P<prefix>.+_)(?P<frame>\d+)$", stem)
    if m:
        return ParsedName(normalize_token(m.group("prefix")), int(m.group("frame")))

    m = re.match(r"^(?P<prefix>.*?)(?P<frame>\d+)$", stem)
    if m:
        return ParsedName(normalize_token(m.group("prefix")), int(m.group("frame")))

    raise ValueError(f"Unable to parse frame name: {stem}")


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_split(src_root: Path, dst_root: Path, split: str, frame_width: int = 6) -> dict:
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    ensure_clean_dir(dst_images)
    ensure_clean_dir(dst_labels)

    records = []
    collisions = Counter()
    link_modes = Counter()
    sequence_counts = Counter()

    image_files = sorted(p for p in src_images.glob(f"*{IMG_EXT}") if p.is_file())
    for image_path in image_files:
        parsed = parse_stem(image_path.stem)
        sequence_counts[parsed.sequence] += 1
        new_stem = f"{parsed.sequence}_img{parsed.frame:0{frame_width}d}"
        new_image = dst_images / f"{new_stem}{IMG_EXT}"
        new_label = dst_labels / f"{new_stem}{LABEL_EXT}"
        if new_image.exists() or new_label.exists():
            collisions[new_stem] += 1
            suffix = collisions[new_stem]
            new_stem = f"{new_stem}_dup{suffix}"
            new_image = dst_images / f"{new_stem}{IMG_EXT}"
            new_label = dst_labels / f"{new_stem}{LABEL_EXT}"

        link_modes[hardlink_or_copy(image_path, new_image)] += 1

        src_label = src_labels / f"{image_path.stem}{LABEL_EXT}"
        if src_label.exists():
            link_modes[hardlink_or_copy(src_label, new_label)] += 1
        else:
            new_label.write_text("", encoding="utf-8")
            link_modes["empty_label"] += 1

        records.append(
            {
                "src_image": str(image_path),
                "dst_image": str(new_image),
                "src_label": str(src_label) if src_label.exists() else None,
                "dst_label": str(new_label),
                "sequence": parsed.sequence,
                "frame": parsed.frame,
            }
        )

    return {
        "split": split,
        "num_images": len(records),
        "num_sequences": len(sequence_counts),
        "top_sequences": sequence_counts.most_common(20),
        "link_modes": dict(link_modes),
        "records": records,
    }


def write_dataset_yaml(dst_root: Path, yaml_path: Path) -> None:
    content = f"""# Auto-generated temporal-friendly small UAVDT dataset
path: {dst_root.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: car
  1: truck
  2: bus
"""
    yaml_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create temporal-friendly canonical filenames for small UAVDT.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/processed/UAVDT"),
        help="Source small UAVDT directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/processed/UAVDT_temporal_small"),
        help="Destination directory with canonical temporal filenames.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("configs/datasets/UAVDT_temporal_small.yaml"),
        help="Output dataset yaml path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    summary = {"src": str(args.src.resolve()), "dst": str(args.dst.resolve()), "splits": []}
    for split in ("train", "val", "test"):
        summary["splits"].append(build_split(args.src, args.dst, split))

    write_dataset_yaml(args.dst, args.yaml)
    summary_path = args.dst / "temporal_rename_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        split_info["split"]: {
            "num_images": split_info["num_images"],
            "num_sequences": split_info["num_sequences"],
            "top_sequences": split_info["top_sequences"][:10],
            "link_modes": split_info["link_modes"],
        }
        for split_info in summary["splits"]
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"dataset_yaml={args.yaml.resolve()}")
    print(f"summary_json={summary_path.resolve()}")


if __name__ == "__main__":
    main()
