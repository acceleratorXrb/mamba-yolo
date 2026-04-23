#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import requests


FILES = {
    "train": ("1NSNapZQHar22OYzQYuXCugA3QlMndzvw", "VisDrone2019-VID-train.zip"),
    "val": ("1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B", "VisDrone2019-VID-val.zip"),
}

HF_MIRROR_FILES = {
    "train": "https://hf-mirror.com/datasets/AndriiDemk/visDrone_copy/resolve/main/VisDrone2019-VID-train.zip",
    "val": "https://hf-mirror.com/datasets/AndriiDemk/visDrone_copy/resolve/main/VisDrone2019-VID-val.zip",
}


def infer_filename(content_disposition: str | None, fallback: str) -> str:
    if not content_disposition:
        return fallback
    match = re.search(r'filename="([^"]+)"', content_disposition)
    return match.group(1) if match else fallback


def download_from_url(url: str, output_path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", "0"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and total_bytes and output_path.stat().st_size == total_bytes:
            print(f"skip {output_path.name}: already downloaded ({total_bytes} bytes)")
            return

        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        written = 0
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total_bytes:
                    percent = written / total_bytes * 100
                    print(f"\r{output_path.name}: {written}/{total_bytes} bytes ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r{output_path.name}: {written} bytes", end="", flush=True)
        print()
        temp_path.replace(output_path)
        print(f"saved {output_path}")


def download_file(file_id: str, output_path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", "0"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and total_bytes and output_path.stat().st_size == total_bytes:
            print(f"skip {output_path.name}: already downloaded ({total_bytes} bytes)")
            return

        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        written = 0
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total_bytes:
                    percent = written / total_bytes * 100
                    print(f"\r{output_path.name}: {written}/{total_bytes} bytes ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r{output_path.name}: {written} bytes", end="", flush=True)
        print()
        temp_path.replace(output_path)
        print(f"saved {output_path}")


def resolve_filename_from_url(url: str, fallback: str) -> str:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        filename = infer_filename(response.headers.get("content-disposition"), fallback)
    return filename


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    target_dir = output_dir / zip_path.stem
    if target_dir.exists():
        print(f"skip extract: already exists at {target_dir}")
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    print(f"extracted {zip_path.name} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VisDrone-VID train/val subsets for local validation.")
    parser.add_argument(
        "--items",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Dataset parts to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/external/visdrone_vid"),
        help="Directory to save downloaded files and extracted subsets.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded zip archives into subset folders.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "mirror", "official"],
        default="auto",
        help="Download source preference. auto=hf-mirror first, then official Google Drive.",
    )
    args = parser.parse_args()

    for item in args.items:
        file_id, fallback_name = FILES[item]
        output_path = args.output_dir / fallback_name
        mirror_url = HF_MIRROR_FILES[item]

        if args.source == "mirror":
            print(f"download {item} from hf-mirror")
            download_from_url(mirror_url, output_path)
        elif args.source == "official":
            print(f"download {item} from official google drive")
            filename = resolve_filename_from_url(
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                fallback_name,
            )
            output_path = args.output_dir / filename
            download_file(file_id, output_path)
        else:
            try:
                print(f"download {item} from hf-mirror")
                download_from_url(mirror_url, output_path)
            except Exception as mirror_exc:
                print(f"hf-mirror failed for {item}: {mirror_exc}")
                print(f"fallback {item} to official google drive")
                filename = resolve_filename_from_url(
                    f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                    fallback_name,
                )
                output_path = args.output_dir / filename
                download_file(file_id, output_path)
        if args.extract:
            extract_zip(output_path, args.output_dir)


if __name__ == "__main__":
    main()
