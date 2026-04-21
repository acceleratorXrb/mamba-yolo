#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

import requests


FILES = {
    "train": ("1NSNapZQHar22OYzQYuXCugA3QlMndzvw", "VisDrone2019-VID-train.zip"),
    "val": ("1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B", "VisDrone2019-VID-val.zip"),
    "test-dev": ("1-BEq--FcjshTF1UwUabby_LHhYj41os5", "VisDrone2019-VID-test-dev.zip"),
}

TOOLKIT_REPO = "https://github.com/VisDrone/VisDrone2018-VID-toolkit.git"


def infer_filename(content_disposition: str | None, fallback: str) -> str:
    if not content_disposition:
        return fallback
    match = re.search(r'filename="([^"]+)"', content_disposition)
    return match.group(1) if match else fallback


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


def ensure_toolkit(output_dir: Path) -> None:
    target = output_dir / "VisDrone2018-VID-toolkit"
    if target.exists():
        print(f"skip toolkit: already exists at {target}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", TOOLKIT_REPO, str(target)], check=True)
    print(f"cloned toolkit to {target}")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    target_dir = output_dir / zip_path.stem
    if target_dir.exists():
        print(f"skip extract: already exists at {target_dir}")
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    print(f"extracted {zip_path.name} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official VisDrone-VID dataset and toolkit.")
    parser.add_argument(
        "--items",
        nargs="+",
        choices=["train", "val", "test-dev", "toolkit"],
        default=["train", "val", "toolkit"],
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
    args = parser.parse_args()

    for item in args.items:
        if item == "toolkit":
            ensure_toolkit(args.output_dir)
            continue

        file_id, fallback_name = FILES[item]
        head = requests.get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            stream=True,
            timeout=120,
        )
        head.raise_for_status()
        filename = infer_filename(head.headers.get("content-disposition"), fallback_name)
        head.close()
        output_path = args.output_dir / filename
        download_file(file_id, output_path)
        if args.extract:
            extract_zip(output_path, args.output_dir)


if __name__ == "__main__":
    main()
