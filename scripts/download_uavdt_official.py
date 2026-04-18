#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests


FILES = {
    "dataset": ("1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc", "UAV-benchmark-M.zip"),
    "toolkit": ("19498uJd7T9w4quwnQEy62nibt3uyT9pq", "UAV-benchmark-MOTD_v1.0.zip"),
    "attributes": ("1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1", "M_attr.zip"),
    "readme": ("1FV7zNa_Z7ivir7Kjw3Cpp2SmeQSinij_", "README.txt"),
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official UAVDT benchmark assets.")
    parser.add_argument(
        "--items",
        nargs="+",
        choices=sorted(FILES),
        default=["dataset", "toolkit", "attributes", "readme"],
        help="Files to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/easyai/桌面/mamba-yolo3/data/raw/uavdt_full"),
        help="Directory to save downloaded files.",
    )
    args = parser.parse_args()

    for item in args.items:
        file_id, fallback_name = FILES[item]
        head = requests.get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            stream=True,
            timeout=120,
        )
        head.raise_for_status()
        filename = infer_filename(head.headers.get("content-disposition"), fallback_name)
        head.close()
        download_file(file_id, args.output_dir / filename)


if __name__ == "__main__":
    main()
