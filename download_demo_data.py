#!/usr/bin/env python3
"""Download and unpack the demo datasets for the match-phase package.

The script uses only the Python standard library. By default it downloads the
public LRZ Sync+Share link used for the example data and extracts the archive
into ``Demo_Datasets`` next to this file.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_URL = "https://syncandshare.lrz.de/getlink/fiSLmtY6A2V4nvbYpiRgw6/"
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when the demo data cannot be downloaded or unpacked."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download demo data into Demo_Datasets/.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Demo-data download URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Demo_Datasets",
        help="Destination folder. Defaults to Demo_Datasets next to this script.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files. By default existing files are kept.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive in the output folder after extraction.",
    )
    return parser.parse_args()


def filename_from_response(response: urllib.response.addinfourl, url: str) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition, flags=re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1)).strip()

    parsed = urllib.parse.urlparse(response.geturl() or url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if name and name not in {"getlink", "download"}:
        return name
    return "demo_data_download"


def download(url: str, work_dir: Path) -> tuple[Path, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "match-phase-demo-data-downloader/1.0",
            "Accept": "application/octet-stream,application/zip,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            filename = filename_from_response(response, url)
            target = work_dir / filename
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            with target.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    print_progress(downloaded, total_bytes)
            print()
            return target, response.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        raise DownloadError(f"Could not download demo data from {url}: {exc}") from exc


def print_progress(downloaded: int, total: int | None) -> None:
    if total:
        percent = downloaded / total * 100
        message = f"\rDownloading: {downloaded / 1_000_000:.1f}/{total / 1_000_000:.1f} MB ({percent:5.1f}%)"
    else:
        message = f"\rDownloading: {downloaded / 1_000_000:.1f} MB"
    print(message, end="", flush=True)


def is_probably_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def safe_destination(output_dir: Path, relative_name: str) -> Path:
    clean = PurePosixPath(relative_name)
    parts = [part for part in clean.parts if part not in {"", ".", "__MACOSX"}]
    if not parts or any(part == ".." for part in parts):
        raise DownloadError(f"Unsafe archive member path: {relative_name}")

    if "Demo_Datasets" in parts:
        parts = parts[parts.index("Demo_Datasets") + 1 :]
    if not parts or parts[-1] == ".DS_Store":
        return output_dir

    destination = (output_dir / Path(*parts)).resolve()
    output_root = output_dir.resolve()
    if destination != output_root and output_root not in destination.parents:
        raise DownloadError(f"Archive member would escape output directory: {relative_name}")
    return destination


def write_file(source, destination: Path, overwrite: bool) -> bool:
    if destination == destination.parent:
        return False
    if destination.exists() and not overwrite:
        print(f"Keeping existing file: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return True


def extract_zip(archive: Path, output_dir: Path, overwrite: bool) -> int:
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            destination = safe_destination(output_dir, info.filename)
            if destination == output_dir:
                continue
            with zf.open(info) as source:
                written += int(write_file(source, destination, overwrite))
    return written


def extract_tar(archive: Path, output_dir: Path, overwrite: bool) -> int:
    written = 0
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            destination = safe_destination(output_dir, member.name)
            if destination == output_dir:
                continue
            source = tf.extractfile(member)
            if source is None:
                continue
            with source:
                written += int(write_file(source, destination, overwrite))
    return written


def unpack_or_copy(downloaded: Path, content_type: str, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(downloaded):
        return extract_zip(downloaded, output_dir, overwrite)
    if tarfile.is_tarfile(downloaded):
        return extract_tar(downloaded, output_dir, overwrite)

    if is_probably_html(downloaded) or "text/html" in content_type.lower():
        raise DownloadError(
            "The URL returned an HTML page rather than a data archive. "
            "Open the link in a browser, copy the direct download URL for the archive, "
            "and pass it with --url."
        )

    destination = output_dir / downloaded.name
    with downloaded.open("rb") as source:
        return int(write_file(source, destination, overwrite))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    print(f"Downloading demo data from: {args.url}")
    print(f"Destination folder: {output_dir}")

    with tempfile.TemporaryDirectory(prefix="match_phase_demo_data_") as tmp:
        tmp_dir = Path(tmp)
        downloaded, content_type = download(args.url, tmp_dir)
        print(f"Downloaded file: {downloaded.name}")
        written = unpack_or_copy(downloaded, content_type, output_dir, args.overwrite)

        if args.keep_archive:
            archive_target = output_dir / downloaded.name
            if args.overwrite or not archive_target.exists():
                shutil.copy2(downloaded, archive_target)
                print(f"Kept archive: {archive_target}")

    print(f"Done. Files written: {written}")
    if written == 0 and not args.overwrite:
        print("No new files were written. Use --overwrite to replace existing files.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
