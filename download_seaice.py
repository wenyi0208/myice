#!/usr/bin/env python3
"""
Download NSIDC NOAA/NSIDC Climate Data Record monthly sea ice concentration files.

Default source:
https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/monthly/

Examples:
  python download_nsidc_g02202_monthly.py
  python download_nsidc_g02202_monthly.py --out-dir data/nsidc_monthly_north
  python download_nsidc_g02202_monthly.py --start-year 1979 --end-year 2024
  python download_nsidc_g02202_monthly.py --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html.parser
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


DEFAULT_URL = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/monthly/"
FILE_RE = re.compile(r"^sic_psn25_(\d{4})(\d{2})_.*_v06r00\.nc$", re.IGNORECASE)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def open_with_retries(
    request: urllib.request.Request,
    *,
    retries: int,
    timeout: int,
) -> urllib.response.addinfourl:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed after {retries + 1} attempts: {last_error}") from last_error


def read_index(url: str, *, retries: int, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "python-nsidc-downloader/1.0"})
    with open_with_retries(request, retries=retries, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def discover_files(
    base_url: str,
    *,
    start_year: int | None,
    end_year: int | None,
    retries: int,
    timeout: int,
) -> list[tuple[str, str, int, int]]:
    html_text = read_index(base_url, retries=retries, timeout=timeout)
    parser = LinkParser()
    parser.feed(html_text)

    files: list[tuple[str, str, int, int]] = []
    seen: set[str] = set()
    for href in parser.links:
        name = Path(urllib.parse.urlparse(href).path).name
        match = FILE_RE.match(name)
        if not match:
            continue

        year = int(match.group(1))
        month = int(match.group(2))
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue

        url = urllib.parse.urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        files.append((url, name, year, month))

    files.sort(key=lambda item: (item[2], item[3], item[1]))
    return files


def remote_size(url: str, *, retries: int, timeout: int) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "python-nsidc-downloader/1.0"},
    )
    try:
        with open_with_retries(request, retries=retries, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_one(
    item: tuple[str, str, int, int],
    *,
    out_dir: Path,
    retries: int,
    timeout: int,
    overwrite: bool,
) -> tuple[str, str]:
    url, name, _, _ = item
    final_path = out_dir / name
    part_path = out_dir / f"{name}.part"

    expected_size = remote_size(url, retries=retries, timeout=timeout)
    if final_path.exists() and not overwrite:
        if expected_size is None or final_path.stat().st_size == expected_size:
            return name, "skipped"

    if overwrite:
        part_path.unlink(missing_ok=True)

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "python-nsidc-downloader/1.0"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(url, headers=headers)
    with open_with_retries(request, retries=retries, timeout=timeout) as response:
        if resume_from > 0 and response.status != 206:
            resume_from = 0
            part_path.unlink(missing_ok=True)

        mode = "ab" if resume_from > 0 else "wb"
        with part_path.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

    if expected_size is not None and part_path.stat().st_size != expected_size:
        return name, f"incomplete ({part_path.stat().st_size}/{expected_size} bytes)"

    if final_path.exists() and overwrite:
        final_path.unlink()
    os.replace(part_path, final_path)
    return name, "downloaded"


def print_file_list(files: Iterable[tuple[str, str, int, int]]) -> None:
    for _, name, _, _ in files:
        print(name)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download monthly northern hemisphere NSIDC G02202 v6 NetCDF files.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Index URL. Default: {DEFAULT_URL}")
    parser.add_argument("--out-dir", default="nsidc_g02202_v6_north_monthly", help="Download folder.")
    parser.add_argument("--start-year", type=int, help="First year to download, inclusive.")
    parser.add_argument("--end-year", type=int, help="Last year to download, inclusive.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads. Default: 4.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per request. Default: 3.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds. Default: 60.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Only list files that would be downloaded.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        print("Error: --start-year cannot be greater than --end-year", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    files = discover_files(
        args.url,
        start_year=args.start_year,
        end_year=args.end_year,
        retries=args.retries,
        timeout=args.timeout,
    )

    if not files:
        print("No matching .nc files found.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} files.")
    if args.dry_run:
        print_file_list(files)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    failures: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(
                download_one,
                item,
                out_dir=out_dir,
                retries=args.retries,
                timeout=args.timeout,
                overwrite=args.overwrite,
            ): item[1]
            for item in files
        }

        done_count = 0
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            done_count += 1
            try:
                file_name, status = future.result()
                print(f"[{done_count}/{len(files)}] {status}: {file_name}")
                if status.startswith("incomplete"):
                    failures.append((file_name, status))
            except Exception as exc:
                print(f"[{done_count}/{len(files)}] failed: {name}: {exc}", file=sys.stderr)
                failures.append((name, str(exc)))

    if failures:
        print("\nSome files failed or were incomplete:", file=sys.stderr)
        for name, reason in failures:
            print(f"  {name}: {reason}", file=sys.stderr)
        return 1

    print(f"\nDone. Files saved in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
