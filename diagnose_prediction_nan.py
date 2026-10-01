#!/usr/bin/env python3
"""Locate input channels that cause NaN predictions near a time boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VAR_KEYS = [
    "siconc",
    "sithick",
    "tas",
    "uo0",
    "uo10",
    "vo0",
    "vo10",
    "zg925",
    "zg850",
    "zg500",
    "zg300",
    "zg100",
    "zg50",
    "zg10",
]


def parse_year_month(value: str) -> tuple[int, int]:
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        year = int(year_text)
        month = int(month_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must use YYYY-MM format") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError("month must be between 01 and 12")
    return year, month


def month_index(
    year_month: tuple[int, int],
    start_year_month: tuple[int, int],
) -> int:
    year, month = year_month
    start_year, start_month = start_year_month
    return (year - start_year) * 12 + (month - start_month)


def add_months(year_month: tuple[int, int], offset: int) -> tuple[int, int]:
    year, month = year_month
    total = year * 12 + month - 1 + offset
    return total // 12, total % 12 + 1


def format_year_month(year_month: tuple[int, int]) -> str:
    return f"{year_month[0]:04d}-{year_month[1]:02d}"


def array_stats(data: np.ndarray, extreme_limit: float) -> dict[str, float | int]:
    data = np.asarray(data)
    finite = np.isfinite(data)
    finite_values = data[finite]
    result: dict[str, float | int] = {
        "size": int(data.size),
        "nan": int(np.isnan(data).sum()),
        "posinf": int(np.isposinf(data).sum()),
        "neginf": int(np.isneginf(data).sum()),
        "finite": int(finite.sum()),
        "extreme": 0,
        "zero": 0,
        "min": np.nan,
        "max": np.nan,
        "maxabs": np.nan,
    }
    if finite_values.size:
        absolute = np.abs(finite_values)
        result.update(
            {
                "extreme": int((absolute > extreme_limit).sum()),
                "zero": int((finite_values == 0).sum()),
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "maxabs": float(absolute.max()),
            }
        )
    return result


def is_suspicious(stats: dict[str, float | int]) -> bool:
    return bool(
        stats["nan"]
        or stats["posinf"]
        or stats["neginf"]
        or stats["extreme"]
    )


def print_stats(
    name: str,
    stats: dict[str, float | int],
    *,
    marker: bool = True,
) -> None:
    size = int(stats["size"])
    finite_pct = float(stats["finite"]) / size * 100 if size else 0.0
    prefix = "!!" if marker and is_suspicious(stats) else "  "
    print(
        f"{prefix} {name:8s} "
        f"finite={finite_pct:7.3f}% "
        f"nan={stats['nan']:7d} "
        f"inf={int(stats['posinf']) + int(stats['neginf']):7d} "
        f"extreme={stats['extreme']:7d} "
        f"min={stats['min']:.6g} "
        f"max={stats['max']:.6g} "
        f"maxabs={stats['maxabs']:.6g}"
    )


def select_stats_dataset(
    climate: dict,
    normalize: dict,
    requested: str | None,
) -> str:
    common = [name for name in climate if name in normalize]
    if requested is not None:
        if requested not in common:
            raise KeyError(
                f"stats dataset {requested!r} is unavailable; candidates: {common}"
            )
        return requested
    if not common:
        raise KeyError("climate.npy and normalize.json have no common dataset")
    if len(common) > 1:
        print(
            f"Warning: multiple statistics datasets found: {common}; "
            f"using {common[0]!r}. Pass --stats-ds-name to select explicitly."
        )
    return common[0]


def normalized_channel(
    arr: np.ndarray,
    channel_index: int,
    time_index: int,
    calendar_month: int,
    climate_stats: dict,
    normalize_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arr[channel_index, time_index], dtype=np.float32)
    if channel_index == 0:
        before_fill = raw
    else:
        key = VAR_KEYS[channel_index]
        clim = np.asarray(climate_stats[key][calendar_month - 1])
        params = normalize_stats[key]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            normalized = (raw - clim - params["mean"]) / params["std"]
            if params["max"] > 1e-8:
                normalized = normalized / params["max"]
        # PreDataset writes into a float32 sample before calling nan_to_num.
        before_fill = np.asarray(normalized, dtype=np.float32)

    after_fill = np.nan_to_num(before_fill, nan=0.0)
    return before_fill, after_fill


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw and normalized validation inputs around the first month "
            "where newly appended data begin."
        )
    )
    parser.add_argument("--val-file", default="./numpy/val/val.npy")
    parser.add_argument("--climate-file", default="./numpy/climate.npy")
    parser.add_argument("--normalize-file", default="./numpy/normalize.json")
    parser.add_argument(
        "--val-start-time",
        type=parse_year_month,
        default=(1979, 1),
        help="calendar month represented by val.npy time index 0",
    )
    parser.add_argument(
        "--boundary-time",
        type=parse_year_month,
        default=(2019, 1),
        help="first month of newly appended data",
    )
    parser.add_argument("--stats-ds-name", default=None)
    parser.add_argument(
        "--raw-limit",
        type=float,
        default=1e10,
        help="absolute raw value above which a cell is flagged",
    )
    parser.add_argument(
        "--normalized-limit",
        type=float,
        default=1e4,
        help="absolute normalized value above which a cell is flagged",
    )
    parser.add_argument(
        "--scan-months",
        type=int,
        default=12,
        help="number of months to scan from the boundary",
    )
    args = parser.parse_args()

    val_path = Path(args.val_file)
    climate_path = Path(args.climate_file)
    normalize_path = Path(args.normalize_file)
    for path in (val_path, climate_path, normalize_path):
        if not path.exists():
            raise SystemExit(f"File does not exist: {path}")

    arr = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 4:
        raise SystemExit(
            f"Expected val.npy shape [variable,time,H,W], got {arr.shape}"
        )
    if arr.shape[0] != len(VAR_KEYS):
        raise SystemExit(
            f"Expected {len(VAR_KEYS)} variables {VAR_KEYS}, got shape {arr.shape}"
        )

    climate = np.load(climate_path, allow_pickle=True).item()
    with normalize_path.open("r", encoding="utf-8") as file:
        normalize = json.load(file)
    stats_name = select_stats_dataset(
        climate, normalize, args.stats_ds_name
    )
    climate_stats = climate[stats_name]
    normalize_stats = normalize[stats_name]

    boundary_index = month_index(args.boundary_time, args.val_start_time)
    if not 0 <= boundary_index < arr.shape[1]:
        raise SystemExit(
            f"Boundary {format_year_month(args.boundary_time)} maps to index "
            f"{boundary_index}, outside val.npy time length {arr.shape[1]}"
        )

    print("Validation input diagnostic")
    print(f"  val file:       {val_path}")
    print(f"  shape:          {arr.shape}")
    print(f"  val start:      {format_year_month(args.val_start_time)}")
    print(f"  boundary:       {format_year_month(args.boundary_time)}")
    print(f"  boundary index: {boundary_index}")
    print(f"  stats dataset:  {stats_name}")

    print("\n1. Raw val.npy values around the boundary")
    raw_suspicious_channels: set[str] = set()
    for offset in (-1, 0, 1):
        time_index = boundary_index + offset
        if not 0 <= time_index < arr.shape[1]:
            continue
        date = add_months(args.val_start_time, time_index)
        print(f"\n[{format_year_month(date)}] time_index={time_index}")
        for channel_index, key in enumerate(VAR_KEYS):
            stats = array_stats(
                arr[channel_index, time_index], args.raw_limit
            )
            print_stats(key, stats)
            if is_suspicious(stats):
                raw_suspicious_channels.add(key)

    print("\n2. Normalized values before/after the first new-data month")
    # A six-month window immediately before the boundary ends at boundary-1.
    # The next window is the first one that includes the boundary month.
    window_starts = [
        ("last window before boundary", boundary_index - 6),
        ("first window including boundary", boundary_index - 5),
    ]
    normalized_suspicious_channels: set[str] = set()
    for label, window_start in window_starts:
        if window_start < 0 or window_start + 6 > arr.shape[1]:
            print(f"\n[{label}] unavailable: start_idx={window_start}")
            continue
        first_date = add_months(args.val_start_time, window_start)
        last_date = add_months(args.val_start_time, window_start + 5)
        print(
            f"\n[{label}] start_idx={window_start}, "
            f"{format_year_month(first_date)}..{format_year_month(last_date)}"
        )
        for channel_index, key in enumerate(VAR_KEYS):
            before_parts = []
            after_parts = []
            for time_index in range(window_start, window_start + 6):
                date = add_months(args.val_start_time, time_index)
                before, after = normalized_channel(
                    arr,
                    channel_index,
                    time_index,
                    date[1],
                    climate_stats,
                    normalize_stats,
                )
                before_parts.append(before)
                after_parts.append(after)
            before_window = np.stack(before_parts)
            after_window = np.stack(after_parts)
            before_stats = array_stats(
                before_window, args.normalized_limit
            )
            after_stats = array_stats(
                after_window, args.normalized_limit
            )
            print_stats(f"{key}:pre", before_stats)
            if is_suspicious(before_stats) or is_suspicious(after_stats):
                normalized_suspicious_channels.add(key)
                if not is_suspicious(before_stats):
                    print_stats(f"{key}:post", after_stats)

    print(
        f"\n3. Scan from {format_year_month(args.boundary_time)} "
        f"for {args.scan_months} month(s)"
    )
    first_issue: dict[str, str] = {}
    scan_end = min(
        arr.shape[1], boundary_index + max(0, args.scan_months)
    )
    for time_index in range(boundary_index, scan_end):
        date = add_months(args.val_start_time, time_index)
        for channel_index, key in enumerate(VAR_KEYS):
            before, after = normalized_channel(
                arr,
                channel_index,
                time_index,
                date[1],
                climate_stats,
                normalize_stats,
            )
            before_stats = array_stats(before, args.normalized_limit)
            after_stats = array_stats(after, args.normalized_limit)
            if (
                is_suspicious(before_stats)
                or is_suspicious(after_stats)
            ) and key not in first_issue:
                first_issue[key] = format_year_month(date)

    if first_issue:
        for key in VAR_KEYS:
            if key in first_issue:
                print(f"!! {key:8s} first suspicious month: {first_issue[key]}")
    else:
        print("No NaN, Inf, or extreme normalized values found in this scan.")

    suspects = sorted(
        raw_suspicious_channels | normalized_suspicious_channels | set(first_issue)
    )
    print("\nConclusion")
    if suspects:
        print("  Suspicious input channel(s): " + ", ".join(suspects))
        print(
            "  '!!' marks the channel/month that should be checked in its "
            "source NetCDF and preprocessing step."
        )
        raise SystemExit(2)
    print(
        "  No input channel exceeded the configured limits. In that case, "
        "inspect the prediction NetCDF or lower --normalized-limit."
    )


if __name__ == "__main__":
    main()
