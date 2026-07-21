#!/usr/bin/env python3
"""Print the calendar range covered by val.npy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + offset
    return total // 12, total % 12 + 1


def format_year_month(year_month: tuple[int, int]) -> str:
    year, month = year_month
    return f"{year:04d}-{month:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看 val.npy 覆盖的时间范围。默认 val.npy 第 0 个时间步为 1979-01。"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="./numpy/val/val.npy",
        help="val.npy 路径，默认: ./numpy/val/val.npy",
    )
    parser.add_argument(
        "--start-time",
        type=parse_year_month,
        default=(1979, 1),
        help="val.npy 第 0 个时间步对应的年月，格式 YYYY-MM，默认: 1979-01",
    )
    parser.add_argument(
        "--time-axis",
        type=int,
        default=1,
        help="时间维所在的轴。项目里的 val.npy 形状通常是 [变量, 时间, H, W]，默认: 1",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    ndim = arr.ndim
    time_axis = args.time_axis
    if time_axis < 0:
        time_axis += ndim
    if not 0 <= time_axis < ndim:
        raise SystemExit(f"--time-axis={args.time_axis} 超出维度范围，数组维度为 {ndim}")

    time_steps = int(arr.shape[time_axis])
    if time_steps <= 0:
        raise SystemExit("时间长度为 0，无法计算覆盖范围")

    start = args.start_time
    end = add_months(start[0], start[1], time_steps - 1)

    print(f"文件: {path}")
    print(f"形状: {arr.shape}")
    print(f"时间轴: axis {time_axis}")
    print(f"时间步数: {time_steps} 个月")
    print(f"覆盖范围: {format_year_month(start)} ~ {format_year_month(end)}")

    first_input_end = add_months(start[0], start[1], 5)
    last_input_start = add_months(start[0], start[1], time_steps - 6)
    if time_steps >= 6:
        print(f"可用 6 个月输入窗口: {format_year_month(start)}~{format_year_month(first_input_end)} 到 "
              f"{format_year_month(last_input_start)}~{format_year_month(end)}")


if __name__ == "__main__":
    main()
