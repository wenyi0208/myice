#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xarray as xr


FILL_VALUE_LIMIT = 1e20


def clean_array(arr: np.ndarray, file_path: Path) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bad = (~np.isfinite(arr)) | (np.abs(arr) > FILL_VALUE_LIMIT)
    if np.any(bad):
        print(f"Warning: replace {int(bad.sum())} invalid/fill values in {file_path}")
        arr = arr.copy()
        arr[bad] = 0.0
    return arr


def validate_time_range(
    time_coord: xr.DataArray,
    file_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    expected_months = (end_year - start_year + 1) * 12
    actual_months = time_coord.size
    expected_codes = np.arange(
        start_year * 12 + 1,
        end_year * 12 + 13,
    )
    actual_codes = (
        np.asarray(time_coord.dt.year.values, dtype=np.int64) * 12
        + np.asarray(time_coord.dt.month.values, dtype=np.int64)
    )

    if actual_months != expected_months or not np.array_equal(
        actual_codes, expected_codes
    ):
        raise ValueError(
            f"{file_path} does not cover every month from "
            f"{start_year}-01 through {end_year}-12: "
            f"found {actual_months}, expected {expected_months}."
        )


def select_validation_period(
    data_var: xr.DataArray,
    file_path: Path,
    var_name: str,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    expected_months = (end_year - start_year + 1) * 12
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    time_coord_name = None
    for candidate in ("time", "time_counter"):
        if candidate in data_var.coords:
            time_coord_name = candidate
            break

    if time_coord_name is not None:
        selected = data_var.sel({time_coord_name: slice(start_date, end_date)})
        validate_time_range(
            selected[time_coord_name],
            file_path,
            start_year,
            end_year,
        )
        return np.squeeze(selected.values)

    arr = np.squeeze(data_var.values)
    if arr.ndim < 3:
        raise ValueError(
            f"{file_path} variable {var_name} has unexpected shape: {arr.shape}"
        )
    if arr.shape[0] != expected_months:
        raise ValueError(
            f"{file_path} variable {var_name} has no time/time_counter "
            f"coordinate, and its first dimension is {arr.shape[0]}, "
            f"not the expected {expected_months} months."
        )
    print(
        f"Warning: {file_path} variable {var_name} has no time/time_counter "
        f"coordinate; treating the first dimension as {start_year}-01 "
        f"through {end_year}-12."
    )
    return arr


def main():
    out_dir = Path("./numpy/val")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Channel order must agree with dataset.VAR_KEYS: zg925, zg850, zg500,
    # zg300, zg100, zg50, zg10. The NetCDF filenames use the same hPa labels.
    variables = [
        {
            "source_dir": "oras5",
            "filename": "siconc.nc",
            "var_name": "siconc",
        },
        {
            "source_dir": "oras5",
            "filename": "sithick.nc",
            "var_name": "sithick",
        },
        {
            "source_dir": "era5",
            "filename": "tas.nc",
            "var_name": "tas",
        },
        {
            "source_dir": "oras5",
            "filename": "uo_0.nc",
            "var_name": "uo",
        },
        {
            "source_dir": "oras5",
            "filename": "uo_10.nc",
            "var_name": "uo",
        },
        {
            "source_dir": "oras5",
            "filename": "vo_0.nc",
            "var_name": "vo",
        },
        {
            "source_dir": "oras5",
            "filename": "vo_10.nc",
            "var_name": "vo",
        },
        {
            "source_dir": "era5",
            "filename": "zg_925.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_850.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_500.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_300.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_100.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_50.nc",
            "var_name": "zg",
        },
        {
            "source_dir": "era5",
            "filename": "zg_10.nc",
            "var_name": "zg",
        },
    ]

    val_start_year = 1979
    val_end_year = 2025

    arrs = []
    for var in variables:
        file_path = Path("./data") / var["source_dir"] / var["filename"]
        ds = xr.open_dataset(file_path)
        data_var = ds[var["var_name"]]
        arr = select_validation_period(
            data_var,
            file_path,
            var["var_name"],
            val_start_year,
            val_end_year,
        )
        arr = clean_array(arr, file_path)

        if var["var_name"] == "siconc":
            if np.nanmax(arr) > 1.5:
                arr = arr / 100.0
            arr[arr < 0] = 0.0

        if var["var_name"] == "zg":
            arr = arr / 9.8
            arr = clean_array(arr, file_path)

        print(
            f"读取 {var['source_dir']}/{var['filename']} ({var['var_name']}) "
            f"形状: {arr.shape}, 月数: {arr.shape[0]}"
        )

        arrs.append(arr)
        ds.close()

    stacked = np.stack(arrs, axis=0).astype(np.float32)
    stacked = stacked[:, :, 1:, :]

    out_path = out_dir / "val.npy"
    np.save(out_path, stacked, allow_pickle=False)
    print(f"\n已保存验证集到 {out_path},形状 {stacked.shape}")


if __name__ == "__main__":
    main()
