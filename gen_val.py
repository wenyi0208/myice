#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    out_dir = Path("./numpy/val")
    out_dir.mkdir(parents=True, exist_ok=True)

    # VAR_KEYS = ["siconc", "sithick", "tas", "uo0", "uo10", "vo0", "vo10", "zg5000"]
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
            "filename": "zg_5000.nc",
            "var_name": "zg",
        },
    ]

    source_start_year = 1958
    val_start_year = 1979
    val_end_year = 2020
    target_time_steps = (val_end_year - val_start_year + 1) * 12
    start_idx = (val_start_year - source_start_year) * 12

    arrs = []
    for var in variables:
        file_path = Path("./data") / var["source_dir"] / var["filename"]
        ds = xr.open_dataset(file_path)
        data_var = ds[var["var_name"]]
        arr = np.squeeze(data_var.values)

        arr = np.nan_to_num(arr, nan=0.0)

        end_idx = start_idx + target_time_steps
        if arr.shape[0] < end_idx:
            raise ValueError(
                f"{file_path} 时间长度不足: 需要至少 {end_idx} 个月 "
                f"({val_start_year}-01 到 {val_end_year}-12), 实际 {arr.shape[0]} 个月"
            )
        arr = arr[start_idx:end_idx, :, :]

        if var["var_name"] == "siconc":
            if np.nanmax(arr) > 1.5:
                arr = arr / 100.0
            arr[arr < 0] = 0.0

        if var["var_name"] == "zg":
            arr = arr / 9.8

        print(f"读取 {var['source_dir']}/{var['filename']} ({var['var_name']}) "
              f"形状: {arr.shape}")

        arrs.append(arr)
        ds.close()

    stacked = np.stack(arrs, axis=0).astype(np.float32)
    stacked = stacked[:, :, 1:, :]

    out_path = out_dir / "val.npy"
    np.save(out_path, stacked, allow_pickle=False)
    print(f"\n已保存验证集到 {out_path},形状 {stacked.shape}")

if __name__ == "__main__":
    main()
