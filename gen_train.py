#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xarray as xr


TARGETS = {"siconc", "sithick", "tas", "uo", "vo", "zg"}


def load_variable(file_path: Path, filename: str) -> np.ndarray:
    ds = xr.open_dataset(file_path)
    try:
        for name in ds.data_vars:
            if name in TARGETS:
                data_var = ds[name]
                break
        else:
            raise ValueError(f"No target variable found in {file_path}")

        arr = np.squeeze(data_var.values)
        arr = arr[:, 1:, :]
        arr = np.asarray(arr, dtype=np.float32)
        np.nan_to_num(arr, nan=0.0, copy=False)

        if filename == "siconc.nc":
            if np.max(arr) > 1.5:
                arr /= 100.0
            arr[arr < 0] = 0.0

        if filename == "sithick.nc":
            arr[arr > 50] = 0.0

        print(f"Selected {name} from {file_path} with shape {arr.shape}")
        return arr
    finally:
        ds.close()


def main():
    data_dir = Path("./data/cmip6")
    out_dir = Path("./numpy/train")
    
    required_files = [
        "siconc.nc",
        "sithick.nc", 
        "tas.nc",
        "uo_0.nc",
        "uo_10.nc", 
        "vo_0.nc",
        "vo_10.nc",
        "zg_925.nc",
        "zg_850.nc",
        "zg_500.nc",
        "zg_300.nc",
        "zg_100.nc",
        "zg_50.nc",
        "zg_10.nc"
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    for ds_dir in dataset_dirs:
        out_path = out_dir / (ds_dir.name + ".npy")
        stacked = None

        if out_path.exists():
            try:
                existing = np.load(out_path, mmap_mode="r", allow_pickle=False)
                if existing.shape[0] == len(required_files):
                    print(f"Skip existing {out_path} with shape {existing.shape}")
                    del existing
                    continue
                del existing
            except Exception as exc:
                print(f"Regenerate unreadable {out_path}: {exc}")

        for channel_idx, filename in enumerate(required_files):
            file_path = ds_dir / filename
            arr = load_variable(file_path, filename)

            if stacked is None:
                shape = (len(required_files),) + arr.shape
                stacked = np.lib.format.open_memmap(
                    out_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
            elif arr.shape != stacked.shape[1:]:
                raise ValueError(
                    f"Shape mismatch in {file_path}: got {arr.shape}, "
                    f"expected {stacked.shape[1:]}"
                )

            stacked[channel_idx] = arr
            stacked.flush()
            del arr

        print(f"Saved {out_path} with shape {stacked.shape}")

    landmask_path = Path("./data/target.nc")
    ds = xr.open_dataset(landmask_path)
    lm = ds["land_mask"].values
    lm = np.where(np.isnan(lm), 0, lm)
    lm = (lm != 0).astype(np.uint8)
    lm = lm[1:, :]
    landmask_out_path = Path("./numpy") / "landmask.npy"
    np.save(landmask_out_path, lm, allow_pickle=False)
    print(f"Saved landmask to {landmask_out_path} with shape {lm.shape}")
    ds.close()

if __name__ == "__main__":
    main()
