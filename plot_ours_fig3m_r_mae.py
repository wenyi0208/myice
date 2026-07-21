#!/usr/bin/env python3
"""Plot an SICNet Fig. 3m-r style MAE panel for our model.

The paper's Fig. 3m-r panels show September sea-ice-concentration MAE
maps for lead months 1-6. This script computes the same style of field
from a prediction NetCDF and an observation NetCDF, then saves PNG/PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def to_float_array(data) -> np.ndarray:
    """Convert masked NetCDF data to float arrays with NaNs."""
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    """Accept either fraction data or percent data."""
    finite_max = np.nanmax(data)
    if finite_max > 1.5:
        return data / 100.0
    return data


def dates_from_dataset(ds: nc.Dataset) -> list:
    time_var = ds.variables["time"]
    calendar = getattr(time_var, "calendar", "standard")
    return list(nc.num2date(time_var[:], time_var.units, calendar=calendar))


def date_key(date) -> tuple[int, int]:
    return int(date.year), int(date.month)


def align_latitudes(pred_ds: nc.Dataset, gt_ds: nc.Dataset) -> tuple[slice, slice]:
    pred_lats = np.asarray(pred_ds.variables["lat"][:])
    gt_lats = np.asarray(gt_ds.variables["lat"][:])
    common = sorted(set(pred_lats.tolist()) & set(gt_lats.tolist()))
    if not common:
        raise ValueError("Prediction and observation latitude coordinates do not overlap.")

    min_lat, max_lat = common[0], common[-1]
    pred_start = int(np.where(pred_lats == min_lat)[0][0])
    pred_stop = int(np.where(pred_lats == max_lat)[0][0]) + 1
    gt_start = int(np.where(gt_lats == min_lat)[0][0])
    gt_stop = int(np.where(gt_lats == max_lat)[0][0]) + 1
    return slice(pred_start, pred_stop), slice(gt_start, gt_stop)


def parse_leads(value: str, available: Iterable[int]) -> list[int]:
    available = list(map(int, available))
    if value.lower() == "all":
        return available

    leads = [int(item.strip()) for item in value.split(",") if item.strip()]
    missing = [lead for lead in leads if lead not in available]
    if missing:
        raise ValueError(f"Requested lead(s) not found in prediction file: {missing}. Available: {available}")
    return leads


def load_landmask(path: str | Path | None, pred_lat_slice: slice, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path:
        return None

    landmask = np.load(path).astype(bool)
    if landmask.shape != expected_shape:
        landmask = landmask[pred_lat_slice, :]
    if landmask.shape != expected_shape:
        raise ValueError(f"Unexpected landmask shape {landmask.shape}; expected {expected_shape}.")
    return landmask


def get_gt_lookup(gt_dates: list) -> dict[tuple[int, int], int]:
    return {date_key(date): idx for idx, date in enumerate(gt_dates)}


def selected_prediction_indices(pred_dates: list, target_month: int, start_year: int | None, end_year: int | None) -> list[int]:
    indices = []
    for idx, date in enumerate(pred_dates):
        year = int(date.year)
        month = int(date.month)
        if month != target_month:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        indices.append(idx)
    return indices


def compute_lead_mae_fields(args) -> dict:
    pred_ds = nc.Dataset(args.pred_file)
    gt_ds = nc.Dataset(args.gt_file)

    try:
        pred_dates = dates_from_dataset(pred_ds)
        gt_dates = dates_from_dataset(gt_ds)
        gt_lookup = get_gt_lookup(gt_dates)

        pred_lat_slice, gt_lat_slice = align_latitudes(pred_ds, gt_ds)
        lats = np.asarray(pred_ds.variables["lat"][pred_lat_slice], dtype=np.float64)
        lons = np.asarray(pred_ds.variables["lon"][:], dtype=np.float64)
        expected_shape = (len(lats), len(lons))
        landmask = load_landmask(args.landmask, pred_lat_slice, expected_shape)

        pred_indices = selected_prediction_indices(pred_dates, args.target_month, args.start_year, args.end_year)
        if not pred_indices:
            raise ValueError("No prediction dates matched the requested target month/year range.")

        matched = []
        for pred_idx in pred_indices:
            key = date_key(pred_dates[pred_idx])
            if key in gt_lookup:
                matched.append((pred_idx, gt_lookup[key]))
        if not matched:
            raise ValueError("No target prediction dates were found in the observation file.")

        leadtime_values = np.asarray(pred_ds.variables[args.lead_dim][:], dtype=int)
        leads = parse_leads(args.leads, leadtime_values)
        lead_to_idx = {int(lead): int(np.where(leadtime_values == lead)[0][0]) for lead in leads}

        pred_var = pred_ds.variables[args.pred_var]
        gt_var = gt_ds.variables[args.gt_var]
        mae_fields = []
        sample_counts = []

        for lead in leads:
            lead_idx = lead_to_idx[lead]
            errors = []
            display_mask = np.zeros(expected_shape, dtype=bool)
            for pred_idx, gt_idx in matched:
                pred = to_float_array(pred_var[pred_idx, lead_idx, pred_lat_slice, :])
                obs = to_float_array(gt_var[gt_idx, gt_lat_slice, :])
                pred = normalize_to_01(pred)
                obs = normalize_to_01(obs)

                valid = np.isfinite(pred) & np.isfinite(obs)
                if landmask is not None:
                    valid &= ~landmask
                if not args.include_open_water:
                    display_mask |= valid & ((pred >= args.ice_threshold) | (obs >= args.ice_threshold))

                err = np.full(expected_shape, np.nan, dtype=np.float64)
                err[valid] = np.abs(pred[valid] - obs[valid]) * 100.0
                errors.append(err)

            error_stack = np.stack(errors, axis=0)
            valid_count = np.sum(np.isfinite(error_stack), axis=0)
            mae = np.full(expected_shape, np.nan, dtype=np.float64)
            np.divide(
                np.nansum(error_stack, axis=0),
                valid_count,
                out=mae,
                where=valid_count > 0,
            )
            if landmask is not None:
                mae = np.where(landmask, np.nan, mae)
            if not args.include_open_water:
                mae = np.where(display_mask, mae, np.nan)
            mae_fields.append(mae)
            sample_counts.append(len(errors))

        return {
            "lats": lats,
            "lons": lons,
            "landmask": landmask,
            "leads": leads,
            "mae_fields": mae_fields,
            "sample_counts": sample_counts,
            "matched_dates": [date_key(pred_dates[pred_idx]) for pred_idx, _ in matched],
        }
    finally:
        pred_ds.close()
        gt_ds.close()


def add_grid_labels(ax, data_proj: ccrs.CRS) -> None:
    gl = ax.gridlines(
        crs=data_proj,
        draw_labels=True,
        linewidth=0.0,
        xlocs=[-120, -100, -80, -60, 60, 80, 100, 120],
        ylocs=[50],
    )
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.xlabel_style = {"size": 7, "weight": "bold"}
    gl.ylabel_style = {"size": 7, "weight": "bold"}
    gl.x_inline = False
    gl.y_inline = False


def panel_label(index: int, start: str) -> str:
    return f"({chr(ord(start.lower()) + index)})"


def plot(args) -> Path:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.5,
            "savefig.dpi": args.dpi,
            "figure.facecolor": "white",
        }
    )

    data = compute_lead_mae_fields(args)
    lons = data["lons"]
    lats = data["lats"]
    lon2d, lat2d = np.meshgrid(lons, lats)

    cmap = mpl.colormaps[args.cmap].copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)

    ncols = len(data["leads"])
    fig = plt.figure(figsize=(ncols * 1.55, 2.85))
    gs = gridspec.GridSpec(2, ncols, height_ratios=[1.0, 0.26])
    gs.update(left=0.035, right=0.985, top=0.82, bottom=0.20, wspace=0.08, hspace=0.02)

    map_proj = ccrs.NorthPolarStereo(central_longitude=args.central_longitude)
    data_proj = ccrs.PlateCarree()

    last_mesh = None
    for col, (lead, mae_field) in enumerate(zip(data["leads"], data["mae_fields"])):
        ax = fig.add_subplot(gs[0, col], projection=map_proj)
        ax.set_extent([-180, 180, args.min_lat, 90], crs=data_proj)
        ax.set_aspect("equal")
        ax.set_facecolor(args.background_color)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        last_mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            mae_field,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=data_proj,
        )

        if data["landmask"] is not None:
            land = np.where(data["landmask"], 1.0, np.nan)
            ax.pcolormesh(
                lon2d,
                lat2d,
                land,
                cmap=mpl.colors.ListedColormap([args.land_color]),
                shading="auto",
                transform=data_proj,
            )
            ax.contour(
                lon2d,
                lat2d,
                data["landmask"].astype(float),
                levels=[0.5],
                colors=args.coastline_color,
                linewidths=args.coastline_width,
                transform=data_proj,
            )

        add_grid_labels(ax, data_proj)
        ax.set_title(f"Lead month={lead}", pad=12)
        ax.text(
            0.02,
            0.97,
            panel_label(col, args.panel_start),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    label_ax = fig.add_subplot(gs[1, : max(1, ncols // 2)])
    label_ax.axis("off")
    label_ax.text(0.0, 0.15, args.model_name, ha="left", va="bottom", fontsize=15, fontweight="bold")

    cbar_ax = fig.add_subplot(gs[1, max(1, ncols // 2) :])
    pos = cbar_ax.get_position()
    new_height = pos.height * args.colorbar_height_scale
    cbar_ax.set_position([pos.x0, pos.y0 + (pos.height - new_height) * 0.72, pos.width, new_height])
    if last_mesh is None:
        raise RuntimeError("No panels were drawn.")
    cbar = fig.colorbar(last_mesh, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("MAE (%)", fontsize=12, fontweight="bold", labelpad=0)
    cbar.set_ticks(np.arange(args.vmin, args.vmax + 0.1, args.tick_step))
    cbar.ax.tick_params(labelsize=9, width=0.5, length=2, pad=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / args.output_name
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    dates = ", ".join(f"{year:04d}-{month:02d}" for year, month in data["matched_dates"])
    print(f"Matched target dates: {dates}")
    print("Lead sample counts:", dict(zip(data["leads"], data["sample_counts"])))
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw an SICNet Fig. 3m-r style MAE map row for our model."
    )
    parser.add_argument(
        "--pred-file",
        default="predictions/200001_201812_20260709_012615_TTT.nc",
        help="Prediction NetCDF with dimensions time, leadtime, lat, lon.",
    )
    parser.add_argument("--gt-file", default="data/oras5/siconc.nc", help="Observation NetCDF file.")
    parser.add_argument("--landmask", default="numpy/landmask.npy", help="Land mask .npy file where 1 means land.")
    parser.add_argument("--pred-var", default="siconc", help="Prediction variable name.")
    parser.add_argument("--gt-var", default="siconc", help="Observation variable name.")
    parser.add_argument("--lead-dim", default="leadtime", help="Lead-time coordinate variable name.")
    parser.add_argument("--leads", default="1,2,3,4,5,6", help="Comma-separated leads, or 'all'.")
    parser.add_argument("--target-month", type=int, default=9, help="Target month to average, 9 for September.")
    parser.add_argument("--start-year", type=int, default=None, help="Optional first target year.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional last target year.")
    parser.add_argument("--model-name", default="ours", help="Row label shown at lower left.")
    parser.add_argument("--panel-start", default="m", help="First panel letter, default mimics Fig. 3m-r.")
    parser.add_argument("--min-lat", type=float, default=50.0, help="Southern latitude limit.")
    parser.add_argument("--central-longitude", type=float, default=0.0, help="Central longitude for polar projection.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Colorbar minimum.")
    parser.add_argument("--vmax", type=float, default=40.0, help="Colorbar maximum.")
    parser.add_argument("--tick-step", type=float, default=5.0, help="Colorbar tick spacing.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap.")
    parser.add_argument("--ice-threshold", type=float, default=0.15, help="SIC threshold for plotting the ice-related MAE area.")
    parser.add_argument("--include-open-water", action="store_true", help="Also plot MAE over open-water grid cells.")
    parser.add_argument("--background-color", default="0.94", help="Map background color for open-water/no-data areas.")
    parser.add_argument("--land-color", default="0.86", help="Land grid color.")
    parser.add_argument("--coastline-color", default="black", help="Land-mask coastline color.")
    parser.add_argument("--coastline-width", type=float, default=0.35, help="Land-mask coastline width.")
    parser.add_argument("--colorbar-height-scale", type=float, default=0.42, help="Relative height of the MAE colorbar.")
    parser.add_argument("--output-dir", default="figures", help="Output directory.")
    parser.add_argument("--output-name", default="ours_fig3m_r_mae", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    args = parser.parse_args()

    if not (1 <= args.target_month <= 12):
        raise SystemExit("--target-month must be between 1 and 12.")
    if len(args.panel_start) != 1 or not args.panel_start.isalpha():
        raise SystemExit("--panel-start must be one alphabetic character.")

    try:
        plot(args)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Try running: conda install -n swin -c conda-forge cartopy netCDF4"
        )


if __name__ == "__main__":
    main()
