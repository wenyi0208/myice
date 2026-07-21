#!/usr/bin/env python3
"""Draw an IceNet Fig. 7 style ice-edge forecast figure for this project.

The paper's Fig. 7 has two rows:
  1. Predicted sea-ice probability/concentration with predicted and observed
     ice edges plus their binary edge-error region.
  2. A three-class forecast region map: confident open water, ice-edge region,
     and confident ice.

This script accepts the current deterministic SIC NetCDF output as input. The
default lower-row classification uses an SIC band around the ice threshold. If
your model output is a calibrated probability of ice occurrence, pass that
variable with --pred-var and add --region-mode probability to use the paper's
probability interval definition.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import netCDF4 as nc
import numpy as np
from scipy.ndimage import zoom


def to_float_array(data) -> np.ndarray:
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data.astype(np.float64)
    if np.nanmax(finite) > 1.5:
        return data / 100.0
    return data


def dates_from_dataset(ds: nc.Dataset) -> list:
    time_var = ds.variables["time"]
    calendar_name = getattr(time_var, "calendar", "standard")
    return list(nc.num2date(time_var[:], time_var.units, calendar=calendar_name))


def date_key(date) -> tuple[int, int]:
    return int(date.year), int(date.month)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def align_latitudes(pred_ds: nc.Dataset, obs_ds: nc.Dataset) -> tuple[slice, slice]:
    pred_lats = np.asarray(pred_ds.variables["lat"][:], dtype=np.float64)
    obs_lats = np.asarray(obs_ds.variables["lat"][:], dtype=np.float64)
    common = sorted(set(pred_lats.tolist()) & set(obs_lats.tolist()))
    if not common:
        raise ValueError("Prediction and observation latitude coordinates do not overlap.")

    min_lat, max_lat = common[0], common[-1]
    pred_start = int(np.where(pred_lats == min_lat)[0][0])
    pred_stop = int(np.where(pred_lats == max_lat)[0][0]) + 1
    obs_start = int(np.where(obs_lats == min_lat)[0][0])
    obs_stop = int(np.where(obs_lats == max_lat)[0][0]) + 1
    return slice(pred_start, pred_stop), slice(obs_start, obs_stop)


def load_landmask(path: str | Path | None, lat_slice: slice, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path:
        return None
    landmask = np.load(path).astype(bool)
    if landmask.shape != expected_shape:
        landmask = landmask[lat_slice, :]
    if landmask.shape != expected_shape:
        raise ValueError(f"Unexpected landmask shape {landmask.shape}; expected {expected_shape}.")
    return landmask


def infer_year(pred_dates: Iterable, months: list[int]) -> int:
    available: dict[int, set[int]] = {}
    for date in pred_dates:
        available.setdefault(int(date.year), set()).add(int(date.month))
    years = [year for year, year_months in available.items() if all(month in year_months for month in months)]
    if not years:
        month_text = ",".join(f"{month:02d}" for month in months)
        raise ValueError(f"No prediction year contains all requested months: {month_text}.")
    return max(years)


def variable_to_probability(var, time_idx: int, lead_idx: int | None, lat_slice: slice) -> np.ndarray:
    if len(var.dimensions) == 4:
        arr = to_float_array(var[time_idx, lead_idx, lat_slice, :])
    elif len(var.dimensions) == 3:
        arr = to_float_array(var[time_idx, lat_slice, :])
    else:
        raise ValueError(f"Expected a 3D or 4D variable, got dimensions {var.dimensions}.")
    return np.clip(normalize_to_01(arr), 0.0, 1.0)


def make_cyclic(field: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field_cyclic, lons_cyclic = add_cyclic_point(field, coord=lons)
    return field_cyclic, lons_cyclic


def resample_for_plot(values: np.ndarray, scale: int, order: int) -> np.ndarray:
    if scale <= 1:
        return values
    values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    return zoom(values, (scale, scale), order=order)


def add_land(ax, lon2d, lat2d, landmask, data_proj, args) -> None:
    if landmask is None:
        return

    land = np.where(landmask, 1.0, np.nan)
    ax.pcolormesh(
        lon2d,
        lat2d,
        land,
        cmap=mpl.colors.ListedColormap([args.land_color]),
        shading="auto",
        transform=data_proj,
        zorder=4,
    )

def draw_continuous_edge(
    ax,
    lon2d,
    lat2d,
    values,
    level,
    color,
    width,
    data_proj,
    zorder,
    path_effects=None,
    linestyle="-",
) -> None:
    values = values.astype(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not (np.nanmin(finite) < level < np.nanmax(finite)):
        return
    contour = ax.contour(
        lon2d,
        lat2d,
        values,
        levels=[level],
        colors=color,
        linewidths=width,
        linestyles=linestyle,
        transform=data_proj,
        zorder=zorder,
    )
    if path_effects:
        if hasattr(contour, "collections"):
            for collection in contour.collections:
                collection.set_path_effects(path_effects)
        elif hasattr(contour, "set_path_effects"):
            contour.set_path_effects(path_effects)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.985,
        0.015,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.7},
        zorder=20,
    )


def setup_map_axis(ax, args, data_proj) -> None:
    ax.set_extent([-180, 180, args.min_lat, 90], crs=data_proj)
    ax.set_facecolor(args.open_water_color)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def read_panels(args) -> dict:
    pred_ds = nc.Dataset(args.pred_file)
    obs_ds = nc.Dataset(args.obs_file)
    try:
        pred_dates = dates_from_dataset(pred_ds)
        obs_dates = dates_from_dataset(obs_ds)
        obs_lookup = {date_key(date): idx for idx, date in enumerate(obs_dates)}

        months = parse_int_list(args.months)
        if len(months) != args.ncols:
            raise ValueError(f"--months must contain exactly {args.ncols} month values.")
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError("--months values must be between 1 and 12.")

        year = args.year if args.year is not None else infer_year(pred_dates, months)

        pred_lat_slice, obs_lat_slice = align_latitudes(pred_ds, obs_ds)
        lats = np.asarray(pred_ds.variables["lat"][pred_lat_slice], dtype=np.float64)
        lons = np.asarray(pred_ds.variables["lon"][:], dtype=np.float64)
        expected_shape = (len(lats), len(lons))
        landmask = load_landmask(args.landmask, pred_lat_slice, expected_shape)

        lead_values = np.asarray(pred_ds.variables[args.lead_dim][:], dtype=int)
        if args.lead not in lead_values:
            raise ValueError(f"Lead {args.lead} is not available. Available leads: {lead_values.tolist()}")
        lead_idx = int(np.where(lead_values == args.lead)[0][0])

        pred_var = pred_ds.variables[args.pred_var]
        obs_var = obs_ds.variables[args.obs_var]
        pred_lookup = {date_key(date): idx for idx, date in enumerate(pred_dates)}

        panels = []
        for month in months:
            key = (year, month)
            if key not in pred_lookup:
                raise ValueError(f"Prediction file does not contain {year:04d}-{month:02d}.")
            if key not in obs_lookup:
                raise ValueError(f"Observation file does not contain {year:04d}-{month:02d}.")

            pred_idx = pred_lookup[key]
            obs_idx = obs_lookup[key]
            pred = variable_to_probability(pred_var, pred_idx, lead_idx, pred_lat_slice)
            obs = normalize_to_01(to_float_array(obs_var[obs_idx, obs_lat_slice, :]))

            valid = np.isfinite(pred) & np.isfinite(obs)
            if landmask is not None:
                valid &= ~landmask

            pred = np.where(valid, pred, np.nan)
            obs = np.where(valid, obs, np.nan)
            pred_ice = pred >= args.ice_threshold
            obs_ice = obs >= args.ice_threshold
            edge_error = valid & (pred_ice != obs_ice)

            lower = args.edge_lower
            if args.region_mode == "probability":
                upper = args.edge_upper if args.edge_upper is not None else 1.0 - lower
            else:
                upper = args.edge_upper if args.edge_upper is not None else args.ice_threshold
            if lower > upper:
                raise ValueError(
                    f"Invalid ice-edge region [{lower:.3f}, {upper:.3f}] for mode {args.region_mode}."
                )
            region = np.full(expected_shape, np.nan, dtype=np.float64)
            region[valid & (pred < lower)] = 0.0
            region[valid & (pred >= lower) & (pred <= upper)] = 1.0
            region[valid & (pred > upper)] = 2.0

            panels.append(
                {
                    "year": year,
                    "month": month,
                    "pred": pred,
                    "obs": obs,
                    "edge_error": edge_error,
                    "region": region,
                }
            )

        return {
            "lats": lats,
            "lons": lons,
            "landmask": landmask,
            "lead": int(args.lead),
            "region_mode": args.region_mode,
            "edge_lower": float(args.edge_lower),
            "edge_upper": float(args.edge_upper if args.edge_upper is not None else upper),
            "panels": panels,
        }
    finally:
        pred_ds.close()
        obs_ds.close()


def plot(args) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "savefig.dpi": args.dpi,
            "figure.facecolor": "white",
        }
    )

    data = read_panels(args)
    lons = data["lons"]
    lats = data["lats"]
    lon2d, lat2d = np.meshgrid(lons, lats)

    land_c = None
    if data["landmask"] is not None:
        land_c, lon_c = make_cyclic(data["landmask"].astype(float), lons)
    else:
        lon_c = np.r_[lons, lons[-1] + (lons[1] - lons[0])]
    lon2d_c, lat2d_c = np.meshgrid(lon_c, lats)
    landmask_c = land_c.astype(bool) if land_c is not None else None

    sip_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sip_ice_style",
        [args.open_water_color, "#2b83ba", "#abd9e9", "#f7fbff", "#ffffff"],
    )
    sip_cmap.set_bad((1, 1, 1, 0))
    region_cmap = mpl.colors.ListedColormap(
        [args.open_water_color, args.edge_region_color, args.confident_ice_color]
    )
    region_cmap.set_bad((1, 1, 1, 0))
    region_norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], region_cmap.N)
    error_cmap = mpl.colors.ListedColormap([args.error_color])
    error_cmap.set_bad((1, 1, 1, 0))

    data_proj = ccrs.PlateCarree()
    map_proj = ccrs.NorthPolarStereo(central_longitude=args.central_longitude)

    fig = plt.figure(figsize=(8.2, 4.35))
    gs = gridspec.GridSpec(
        2,
        args.ncols + 1,
        width_ratios=[1.0] * args.ncols + [0.085],
        hspace=0.105,
        wspace=0.055,
        left=0.055,
        right=0.945,
        top=0.89,
        bottom=0.12,
    )

    last_sip_mesh = None
    last_region_mesh = None
    letters = [chr(ord("a") + i) for i in range(args.ncols * 2)]

    for col, panel in enumerate(data["panels"]):
        pred_c, _ = make_cyclic(panel["pred"], lons)
        obs_c, _ = make_cyclic(panel["obs"], lons)
        error_c, _ = make_cyclic(np.where(panel["edge_error"], 1.0, np.nan), lons)
        region_c, _ = make_cyclic(panel["region"], lons)
        pred_edge = resample_for_plot(pred_c, args.render_scale, 1)
        obs_edge = resample_for_plot(obs_c, args.render_scale, 1)
        lon_edge = np.linspace(float(lon_c[0]), float(lon_c[-1]), pred_edge.shape[1])
        lat_edge = np.linspace(float(lats[0]), float(lats[-1]), pred_edge.shape[0])
        lon2d_edge, lat2d_edge = np.meshgrid(lon_edge, lat_edge)

        ax = fig.add_subplot(gs[0, col], projection=map_proj)
        setup_map_axis(ax, args, data_proj)
        last_sip_mesh = ax.pcolormesh(
            lon2d_c,
            lat2d_c,
            pred_c,
            cmap=sip_cmap,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            transform=data_proj,
            zorder=1,
        )
        ax.pcolormesh(
            lon2d_c,
            lat2d_c,
            error_c,
            cmap=error_cmap,
            shading="auto",
            transform=data_proj,
            zorder=8,
            alpha=args.error_alpha,
        )
        add_land(ax, lon2d_c, lat2d_c, landmask_c, data_proj, args)
        draw_continuous_edge(
            ax,
            lon2d_edge,
            lat2d_edge,
            obs_edge,
            args.ice_threshold,
            args.obs_edge_color,
            1.25,
            data_proj,
            12,
        )
        draw_continuous_edge(
            ax,
            lon2d_edge,
            lat2d_edge,
            pred_edge,
            args.ice_threshold,
            args.pred_edge_color,
            1.35,
            data_proj,
            13,
            [pe.Stroke(linewidth=2.0, foreground="0.25"), pe.Normal()],
            linestyle="--",
        )
        ax.set_title(
            f"Forecast month: {calendar.month_abbr[panel['month']]} {panel['year']}\n"
            f"Leadtime = {data['lead']} month",
            pad=5,
        )
        panel_label(ax, letters[col])

        if col == 0:
            legend = ax.legend(
                handles=[
                    Patch(facecolor=args.error_color, edgecolor="none", label="Ice edge error region"),
                    Line2D([0], [0], color=args.obs_edge_color, lw=1.5, label="Observed ice edge"),
                    Line2D([0], [0], color=args.pred_edge_color, lw=1.7, linestyle="--", label="Predicted ice edge"),
                ],
                loc="lower left",
                fontsize=6,
                frameon=True,
                framealpha=0.82,
                borderpad=0.3,
                handlelength=2.2,
            )
            legend.get_frame().set_linewidth(0.3)

        ax2 = fig.add_subplot(gs[1, col], projection=map_proj)
        setup_map_axis(ax2, args, data_proj)
        last_region_mesh = ax2.pcolormesh(
            lon2d_c,
            lat2d_c,
            region_c,
            cmap=region_cmap,
            norm=region_norm,
            shading="auto",
            transform=data_proj,
            zorder=1,
        )
        add_land(ax2, lon2d_c, lat2d_c, landmask_c, data_proj, args)
        draw_continuous_edge(
            ax2,
            lon2d_edge,
            lat2d_edge,
            obs_edge,
            args.ice_threshold,
            args.obs_edge_color,
            1.25,
            data_proj,
            12,
        )
        panel_label(ax2, letters[col + args.ncols])

        if col == 0:
            legend = ax2.legend(
                handles=[Line2D([0], [0], color=args.obs_edge_color, lw=1.5, label="Observed ice edge")],
                loc="lower left",
                fontsize=6,
                frameon=True,
                framealpha=0.82,
                borderpad=0.3,
                handlelength=2.2,
            )
            legend.get_frame().set_linewidth(0.3)

    if last_sip_mesh is None or last_region_mesh is None:
        raise RuntimeError("No panels were drawn.")

    cax1 = fig.add_subplot(gs[0, -1])
    cbar1 = fig.colorbar(last_sip_mesh, cax=cax1)
    cbar1.set_label(args.probability_label, rotation=0, labelpad=24, va="center")
    cbar1.set_ticks(np.linspace(0, 1, 6))
    cbar1.ax.tick_params(length=2.5, width=0.6)

    cax2 = fig.add_subplot(gs[1, -1])
    cbar2 = fig.colorbar(last_region_mesh, cax=cax2, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(
        ["Confident\nopen water\nregion", "Ice edge\nregion", "Confident ice\nregion"]
    )
    cbar2.ax.tick_params(length=0, pad=8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / args.output_name
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    target_text = ", ".join(
        f"{panel['year']:04d}-{panel['month']:02d}" for panel in data["panels"]
    )
    print(f"Target months: {target_text}")
    print(f"Ice threshold: {args.ice_threshold:.3f}")
    print(f"Region mode: {data['region_mode']}")
    print(f"Ice-edge region: [{data['edge_lower']:.3f}, {data['edge_upper']:.3f}]")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an IceNet Fig. 7 style forecast figure.")
    parser.add_argument("--pred-file", default="predictions/200001_201812_20260709_012615_TTT.nc")
    parser.add_argument("--obs-file", default="data/oras5/siconc.nc")
    parser.add_argument("--landmask", default="numpy/landmask.npy")
    parser.add_argument("--pred-var", default="siconc")
    parser.add_argument("--obs-var", default="siconc")
    parser.add_argument("--lead-dim", default="leadtime")
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--year", type=int, default=None, help="Default: latest year with all requested months.")
    parser.add_argument("--months", default="7,8,9", help="Comma-separated forecast months.")
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--ice-threshold", type=float, default=0.15)
    parser.add_argument("--edge-lower", type=float, default=0.036)
    parser.add_argument("--edge-upper", type=float, default=None)
    parser.add_argument(
        "--region-mode",
        choices=["sic", "probability"],
        default="sic",
        help=(
            "sic: edge region is [edge-lower, ice-threshold] by default. "
            "probability: edge region is [edge-lower, 1-edge-lower] by default, as in the paper."
        ),
    )
    parser.add_argument("--min-lat", type=float, default=50.0)
    parser.add_argument("--central-longitude", type=float, default=0.0)
    parser.add_argument("--probability-label", default="SIC")
    parser.add_argument("--open-water-color", default="#08306b")
    parser.add_argument("--confident-ice-color", default="#f7fbff")
    parser.add_argument("--edge-region-color", default="#74c476")
    parser.add_argument("--error-color", default="#ff7f0e")
    parser.add_argument("--error-alpha", type=float, default=0.95)
    parser.add_argument("--land-color", default="0.55")
    parser.add_argument("--coastline-color", default="white")
    parser.add_argument("--coastline-width", type=float, default=0.25)
    parser.add_argument("--obs-edge-color", default="black")
    parser.add_argument("--pred-edge-color", default="white")
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-name", default="my_model_fig7_ice_edge")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not 0.0 <= args.ice_threshold <= 1.0:
        raise SystemExit("--ice-threshold must be in [0, 1].")
    if not 0.0 <= args.edge_lower <= 1.0:
        raise SystemExit("--edge-lower must be in [0, 1].")
    if args.edge_upper is not None and not 0.0 <= args.edge_upper <= 1.0:
        raise SystemExit("--edge-upper must be in [0, 1].")
    if args.edge_upper is not None and args.edge_lower > args.edge_upper:
        raise SystemExit("--edge-lower cannot be larger than --edge-upper.")
    if args.render_scale < 1:
        raise SystemExit("--render-scale must be at least 1.")

    try:
        plot(args)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Try running: conda install -n swin -c conda-forge cartopy netCDF4"
        )


if __name__ == "__main__":
    main()
