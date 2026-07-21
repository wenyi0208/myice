#!/usr/bin/env python3
import argparse
import contextlib
import numpy as np
import netCDF4 as nc
from pathlib import Path


def load_data(pred_path: str, gt_path: str):
    """加载预测和真实数据"""
    print(f"加载预测文件: {pred_path}")
    ds_pred = nc.Dataset(pred_path, 'r')
    print(f"加载真实数据: {gt_path}")
    ds_gt = nc.Dataset(gt_path, 'r')
    return ds_pred, ds_gt


def get_valid_mask(pred: np.ndarray, gt: np.ndarray, land_mask: np.ndarray = None) -> np.ndarray:
    """获取有效像素掩码（排除陆地和NaN）"""
    mask = ~np.isnan(pred) & ~np.isnan(gt)
    if land_mask is not None:
        mask &= ~land_mask
    return mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray, land_mask: np.ndarray = None, climatology: np.ndarray = None) -> tuple:
    """Compute MAE, RMSE, and anomaly correlation coefficient (ACC)."""
    mask = get_valid_mask(pred, gt, land_mask)
    if climatology is not None:
        mask &= ~np.isnan(climatology)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan

    pred_valid = pred[mask]
    gt_valid = gt[mask]
    diff = pred_valid - gt_valid
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())

    acc = np.nan
    if climatology is not None:
        clim_valid = climatology[mask]
        pred_anom = pred_valid - clim_valid
        gt_anom = gt_valid - clim_valid
        denom = np.sqrt(np.sum(pred_anom ** 2) * np.sum(gt_anom ** 2))
        if pred_anom.size >= 2 and denom > 0:
            acc = np.sum(pred_anom * gt_anom) / denom

    return mae, rmse, acc

def get_dates(ds: nc.Dataset) -> list:
    """从NetCDF数据集获取日期列表"""
    time = ds.variables['time'][:]
    units = ds.variables['time'].units
    calendar = ds.variables['time'].calendar
    return nc.num2date(time, units, calendar=calendar)


def align_lat_range(lat_pred: np.ndarray, lat_gt: np.ndarray) -> tuple:
    """对齐两个数据集的纬度范围，返回裁剪索引"""
    common = sorted(set(lat_pred) & set(lat_gt))
    if not common:
        return None, None, None
    
    min_lat, max_lat = min(common), max(common)
    p0 = np.where(lat_pred == min_lat)[0][0]
    p1 = np.where(lat_pred == max_lat)[0][0] + 1
    g0 = np.where(lat_gt == min_lat)[0][0]
    g1 = np.where(lat_gt == max_lat)[0][0] + 1
    """P：预测数据的维度切片，g：真实数据的维度切片，min/max_lat：实际公共维度切片"""
    return (p0, p1), (g0, g1), (min_lat, max_lat)


def find_matching_indices(dates_pred: list, dates_gt: list) -> list:
    """找到预测和真实数据中对应时间的索引"""
    pred_dates = {(d.year, d.month) for d in dates_pred}
    return [i for i, d in enumerate(dates_gt) if (d.year, d.month) in pred_dates]


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    """将数据标准化到0-1范围"""
    if np.nanmax(data) > 1.5:
        return data / 100.0
    return data


def to_float_array(data: np.ndarray) -> np.ndarray:
    """Convert NetCDF masked arrays to normal arrays with NaN fill values."""
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)

def compute_monthly_climatology(siconc_gt: np.ndarray, dates_gt: list) -> dict:
    """Compute monthly climatology fields from the ground-truth dataset."""
    climatology = {}
    for month in range(1, 13):
        indices = [idx for idx, date in enumerate(dates_gt) if date.month == month]
        if indices:
            climatology[month] = np.nanmean(siconc_gt[indices], axis=0)
    return climatology


def evaluate(pred_file: str, gt_file: str, landmask_path: str = None):
    """评估预测结果，支持带lead time的预测数据"""
    ds_pred, ds_gt = load_data(pred_file, gt_file)
    
    # 加载数据
    dates_pred = get_dates(ds_pred)
    dates_gt = get_dates(ds_gt)
    siconc_pred = to_float_array(ds_pred.variables['siconc'][:])
    siconc_gt = to_float_array(ds_gt.variables['siconc'][:])
    
    print(f"\n预测: {dates_pred[0]} ~ {dates_pred[-1]} ({len(dates_pred)} 个月)")
    print(f"真实: {dates_gt[0]} ~ {dates_gt[-1]} ({len(dates_gt)} 个月)")
    
    # 检测lead time维度
    has_leadtime = siconc_pred.ndim == 4
    num_leadtime = siconc_pred.shape[1] if has_leadtime else 1
    print(f"Lead time: {num_leadtime} 个预见期")
    
    # 纬度对齐
    lat_pred, lat_gt = ds_pred.variables['lat'][:], ds_gt.variables['lat'][:]
    pred_range, gt_range, lat_range = align_lat_range(lat_pred, lat_gt)
    
    # 裁剪数据（对齐公共纬度范围）
    siconc_pred = siconc_pred[:, :, pred_range[0]:pred_range[1], :] if has_leadtime \
                  else siconc_pred[:, pred_range[0]:pred_range[1], :]
    siconc_gt = siconc_gt[:, gt_range[0]:gt_range[1], :]
    
    # 单位转换
    siconc_pred = normalize_to_01(siconc_pred)
    monthly_climatology = compute_monthly_climatology(siconc_gt, dates_gt)
    
    # 海陆掩码（先按公共纬度范围裁剪）
    land_mask = None
    if landmask_path and Path(landmask_path).exists():
        land_mask = (np.load(landmask_path) == 1)[pred_range[0]:pred_range[1], :]
    
    print(f"数据形状: 预测 {siconc_pred.shape}, 真实 {siconc_gt.shape}")
    
    # 匹配时间索引
    matched = find_matching_indices(dates_pred, dates_gt)
    if not matched:
        print("错误: 无匹配时间点!")
        return None
    
    print(f"匹配: {len(matched)} 个时间点")
    
    # 计算指标
    results = {}
    for lead_idx in range(num_leadtime):
        lead = lead_idx + 1
        mae_list, rmse_list, acc_list = [], [], []
        
        for p_idx, g_idx in zip(range(len(dates_pred)), matched):
            pred = siconc_pred[p_idx, lead_idx] if has_leadtime else siconc_pred[p_idx]
            climatology = monthly_climatology.get(dates_gt[g_idx].month)
            mae, rmse, acc = compute_metrics(pred, siconc_gt[g_idx], land_mask, climatology)
            mae_list.append(mae)
            rmse_list.append(rmse)
            acc_list.append(acc)
        
        results[lead] = {
            'mae': mae_list, 'rmse': rmse_list, 'acc': acc_list,
            'overall_mae': np.nanmean(mae_list), 'std_mae': np.nanstd(mae_list),
            'overall_rmse': np.nanmean(rmse_list), 'std_rmse': np.nanstd(rmse_list),
            'overall_acc': np.nanmean(acc_list), 'std_acc': np.nanstd(acc_list),
            'dates': [(d.year, d.month) for d in dates_pred]
        }
    
    # 计算指标 - 收集所有年月和lead time的数据
    # (year, month, lead) -> {'mae': [...], 'rmse': [...]}
    monthly_lead_stats = {}
    all_mae = []
    all_rmse = []
    all_acc = []
    
    for lead_idx in range(num_leadtime):
        lead = lead_idx + 1
        
        for p_idx, g_idx in zip(range(len(dates_pred)), matched):
            pred = siconc_pred[p_idx, lead_idx] if has_leadtime else siconc_pred[p_idx]
            climatology = monthly_climatology.get(dates_gt[g_idx].month)
            mae, rmse, acc = compute_metrics(pred, siconc_gt[g_idx], land_mask, climatology)
            
            date = (dates_pred[p_idx].year, dates_pred[p_idx].month)
            monthly_lead_stats.setdefault((date[0], date[1]), {})[lead] = (mae, rmse, acc)
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_acc.append(acc)
    
    # 输出结果
    print(f"\n========== 评估结果 (MAE) ==========")
    
    # 打印表头
    header = f"{'年月':^10}"
    for lead in range(1, num_leadtime + 1):
        header += f"{f'Lead-{lead}':^10}"
    print(header)
    print("-" * len(header))
    
    # 按年月打印表格
    for (year, month) in sorted(monthly_lead_stats.keys()):
        row = f"{year}-{month:02d}"
        for lead in range(1, num_leadtime + 1):
            mae, rmse, acc = monthly_lead_stats[(year, month)][lead]
            row += f"{mae*100:>9.2f}%"
        print(row)
    
    # 每行 lead time 的总体 MAE
    print("-" * len(header))
    row = f"{'总体':^10}"
    for lead in range(1, num_leadtime + 1):
        lead_maes = [monthly_lead_stats[(y,m)][lead][0] for (y,m) in monthly_lead_stats]
        row += f"{np.nanmean(lead_maes)*100:>9.2f}%"
    print(row)
    
    # 打印 RMSE 表格
    print(f"\n========== 评估结果 (RMSE) ==========")
    print(header)
    print("-" * len(header))
    
    for (year, month) in sorted(monthly_lead_stats.keys()):
        row = f"{year}-{month:02d}"
        for lead in range(1, num_leadtime + 1):
            mae, rmse, acc = monthly_lead_stats[(year, month)][lead]
            row += f"{rmse*100:>9.2f}%"
        print(row)
    
    print("-" * len(header))
    row = f"{'总体':^10}"
    for lead in range(1, num_leadtime + 1):
        lead_rmses = [monthly_lead_stats[(y,m)][lead][1] for (y,m) in monthly_lead_stats]
        row += f"{np.nanmean(lead_rmses)*100:>9.2f}%"
    print(row)
    
    # 打印 anomaly correlation coefficient (ACC) 表格
    print(f"\n========== 评估结果 (ACC) ==========")
    print(header)
    print("-" * len(header))
    
    for (year, month) in sorted(monthly_lead_stats.keys()):
        row = f"{year}-{month:02d}"
        for lead in range(1, num_leadtime + 1):
            mae, rmse, acc = monthly_lead_stats[(year, month)][lead]
            row += f"{acc:>10.4f}"
        print(row)
    
    print("-" * len(header))
    row = f"{'总体':^10}"
    for lead in range(1, num_leadtime + 1):
        lead_accs = [monthly_lead_stats[(y,m)][lead][2] for (y,m) in monthly_lead_stats]
        row += f"{np.nanmean(lead_accs):>10.4f}"
    print(row)
    
    # 总体统计
    print("\n========== 总体统计 ==========")
    print(f"  MAE: {np.nanmean(all_mae)*100:.2f}% ± {np.nanstd(all_mae)*100:.2f}%")
    print(f"  RMSE: {np.nanmean(all_rmse)*100:.2f}% ± {np.nanstd(all_rmse)*100:.2f}%")
    print(f"  ACC: {np.nanmean(all_acc):.4f} ± {np.nanstd(all_acc):.4f}")
    print(f"  样本数: {len(all_mae)} (lead time x 时间步)")
    
    ds_pred.close()
    ds_gt.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='评估预测结果')
    parser.add_argument('--pred-file', type=str, 
                        default='./predictions/201501_201812_TTT.nc')
    parser.add_argument('--gt-file', type=str, default='./data/oras5/siconc.nc')
    parser.add_argument('--landmask', type=str, default='./numpy/landmask.npy')
    parser.add_argument('--output-log', type=str, default='./log/gpu_775499_TTT.out',
                        help='评估结果追加写入的日志文件')
    args = parser.parse_args()

    output_log = Path(args.output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    has_content = output_log.exists() and output_log.stat().st_size > 0
    with open(output_log, "a", encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        with contextlib.redirect_stdout(f):
            evaluate(args.pred_file, args.gt_file, args.landmask)


if __name__ == '__main__':
    main()
