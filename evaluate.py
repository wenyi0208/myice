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


def compute_metrics(pred: np.ndarray, gt: np.ndarray, land_mask: np.ndarray = None) -> tuple:
    """计算MAE和RMSE"""
    mask = get_valid_mask(pred, gt, land_mask)
    if mask.sum() == 0:
        return np.nan, np.nan
    
    diff = pred[mask] - gt[mask]
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    return mae, rmse


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


def evaluate(pred_file: str, gt_file: str, landmask_path: str = None):
    """评估预测结果，支持带lead time的预测数据"""
    ds_pred, ds_gt = load_data(pred_file, gt_file)
    
    # 加载数据
    dates_pred = get_dates(ds_pred)
    dates_gt = get_dates(ds_gt)
    siconc_pred = ds_pred.variables['siconc'][:]
    siconc_gt = ds_gt.variables['siconc'][:]
    
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
    
    # 海陆掩码（先按公共纬度范围裁剪）
    land_mask = None
    if landmask_path and Path(landmask_path).exists():
        land_mask = (np.load(landmask_path) == 1)[pred_range[0]:pred_range[1], :]
    
    # nh_lat_count = 60
    # if has_leadtime:
    #     siconc_pred = siconc_pred[:, :, -nh_lat_count:, :]
    # else:
    #     siconc_pred = siconc_pred[:, -nh_lat_count:, :]
    # siconc_gt = siconc_gt[:, -nh_lat_count:, :]
    # if land_mask is not None:
    #     land_mask = land_mask[-nh_lat_count:, :]
    
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
        mae_list, rmse_list = [], []
        
        for p_idx, g_idx in zip(range(len(dates_pred)), matched):
            pred = siconc_pred[p_idx, lead_idx] if has_leadtime else siconc_pred[p_idx]
            mae, rmse = compute_metrics(pred, siconc_gt[g_idx], land_mask)
            mae_list.append(mae)
            rmse_list.append(rmse)
        
        results[lead] = {
            'mae': mae_list, 'rmse': rmse_list,
            'overall_mae': np.mean(mae_list), 'std_mae': np.std(mae_list),
            'overall_rmse': np.mean(rmse_list), 'std_rmse': np.std(rmse_list),
            'dates': [(d.year, d.month) for d in dates_pred]
        }
    
    # 计算指标 - 收集所有年月和lead time的数据
    # (year, month, lead) -> {'mae': [...], 'rmse': [...]}
    monthly_lead_stats = {}
    all_mae = []
    all_rmse = []
    
    for lead_idx in range(num_leadtime):
        lead = lead_idx + 1
        
        for p_idx, g_idx in zip(range(len(dates_pred)), matched):
            pred = siconc_pred[p_idx, lead_idx] if has_leadtime else siconc_pred[p_idx]
            mae, rmse = compute_metrics(pred, siconc_gt[g_idx], land_mask)
            
            date = (dates_pred[p_idx].year, dates_pred[p_idx].month)
            monthly_lead_stats.setdefault((date[0], date[1]), {})[lead] = (mae, rmse)
            all_mae.append(mae)
            all_rmse.append(rmse)
    
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
            mae, rmse = monthly_lead_stats[(year, month)][lead]
            row += f"{mae*100:>9.2f}%"
        print(row)
    
    # 每行 lead time 的总体 MAE
    print("-" * len(header))
    row = f"{'总体':^10}"
    for lead in range(1, num_leadtime + 1):
        lead_maes = [monthly_lead_stats[(y,m)][lead][0] for (y,m) in monthly_lead_stats]
        row += f"{np.mean(lead_maes)*100:>9.2f}%"
    print(row)
    
    # 打印 RMSE 表格
    print(f"\n========== 评估结果 (RMSE) ==========")
    print(header)
    print("-" * len(header))
    
    for (year, month) in sorted(monthly_lead_stats.keys()):
        row = f"{year}-{month:02d}"
        for lead in range(1, num_leadtime + 1):
            mae, rmse = monthly_lead_stats[(year, month)][lead]
            row += f"{rmse*100:>9.2f}%"
        print(row)
    
    print("-" * len(header))
    row = f"{'总体':^10}"
    for lead in range(1, num_leadtime + 1):
        lead_rmses = [monthly_lead_stats[(y,m)][lead][1] for (y,m) in monthly_lead_stats]
        row += f"{np.mean(lead_rmses)*100:>9.2f}%"
    print(row)
    
    # 总体统计
    print("\n========== 总体统计 ==========")
    print(f"  MAE: {np.mean(all_mae)*100:.2f}% ± {np.std(all_mae)*100:.2f}%")
    print(f"  RMSE: {np.mean(all_rmse)*100:.2f}% ± {np.std(all_rmse)*100:.2f}%")
    print(f"  样本数: {len(all_mae)} (lead time x 时间步)")
    
    ds_pred.close()
    ds_gt.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='评估预测结果')
    parser.add_argument('--pred-file', type=str, 
                        default='./predictions/197912_202512_20260721_230027.nc')
    parser.add_argument('--gt-file', type=str, default='./data/oras5/siconc.nc')
    parser.add_argument('--landmask', type=str, default='./numpy/landmask.npy')
    parser.add_argument('--output-log', type=str, default='./log/test.out',
                        help='评估结果追加写入的日志文件')
    args = parser.parse_args()

    output_log = Path(args.output_log)
    has_content = output_log.exists() and output_log.stat().st_size > 0
    with open(output_log, "a", encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        with contextlib.redirect_stdout(f):
            evaluate(args.pred_file, args.gt_file, args.landmask)


if __name__ == '__main__':
    main()
