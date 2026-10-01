#!/usr/bin/env python3
import argparse
from pathlib import Path
import cftime
import json
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from datetime import datetime

from model_grouped import model as create_model

# Must match dataset.ZG_LEVELS and the hPa-based zg filenames used by the
# data-generation scripts: zg_925.nc ... zg_10.nc.
ZG_LEVELS = (925, 850, 500, 300, 100, 50, 10)
INPUT_VAR_KEYS = [
    "siconc",
    "sithick",
    "tas",
    "uo0",
    "uo10",
    "vo0",
    "vo10",
    *(f"zg{level}" for level in ZG_LEVELS),
]
SAMPLE_CHANNELS = len(INPUT_VAR_KEYS) + 1
LANDMASK_CHANNEL_IDX = len(INPUT_VAR_KEYS)

# 显存优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== 辅助函数 ==========

def parse_time(time_str: str) -> tuple:
    """解析 YYYY-MM 格式时间"""
    y, m = map(int, time_str.split('-'))
    if not 1 <= m <= 12:
        raise ValueError(f"月份错误: {m}")
    return y, m


def to_month_index(year: int, month: int, start_year: int = 1979, start_month: int = 1) -> int:
    """将年月转换为从起始时间的月份索引"""
    return (year - start_year) * 12 + (month - start_month)


def from_month_index(idx: int, start_year: int = 1979, start_month: int = 1) -> tuple:
    """将月份索引转换为年月"""
    total = start_year * 12 + start_month + idx - 1
    return (total - 1) // 12 + 1, (total - 1) % 12 + 1


# ========== 数据集类 ==========

class PreDataset:
    """预测用数据集 - 直接加载任意时间范围的输入数据"""

    VAR_KEYS = INPUT_VAR_KEYS

    def __init__(
        self,
        val_numpy_path: Path,
        train_data_dir: Path,
        normalize_path: Path,
        stats_ds_name: str | None = None
    ):
        self.arr = np.load(val_numpy_path)
        if self.arr.ndim != 4:
            raise ValueError(
                f"{val_numpy_path} must have shape [variable, time, H, W], "
                f"got {self.arr.shape}"
            )
        if self.arr.shape[0] != len(self.VAR_KEYS):
            raise ValueError(
                f"{val_numpy_path} has {self.arr.shape[0]} variable channels, "
                f"but this model expects {len(self.VAR_KEYS)}: {self.VAR_KEYS}. "
                "Regenerate val.npy with gen_val.py."
            )
        self.T, self.H, self.W = self.arr.shape[1], self.arr.shape[2], self.arr.shape[3]
        
        # 加载标准化参数
        with open(normalize_path, "r") as f:
            self.normalize = json.load(f)
        
        # 加载气候态
        climate_path = train_data_dir.parent / "climate.npy"
        climate_data = np.load(climate_path, allow_pickle=True).item()
        available = list(climate_data.keys())
        if stats_ds_name is None:
            if len(available) > 1:
                print(f"警告: climate.npy 包含多个数据集 {available}，默认使用第一个: {available[0]}")
            self.ds_name = available[0]
        else:
            if stats_ds_name not in climate_data:
                raise KeyError(f"--stats-ds-name={stats_ds_name} 不存在于 climate.npy，候选: {available}")
            self.ds_name = stats_ds_name

        if self.ds_name not in self.normalize:
            raise KeyError(f"normalize.json 中不存在数据集 {self.ds_name}，候选: {list(self.normalize.keys())}")
        self.climate = {k: v for k, v in climate_data[self.ds_name].items()}
        missing_climate = [
            key for key in self.VAR_KEYS[1:] if key not in self.climate
        ]
        missing_normalize = [
            key
            for key in self.VAR_KEYS[1:]
            if key not in self.normalize[self.ds_name]
        ]
        if missing_climate or missing_normalize:
            raise ValueError(
                "Training statistics do not cover every input variable. "
                f"Missing climate keys: {missing_climate}; "
                f"missing normalization keys: {missing_normalize}. "
                "Regenerate training NumPy data with gen_val.py/gen_train.py, then "
                "rebuild climate.npy and normalize.json."
            )
        
        # 加载 landmask
        self.landmask = np.load(train_data_dir.parent / "landmask.npy")

    def get_input(self, start_idx: int) -> np.ndarray:
        """获取输入样本 [C, 6, H, W]"""
        if start_idx < 0 or (start_idx + 6) > self.T:
            raise IndexError(
                f"start_idx 越界: {start_idx}, 需要满足 0 <= start_idx <= {self.T - 6} "
                f"(因为输入窗口长度为 6，数据总长度 T={self.T})"
            )
        H, W = self.H, self.W
        sample = np.zeros((SAMPLE_CHANNELS, 6, H, W), dtype=np.float32)
        
        start_mod = start_idx % 12
        month_indices = (np.arange(6) + start_mod) % 12
        
        # siconc (不标准化)
        sample[0] = self.arr[0, start_idx:start_idx + 6]
        
        # 其他变量 (标准化 + 除以最大值缩放)
        for vidx, key in enumerate(self.VAR_KEYS[1:], 1):
            var_slice = self.arr[vidx, start_idx:start_idx + 6]
            clim = np.array(self.climate[key])
            mean = self.normalize[self.ds_name][key]["mean"]
            std = self.normalize[self.ds_name][key]["std"]
            max_val = self.normalize[self.ds_name][key]["max"]
            anomaly = var_slice - clim[month_indices]
            normalized = (anomaly - mean) / std
            if max_val > 1e-8:
                normalized = normalized / max_val
            sample[vidx] = normalized
        
        # landmask
        sample[LANDMASK_CHANNEL_IDX] = self.landmask.astype(np.float32)
        
        return np.nan_to_num(sample, nan=0.0)


# ========== 预测函数 ==========

def predict_batch(model: nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int = 12) -> np.ndarray:
    """批量预测 [B, 9, 6, H, W] -> [B, 6, H, W]，每 batch_size 个样本预测一次"""
    model.eval()
    B = inputs.shape[0]
    all_outputs = []

    with torch.no_grad():
        for i in range(0, B, batch_size):
            batch_input = inputs[i:i + batch_size].to(device)
            output = model(batch_input)
            all_outputs.append(output[:, 0, :, :, :].cpu().numpy())

    return np.concatenate(all_outputs, axis=0)


def create_netcdf(predictions: np.ndarray, start_year: int, start_month: int, output_path: Path):
    """创建 NetCDF 文件"""
    T, leadtime, H, W = predictions.shape
    lon = np.arange(0, W, dtype=np.float64)
    lat = np.arange(-59, -59 + H, dtype=np.float64)
    leadtime_arr = np.arange(1, leadtime + 1, dtype=np.int32)
    
    # 生成时间坐标
    time = []
    y, m = start_year, start_month
    for _ in range(T):
        time.append(cftime.DatetimeNoLeap(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    
    ds = xr.Dataset(
        {'siconc': (['time', 'leadtime', 'lat', 'lon'], predictions,
                    {'long_name': 'Sea Ice Concentration', 'units': '%'})},
        coords={'time': time, 'leadtime': leadtime_arr, 'lat': lat, 'lon': lon}
    )
    ds.lon.attrs = {'units': 'degrees_east'}
    ds.lat.attrs = {'units': 'degrees_north'}
    ds.time.attrs = {'standard_name': 'time'}
    ds.siconc.attrs['missing_value'] = np.nan
    
    ds.to_netcdf(output_path)
    print(f"保存: {output_path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='Swin Transformer 预测')
    parser.add_argument('--start-time', type=str, default='1979-12', help='起始时间 YYYY-MM')
    parser.add_argument('--end-time', type=str, default='2025-12', help='结束时间 YYYY-MM')
    parser.add_argument(
        '--val-start-time',
        type=str,
        default='1979-01',
        help='val.npy 的第 0 个时间步对应的年月 (默认: 1979-01，与 gen_val.py 一致)'
    )
    parser.add_argument('--val-data-dir', type=str, default='./numpy/val/val.npy')
    parser.add_argument('--train-data-dir', type=str, default='./numpy/train')
    parser.add_argument('--normalize-path', type=str, default='./numpy/normalize.json')
    parser.add_argument(
        '--stats-ds-name',
        type=str,
        default=None,
        help='climate.npy/normalize.json 中用于标准化的训练数据集名称（多数据集时建议指定）'
    )
    parser.add_argument('--model-path', type=str, default='./weights/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 解析时间
    start_year, start_month = parse_time(args.start_time)
    end_year, end_month = parse_time(args.end_time)
    val0_year, val0_month = parse_time(args.val_start_time)
    num_steps = (end_year - start_year) * 12 + (end_month - start_month) + 1
    
    print(f"预测范围: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d} ({num_steps} 个月)")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    model = create_model(in_chans=SAMPLE_CHANNELS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载 (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # 加载数据
    dataset = PreDataset(
        Path(args.val_data_dir),
        Path(args.train_data_dir),
        Path(args.normalize_path),
        stats_ds_name=args.stats_ds_name
    )
    H, W = dataset.H, dataset.W
    land_mask = (dataset.landmask == 1)
    print(f"数据: {dataset.T} 个月, H={H}, W={W}")
    
    # 计算输入窗口索引范围
    start_input_idx = to_month_index(start_year, start_month, start_year=val0_year, start_month=val0_month) - 11
    end_input_idx = to_month_index(end_year, end_month, start_year=val0_year, start_month=val0_month) - 6
    num_windows = end_input_idx - start_input_idx + 1
    print(f"输入窗口: {num_windows} 个 (索引 {start_input_idx} ~ {end_input_idx})")

    # 边界检查
    if num_windows <= 0:
        raise ValueError(f"预测时间范围非法: start={args.start_time}, end={args.end_time}")
    if start_input_idx < 0:
        raise ValueError(
            f"start_time={args.start_time} 太早，导致 start_input_idx={start_input_idx} < 0。"
            f"请检查 --val-start-time={args.val_start_time} 是否正确，或调整预测起始时间。"
        )
    if (end_input_idx + 6) > dataset.T:
        raise ValueError(
            f"end_time={args.end_time} 太晚/val 数据不足，导致 end_input_idx+6={end_input_idx + 6} > T={dataset.T}。"
            f"请检查 val.npy 覆盖的时间范围或调整预测结束时间。"
        )
    
    # 直接构建所有窗口的输入张量
    batch_inputs = torch.stack([
        torch.from_numpy(dataset.get_input(start_input_idx + i)).float()
        for i in range(num_windows)
    ])
    print(f"批量输入: {batch_inputs.shape}")
    
    # 批量预测
    all_outputs = predict_batch(model, batch_inputs, device, batch_size=12)
    all_outputs = np.where(land_mask, np.nan, all_outputs)

    print(f"批量预测完成, 形状: {all_outputs.shape}")
    
    # 重组结果: [num_steps, 6, H, W]
    predictions = np.full((num_steps, 6, H, W), np.nan, dtype=np.float32)

    for t_idx in range(num_steps):
        for lead in range(1, 7):
            predictions[t_idx, lead - 1] = all_outputs[t_idx - lead + 6, lead - 1]
    
    print(f"重组后: {predictions.shape}")
    
    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{start_year}{start_month:02d}_{end_year}{end_month:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
    create_netcdf(predictions, start_year, start_month, output_dir / output_file)
    print(f"完成!")


if __name__ == '__main__':
    main()
