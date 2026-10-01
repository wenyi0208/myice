#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List

import numpy as np
from torch.utils.data import Dataset

from numba import njit, prange

# 常量定义
# Geopotential-height files now use hPa in their filenames, e.g. zg_925.nc.
# Keep this channel order identical to gen_train.py and gen_val.py.
ZG_LEVELS = (925, 850, 500, 300, 100, 50, 10)
VAR_KEYS = (
    "siconc",
    "sithick",
    "tas",
    "uo0",
    "uo10",
    "vo0",
    "vo10",
    *(f"zg{level}" for level in ZG_LEVELS),
)
MONTHS_PER_YEAR = 12
CLIMATE_COMPUTATION_YEARS = 35  # 用于计算气候态的年数（420个月 = 35年）
CLIMATE_TIME_STEPS = CLIMATE_COMPUTATION_YEARS * MONTHS_PER_YEAR  # 420
SAMPLE_CHANNELS = len(VAR_KEYS) + 1  # variables + landmask
LANDMASK_CHANNEL_IDX = len(VAR_KEYS)
SICONC_VAR_IDX = 0  # siconc变量在VAR_KEYS中的索引
EPSILON = 1e-8  # 用于避免除零的小常数
FILL_VALUE_LIMIT = 1e20


def _sanitize_array(arr: np.ndarray, fill_value_limit: float = FILL_VALUE_LIMIT) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bad = (~np.isfinite(arr)) | (np.abs(arr) > fill_value_limit)
    if np.any(bad):
        arr = arr.copy()
        arr[bad] = 0.0
    return arr


@njit(parallel=True)
def _normalize_jit(
    var_slice: np.ndarray,
    clim: np.ndarray,
    mean: float,
    std: float,
    start_mod: int
) -> np.ndarray:
    """
    使用 Numba 加速的标准化计算。
    
    Args:
        var_slice: 变量时间序列切片，形状为 (T, H, W)
        clim: 气候态数据，形状为 (12, H, W)
        mean: 异常值的均值
        std: 异常值的标准差
        start_mod: 起始时间的月份偏移量 (0-11)
        
    Returns:
        标准化后的数组，形状为 (T, H, W)
    """
    T, H, W = var_slice.shape
    out = np.empty((T, H, W), dtype=np.float32)
    for t in prange(T):
        month = (start_mod + t) % MONTHS_PER_YEAR
        for h in range(H):
            for w in range(W):
                anomaly = var_slice[t, h, w] - clim[month, h, w]
                out[t, h, w] = (anomaly - mean) / std
    return out


# 预热 Numba 编译（使用示例形状）
_normalize_jit(
    np.zeros((MONTHS_PER_YEAR, 150, 360), dtype=np.float32),
    np.zeros((MONTHS_PER_YEAR, 150, 360), dtype=np.float32),
    0.0, 1.0, 0
)


def _normalize(
    var_slice: np.ndarray,
    clim: np.ndarray,
    mean: float,
    std: float,
    start_mod: int = 0
) -> np.ndarray:
    """
    对变量进行标准化处理：先计算异常值（减去气候态），再标准化。
    
    使用 Numba 加速计算。
    
    Args:
        var_slice: 变量时间序列切片，形状为 (T, H, W)
        clim: 气候态数据，形状为 (12, H, W) 或列表
        mean: 异常值的均值
        std: 异常值的标准差
        start_mod: 起始时间的月份偏移量 (0-11)
        
    Returns:
        标准化后的数组，形状为 (T, H, W)
    """
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or abs(std) <= EPSILON:
        return np.zeros_like(var_slice, dtype=np.float32)

    var_slice = _sanitize_array(var_slice)
    clim = _sanitize_array(np.array(clim, dtype=np.float32))
    normalized = _normalize_jit(var_slice, clim, float(mean), float(std), start_mod)
    return _sanitize_array(normalized)


def _compute_climate(arr: np.ndarray, var_keys: List[str]) -> Dict[str, List]:
    """
    使用整个训练数据计算气候态（严格无数据泄露）
    """
    climate_dict = {}

    for var_idx, var_key in enumerate(var_keys):
        monthly_climates = []

        for month in range(MONTHS_PER_YEAR):
            #  直接用整个时间序列
            month_data = _sanitize_array(arr[var_idx][month::MONTHS_PER_YEAR, ...])
            monthly_mean = np.nanmean(month_data, axis=0)
            monthly_mean = _sanitize_array(monthly_mean)

            monthly_climates.append(monthly_mean.tolist())

        climate_dict[var_key] = monthly_climates

    return climate_dict


def _load_json(path: Path) -> Dict:
    """
    加载 JSON 文件。
    
    Args:
        path: JSON 文件路径
        
    Returns:
        解析后的字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class BaseDataset(Dataset):
    """
    数据集基类，提供通用的数据加载和预处理功能。
    
    功能包括：
    - 加载气候态数据
    - 加载标准化参数
    - 加载 landmask
    - 准备训练样本和标签
    """
    # 变量名称列表，按数据数组中的顺序排列
    VAR_KEYS = list(VAR_KEYS)
    
    # 窗口大小配置
    INPUT_LEN = 6  # 输入时间步数
    OUTPUT_LEN = 6  # 输出时间步数

    def __init__(self, numpy_dir: Path, normalize_path: Path, climate_path: Path):
        """
        初始化数据集基类。
        
        Args:
            numpy_dir: numpy 数据文件目录
            normalize_path: 标准化参数 JSON 文件路径
            climate_path: 气候态数据 .npy 文件路径
        """
        self.numpy_dir = Path(numpy_dir)
        self.input_len = self.INPUT_LEN
        self.output_len = self.OUTPUT_LEN
        self.window_size = self.input_len + self.output_len

        # 加载 landmask
        self.landmask = self._load_landmask(numpy_dir)
        
        # 加载气候态数据
        self.climate = self._load_climate(climate_path)
        
        # 加载标准化参数
        self.normalize = self._load_normalize_params(normalize_path)

    def _load_landmask(self, numpy_dir: Path) -> Optional[np.ndarray]:
        """
        加载 landmask 数据。
        
        Args:
            numpy_dir: numpy 数据文件目录
            
        Returns:
            landmask 数组，如果文件不存在则返回 None
        """
        landmask_path = numpy_dir.parent / "landmask.npy"
        if landmask_path.exists():
            return np.load(landmask_path)
        return None

    def _load_climate(self, climate_path: Path) -> Optional[Dict[str, Dict[str, List]]]:
        """
        加载气候态数据。
        
        从 .npy 文件加载所有数据集的气候态，确保所有值都是列表格式。
        
        Args:
            climate_path: 气候态数据文件路径
            
        Returns:
            嵌套字典：{数据集名: {变量名: [12个月的气候态列表]}}，如果文件不存在则返回 None
        """
        if not climate_path.exists():
            return None
        
        climate_data = np.load(climate_path, allow_pickle=True).item()
        climate_dict = {}
        
        for dataset_name, dataset_climate in climate_data.items():
            climate_dict[dataset_name] = {
                var_key: var_value.tolist() if isinstance(var_value, np.ndarray) else var_value
                for var_key, var_value in dataset_climate.items()
            }
        
        return climate_dict

    def _load_normalize_params(self, normalize_path: Path) -> Optional[Dict]:
        """
        加载标准化参数。
        
        Args:
            normalize_path: 标准化参数 JSON 文件路径
            
        Returns:
            标准化参数字典，如果文件不存在则返回 None
        """
        if normalize_path.exists():
            return _load_json(normalize_path)
        return None

    def _has_required_stats(self, stats: Optional[Dict]) -> bool:
        if not stats or "cmip6_mean" not in stats:
            return False
        return all(
            key in stats["cmip6_mean"]
            for key in self.VAR_KEYS
            if key != "siconc"
        )

    def _validate_array_channels(self, path: Path) -> int:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if arr.ndim != 4:
                raise ValueError(
                    f"{path} must have shape [variable, time, H, W], got {arr.shape}"
                )
            expected = len(self.VAR_KEYS)
            if arr.shape[0] != expected:
                raise ValueError(
                    f"{path} has {arr.shape[0]} variable channels, but the current "
                    f"configuration expects {expected}: {self.VAR_KEYS}. "
                    "Regenerate the NumPy data with gen_train.py or gen_val.py."
                )
            return int(arr.shape[1])
        finally:
            del arr

    def _prepare_sample(
        self,
        arr: np.ndarray,
        start: int,
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        准备一个训练样本和对应的标签。
        
        处理流程：
        1. siconc: 直接使用原始值，不做标准化
        2. 其他变量: 计算异常值 -> 标准化 -> 缩放到 [-1, 1]
        3. landmask: 添加到样本的最后一个通道
        
        Args:
            arr: 数据数组，形状为 (num_vars, T, H, W)
            start: 窗口起始时间索引
            dataset_name: 数据集名称，用于获取对应的气候态和标准化参数
            
        Returns:
            (sample, label, month_idx) 元组：
            - sample: 输入样本，形状为 (SAMPLE_CHANNELS, input_len, H, W)
            - label: 标签（siconc），形状为 (1, output_len, H, W)
            - month_idx: 标签起始月份索引 (0-11)
        """
        height, width = arr.shape[2], arr.shape[3]
        sample = np.zeros((SAMPLE_CHANNELS, self.input_len, height, width), dtype=np.float32)
        label = np.zeros((1, self.output_len, height, width), dtype=np.float32)

        # 处理 siconc（原始值，不做任何处理）
        self._prepare_siconc(arr, start, sample, label)

        # 计算月份偏移量（用于选择正确的气候态月份）
        start_month_offset = start % MONTHS_PER_YEAR
        label_start_month = (start + self.input_len) % MONTHS_PER_YEAR

        # 处理其他变量（标准化并缩放）
        self._prepare_other_variables(
            arr, start, dataset_name, start_month_offset, sample
        )

        # 添加 landmask
        self._add_landmask(sample, height, width)

        # 处理 NaN 值
        sample = _sanitize_array(sample)
        label = _sanitize_array(label)
        
        return sample, label, label_start_month

    def _prepare_siconc(
        self,
        arr: np.ndarray,
        start: int,
        sample: np.ndarray,
        label: np.ndarray
    ) -> None:
        """
        准备 siconc 变量的样本和标签（使用原始值）。
        
        Args:
            arr: 数据数组
            start: 窗口起始时间索引
            sample: 样本数组（将被修改）
            label: 标签数组（将被修改）
        """
        siconc_window = _sanitize_array(arr[SICONC_VAR_IDX, start:start + self.window_size])
        sample[SICONC_VAR_IDX] = siconc_window[:self.input_len]
        label[0] = siconc_window[self.input_len:]

    def _prepare_other_variables(
        self,
        arr: np.ndarray,
        start: int,
        dataset_name: str,
        start_month_offset: int,
        sample: np.ndarray
    ) -> None:
        """
        准备其他变量的样本（标准化并缩放到 [-1, 1]）。
        
        Args:
            arr: 数据数组
            start: 窗口起始时间索引
            dataset_name: 数据集名称
            start_month_offset: 起始时间的月份偏移量
            sample: 样本数组（将被修改）
        """
        for var_idx, var_key in enumerate(self.VAR_KEYS):
            if var_key == "siconc":
                continue
            
            # 提取变量时间窗口
            var_window = _sanitize_array(arr[var_idx, start:start + self.window_size])
            
            # 获取该变量的气候态和标准化参数
            climate = self.climate["cmip6_mean"][var_key]
            norm_params = self.normalize["cmip6_mean"][var_key]
            
            # 标准化：计算异常值并标准化
            normalized = _normalize(
                var_window,
                climate,
                norm_params["mean"],
                norm_params["std"],
                start_mod=start_month_offset
            )
            
            # 缩放到 [-1, 1]（如果 max_val 足够大）
            max_val = norm_params["max"]
            if np.isfinite(max_val) and max_val > EPSILON:
                normalized = normalized / max_val
            normalized = _sanitize_array(normalized)
            
            # 只取输入部分（前 input_len 个时间步）
            sample[var_idx] = normalized[:self.input_len]

    def _add_landmask(self, sample: np.ndarray, height: int, width: int) -> None:
        """
        将 landmask 添加到样本的最后一个通道。
        
        Args:
            sample: 样本数组（将被修改）
            height: 高度
            width: 宽度
        """
        if self.landmask is None:
            return
        
        landmask = self.landmask
        # 确保 landmask 形状正确
        if landmask.shape != (height, width) and landmask.size == height * width:
            landmask = landmask.reshape((height, width))
        
        # 将 landmask 广播到所有时间步
        sample[LANDMASK_CHANNEL_IDX] = landmask.astype(np.float32)


class TrainDataset(BaseDataset):
    """
    训练数据集类。
    
    从多个 numpy 文件中加载训练数据，自动计算或加载气候态和标准化参数。
    """
    
    def __init__(
        self,
        numpy_dir: Path = Path.cwd() / "numpy" / "train",
        normalize_path: Path = Path.cwd() / "numpy" / "normalize.json"
    ):
        """
        初始化训练数据集。
        
        Args:
            numpy_dir: 训练数据 numpy 文件目录
            normalize_path: 标准化参数 JSON 文件路径
        """
        climate_path = numpy_dir.parent / "climate.npy"
        super().__init__(numpy_dir, normalize_path, climate_path)

        # 加载所有数据集文件
        self.dataset_files = sorted(numpy_dir.glob("*.npy"))
        self.dataset_names = [path.stem for path in self.dataset_files]
        if not self.dataset_files:
            raise ValueError(f"No training npy files found in {numpy_dir}")
        for path in self.dataset_files:
            self._validate_array_channels(path)

        # 如果气候态数据不存在，就计算并保存它
        if not self._has_required_stats(self.climate):
            self._compute_and_save_climate(climate_path) # 生成 climate.npy

        # 计算或加载标准化参数
        if not self._has_required_stats(self.normalize):
            self._compute_and_save_normalize_params(normalize_path)

        # 计算数据集大小
        self._compute_dataset_size()

    def _compute_and_save_climate(self, climate_path: Path) -> None:
        print("计算 climatology（统一CMIP6）...")

        all_climates = []

        for dataset_path in self.dataset_files:
            arr = np.load(dataset_path)
            climate = _compute_climate(arr, self.VAR_KEYS)
            all_climates.append(climate)

        #  计算平均 climate（关键）
        mean_climate = {}

        for var_key in self.VAR_KEYS:
            monthly_stack = []

            for month in range(MONTHS_PER_YEAR):
                # 收集所有模型该月数据
                month_maps = [
                    np.array(climate[var_key][month])
                    for climate in all_climates
                ]
                month_mean = np.mean(month_maps, axis=0)
                monthly_stack.append(month_mean.tolist())

            mean_climate[var_key] = monthly_stack

        #  存一个统一 key
        self.climate = {"cmip6_mean": mean_climate}

        np.save(climate_path, self.climate, allow_pickle=True)
        print(f" climatology 已保存: {climate_path}")

    def _compute_and_save_normalize_params(self, normalize_path: Path) -> None:
        """
        使用所有 CMIP6 训练数据计算统一 normalize 参数
        
        标准化参数包括：均值、标准差和全局最大值（用于缩放到 [-1, 1]）。
        
        Args:
            normalize_path: 保存标准化参数的文件路径
        """
        print("计算统一标准化参数（CMIP6 global normalize）...")

        # 用于累积所有数据
        all_data = {var: [] for var in self.VAR_KEYS if var != "siconc"}

        # 1 收集所有模型的数据
        for dataset_path in self.dataset_files:
            arr = np.load(dataset_path)
            dataset_name = dataset_path.stem

            for var_idx, var_key in enumerate(self.VAR_KEYS):
                if var_key == "siconc":
                    continue

                #  取该变量
                var_data = _sanitize_array(arr[var_idx])  # (T, H, W)

                #  取对应 climate
                climate = _sanitize_array(np.array(self.climate["cmip6_mean"][var_key]))

                time_steps = var_data.shape[0]
                month_idx = np.arange(time_steps) % MONTHS_PER_YEAR
                climate_selected = np.take(climate, month_idx, axis=0)

                anomaly = _sanitize_array(var_data - climate_selected)

                all_data[var_key].append(anomaly)

        #  拼接所有数据
        normalize = {"cmip6_mean": {}}

        for var_key, data_list in all_data.items():
            print(f"Processing {var_key}...")

            # 拼接成一个大数组
            combined = _sanitize_array(np.concatenate(data_list, axis=0))  # (total_T, H, W)

            mean = np.nanmean(combined)
            std = np.nanstd(combined)
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std <= EPSILON:
                std = 1.0

            normalized = (combined - mean) / (std + EPSILON)
            normalized = _sanitize_array(normalized)
            max_val = np.nanmax(np.abs(normalized))
            if not np.isfinite(max_val) or max_val <= EPSILON:
                max_val = 1.0

            normalize["cmip6_mean"][var_key] = {
                "mean": float(mean),
                "std": float(std),
                "max": float(max_val)
            }

        # 保存
        self.normalize = normalize

        with open(normalize_path, "w", encoding="utf-8") as f:
            json.dump(self.normalize, f)

        print(f"统一 normalize 已保存: {normalize_path}")

    def _compute_dataset_normalize_params(
        self,
        arr: np.ndarray,
        dataset_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        计算单个数据集的标准化参数。
        
        Args:
            arr: 数据数组，形状为 (num_vars, T, H, W)
            dataset_name: 数据集名称
            
        Returns:
            标准化参数字典：{变量名: {"mean": float, "std": float, "max": float}}
        """
        normalize_params = {}
        
        for var_idx, var_key in enumerate(self.VAR_KEYS):
            if var_key == "siconc":
                continue
            
            # 计算异常值
            climate = _sanitize_array(np.array(self.climate[dataset_name][var_key]))
            time_steps = arr.shape[1]
            month_indices = np.arange(time_steps) % MONTHS_PER_YEAR
            climate_selected = np.take(climate, month_indices, axis=0)
            anomaly = _sanitize_array(arr[var_idx] - climate_selected)
            
            # 计算标准化后的值
            anomaly_mean = np.nanmean(anomaly)
            anomaly_std = np.nanstd(anomaly)
            if not np.isfinite(anomaly_mean):
                anomaly_mean = 0.0
            if not np.isfinite(anomaly_std) or anomaly_std <= EPSILON:
                anomaly_std = 1.0
            normalized = _sanitize_array((anomaly - anomaly_mean) / anomaly_std)
            max_val = np.nanmax(np.abs(normalized))
            if not np.isfinite(max_val) or max_val <= EPSILON:
                max_val = 1.0
            
            # 保存参数
            normalize_params[var_key] = {
                "mean": float(anomaly_mean),
                "std": float(anomaly_std),
                "max": float(max_val)
            }
        
        return normalize_params

    def _compute_dataset_size(self) -> None:
        """计算数据集大小（窗口数和数据集数量）。"""
        # 找到所有数据集中最短的时间长度
        self.min_time = min(
            self._validate_array_channels(path)
            for path in self.dataset_files
        )
        self.num_windows = max(0, self.min_time - self.window_size + 1)
        self.num_datasets = len(self.dataset_files)

    def __len__(self) -> int:
        """返回数据集的总大小（窗口数 × 数据集数）。"""
        return self.num_windows * self.num_datasets

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        获取一个训练样本。
        
        Args:
            idx: 样本索引
            
        Returns:
            (sample, label, month_idx) 元组
        """
        idx = idx % len(self)
        window_idx = idx // self.num_datasets
        dataset_idx = idx % self.num_datasets
        dataset_name = self.dataset_names[dataset_idx]

        arr = np.load(self.dataset_files[dataset_idx], mmap_mode='r')
        return self._prepare_sample(arr, int(window_idx), dataset_name)


class ValDataset(BaseDataset):
    """
    验证数据集类（支持多个 npy 文件，统一使用 train 的统计参数）
    """

    def __init__(
        self,
        val_numpy_path: Path = Path.cwd() / "numpy" / "val",
        train_numpy_dir: Path = Path.cwd() / "numpy" / "train",
        normalize_path: Path = Path.cwd() / "numpy" / "normalize.json"
    ):
        climate_path = train_numpy_dir.parent / "climate.npy"
        super().__init__(train_numpy_dir, normalize_path, climate_path)

        # ---------------------------
        # 加载所有 val npy 文件
        # ---------------------------
        val_path = Path(val_numpy_path)
        if val_path.is_dir():
            self.dataset_files = sorted(val_path.glob("*.npy"))
        elif val_path.is_file():
            self.dataset_files = [val_path]
        else:
            raise FileNotFoundError(f"Validation path does not exist: {val_path}")
        self.dataset_names = [p.stem for p in self.dataset_files]

        if len(self.dataset_files) == 0:
            raise ValueError(f"No validation npy files found in {val_path}")

        # ---------------------------
        # 计算时间窗口
        # ---------------------------
        self.min_time = min(
            self._validate_array_channels(path)
            for path in self.dataset_files
        )

        self.num_windows = max(0, self.min_time - self.window_size + 1)
        self.num_datasets = len(self.dataset_files)

        # ---------------------------
        # 关键：统一使用 train 的统计参数
        # ---------------------------
        self.ref_dataset_name = list(self.climate.keys())[0]

    def __len__(self) -> int:
        return self.num_windows * self.num_datasets

    def __getitem__(self, idx: int):
        idx = idx % len(self)

        window_idx = idx // self.num_datasets
        dataset_idx = idx % self.num_datasets

        arr = np.load(self.dataset_files[dataset_idx], mmap_mode='r')

        # 关键：使用 train 的 dataset_name
        dataset_name = self.ref_dataset_name

        return self._prepare_sample(arr, int(window_idx), dataset_name)
