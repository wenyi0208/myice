#!/bin/bash
# ===================================================
# GPU计算作业示例 - 深度学习训练
# 适用队列：qgpu_3090, qgpu_4090, qgpu_a800
# ===================================================

#SBATCH --job-name=gpu_training
#SBATCH --output=gpu_%j.out
#SBATCH --error=gpu_%j.err

# GPU资源配置
#SBATCH --partition=qgpu_3090       # GPU分区选择
#SBATCH --nodes=1                   # GPU作业通常单节点
#SBATCH --ntasks-per-node=1         # 单任务运行
#SBATCH --cpus-per-task=4           # CPU核心数（建议2-4核/GPU）
#SBATCH --gres=gpu:1                # GPU卡数量（1-4卡）
#SBATCH --mem=128G                   # 内存大小（建议16-64GB）
#SBATCH --time=36:00:00             # 最大运行时间

# 通知设置
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=932238083@qq.com

# 环境初始化
echo "========== GPU作业开始 =========="
echo "作业ID: $SLURM_JOB_ID"
echo "GPU节点: $(hostname)"
echo "开始时间: $(date)"

# 检查GPU状态
nvidia-smi
echo "GPU驱动信息已显示"

# 执行GPU计算任务
echo "开始执行深度学习训练..."
python -u train_TTT_5.py

echo "========== GPU作业完成 =========="
echo "结束时间: $(date)"