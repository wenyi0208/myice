#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
#SBATCH --partition=qgpu_3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

echo "========== 预测作业开始 =========="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $(hostname)"
echo "开始时间: $(date)"

nvidia-smi

python -u predict.py

echo "========== 预测作业结束 =========="
echo "结束时间: $(date)"
