#!/bin/bash
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node-5090-2
#SBATCH -J gpu_icetraining
#SBATCH -o gpu_%j.out
#SBATCH -e gpu_%j.err
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

echo "========== GPU job started =========="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "SLURM_JOB_GPUS before export: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES before export: ${CUDA_VISIBLE_DEVICES:-unset}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "SLURM_JOB_GPUS after export: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES after export: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTORCH_NVML_BASED_CUDA_CHECK: ${PYTORCH_NVML_BASED_CUDA_CHECK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "---------- NVIDIA device files ----------"
ls -l /dev/nvidia* || true
echo "-----------------------------------------"

echo "---------- NVIDIA driver info ----------"
cat /proc/driver/nvidia/version || true
nvidia-smi -L || true
echo "----------------------------------------"

echo "---------- GPU device info ----------"
nvidia-smi || true
echo "-------------------------------------"

python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    sys.exit("No CUDA devices are visible to PyTorch.")
try:
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    x = torch.empty(1, device="cuda")
    print("CUDA tensor test:", x.device)
except Exception as exc:
    print("CUDA runtime test failed:", repr(exc))
    sys.exit(1)
PY

echo "Starting training..."
python -u train_TTT_5.py --device cuda --num-workers 0

echo "========== GPU job finished =========="
echo "End time: $(date)"
