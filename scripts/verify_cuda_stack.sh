#!/usr/bin/env bash
set -euo pipefail

echo "host: $(hostname)"
echo "kernel: $(uname -r)"
echo "nvidia-smi:"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  /usr/local/cuda/bin/nvcc --version | tail -n 1
else
  nvcc --version | tail -n 1
fi

cd /home/lence/MyLabs-LLC/Axion
uv run python - <<'PY'
import torch

print('torch:', torch.__version__)
print('torch_cuda_runtime:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

dpkg -l | grep -E '^(ii)  (cuda-toolkit-13-2|nvidia-driver-595|nvidia-utils-595|nvidia-dkms-595|nvidia-compute-utils-595)' || true
