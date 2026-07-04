#!/usr/bin/env bash
set -euo pipefail

echo "hostname: $(hostname)"
PYTHON=${PYTHON:-python3}
UV=${UV:-/home/lence/.local/bin/uv}
echo "python: $($PYTHON --version 2>&1)"
echo "uv: $($UV --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found" >&2
  exit 1
fi
$UV run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
