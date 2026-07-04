#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
CUDA_KEYRING_DEB=/tmp/cuda-keyring_1.1-1_all.deb
CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

apt-get update
apt-get install -y ca-certificates curl gnupg build-essential dkms "linux-headers-$(uname -r)"

if ! apt-cache policy cuda-toolkit-13-2 | grep -q 'developer.download.nvidia.com'; then
  curl -fsSL "$CUDA_KEYRING_URL" -o "$CUDA_KEYRING_DEB"
  dpkg -i "$CUDA_KEYRING_DEB"
  apt-get update
fi

# Remove Ubuntu's older CUDA 12 toolkit packages if present; keep driver packages.
apt-get remove -y \
  nvidia-cuda-toolkit \
  nvidia-cuda-dev \
  nvidia-cuda-toolkit-doc \
  nvidia-cuda-gdb \
  nvidia-profiler \
  nvidia-visual-profiler \
  nvidia-opencl-dev || true

apt-get install -y \
  nvidia-driver-595 \
  nvidia-dkms-595 \
  nvidia-utils-595 \
  nvidia-compute-utils-595 \
  nvidia-settings \
  nvidia-prime \
  cuda-toolkit-13-2

if [[ -d /usr/local/cuda-13.2 ]]; then
  ln -sfn /usr/local/cuda-13.2 /usr/local/cuda
fi

modprobe nvidia || true

echo '=== nvidia-smi ==='
nvidia-smi || true

echo '=== nvcc ==='
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  /usr/local/cuda/bin/nvcc --version
else
  command -v nvcc && nvcc --version || true
fi

echo '=== packages ==='
dpkg -l | grep -E '^(ii)  (cuda-toolkit-13-2|nvidia-driver-595|nvidia-utils-595|nvidia-dkms-595|nvidia-compute-utils-595)' || true

echo 'If nvidia-smi still shows an old driver, reboot this node.'
