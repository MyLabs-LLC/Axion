#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/axion_1m.yaml}
STEPS=${AXION_STEPS:-2}
HOSTS=(${AXION_HOSTS:-oblivion oblivion1 oblivion2})
MASTER_ADDR=${AXION_MASTER_ADDR:-${HOSTS[0]}}
MASTER_PORT=${AXION_MASTER_PORT:-29500}
IFACE=${AXION_IFACE:-}
IFACES=(${AXION_IFACES:-})
ROOT=${AXION_ROOT:-$PWD}
UV=${AXION_UV:-/home/lence/.local/bin/uv}
CPU=${AXION_CPU:-0}
COLLECT_RELICS=${AXION_COLLECT_RELICS:-1}

if [[ ${#HOSTS[@]} -ne 3 ]]; then
  echo "AXION_HOSTS must contain exactly three hosts" >&2
  exit 2
fi

for rank in 0 1 2; do
  host=${HOSTS[$rank]}
  env_args="NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN PYTHONPATH=$ROOT/src"
  if [[ "$CPU" == "1" ]]; then
    env_args="$env_args CUDA_VISIBLE_DEVICES="
  fi
  rank_iface="$IFACE"
  if [[ ${#IFACES[@]} -eq 3 ]]; then
    rank_iface="${IFACES[$rank]}"
  fi
  if [[ -n "$rank_iface" ]]; then
    env_args="$env_args NCCL_SOCKET_IFNAME=$rank_iface GLOO_SOCKET_IFNAME=$rank_iface"
  fi
  ssh "$host" "cd '$ROOT' && $env_args '$UV' run torchrun --nnodes=3 --nproc-per-node=1 --node-rank=$rank --master-addr='$MASTER_ADDR' --master-port='$MASTER_PORT' -m axion.train_pipeline --config '$CONFIG' --steps '$STEPS'" &
done

wait

if [[ "$COLLECT_RELICS" == "1" ]]; then
  latest_run=$(ls -td runs/* 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_run" ]]; then
    run_name=$(basename "$latest_run")
    mkdir -p "relics/$run_name"
    for rank in 1 2; do
      host=${HOSTS[$rank]}
      rsync -az "$host:$ROOT/relics/$run_name/" "relics/$run_name/"
    done
  fi
fi
