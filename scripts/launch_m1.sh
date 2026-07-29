#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/axion_1m.yaml}
STEPS=${AXION_STEPS:-2}
RUN_NAME=${AXION_RUN_NAME:-}
CHECKPOINT_EVERY=${AXION_CHECKPOINT_EVERY:-0}
KEEP_CHECKPOINTS=${AXION_KEEP_CHECKPOINTS:-2}
EVAL_EVERY=${AXION_EVAL_EVERY:-0}
VAL_BATCHES=${AXION_VAL_BATCHES:-4}
RESUME=${AXION_RESUME:-0}
RESUME_STEP=${AXION_RESUME_STEP:-}
HOSTS=(${AXION_HOSTS:-oblivion oblivion1 oblivion2})
MASTER_ADDR=${AXION_MASTER_ADDR:-${HOSTS[0]}}
MASTER_PORT=${AXION_MASTER_PORT:-29500}
IFACE=${AXION_IFACE:-}
IFACES=(${AXION_IFACES:-})
ROOT=${AXION_ROOT:-$PWD}
UV=${AXION_UV:-/home/lence/.local/bin/uv}
CPU=${AXION_CPU:-0}
COLLECT_RELICS=${AXION_COLLECT_RELICS:-1}
NCCL_DEBUG_LEVEL=${AXION_NCCL_DEBUG:-WARN}
pids=()

if [[ ${#HOSTS[@]} -ne 3 ]]; then
  echo "AXION_HOSTS must contain exactly three hosts" >&2
  exit 2
fi

for rank in 0 1 2; do
  host=${HOSTS[$rank]}
  env_args="NCCL_IB_DISABLE=1 NCCL_DEBUG=$NCCL_DEBUG_LEVEL PYTHONPATH=$ROOT/src"
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
  extra_args="--keep-checkpoints '$KEEP_CHECKPOINTS' --eval-every '$EVAL_EVERY' --val-batches '$VAL_BATCHES'"
  if [[ -n "$RUN_NAME" ]]; then
    extra_args="$extra_args --run-name '$RUN_NAME'"
  fi
  if [[ "$CHECKPOINT_EVERY" != "0" ]]; then
    extra_args="$extra_args --checkpoint-every '$CHECKPOINT_EVERY'"
  fi
  if [[ "$RESUME" == "1" ]]; then
    extra_args="$extra_args --resume"
    if [[ -n "$RESUME_STEP" ]]; then
      extra_args="$extra_args --resume-step '$RESUME_STEP'"
    fi
  fi
  ssh "$host" "cd '$ROOT' && $env_args '$UV' run torchrun --nnodes=3 --nproc-per-node=1 --node-rank=$rank --master-addr='$MASTER_ADDR' --master-port='$MASTER_PORT' -m axion.train_pipeline --config '$CONFIG' --steps '$STEPS' $extra_args" &
  pids[$rank]=$!
done

status=0
for rank in 0 1 2; do
  if wait "${pids[$rank]}"; then
    :
  else
    code=$?
    echo "rank $rank on ${HOSTS[$rank]} failed with exit code $code" >&2
    status=$code
  fi
done

if [[ "$status" != "0" ]]; then
  exit "$status"
fi

if [[ "$COLLECT_RELICS" == "1" ]]; then
  run_name="$RUN_NAME"
  if [[ -z "$run_name" ]]; then
    latest_run=$(ls -td runs/* 2>/dev/null | head -n 1 || true)
    run_name=$(basename "$latest_run")
  fi
  if [[ -n "$run_name" ]]; then
    mkdir -p "relics/$run_name"
    for rank in 1 2; do
      host=${HOSTS[$rank]}
      rsync -a "$host:$ROOT/relics/$run_name/" "relics/$run_name/"
    done
  fi
fi
