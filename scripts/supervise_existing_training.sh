#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=""
TARGET_STEPS=""
METRICS_PATH=""
POLL_SECONDS=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME=$2
      shift 2
      ;;
    --target-steps)
      TARGET_STEPS=$2
      shift 2
      ;;
    --metrics)
      METRICS_PATH=$2
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_NAME" || -z "$TARGET_STEPS" || -z "$METRICS_PATH" || $# -eq 0 ]]; then
  echo "usage: $0 --run-name NAME --target-steps STEPS --metrics PATH -- COMMAND ..." >&2
  exit 2
fi

active_rank_pid() {
  pgrep -f "axion\.train_pipeline.*--run-name[[:space:]]+${RUN_NAME}" | head -n 1 || true
}

completed_steps() {
  if [[ ! -s "$METRICS_PATH" ]]; then
    return 1
  fi
  local step
  step=$(tail -n 1 "$METRICS_PATH" | sed -nE 's/.*"optimizer_step": ([0-9]+).*/\1/p')
  [[ -n "$step" && "$step" -ge "$TARGET_STEPS" ]]
}

pid=$(active_rank_pid)
if [[ -n "$pid" ]]; then
  echo "adopting existing Axion training rank pid=$pid run=$RUN_NAME" >&2
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  if completed_steps; then
    echo "adopted Axion training reached target step=$TARGET_STEPS" >&2
    exit 0
  fi
  echo "adopted Axion training exited before target; requesting recovery" >&2
  exit 1
fi

exec "$@"
