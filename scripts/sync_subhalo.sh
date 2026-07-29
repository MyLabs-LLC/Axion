#!/usr/bin/env bash
set -euo pipefail

HOSTS=(${AXION_HOSTS:-oblivion oblivion1 oblivion2})
ROOT=${AXION_ROOT:-$PWD}

if [[ ${#HOSTS[@]} -ne 3 ]]; then
  echo "AXION_HOSTS must contain exactly three hosts" >&2
  exit 2
fi

for host in "${HOSTS[@]:1}"; do
  ssh "$host" "mkdir -p '$ROOT'"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude 'data/' \
    --exclude 'runs/' \
    --exclude 'relics/' \
    --exclude 'experiments/' \
    --exclude 'reports/' \
    --exclude 'hf_export/' \
    --exclude 'mlruns/' \
    --exclude 'mlartifacts/' \
    --exclude 'mlflow.db' \
    ./ "$host:$ROOT/"
done
