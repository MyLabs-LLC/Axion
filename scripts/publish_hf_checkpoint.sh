#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: publish_hf_checkpoint.sh --run-name NAME --step STEP --repo REPO_ID [options]

Wait for a completed three-stage checkpoint, export it as a Hugging Face
Transformers model, then upload it to the requested Hub model repository.

Options:
  --root PATH             Axion checkout and matching remote checkout path.
  --hosts HOST0 HOST1 HOST2
                          Stage hosts (default: oblivion oblivion1 oblivion2).
  --poll-seconds SECONDS  Checkpoint availability polling interval (default: 60).
  --uv-bin PATH           uv executable (default: uv from PATH).
  --hf-bin PATH           hf executable (default: hf from PATH).
EOF
}

require_value() {
  if [[ $# -lt 2 || -z ${2:-} ]]; then
    echo "missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

ROOT=$PWD
RUN_NAME=
STEP=
REPO=
HOSTS=(oblivion oblivion1 oblivion2)
POLL_SECONDS=60
UV_BIN=$(command -v uv || true)
HF_BIN=$(command -v hf || true)

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-name)
      require_value "$1" "${2:-}"
      RUN_NAME=$2
      shift 2
      ;;
    --step)
      require_value "$1" "${2:-}"
      STEP=$2
      shift 2
      ;;
    --repo)
      require_value "$1" "${2:-}"
      REPO=$2
      shift 2
      ;;
    --root)
      require_value "$1" "${2:-}"
      ROOT=$2
      shift 2
      ;;
    --hosts)
      if [[ $# -lt 4 ]]; then
        echo "--hosts needs exactly three host names" >&2
        exit 2
      fi
      HOSTS=("$2" "$3" "$4")
      shift 4
      ;;
    --poll-seconds)
      require_value "$1" "${2:-}"
      POLL_SECONDS=$2
      shift 2
      ;;
    --uv-bin)
      require_value "$1" "${2:-}"
      UV_BIN=$2
      shift 2
      ;;
    --hf-bin)
      require_value "$1" "${2:-}"
      HF_BIN=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z $RUN_NAME || -z $STEP || -z $REPO ]]; then
  echo "--run-name, --step, and --repo are required" >&2
  usage >&2
  exit 2
fi
if [[ ! $STEP =~ ^[1-9][0-9]*$ ]]; then
  echo "--step must be a positive integer" >&2
  exit 2
fi
if [[ ! $POLL_SECONDS =~ ^[1-9][0-9]*$ ]]; then
  echo "--poll-seconds must be a positive integer" >&2
  exit 2
fi
if [[ ${#HOSTS[@]} -ne 3 ]]; then
  echo "--hosts must identify exactly three stage hosts" >&2
  exit 2
fi
if [[ ! -x $UV_BIN || ! -x $HF_BIN ]]; then
  echo "uv and hf executables must be available" >&2
  exit 2
fi

ROOT=$(cd "$ROOT" && pwd -P)
printf -v STEP_ID '%08d' "$STEP"

while :; do
  ready=1
  for stage in 0 1 2; do
    remote="$ROOT/relics/$RUN_NAME/stage${stage}-step${STEP_ID}.pt"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${HOSTS[$stage]}" \
      "test -f \"$remote\""; then
      ready=0
      break
    fi
  done
  if [[ $ready -eq 1 ]]; then
    break
  fi
  echo "waiting for complete checkpoint step $STEP"
  sleep "$POLL_SECONDS"
done

cd "$ROOT"
OUT="hf_export/${RUN_NAME}-step${STEP}"
if [[ -e $OUT && ! -f "$OUT/axion_export_manifest.json" ]]; then
  echo "incomplete existing export at $OUT; remove it before retrying" >&2
  exit 1
fi
if [[ ! -f "$OUT/axion_export_manifest.json" ]]; then
  "$UV_BIN" run --with safetensors python scripts/export_hf.py \
    --run-name "$RUN_NAME" \
    --step "$STEP" \
    --root "$ROOT" \
    --hosts "${HOSTS[@]}" \
    --dtype bfloat16
fi

"$HF_BIN" upload-large-folder "$REPO" "$OUT" --type model --num-workers 4 --no-bars
echo "published $REPO from checkpoint step $STEP"
