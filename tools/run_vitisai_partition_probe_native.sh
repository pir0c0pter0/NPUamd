#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AMD_DIR="${AMD_DIR:-/var/home/mariostjr/amd-rai-linux/ryzen14}"
CONTAINER_NAME="${CONTAINER_NAME:-vitis-ubuntu22-hub}"
MODEL_PATH="${MODEL_PATH:-$HUB_DIR/runtime/ubuntu22/models/probe_small_cnn_opset17.onnx}"
CACHE_DIR="${CACHE_DIR:-$HUB_DIR/runtime/ubuntu22/cache}"
CONFIG_FILE="${CONFIG_FILE:-$AMD_DIR/vaip_config_npu_2_3.json}"
HUB_DIR_IN_CONTAINER="${HUB_DIR_IN_CONTAINER:-/work/hub}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

ensure_model() {
  local tmp_model_in_container

  if [[ -f "$MODEL_PATH" ]]; then
    return 0
  fi

  if [[ "$MODEL_PATH" != "$HUB_DIR/"* ]]; then
    printf '[FAIL] model path must live under HUB_DIR so it can be reached from the mounted container: %s\n' "$MODEL_PATH" >&2
    exit 1
  fi

  tmp_model_in_container="/tmp/$(basename "$MODEL_PATH")"

  podman exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
python3 '$HUB_DIR_IN_CONTAINER/tools/make_probe_cnn_model.py' '$tmp_model_in_container' >/dev/null
cat '$tmp_model_in_container'
" >"$MODEL_PATH"
}

main() {
  local cache_key
  local model_path="$MODEL_PATH"

  if (($# > 0)) && [[ "$1" != *=* ]]; then
    model_path="$1"
    shift
  fi

  require_file "$HUB_DIR/tools/make_probe_cnn_model.py" "probe CNN generator"
  require_file "$CONFIG_FILE" "VitisAI config"

  MODEL_PATH="$model_path"
  mkdir -p "$(dirname "$MODEL_PATH")" "$CACHE_DIR"
  ensure_model
  require_file "$MODEL_PATH" "probe CNN model"

  cache_key="$(basename "${MODEL_PATH%.onnx}")"

  bash "$HUB_DIR/tools/run_vitisai_probe_native.sh" \
    "$MODEL_PATH" \
    "config_file=$CONFIG_FILE" \
    "cache_dir=$CACHE_DIR" \
    "cache_key=$cache_key" \
    "enable_cache_file_io_in_mem=0" \
    "$@"
}

main "$@"
