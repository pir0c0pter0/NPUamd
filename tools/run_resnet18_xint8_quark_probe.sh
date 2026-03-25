#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${CONTAINER_NAME:-vitis-ubuntu22-hub}"
CONTAINER_OVERLAY_DIR="${CONTAINER_OVERLAY_DIR:-$HOME/.local/share/containers/storage/overlay}"
FP32_TMP_IN_CONTAINER="${FP32_TMP_IN_CONTAINER:-/tmp/resnet18_fp32_opset13_hub.onnx}"
XINT8_TMP_IN_CONTAINER="${XINT8_TMP_IN_CONTAINER:-/tmp/resnet18_xint8_quark_hub.onnx}"
FP32_MODEL="${FP32_MODEL:-$HUB_DIR/runtime/ubuntu22/models/resnet18_fp32_opset13.onnx}"
XINT8_MODEL="${XINT8_MODEL:-$HUB_DIR/runtime/ubuntu22/models/resnet18_xint8_quark.onnx}"
CACHE_DIR="${CACHE_DIR:-$HUB_DIR/runtime/ubuntu22/cache}"
CACHE_KEY="${CACHE_KEY:-resnet18_xint8_quark}"
SAMPLE_COUNT="${SAMPLE_COUNT:-32}"
SEED="${SEED:-1234}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

copy_from_overlay() {
  local container_path="$1"
  local destination="$2"
  local src
  local upper_dir

  upper_dir="$(podman inspect "$CONTAINER_NAME" --format '{{.GraphDriver.Data.UpperDir}}' 2>/dev/null || true)"

  if [[ -n "$upper_dir" && -f "$upper_dir${container_path}" ]]; then
    src="$upper_dir${container_path}"
  else
    src="$(
      find "$CONTAINER_OVERLAY_DIR" \
        -path "*/diff${container_path}" \
        -printf '%T@ %p\n' \
        2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
  fi

  if [[ -z "$src" || ! -f "$src" ]]; then
    printf '[FAIL] could not materialize container file from overlay: %s\n' "$container_path" >&2
    exit 1
  fi

  cp -f "$src" "$destination"
}

ensure_container() {
  podman exec "$CONTAINER_NAME" bash -lc 'true' >/dev/null
}

ensure_python_stack() {
  local modules_ok=0

  if podman exec "$CONTAINER_NAME" bash -lc 'python3 - <<'"'"'PY'"'"'
import importlib.util
import sys

required = ["torch", "torchvision", "onnxruntime", "quark"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("missing:", ",".join(missing))
    sys.exit(1)
PY' >/dev/null 2>&1; then
    modules_ok=1
  fi

  if [[ "$modules_ok" -eq 1 ]]; then
    return 0
  fi

  podman exec "$CONTAINER_NAME" bash -lc '
set -euo pipefail
python3 -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0
python3 -m pip install --no-cache-dir onnxruntime==1.20.1
python3 -m pip install --no-cache-dir amd-quark==0.10
'
}

main() {
  mkdir -p "$(dirname "$FP32_MODEL")" "$(dirname "$XINT8_MODEL")" "$CACHE_DIR"

  require_file "$HUB_DIR/tools/make_resnet18_xint8_quark_model.py" "ResNet18 Quark generator"
  require_file "$HUB_DIR/tools/run_vitisai_probe_native.sh" "native probe runner"

  ensure_container
  ensure_python_stack

  podman exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
rm -f '$FP32_TMP_IN_CONTAINER' '$XINT8_TMP_IN_CONTAINER'
python3 /work/hub/tools/make_resnet18_xint8_quark_model.py \
  '$FP32_TMP_IN_CONTAINER' \
  '$XINT8_TMP_IN_CONTAINER' \
  --sample-count '$SAMPLE_COUNT' \
  --seed '$SEED'
"

  copy_from_overlay "$FP32_TMP_IN_CONTAINER" "$FP32_MODEL"
  copy_from_overlay "$XINT8_TMP_IN_CONTAINER" "$XINT8_MODEL"

  bash "$HUB_DIR/tools/run_vitisai_probe_native.sh" \
    "$XINT8_MODEL" \
    "cache_dir=$CACHE_DIR" \
    "cache_key=$CACHE_KEY" \
    "enable_cache_file_io_in_mem=0" \
    "$@"
}

main "$@"
