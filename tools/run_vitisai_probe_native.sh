#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AMD_DIR="${AMD_DIR:-/var/home/mariostjr/amd-rai-linux/ryzen14}"
CONTAINER_OVERLAY_DIR="${CONTAINER_OVERLAY_DIR:-$HOME/.local/share/containers/storage/overlay}"
BUILD_DIR="${BUILD_DIR:-$HUB_DIR/.build}"
NATIVE_RUNTIME_DIR="${NATIVE_RUNTIME_DIR:-$HUB_DIR/runtime/ubuntu22}"
NATIVE_LIB_DIR="${NATIVE_RUNTIME_DIR}/lib"
DEFAULT_MODEL="${NATIVE_RUNTIME_DIR}/models/minimal_identity_ir10.onnx"
PROBE_BIN="${BUILD_DIR}/probe_vitisai_native"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

find_compat_libdir() {
  local libpython
  local libdir

  while IFS= read -r libpython; do
    libdir="$(dirname "$libpython")"
    if [[ -f "$libdir/libboost_filesystem.so.1.74.0" ]] && \
       [[ -f "$libdir/libboost_program_options.so.1.74.0" ]] && \
       [[ -f "$libdir/libboost_system.so.1.74.0" ]]; then
      printf '%s\n' "$libdir"
      return 0
    fi
  done < <(
    find "$CONTAINER_OVERLAY_DIR" \
      -path '*/diff/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0' \
      2>/dev/null | sort
  )

  return 1
}

find_default_model() {
  find "$CONTAINER_OVERLAY_DIR" \
    -path '*/diff/tmp/minimal_identity_ir10.onnx' \
    2>/dev/null | sort | head -n 1
}

materialize_native_runtime() {
  local compat_libdir
  local model_src

  mkdir -p "$BUILD_DIR" "$NATIVE_LIB_DIR" "$(dirname "$DEFAULT_MODEL")"

  if [[ ! -f "$NATIVE_LIB_DIR/libpython3.10.so.1.0" ]] || \
     [[ ! -f "$NATIVE_LIB_DIR/libboost_filesystem.so.1.74.0" ]] || \
     [[ ! -f "$NATIVE_LIB_DIR/libboost_program_options.so.1.74.0" ]] || \
     [[ ! -f "$NATIVE_LIB_DIR/libboost_system.so.1.74.0" ]]; then
    compat_libdir="$(find_compat_libdir || true)"
    if [[ -z "$compat_libdir" ]]; then
      printf '[FAIL] could not find a local Ubuntu 22.04 compat lib directory under %s\n' "$CONTAINER_OVERLAY_DIR" >&2
      exit 1
    fi

    cp -f "$compat_libdir/libpython3.10.so.1.0" "$NATIVE_LIB_DIR/"
    cp -f "$compat_libdir/libboost_filesystem.so.1.74.0" "$NATIVE_LIB_DIR/"
    cp -f "$compat_libdir/libboost_program_options.so.1.74.0" "$NATIVE_LIB_DIR/"
    cp -f "$compat_libdir/libboost_system.so.1.74.0" "$NATIVE_LIB_DIR/"
  fi

  if [[ ! -f "$DEFAULT_MODEL" ]]; then
    model_src="$(find_default_model || true)"
    if [[ -z "$model_src" ]]; then
      printf '[FAIL] could not find minimal_identity_ir10.onnx under %s\n' "$CONTAINER_OVERLAY_DIR" >&2
      printf '[FAIL] pass an explicit MODEL_PATH to this script or keep the working container overlay available\n' >&2
      exit 1
    fi

    cp -f "$model_src" "$DEFAULT_MODEL"
  fi
}

main() {
  local model_path="$DEFAULT_MODEL"
  local provider_args=()
  local runtime_ld_path

  materialize_native_runtime

  require_file "$AMD_DIR/libonnxruntime.so.1.20.1" "AMD ONNX Runtime"
  require_file "$HUB_DIR/tools/probe_vitisai.c" "probe source"
  require_file "$HUB_DIR/third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h" "ORT C API header"

  if (($# > 0)); then
    if [[ "$1" == *=* ]]; then
      provider_args=("$@")
    else
      model_path="$1"
      shift
      provider_args=("$@")
    fi
  fi

  require_file "$model_path" "model"

  gcc \
    -I"$HUB_DIR/third_party/onnxruntime/include" \
    -L"$AMD_DIR" \
    -Wl,-rpath,"$AMD_DIR" \
    -o "$PROBE_BIN" \
    "$HUB_DIR/tools/probe_vitisai.c" \
    -lonnxruntime

  runtime_ld_path="$AMD_DIR:$NATIVE_LIB_DIR"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    runtime_ld_path="$runtime_ld_path:$LD_LIBRARY_PATH"
  fi

  printf '[INFO] using AMD userspace: %s\n' "$AMD_DIR"
  printf '[INFO] using compat lib dir: %s\n' "$NATIVE_LIB_DIR"
  printf '[INFO] using model: %s\n' "$model_path"

  LD_LIBRARY_PATH="$runtime_ld_path" \
    "$PROBE_BIN" "$model_path" "${provider_args[@]}"
}

main "$@"
