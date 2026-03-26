#!/usr/bin/env bash
# Probe Whisper encoder through VitisAI EP on host Linux
# Tests: amd/whisper-tiny-en-onnx-npu encoder via VAIML BF16 path
set -euo pipefail

HUB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AMD_LIB="/var/home/mariostjr/amd-rai-linux/ryzen14"
PROBE_SRC="$HUB_DIR/tools/probe_vitisai.c"
PROBE_BIN="$HUB_DIR/runtime/whisper/probe_whisper"
ORT_INCLUDE="$HUB_DIR/third_party/onnxruntime/include"

MODEL="$HUB_DIR/runtime/whisper/models/tiny_en_encoder.onnx"
CONFIG="$HUB_DIR/runtime/whisper/vaiep_config_whisper.json"
CACHE_DIR="$HUB_DIR/runtime/whisper/cache/tiny_en_encoder"

echo "=== Compiling probe ==="
gcc -O2 -o "$PROBE_BIN" "$PROBE_SRC" \
    -I"$ORT_INCLUDE" \
    -L"$AMD_LIB" \
    -lonnxruntime \
    -Wl,-rpath,"$AMD_LIB"

echo "=== Probe compiled: $PROBE_BIN ==="

# Create cache dir
mkdir -p "$CACHE_DIR"

# Locate deps from Ubuntu overlay (same as native probe)
UBUNTU_LIB="$HUB_DIR/runtime/ubuntu22/lib"
EXTRA_LIBS=""
if [ -d "$UBUNTU_LIB" ]; then
    EXTRA_LIBS=":$UBUNTU_LIB"
fi

export LD_LIBRARY_PATH="${AMD_LIB}${EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Do NOT set XLNX_VART_FIRMWARE to a directory - VAIML treats it as xclbin file path
# The runtime should find xclbin overlays automatically via VAIP

echo "=== Running encoder probe ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG"
echo "  cache:  $CACHE_DIR"
echo ""

"$PROBE_BIN" "$MODEL" \
    "config_file=$CONFIG" \
    "cache_dir=$CACHE_DIR" \
    "cache_key=whisper_tiny_en_encoder" \
    2>&1

echo ""
echo "=== Checking EP report ==="
REPORT="$CACHE_DIR/vitisai_ep_report.json"
if [ -f "$REPORT" ]; then
    echo "EP report found:"
    cat "$REPORT"
else
    echo "No EP report generated yet"
    # Check for any report in cache
    find "$CACHE_DIR" -name "*.json" -type f 2>/dev/null
fi
