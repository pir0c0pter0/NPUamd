#!/usr/bin/env bash
# Probe Whisper XINT8 encoder through VitisAI EP on host Linux
# Uses the DPU/DD config path (same as proven ResNet18 XINT8)
set -euo pipefail

HUB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AMD_LIB="/var/home/mariostjr/amd-rai-linux/ryzen14"
PROBE_BIN="$HUB_DIR/runtime/whisper/probe_whisper"
ORT_INCLUDE="$HUB_DIR/third_party/onnxruntime/include"
PROBE_SRC="$HUB_DIR/tools/probe_vitisai.c"

MODEL="$HUB_DIR/runtime/whisper/models/tiny_en_encoder_xint8.onnx"
CONFIG="$AMD_LIB/vaip_config_npu_2_3.json"
CACHE_DIR="$HUB_DIR/runtime/whisper/cache/tiny_en_encoder_xint8"

# Compile if needed
if [ ! -f "$PROBE_BIN" ]; then
    echo "=== Compiling probe ==="
    gcc -O2 -o "$PROBE_BIN" "$PROBE_SRC" \
        -I"$ORT_INCLUDE" \
        -L"$AMD_LIB" \
        -lonnxruntime \
        -Wl,-rpath,"$AMD_LIB"
fi

mkdir -p "$CACHE_DIR"

UBUNTU_LIB="$HUB_DIR/runtime/ubuntu22/lib"
EXTRA_LIBS=""
if [ -d "$UBUNTU_LIB" ]; then
    EXTRA_LIBS=":$UBUNTU_LIB"
fi

export LD_LIBRARY_PATH="${AMD_LIB}${EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Running Whisper encoder XINT8 probe ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG (DPU/DD path)"
echo "  cache:  $CACHE_DIR"
echo ""

"$PROBE_BIN" "$MODEL" \
    "config_file=$CONFIG" \
    "cache_dir=$CACHE_DIR" \
    "cache_key=whisper_tiny_en_encoder_xint8" \
    2>&1

echo ""
echo "=== Checking EP report ==="
REPORT_DIR="$CACHE_DIR/whisper_tiny_en_encoder_xint8"
REPORT="$REPORT_DIR/vitisai_ep_report.json"
if [ -f "$REPORT" ]; then
    echo "EP report found. Device stats:"
    python3 -c "
import json, sys
with open('$REPORT') as f:
    report = json.load(f)
for ds in report.get('deviceStat', []):
    print(f\"  {ds['name']}: {ds['nodeNum']} ops\")
" 2>/dev/null || cat "$REPORT" | head -40
else
    echo "Searching for report..."
    find "$CACHE_DIR" -name "vitisai_ep_report.json" -type f 2>/dev/null
fi
