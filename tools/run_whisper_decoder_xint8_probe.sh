#!/usr/bin/env bash
# Probe Whisper XINT8 decoder through VitisAI EP on host Linux
# Uses a generic C probe so it can exercise both the original decoder and
# derived multi-input body submodels without depending on Python ORT bindings.
set -euo pipefail

HUB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AMD_LIB="/var/home/mariostjr/amd-rai-linux/ryzen14"
UBUNTU_LIB="$HUB_DIR/runtime/ubuntu22/lib"
PROBE_BIN="$HUB_DIR/runtime/whisper/probe_whisper_decoder"
ORT_INCLUDE="$HUB_DIR/third_party/onnxruntime/include"
PROBE_SRC="$HUB_DIR/tools/probe_whisper_decoder.c"

MODEL="${MODEL:-$HUB_DIR/runtime/whisper/models/tiny_en_decoder_xint8.onnx}"
CONFIG="${1:-$AMD_LIB/vaip_config_npu_2_3.json}"
CACHE_ROOT="${CACHE_ROOT:-$HUB_DIR/runtime/whisper/cache}"
CACHE_NAME="${CACHE_NAME:-xint8_decoder}"
CACHE_KEY="${CACHE_KEY:-whisper_dec_xint8}"
CACHE_DIR="$CACHE_ROOT/$CACHE_NAME"

mkdir -p "$CACHE_DIR"

if [ ! -f "$PROBE_BIN" ] || [ "$PROBE_SRC" -nt "$PROBE_BIN" ]; then
    echo "=== Compiling decoder probe ==="
    gcc -O2 -o "$PROBE_BIN" "$PROBE_SRC" \
        -I"$ORT_INCLUDE" \
        -L"$AMD_LIB" \
        -lonnxruntime \
        -Wl,-rpath,"$AMD_LIB"
fi

export LD_LIBRARY_PATH="${AMD_LIB}:${UBUNTU_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== Whisper decoder XINT8 probe ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG"
echo "  cache:  $CACHE_DIR"
echo "  key:    $CACHE_KEY"
echo ""

"$PROBE_BIN" "$MODEL" \
    "config_file=$CONFIG" \
    "cache_dir=$CACHE_DIR" \
    "cache_key=$CACHE_KEY" \
    2>&1

echo ""
echo "=== Checking EP report ==="
REPORT_DIR="$CACHE_DIR/$CACHE_KEY"
REPORT="$REPORT_DIR/vitisai_ep_report.json"
if [ -f "$REPORT" ]; then
    python3 - <<PY
import json
with open("$REPORT") as f:
    report = json.load(f)
for ds in report.get("deviceStat", []):
    print(f"  {ds['name']}: {ds['nodeNum']} ops")
print(f"  subgraphStat: {report.get('subgraphStat')}")
PY
else
    echo "No EP report found"
    find "$CACHE_DIR" -name "vitisai_ep_report.json" -type f 2>/dev/null || true
fi
