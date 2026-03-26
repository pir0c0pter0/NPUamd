#!/usr/bin/env bash
# Full Whisper transcription path:
# - encoder on AMD NPU through native C API
# - decoder on CPU in the helper Ubuntu 22.04 container
set -euo pipefail

HUB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-vitis-ubuntu22-hub}"
ENCODER_BIN="/tmp/whisper_encode_dump"
ENCODER_SRC="$HUB_DIR/tools/whisper_encode_dump.c"
ORT_INCLUDE="$HUB_DIR/third_party/onnxruntime/include"

ensure_container() {
    if podman ps --format '{{.Names}}' | rg -x "$CONTAINER_NAME" >/dev/null 2>&1; then
        return
    fi
    if podman ps -a --format '{{.Names}}' | rg -x "$CONTAINER_NAME" >/dev/null 2>&1; then
        podman start "$CONTAINER_NAME" >/dev/null
        return
    fi

    podman run -d --name "$CONTAINER_NAME" \
        --security-opt label=disable \
        --device /dev/accel/accel0 \
        -v /sys/class/accel:/sys/class/accel:ro \
        -v /var/home/mariostjr/amd-rai-linux:/work/amd-rai-linux:ro \
        -v "$HUB_DIR":/work/hub \
        docker.io/library/ubuntu:22.04 \
        sleep infinity >/dev/null
}

ensure_deps() {
    podman exec "$CONTAINER_NAME" bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update >/dev/null
apt-get install -y --no-install-recommends \
  build-essential python3 python3-pip \
  libboost-filesystem1.74.0 \
  libboost-program-options1.74.0 \
  libboost-system1.74.0 \
  libsndfile1 >/dev/null
python3 -m pip install --disable-pip-version-check --no-cache-dir \
  onnxruntime==1.22.1 onnx transformers soundfile sentencepiece >/dev/null
'
}

compile_encoder() {
    podman exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
gcc -O2 -o /tmp/whisper_encode_dump \
  /work/hub/tools/whisper_encode_dump.c \
  -I/work/hub/third_party/onnxruntime/include \
  -L/work/amd-rai-linux/ryzen14 \
  -lonnxruntime \
  -Wl,-rpath,/work/amd-rai-linux/ryzen14
"
}

run_in_container() {
    local mode="$1"
    local audio_path="${2:-}"
    if [ "$mode" = "test" ]; then
        podman exec "$CONTAINER_NAME" bash -lc '
set -euo pipefail
export WHISPER_HUB_DIR=/work/hub
export WHISPER_ENCODER_BIN=/tmp/whisper_encode_dump
export WHISPER_CACHE_DIR=/tmp/whisper_cache
export WHISPER_TMP_DIR=/tmp/whisper_tmp
python3 /work/hub/tools/run_whisper_hybrid_transcribe.py --test -v
'
        return
    fi

    podman cp "$audio_path" "$CONTAINER_NAME":/tmp/whisper_input.wav >/dev/null
    podman exec "$CONTAINER_NAME" bash -lc '
set -euo pipefail
export WHISPER_HUB_DIR=/work/hub
export WHISPER_ENCODER_BIN=/tmp/whisper_encode_dump
export WHISPER_CACHE_DIR=/tmp/whisper_cache
export WHISPER_TMP_DIR=/tmp/whisper_tmp
python3 /work/hub/tools/run_whisper_hybrid_transcribe.py --audio /tmp/whisper_input.wav -v
'
}

main() {
    local mode="test"
    local audio_path=""
    if [ "${1:-}" = "--audio" ]; then
        mode="audio"
        audio_path="${2:-}"
        if [ -z "$audio_path" ] || [ ! -f "$audio_path" ]; then
            echo "[FAIL] provide an existing WAV path after --audio" >&2
            exit 2
        fi
    elif [ "${1:-}" = "--test" ] || [ $# -eq 0 ]; then
        mode="test"
    else
        echo "usage: $0 [--test] [--audio /abs/path/file.wav]" >&2
        exit 2
    fi

    ensure_container
    ensure_deps
    if [ ! -f "$ENCODER_BIN" ] || [ "$ENCODER_SRC" -nt "$ENCODER_BIN" ]; then
        compile_encoder
    fi
    run_in_container "$mode" "$audio_path"
}

main "$@"
