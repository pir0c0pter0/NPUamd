#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-gpu-pytorch}"
IMAGE="${IMAGE:-docker.io/rocm/pytorch:latest}"
HF_HOME_DIR="${HF_HOME_DIR:-/work/hub/.cache/huggingface_rocm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
PROMPT="${PROMPT:-Responda em portugues em duas frases: qual o estado atual da NPU AMD neste host Linux?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"

ensure_container() {
  if ! podman container exists "$CONTAINER_NAME"; then
    podman run -d \
      --name "$CONTAINER_NAME" \
      --security-opt label=disable \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --device /dev/kfd \
      --device /dev/dri \
      --group-add keep-groups \
      --ipc=host \
      -v "$HUB_DIR:/work/hub" \
      "$IMAGE" \
      sleep infinity >/dev/null
  fi

  if [[ "$(podman inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
    podman start "$CONTAINER_NAME" >/dev/null
  fi
}

install_python_bits() {
  podman exec "$CONTAINER_NAME" bash -lc '
    python3 -m pip install --no-cache-dir -q transformers accelerate sentencepiece safetensors
    python3 -m pip uninstall -y torchvision >/dev/null 2>&1 || true
  '
}

run_model() {
  podman exec "$CONTAINER_NAME" bash -lc "
    export HF_HOME='$HF_HOME_DIR'
    python3 /work/hub/tools/run_qwen3_gpu_transformers.py \
      --model-id '$MODEL_ID' \
      --prompt '$PROMPT' \
      --max-new-tokens '$MAX_NEW_TOKENS' \
      --dtype '$DTYPE'
  "
}

ensure_container
install_python_bits
run_model
