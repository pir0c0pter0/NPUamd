#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${CONTAINER_NAME:-llm-gpu-pytorch}"
IMAGE="${IMAGE:-docker.io/rocm/pytorch:latest}"
HF_HOME_DIR="${HF_HOME_DIR:-/work/hub/.cache/huggingface_rocm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
PROMPT="${PROMPT:-Responda em portugues em duas frases: qual o estado atual da NPU AMD neste host Linux?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-none}"
EXTRA_PKGS="${EXTRA_PKGS:-}"

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
  local base_pkgs="transformers accelerate sentencepiece safetensors"
  local quant_pkgs=""

  if [[ "$QUANTIZATION" == "gptq" ]]; then
    quant_pkgs="gptqmodel"
  elif [[ "$QUANTIZATION" == "bnb4" ]]; then
    quant_pkgs="bitsandbytes"
  fi

  if [[ -n "$EXTRA_PKGS" ]]; then
    quant_pkgs="$quant_pkgs $EXTRA_PKGS"
  fi

  podman exec "$CONTAINER_NAME" bash -lc "
    python3 -m pip install --no-cache-dir -q $base_pkgs
    python3 -m pip uninstall -y torchvision >/dev/null 2>&1 || true
  "

  if [[ -n "$quant_pkgs" ]]; then
    podman exec "$CONTAINER_NAME" bash -lc "
      BUILD_CUDA_EXT=0 python3 -m pip install --no-cache-dir -q $quant_pkgs || {
        printf '[WARN] quantization package install failed, trying without build isolation...\n' >&2
        python3 -m pip install --no-cache-dir --no-build-isolation -q $quant_pkgs
      }
    "
  fi
}

run_model() {
  podman exec "$CONTAINER_NAME" bash -lc "
    export HF_HOME='$HF_HOME_DIR'
    python3 /work/hub/tools/run_llm_gpu_transformers.py \
      --model-id '$MODEL_ID' \
      --prompt '$PROMPT' \
      --max-new-tokens '$MAX_NEW_TOKENS' \
      --dtype '$DTYPE' \
      --quantization '$QUANTIZATION'
  "
}

ensure_container
install_python_bits
run_model
