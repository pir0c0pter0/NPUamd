#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-R1-Distill-Llama-70B on AMD iGPU via ROCm container
# Uses GPTQ INT4 quantization (~37-42 GB VRAM) to fit in 64 GB iGPU

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
export QUANTIZATION="${QUANTIZATION:-none}"
export DTYPE="${DTYPE:-bfloat16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export PROMPT="${PROMPT:-Explain in two sentences what an AMD NPU is and how it differs from a GPU.}"
export CONTAINER_NAME="${CONTAINER_NAME:-llm-gpu-pytorch}"

if [[ "$MODEL_ID" == *gptq* || "$MODEL_ID" == *GPTQ* ]]; then
  export QUANTIZATION="gptq"
  export DTYPE="float16"
fi

printf '[INFO] DeepSeek-R1-Distill-Llama-70B GPU runner\n'
printf '[INFO] model=%s\n' "$MODEL_ID"
printf '[INFO] quantization=%s\n' "$QUANTIZATION"
printf '[INFO] dtype=%s\n' "$DTYPE"
printf '[INFO] max_new_tokens=%s\n' "$MAX_NEW_TOKENS"

exec bash "$HUB_DIR/tools/run_llm_gpu_measured.sh" "$@"
