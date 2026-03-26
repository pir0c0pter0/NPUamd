#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RUN_DIR="${RUN_DIR:-$HUB_DIR/runtime/llm_linux/run_qwen3_14b_hybrid}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-14B-onnx-ryzenai-1.7-hybrid}"
export MODEL_DIR="${MODEL_DIR:-$HUB_DIR/runtime/llm_linux/models/$MODEL_NAME}"

exec bash "$HUB_DIR/tools/run_oga_llm_linux.sh" "$@"
