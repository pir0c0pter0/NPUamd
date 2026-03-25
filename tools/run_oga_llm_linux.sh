#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-$HUB_DIR/runtime/llm_linux/run_phi35}"
MODEL_NAME="${MODEL_NAME:-Phi-3.5-mini-instruct-onnx-ryzenai-npu}"
DEFAULT_MODEL_DIR="$RUN_DIR/$MODEL_NAME"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
RYZEN_AI_VENV="${RYZEN_AI_VENV:-${RYZEN_AI_INSTALLATION_PATH:-}}"
DEPLOYMENT_SRC="${DEPLOYMENT_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/deployment}}"
MODEL_BENCHMARK_SRC="${MODEL_BENCHMARK_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/LLM/examples/model_benchmark}}"
PROMPT_SRC="${PROMPT_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/LLM/examples/amd_genai_prompt.txt}}"
DEPLOYMENT_DIR="$RUN_DIR/deployment"
MODEL_BENCHMARK="$RUN_DIR/model_benchmark"
DEFAULT_PROMPT_FILE="$RUN_DIR/amd_genai_prompt.txt"
PROMPT_FILE="${PROMPT_FILE:-$DEFAULT_PROMPT_FILE}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
PREPARE_ONLY=0
FORCE_STAGE=0
EXTRA_ARGS=()

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

require_glob() {
  local pattern="$1"
  local label="$2"
  local matches=()
  shopt -s nullglob
  matches=($pattern)
  shopt -u nullglob
  if ((${#matches[@]} == 0)); then
    printf '[FAIL] missing %s matching: %s\n' "$label" "$pattern" >&2
    exit 1
  fi
}

is_lfs_pointer() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  [[ "$(sed -n '1p' "$path" 2>/dev/null || true)" == "version https://git-lfs.github.com/spec/v1" ]]
}

require_materialized_file() {
  local path="$1"
  local label="$2"
  require_file "$path" "$label"
  if is_lfs_pointer "$path"; then
    printf '[FAIL] %s is still a git-lfs pointer: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_oga_llm_linux.sh [options] [-- extra model_benchmark args]

Options:
  --model-dir PATH       Path to the materialized AMD OGA model directory
  --prompt-length N      Prompt length passed to model_benchmark (default: 128)
  --prompt-file PATH     Prompt file passed to model_benchmark
  --prepare-only         Stage runtime files and patch the model, but do not run
  --force-stage          Re-copy deployment/, model_benchmark and prompt file
  -h, --help             Show this help

Environment:
  RYZEN_AI_VENV or RYZEN_AI_INSTALLATION_PATH
    Must point to the Linux venv from the official AMD install, for example:
    /path/to/ryzen_ai-1.6.1/venv
EOF
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --model-dir)
        MODEL_DIR="$2"
        shift 2
        ;;
      --prompt-length)
        PROMPT_LENGTH="$2"
        shift 2
        ;;
      --prompt-file)
        PROMPT_FILE="$2"
        shift 2
        ;;
      --prepare-only)
        PREPARE_ONLY=1
        shift
        ;;
      --force-stage)
        FORCE_STAGE=1
        shift
        ;;
      --)
        shift
        EXTRA_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

stage_runtime() {
  if [[ -z "$RYZEN_AI_VENV" ]]; then
    printf '[FAIL] missing RYZEN_AI_VENV/RYZEN_AI_INSTALLATION_PATH.\n' >&2
    printf '[FAIL] the official Linux LLM flow expects a Ryzen AI venv such as <TARGET-PATH>/ryzen_ai-1.6.1/venv\n' >&2
    exit 1
  fi

  require_dir "$RYZEN_AI_VENV" "Ryzen AI Linux venv"
  require_dir "$DEPLOYMENT_SRC" "deployment source"
  require_file "$MODEL_BENCHMARK_SRC" "model_benchmark source"
  require_file "$PROMPT_SRC" "amd_genai_prompt.txt source"

  mkdir -p "$RUN_DIR"

  if [[ "$FORCE_STAGE" -eq 1 && -d "$DEPLOYMENT_DIR" ]]; then
    rm -rf "$DEPLOYMENT_DIR"
  fi

  if [[ ! -d "$DEPLOYMENT_DIR" ]]; then
    cp -a "$DEPLOYMENT_SRC" "$DEPLOYMENT_DIR"
  fi

  if [[ "$FORCE_STAGE" -eq 1 || ! -f "$MODEL_BENCHMARK" ]]; then
    cp -f "$MODEL_BENCHMARK_SRC" "$MODEL_BENCHMARK"
    chmod +x "$MODEL_BENCHMARK"
  fi

  if [[ "$PROMPT_FILE" == "$DEFAULT_PROMPT_FILE" ]]; then
    if [[ "$FORCE_STAGE" -eq 1 || ! -f "$DEFAULT_PROMPT_FILE" ]]; then
      cp -f "$PROMPT_SRC" "$DEFAULT_PROMPT_FILE"
    fi
  fi

  require_glob "$DEPLOYMENT_DIR/lib/libonnxruntime-genai.so*" "Linux OGA runtime"
  require_glob "$DEPLOYMENT_DIR/lib/libonnx_custom_ops.so*" "Linux OGA custom ops library"
  require_glob "$DEPLOYMENT_DIR/lib/libryzen_mm.so*" "Linux OGA custom allocator"
}

link_model_into_run_dir() {
  local abs_model_dir
  local link_path

  abs_model_dir="$(realpath "$MODEL_DIR")"
  MODEL_NAME="$(basename "$abs_model_dir")"
  link_path="$RUN_DIR/$MODEL_NAME"

  if [[ "$abs_model_dir" != "$link_path" ]]; then
    ln -sfn "$abs_model_dir" "$link_path"
  fi
}

validate_model() {
  require_dir "$MODEL_DIR" "OGA model directory"
  require_materialized_file "$MODEL_DIR/genai_config.json" "genai_config.json"
  require_materialized_file "$MODEL_DIR/fusion.onnx" "fusion.onnx"
  require_materialized_file "$MODEL_DIR/fusion.onnx.data" "fusion.onnx.data"
  require_materialized_file "$MODEL_DIR/prefill.bin" "prefill.bin"
  require_materialized_file "$MODEL_DIR/prefill.pb.bin" "prefill.pb.bin"
  require_materialized_file "$MODEL_DIR/.cache/MatMulNBits_2_0_meta.json" "MatMulNBits_2_0_meta.json"
}

patch_model_for_linux() {
  require_file "$HUB_DIR/tools/patch_oga_linux_model.py" "Linux OGA patch helper"
  python3 "$HUB_DIR/tools/patch_oga_linux_model.py" \
    "$MODEL_DIR" \
    --custom-ops-library "deployment/lib/libonnx_custom_ops.so"
}

run_model_benchmark() {
  local runtime_ld_path

  require_file "$PROMPT_FILE" "prompt file"
  require_file "$MODEL_BENCHMARK" "model_benchmark"

  runtime_ld_path="$DEPLOYMENT_DIR/lib"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    runtime_ld_path="$runtime_ld_path:$LD_LIBRARY_PATH"
  fi

  printf '[INFO] using Ryzen AI Linux venv: %s\n' "$RYZEN_AI_VENV"
  printf '[INFO] using run dir: %s\n' "$RUN_DIR"
  printf '[INFO] using model dir: %s\n' "$MODEL_DIR"
  printf '[INFO] using prompt file: %s\n' "$PROMPT_FILE"
  printf '[INFO] using prompt length: %s\n' "$PROMPT_LENGTH"

  (
    cd "$RUN_DIR"
    LD_LIBRARY_PATH="$runtime_ld_path" \
      "$MODEL_BENCHMARK" \
      -i "$MODEL_NAME/" \
      -l "$PROMPT_LENGTH" \
      -f "$(realpath --relative-to="$RUN_DIR" "$PROMPT_FILE")" \
      "${EXTRA_ARGS[@]}"
  )
}

main() {
  parse_args "$@"
  stage_runtime
  validate_model
  patch_model_for_linux
  link_model_into_run_dir

  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    printf '[INFO] Linux OGA run tree prepared under %s\n' "$RUN_DIR"
    exit 0
  fi

  run_model_benchmark
}

main "$@"
