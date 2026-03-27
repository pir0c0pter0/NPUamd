#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-}"
MODEL_NAME="${MODEL_NAME:-Phi-3.5-mini-instruct-onnx-ryzenai-npu}"
MODEL_DIR="${MODEL_DIR:-}"
RYZEN_AI_VENV="${RYZEN_AI_VENV:-${RYZEN_AI_INSTALLATION_PATH:-}}"
DEPLOYMENT_SRC="${DEPLOYMENT_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/deployment}}"
MODEL_BENCHMARK_SRC="${MODEL_BENCHMARK_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/LLM/examples/model_benchmark}}"
PROMPT_SRC="${PROMPT_SRC:-${RYZEN_AI_VENV:+$RYZEN_AI_VENV/LLM/examples/amd_genai_prompt.txt}}"
DEPLOYMENT_DIR=""
MODEL_BENCHMARK=""
DEFAULT_PROMPT_FILE=""
PROMPT_FILE="${PROMPT_FILE:-}"
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

file_is_materialized() {
  local path="$1"
  [[ -f "$path" && -s "$path" ]] && ! is_lfs_pointer "$path"
}

glob_has_materialized_match() {
  local pattern="$1"
  local matches=()
  local match
  shopt -s nullglob
  matches=($pattern)
  shopt -u nullglob
  if ((${#matches[@]} == 0)); then
    return 1
  fi
  for match in "${matches[@]}"; do
    if file_is_materialized "$match"; then
      return 0
    fi
  done
  return 1
}

is_lfs_pointer() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  python3 - "$path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    with path.open("rb") as handle:
        first_line = handle.readline(128)
except OSError:
    raise SystemExit(1)

if first_line.rstrip(b"\r\n") == b"version https://git-lfs.github.com/spec/v1":
    raise SystemExit(0)

raise SystemExit(1)
PY
}

require_materialized_file() {
  local path="$1"
  local label="$2"
  require_file "$path" "$label"
  if [[ ! -s "$path" ]]; then
    printf '[FAIL] %s is empty: %s\n' "$label" "$path" >&2
    exit 1
  fi
  if is_lfs_pointer "$path"; then
    printf '[FAIL] %s is still a git-lfs pointer: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

require_materialized_glob() {
  local pattern="$1"
  local label="$2"
  if ! glob_has_materialized_match "$pattern"; then
    printf '[FAIL] missing materialized %s matching: %s\n' "$label" "$pattern" >&2
    exit 1
  fi
}

require_executable_file() {
  local path="$1"
  local label="$2"
  require_materialized_file "$path" "$label"
  if [[ ! -x "$path" ]]; then
    printf '[FAIL] %s is not executable: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

collect_required_model_files() {
  local model_dir="$1"
  python3 - "$model_dir" <<'PY'
import json
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def add_required(required: list[str], seen: set[str], rel_path: str) -> None:
    normalized = rel_path.replace("\\", "/").strip()
    if not normalized:
        return
    normalized = normalized.removeprefix("./")
    if normalized not in seen:
        seen.add(normalized)
        required.append(normalized)


model_dir = Path(sys.argv[1]).resolve()
config_path = model_dir / "genai_config.json"

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    fail(f"missing genai_config.json: {config_path}")
except json.JSONDecodeError as exc:
    fail(f"invalid genai_config.json: {config_path}: {exc}")

decoder = config.get("model", {}).get("decoder")
if not isinstance(decoder, dict):
    fail(f"missing model.decoder in {config_path}")

decoder_filename = decoder.get("filename")
if not isinstance(decoder_filename, str) or not decoder_filename.strip():
    fail(f"missing model.decoder.filename in {config_path}")

required: list[str] = []
seen: set[str] = set()
add_required(required, seen, decoder_filename)

onnx_path = model_dir / decoder_filename
onnx_sidecar = model_dir / f"{decoder_filename}.data"
if onnx_sidecar.exists():
    add_required(required, seen, f"{decoder_filename}.data")

provider_options = decoder.get("session_options", {}).get("provider_options", [])
if isinstance(provider_options, list):
    for provider_entry in provider_options:
        if not isinstance(provider_entry, dict):
            continue
        for provider_config in provider_entry.values():
            if not isinstance(provider_config, dict):
                continue
            external_data_file = provider_config.get("external_data_file")
            if not isinstance(external_data_file, str) or not external_data_file.strip():
                continue
            add_required(required, seen, external_data_file)
            external_data_name = Path(external_data_file.replace("\\", "/")).name
            if external_data_name.endswith(".pb.bin"):
                external_weights = external_data_name[: -len(".pb.bin")] + ".bin"
                if (model_dir / external_weights).exists():
                    add_required(required, seen, external_weights)

if onnx_path.is_file():
    referenced_files: set[str] = set()
    for match in re.finditer(rb"([A-Za-z0-9_.\\/-]+\.(?:bin|data))", onnx_path.read_bytes()):
        candidate = match.group(1).decode("utf-8", errors="ignore").replace("\\", "/")
        candidate_path = model_dir / candidate
        if candidate_path.exists():
            referenced_files.add(candidate)
            continue

        basename = Path(candidate).name
        if basename and (model_dir / basename).exists():
            referenced_files.add(basename)

    for candidate in sorted(referenced_files):
        add_required(required, seen, candidate)

dd_plugins_dir = model_dir / "dd_plugins"
if dd_plugins_dir.is_dir():
    plugin_files = sorted(
        path.relative_to(model_dir).as_posix()
        for path in dd_plugins_dir.rglob("*")
        if path.is_file()
    )
    if not plugin_files:
        fail(f"dd_plugins exists but is empty: {dd_plugins_dir}")
    for rel_path in plugin_files:
        add_required(required, seen, rel_path)

for rel_path in required:
    print(rel_path)
PY
}

model_uses_provider() {
  local model_dir="$1"
  local provider_name="$2"
  python3 - "$model_dir" "$provider_name" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
provider_name = sys.argv[2]
config_path = model_dir / "genai_config.json"

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

provider_options = (
    config.get("model", {})
    .get("decoder", {})
    .get("session_options", {})
    .get("provider_options", [])
)

if not isinstance(provider_options, list):
    raise SystemExit(1)

for entry in provider_options:
    if isinstance(entry, dict) and provider_name in entry:
        raise SystemExit(0)

raise SystemExit(1)
PY
}

finalize_paths() {
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$HUB_DIR/runtime/llm_linux/run_phi35"
  fi

  DEPLOYMENT_DIR="$RUN_DIR/deployment"
  MODEL_BENCHMARK="$RUN_DIR/model_benchmark"
  DEFAULT_PROMPT_FILE="$RUN_DIR/amd_genai_prompt.txt"

  if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_FILE="$DEFAULT_PROMPT_FILE"
  fi

  if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="$RUN_DIR/$MODEL_NAME"
  fi
}

runtime_stage_needs_refresh() {
  if [[ ! -d "$DEPLOYMENT_DIR" ]]; then
    return 0
  fi
  if ! glob_has_materialized_match "$DEPLOYMENT_DIR/lib/libonnxruntime-genai.so*"; then
    return 0
  fi
  if ! glob_has_materialized_match "$DEPLOYMENT_DIR/lib/libonnx_custom_ops.so*"; then
    return 0
  fi
  if ! glob_has_materialized_match "$DEPLOYMENT_DIR/lib/libryzen_mm.so*"; then
    return 0
  fi
  return 1
}

print_missing_runtime_help() {
  printf '[FAIL] OGA runtime staging under %s is incomplete or still placeholder-only.\n' "$RUN_DIR" >&2
  printf '[FAIL] expected non-empty deployment/lib/libonnxruntime-genai.so, libonnx_custom_ops.so, libryzen_mm.so, model_benchmark and prompt file.\n' >&2
  printf '[FAIL] point RYZEN_AI_VENV/RYZEN_AI_INSTALLATION_PATH at the official Linux venv and rerun.\n' >&2
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_oga_llm_linux.sh [options] [-- extra model_benchmark args]

Options:
  --run-dir PATH         Run directory used for staged runtime + symlinked model
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
      --run-dir)
        RUN_DIR="$2"
        shift 2
        ;;
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
  mkdir -p "$RUN_DIR"

  if [[ "$FORCE_STAGE" -eq 1 ]] || runtime_stage_needs_refresh || ! file_is_materialized "$MODEL_BENCHMARK" || ! file_is_materialized "$PROMPT_FILE"; then
    if [[ -z "$RYZEN_AI_VENV" ]]; then
      print_missing_runtime_help
      printf '[FAIL] missing RYZEN_AI_VENV/RYZEN_AI_INSTALLATION_PATH.\n' >&2
      printf '[FAIL] the official Linux LLM flow expects a Ryzen AI venv such as <TARGET-PATH>/ryzen_ai-1.6.1/venv\n' >&2
      exit 1
    fi

    require_dir "$RYZEN_AI_VENV" "Ryzen AI Linux venv"
    require_dir "$DEPLOYMENT_SRC" "deployment source"
    require_file "$MODEL_BENCHMARK_SRC" "model_benchmark source"
    require_file "$PROMPT_SRC" "amd_genai_prompt.txt source"

    if [[ -d "$DEPLOYMENT_DIR" ]]; then
      rm -rf "$DEPLOYMENT_DIR"
    fi
    cp -a "$DEPLOYMENT_SRC" "$DEPLOYMENT_DIR"
    cp -f "$MODEL_BENCHMARK_SRC" "$MODEL_BENCHMARK"
    chmod +x "$MODEL_BENCHMARK"

    if [[ "$PROMPT_FILE" == "$DEFAULT_PROMPT_FILE" ]]; then
      cp -f "$PROMPT_SRC" "$DEFAULT_PROMPT_FILE"
    elif [[ ! -f "$PROMPT_FILE" ]]; then
      printf '[FAIL] custom prompt file does not exist: %s\n' "$PROMPT_FILE" >&2
      exit 1
    fi
  fi

  require_materialized_glob "$DEPLOYMENT_DIR/lib/libonnxruntime-genai.so*" "Linux OGA runtime"
  require_materialized_glob "$DEPLOYMENT_DIR/lib/libonnx_custom_ops.so*" "Linux OGA custom ops library"
  require_materialized_glob "$DEPLOYMENT_DIR/lib/libryzen_mm.so*" "Linux OGA custom allocator"
  require_executable_file "$MODEL_BENCHMARK" "model_benchmark"
  require_materialized_file "$PROMPT_FILE" "prompt file"
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
  local rel_path
  local required_files=()

  require_dir "$MODEL_DIR" "OGA model directory"
  require_materialized_file "$MODEL_DIR/genai_config.json" "genai_config.json"

  mapfile -t required_files < <(collect_required_model_files "$MODEL_DIR")
  if ((${#required_files[@]} == 0)); then
    printf '[FAIL] could not infer required OGA model files from %s/genai_config.json\n' "$MODEL_DIR" >&2
    exit 1
  fi

  for rel_path in "${required_files[@]}"; do
    require_materialized_file "$MODEL_DIR/$rel_path" "OGA model artifact ($rel_path)"
  done
}

patch_model_for_linux() {
  require_file "$HUB_DIR/tools/patch_oga_linux_model.py" "Linux OGA patch helper"
  python3 "$HUB_DIR/tools/patch_oga_linux_model.py" \
    "$MODEL_DIR" \
    --custom-ops-library "deployment/lib/libonnx_custom_ops.so"
}

validate_runtime_for_model() {
  if model_uses_provider "$MODEL_DIR" "RyzenAI"; then
    require_materialized_glob "$DEPLOYMENT_DIR/lib/libonnxruntime_providers_ryzenai.so*" "RyzenAI provider library"
  fi
}

run_model_benchmark() {
  local runtime_ld_path

  require_materialized_file "$PROMPT_FILE" "prompt file"
  require_executable_file "$MODEL_BENCHMARK" "model_benchmark"

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
  finalize_paths
  stage_runtime
  validate_model
  patch_model_for_linux
  validate_runtime_for_model
  link_model_into_run_dir

  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    printf '[INFO] Linux OGA run tree prepared under %s\n' "$RUN_DIR"
    exit 0
  fi

  run_model_benchmark
}

main "$@"
