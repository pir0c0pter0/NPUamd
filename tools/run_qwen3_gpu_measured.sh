#!/usr/bin/env bash
set -euo pipefail

HUB_DIR="${HUB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
PROMPT="${PROMPT:-Responda em portugues em duas frases: qual o estado atual da NPU AMD neste host Linux?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DTYPE="${DTYPE:-bfloat16}"
INTERVAL="${INTERVAL:-1}"
OUT_DIR="${OUT_DIR:-$HUB_DIR/runtime/gpu_runs}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-gpu-pytorch}"
RUNNER="$HUB_DIR/tools/run_qwen3_gpu_container.sh"
MONITOR="$HUB_DIR/tools/monitor_apu_usage.sh"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[FAIL] missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

slugify() {
  local value="$1"
  value="${value//\//__}"
  value="${value// /_}"
  printf '%s' "$value"
}

find_host_pid() {
  local model="$1"
  ps -eo pid=,args= | awk -v model="$model" '
    $0 ~ /python3 \/work\/hub\/tools\/run_qwen3_gpu_transformers\.py/ && index($0, model) {
      print $1
      exit
    }
  '
}

summarize_csv() {
  local path="$1"
  python3 - "$path" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    print("[WARN] monitor CSV is empty")
    raise SystemExit(0)

def max_float(key):
    values = []
    for row in rows:
      raw = row.get(key, "na")
      if raw not in ("", "na", None):
        values.append(float(raw))
    return max(values) if values else None

summary = {
    "samples": len(rows),
    "max_cpu_pct": max_float("cpu_pct"),
    "max_rss_mib": max_float("rss_mib"),
    "max_gpu_busy_pct": max_float("gpu_busy_pct"),
    "max_gpu_vram_used_gib": max_float("gpu_vram_used_gib"),
    "max_gpu_power_w": max_float("gpu_power_w"),
}

for key, value in summary.items():
    if value is None:
        print(f"{key}=na")
    elif key == "samples":
        print(f"{key}={int(value)}")
    else:
        print(f"{key}={value:.2f}")
PY
}

main() {
  local ts model_slug log_path monitor_path runner_pid host_pid

  require_file "$RUNNER" "GPU runner"
  require_file "$MONITOR" "APU monitor"
  mkdir -p "$OUT_DIR"

  ts="$(date +%Y%m%dT%H%M%S)"
  model_slug="$(slugify "$MODEL_ID")"
  log_path="$OUT_DIR/${model_slug}_${ts}.log"
  monitor_path="$OUT_DIR/${model_slug}_${ts}.csv"

  printf '[INFO] model=%s\n' "$MODEL_ID"
  printf '[INFO] log=%s\n' "$log_path"
  printf '[INFO] monitor=%s\n' "$monitor_path"

  (
    cd "$HUB_DIR"
    MODEL_ID="$MODEL_ID" \
    PROMPT="$PROMPT" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    DTYPE="$DTYPE" \
      bash "$RUNNER"
  ) >"$log_path" 2>&1 &
  runner_pid="$!"

  host_pid=""
  for _ in $(seq 1 120); do
    host_pid="$(find_host_pid "$MODEL_ID" || true)"
    if [[ -n "$host_pid" ]]; then
      break
    fi
    if ! kill -0 "$runner_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if [[ -n "$host_pid" ]]; then
    printf '[INFO] host_pid=%s\n' "$host_pid"
    (
      cd "$HUB_DIR"
      bash "$MONITOR" "$host_pid" "$INTERVAL" >"$monitor_path"
    ) &
  else
    printf '[WARN] could not resolve host pid for %s; monitor not started\n' "$MODEL_ID" >&2
  fi

  wait "$runner_pid"
  printf '[INFO] run finished for %s\n' "$MODEL_ID"

  if [[ -f "$monitor_path" ]]; then
    summarize_csv "$monitor_path"
  fi
}

main "$@"
