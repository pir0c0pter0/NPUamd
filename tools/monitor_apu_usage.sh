#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'EOF' >&2
Usage:
  bash tools/monitor_apu_usage.sh <pid> [interval_seconds]
EOF
  exit 1
fi

PID="$1"
INTERVAL="${2:-1}"
GPU_SYSFS_BASE="${GPU_SYSFS_BASE:-/sys/class/drm/card1/device}"
NPU_SYSFS_BASE="${NPU_SYSFS_BASE:-/sys/class/accel/accel0}"

read_sysfs() {
  local path="$1"
  if [[ -f "$path" ]]; then
    cat "$path"
  else
    printf 'na'
  fi
}

read_first_matching_file() {
  local pattern="$1"
  local matches=()

  shopt -s nullglob
  matches=($pattern)
  shopt -u nullglob

  if ((${#matches[@]} == 0)); then
    printf 'na'
    return
  fi

  read_sysfs "${matches[0]}"
}

bytes_to_gib() {
  local value="$1"
  if [[ "$value" == "na" ]]; then
    printf 'na'
    return
  fi
  awk -v v="$value" 'BEGIN { printf "%.2f", v / 1024 / 1024 / 1024 }'
}

microwatts_to_watts() {
  local value="$1"
  if [[ "$value" == "na" ]]; then
    printf 'na'
    return
  fi
  awk -v v="$value" 'BEGIN { printf "%.2f", v / 1000000 }'
}

printf 'timestamp,pid,cpu_pct,rss_mib,gpu_busy_pct,gpu_vram_used_gib,gpu_vram_total_gib,gpu_power_w,npu_runtime_status,npu_runtime_active_ms\n'

while kill -0 "$PID" 2>/dev/null; do
  timestamp="$(date --iso-8601=seconds)"
  read -r cpu_pct rss_kib < <(ps -p "$PID" -o %cpu=,rss= 2>/dev/null || printf 'na na\n')
  gpu_busy="$(read_sysfs "$GPU_SYSFS_BASE/gpu_busy_percent")"
  gpu_vram_used="$(read_sysfs "$GPU_SYSFS_BASE/mem_info_vram_used")"
  gpu_vram_total="$(read_sysfs "$GPU_SYSFS_BASE/mem_info_vram_total")"
  gpu_power="$(read_first_matching_file "$GPU_SYSFS_BASE/hwmon/hwmon*/power1_average")"
  npu_runtime_status="$(read_sysfs "$NPU_SYSFS_BASE/power/runtime_status")"
  npu_runtime_active_ms="$(read_sysfs "$NPU_SYSFS_BASE/power/runtime_active_time")"
  rss_mib="$(awk -v rss="${rss_kib:-0}" 'BEGIN { if (rss == "na") print "na"; else printf "%.2f", rss / 1024 }')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" \
    "$PID" \
    "${cpu_pct:-na}" \
    "$rss_mib" \
    "$gpu_busy" \
    "$(bytes_to_gib "$gpu_vram_used")" \
    "$(bytes_to_gib "$gpu_vram_total")" \
    "$(microwatts_to_watts "$gpu_power")" \
    "$npu_runtime_status" \
    "$npu_runtime_active_ms"

  sleep "$INTERVAL"
done
