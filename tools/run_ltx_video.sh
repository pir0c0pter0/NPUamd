#!/bin/bash
# LTX-2.3 video generation on Radeon 8060S (gfx1151) via ROCm 7.2
# Usage: ./run_ltx_video.sh "prompt" output.mp4 [num_frames] [height] [width]
set -euo pipefail

LTX_DIR="/home/mariostjr/Documentos/hubs/LTX-2"
PROMPT="${1:?Usage: $0 \"prompt\" output.mp4 [num_frames] [height] [width]}"
OUTPUT="${2:?Usage: $0 \"prompt\" output.mp4 [num_frames] [height] [width]}"
NUM_FRAMES="${3:-249}"   # 249 = ~10s at 24fps
HEIGHT="${4:-512}"
WIDTH="${5:-768}"

export LD_PRELOAD="/usr/lib/libpthread.so.0 /opt/rocm/lib/librocprofiler-sdk.so.1"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

cd "$LTX_DIR"
exec .venv/bin/python run_ltx_distilled.py \
  --distilled-checkpoint-path models/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root models/gemma3 \
  --lora models/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --prompt "$PROMPT" \
  --output-path "$OUTPUT" \
  --num-frames "$NUM_FRAMES" \
  --height "$HEIGHT" \
  --width "$WIDTH"
