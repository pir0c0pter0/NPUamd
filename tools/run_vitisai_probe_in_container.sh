#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vitis-ubuntu22}"
HUB_DIR="${HUB_DIR:-/var/home/mariostjr/Documents/hubs/NPUamd}"
AMD_DIR="${AMD_DIR:-/work/amd-rai-linux/ryzen14}"
MODEL_PATH_IN_CONTAINER="${MODEL_PATH_IN_CONTAINER:-/tmp/minimal_identity_ir10.onnx}"

podman exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip build-essential libboost-filesystem1.74.0 libboost-program-options1.74.0 libboost-system1.74.0 >/dev/null
python3 -m pip install --no-cache-dir onnx >/dev/null
python3 '${HUB_DIR}/tools/make_minimal_identity_model.py' '${MODEL_PATH_IN_CONTAINER}' >/dev/null
gcc \
  -I'${HUB_DIR}/third_party/onnxruntime/include' \
  -L'${AMD_DIR}' \
  -Wl,-rpath,'${AMD_DIR}' \
  -o /tmp/probe_vitisai \
  '${HUB_DIR}/tools/probe_vitisai.c' \
  -lonnxruntime
export LD_LIBRARY_PATH='${AMD_DIR}'
/tmp/probe_vitisai '${MODEL_PATH_IN_CONTAINER}' \"\$@\"
" -- "$@"
