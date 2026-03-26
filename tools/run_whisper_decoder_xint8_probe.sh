#!/usr/bin/env bash
# Probe Whisper XINT8 decoder through VitisAI EP on host Linux
# Decoder has 2 inputs - uses Python probe instead of C probe
set -euo pipefail

HUB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AMD_LIB="/var/home/mariostjr/amd-rai-linux/ryzen14"
UBUNTU_LIB="$HUB_DIR/runtime/ubuntu22/lib"
MODEL="$HUB_DIR/runtime/whisper/models/tiny_en_decoder_xint8.onnx"
CONFIG="$AMD_LIB/vaip_config_npu_2_3.json"
CACHE_DIR="$HUB_DIR/runtime/whisper/cache/xint8_decoder"

mkdir -p "$CACHE_DIR"

export LD_LIBRARY_PATH="${AMD_LIB}:${UBUNTU_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${AMD_LIB}:${PYTHONPATH:-}"

echo "=== Whisper decoder XINT8 probe ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG"
echo ""

python3 -c "
import sys, os, time, json
sys.path.insert(0, '$AMD_LIB')
os.environ['LD_LIBRARY_PATH'] = '${AMD_LIB}:${UBUNTU_LIB}:' + os.environ.get('LD_LIBRARY_PATH', '')

import numpy as np
import onnxruntime as ort

print(f'ORT version: {ort.__version__}')
print(f'EPs: {ort.get_available_providers()}')

provider = 'VitisAIExecutionProvider'
opts = {
    'config_file': '$CONFIG',
    'cache_dir': '$CACHE_DIR',
    'cache_key': 'whisper_dec_xint8',
}

print('Loading decoder session...')
session = ort.InferenceSession('$MODEL', providers=[(provider, opts)])
print('Session created.')

# Print inputs/outputs
for inp in session.get_inputs():
    print(f'  Input:  {inp.name} {inp.type} {inp.shape}')
for out in session.get_outputs():
    print(f'  Output: {out.name} {out.type} {out.shape}')

# Create dummy inputs
x = np.full((1, 448), 50256, dtype=np.int64)  # EOT padding
x[0, 0] = 50257  # SOT
xa = np.random.randn(1, 1500, 384).astype(np.float32) * 0.3

print('Running inference...')
t0 = time.perf_counter()
outputs = session.run(None, {'x': x, 'xa': xa})
t1 = time.perf_counter()
print(f'Inference done in {t1-t0:.3f}s')
print(f'Output shape: {outputs[0].shape}')
print(f'Output[0,0,:5]: {outputs[0][0, 0, :5]}')

# Check EP report
report_path = os.path.join('$CACHE_DIR', 'whisper_dec_xint8', 'vitisai_ep_report.json')
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
    print()
    print('=== EP Report ===')
    for ds in report.get('deviceStat', []):
        print(f\"  {ds['name']}: {ds['nodeNum']} ops\")
else:
    print('No EP report found')
    import glob
    for f in glob.glob(os.path.join('$CACHE_DIR', '**', 'vitisai_ep_report.json'), recursive=True):
        print(f'  Found: {f}')
" 2>&1
