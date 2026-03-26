#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import importlib.util
import pathlib
import re
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def have_module(name):
    return importlib.util.find_spec(name) is not None


def memtotal_gib():
    meminfo = pathlib.Path("/proc/meminfo").read_text(encoding="utf-8")
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", meminfo, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)) / 1024 / 1024


def kernel_vram_mib():
    proc = run(["journalctl", "-b", "-k"])
    if not proc or proc.returncode != 0:
        return None
    match = re.search(r"VRAM:\s+(\d+)M", proc.stdout)
    if not match:
        return None
    return int(match.group(1))


failures = []

gpu_nodes = [pathlib.Path("/dev/kfd"), pathlib.Path("/dev/dri/renderD128")]
for node in gpu_nodes:
    if not node.exists():
        failures.append(f"missing device node: {node}")

modules = {
    "torch": have_module("torch"),
    "transformers": have_module("transformers"),
    "accelerate": have_module("accelerate"),
}

for name, present in modules.items():
    if not present:
        failures.append(f"missing python module: {name}")

rocminfo = run(["rocminfo"])
if rocminfo is None:
    failures.append("missing rocminfo in PATH")
elif rocminfo.returncode != 0:
    failures.append("rocminfo exists but failed")

torch_summary = None
if modules["torch"]:
    probe = run([
        "python3",
        "-c",
        (
            "import torch; "
            "print('cuda_available', torch.cuda.is_available()); "
            "print('hip', getattr(torch.version, 'hip', None)); "
            "print('device_count', torch.cuda.device_count()); "
            "print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
        ),
    ])
    torch_summary = (probe.returncode, probe.stdout.strip(), probe.stderr.strip()) if probe else None
    if probe is None or probe.returncode != 0:
        failures.append("torch is installed but GPU probe failed")

print("[INFO] Qwen3 GPU preflight")
mem_gib = memtotal_gib()
if mem_gib is not None:
    print(f"[INFO] system RAM visible to CPU: {mem_gib:.1f} GiB")
vram_mib = kernel_vram_mib()
if vram_mib is not None:
    print(f"[INFO] iGPU VRAM reported by kernel: {vram_mib / 1024:.1f} GiB")

for name, present in modules.items():
    print(f"[INFO] python module {name}: {'yes' if present else 'no'}")
print(f"[INFO] /dev/kfd: {'yes' if pathlib.Path('/dev/kfd').exists() else 'no'}")
print(f"[INFO] /dev/dri/renderD128: {'yes' if pathlib.Path('/dev/dri/renderD128').exists() else 'no'}")
print(f"[INFO] rocminfo: {'yes' if rocminfo and rocminfo.returncode == 0 else 'no'}")

if torch_summary:
    _, stdout, stderr = torch_summary
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

if failures:
    print("[FAIL] GPU path is not ready for a large Qwen3 run.", file=sys.stderr)
    for item in failures:
        print(f"[FAIL] {item}", file=sys.stderr)
    print("[FAIL] This host currently has the kernel GPU nodes, but not the ROCm/PyTorch user-space stack.", file=sys.stderr)
    raise SystemExit(1)

print("[OK] GPU path passed the basic preflight.")
PY
