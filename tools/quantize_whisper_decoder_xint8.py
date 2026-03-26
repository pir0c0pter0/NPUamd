#!/usr/bin/env python3
"""Quantize Whisper tiny-en decoder to XINT8 using Quark for NPU offload."""

import os
import sys
import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

DECODER_FP32 = "/work/hub/runtime/whisper/models/tiny_en_decoder.onnx"
DECODER_XINT8 = "/tmp/tiny_en_decoder_xint8.onnx"


class WhisperDecoderCalibReader(CalibrationDataReader):
    """Calibration data for decoder: x (int64 [1,448]) + xa (float [1,1500,384])."""

    def __init__(self, sample_count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        batches = []
        for _ in range(sample_count):
            # x: token IDs (int64), simulate realistic range [0, 51864)
            x = rng.integers(0, 51864, size=(1, 448), dtype=np.int64)
            # xa: encoder hidden states (float32), simulate output from encoder
            xa = rng.standard_normal((1, 1500, 384)).astype(np.float32) * 0.3
            batches.append({"x": x, "xa": xa})
        self._batches = iter(batches)

    def get_next(self):
        return next(self._batches, None)


def main():
    if not os.path.exists(DECODER_FP32):
        print(f"[ERROR] Decoder not found: {DECODER_FP32}")
        sys.exit(1)

    print(f"[INFO] Quantizing {DECODER_FP32}")
    print(f"[INFO] Target: XINT8 for Ryzen AI NPU")

    config = Config(global_quant_config=get_default_config("XINT8"))
    quantizer = ModelQuantizer(config)

    print("[INFO] Running Quark XINT8 quantization with 20 calibration samples...")
    quantizer.quantize_model(
        DECODER_FP32,
        DECODER_XINT8,
        calibration_data_reader=WhisperDecoderCalibReader(sample_count=20, seed=42),
    )

    if os.path.exists(DECODER_XINT8):
        size_mb = os.path.getsize(DECODER_XINT8) / (1024 * 1024)
        print(f"[OK] Quantized model saved: {DECODER_XINT8} ({size_mb:.1f} MB)")
    else:
        print("[FAIL] Quantized model not produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
