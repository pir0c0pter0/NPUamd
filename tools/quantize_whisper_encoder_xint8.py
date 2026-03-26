#!/usr/bin/env python3
"""Quantize Whisper tiny-en encoder to XINT8 using Quark for NPU offload.
Same proven pattern as ResNet18 XINT8 quantization."""

import os
import sys
import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

ENCODER_FP32 = "/work/hub/runtime/whisper/models/tiny_en_encoder.onnx"
ENCODER_XINT8 = "/tmp/tiny_en_encoder_xint8.onnx"


class WhisperEncoderCalibReader(CalibrationDataReader):
    """Calibration data mimicking log-mel spectrogram input [1, 80, 3000]."""

    def __init__(self, sample_count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Whisper encoder expects [1, 80, 3000] float32 (mel spectrogram)
        self._batches = iter(
            [{"x": rng.standard_normal((1, 80, 3000)).astype(np.float32) * 0.5}
             for _ in range(sample_count)]
        )

    def get_next(self):
        return next(self._batches, None)


def main():
    if not os.path.exists(ENCODER_FP32):
        print(f"[ERROR] Encoder not found: {ENCODER_FP32}")
        sys.exit(1)

    print(f"[INFO] Quantizing {ENCODER_FP32}")
    print(f"[INFO] Target: XINT8 for Ryzen AI NPU")

    config = Config(global_quant_config=get_default_config("XINT8"))
    quantizer = ModelQuantizer(config)

    print("[INFO] Running Quark XINT8 quantization with 32 calibration samples...")
    quantizer.quantize_model(
        ENCODER_FP32,
        ENCODER_XINT8,
        calibration_data_reader=WhisperEncoderCalibReader(sample_count=32, seed=42),
    )

    if os.path.exists(ENCODER_XINT8):
        size_mb = os.path.getsize(ENCODER_XINT8) / (1024 * 1024)
        print(f"[OK] Quantized model saved: {ENCODER_XINT8} ({size_mb:.1f} MB)")
    else:
        print("[FAIL] Quantized model not produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
