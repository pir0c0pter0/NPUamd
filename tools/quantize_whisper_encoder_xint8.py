#!/usr/bin/env python3
"""Quantize Whisper tiny-en encoder to XINT8 using speech-like calibration data."""

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config
from transformers import WhisperFeatureExtractor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HUB_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT = os.path.join(HUB_DIR, "runtime", "whisper", "models", "tiny_en_encoder.onnx")
DEFAULT_OUTPUT = os.path.join(HUB_DIR, "runtime", "whisper", "models", "tiny_en_encoder_xint8.onnx")
DEFAULT_AUDIO_GLOBS = [
    os.path.join(HUB_DIR, "runtime", "whisper", "*.wav"),
    os.path.join(HUB_DIR, "runtime", "whisper", "*.flac"),
]
SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        audio = audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac
    return np.asarray(audio, dtype=np.float32)


def shift_audio(audio: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return audio.copy()
    out = np.zeros_like(audio)
    if shift > 0:
        out[shift:] = audio[: len(audio) - shift]
        return out
    shift = -shift
    out[: len(audio) - shift] = audio[shift:]
    return out


def resolve_audio_paths(patterns: list[str]) -> list[str]:
    seen = set()
    paths = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path) and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


class WhisperEncoderCalibReader(CalibrationDataReader):
    """Calibration data based on real speech features plus light augmentations."""

    def __init__(self, audio_paths: list[str], sample_count: int, seed: int) -> None:
        if not audio_paths:
            raise ValueError("no calibration audio files found")

        rng = np.random.default_rng(seed)
        extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
        base_audios = [load_audio(path) for path in audio_paths]
        batches = []

        for i in range(sample_count):
            audio = base_audios[i % len(base_audios)].copy()
            if audio.size == 0:
                audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

            gain_db = rng.uniform(-10.0, 6.0)
            audio *= float(10.0 ** (gain_db / 20.0))

            shift = int(rng.integers(-SAMPLE_RATE // 4, SAMPLE_RATE // 4 + 1))
            audio = shift_audio(audio, shift)

            noise_sigma = float(rng.uniform(0.0, 0.01))
            if noise_sigma > 0.0:
                audio = audio + rng.standard_normal(audio.shape).astype(np.float32) * noise_sigma

            audio = np.clip(audio, -1.0, 1.0)
            features = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"]
            batches.append({"x": np.asarray(features, dtype=np.float32)})

        self._batches = iter(batches)

    def get_next(self):
        return next(self._batches, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Whisper tiny.en encoder to XINT8")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the FP32 encoder ONNX")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the output XINT8 ONNX")
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="Glob for calibration audio files; can be repeated. Defaults to runtime/whisper/*.wav and *.flac",
    )
    parser.add_argument("--samples", type=int, default=64, help="Number of calibration batches")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic augmentations")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    audio_patterns = args.audio or DEFAULT_AUDIO_GLOBS
    audio_paths = resolve_audio_paths(audio_patterns)

    if not os.path.exists(input_path):
        print(f"[ERROR] Encoder not found: {input_path}")
        sys.exit(1)
    if not audio_paths:
        print("[ERROR] No calibration audio files found.")
        for pattern in audio_patterns:
            print(f"  pattern: {pattern}")
        sys.exit(1)

    print(f"[INFO] Quantizing {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Calibration audio files ({len(audio_paths)}):")
    for path in audio_paths:
        print(f"  - {path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config = Config(global_quant_config=get_default_config("XINT8"))
    quantizer = ModelQuantizer(config)

    print(f"[INFO] Running Quark XINT8 quantization with {args.samples} speech-like calibration samples...")
    quantizer.quantize_model(
        input_path,
        output_path,
        calibration_data_reader=WhisperEncoderCalibReader(
            audio_paths=audio_paths,
            sample_count=args.samples,
            seed=args.seed,
        ),
    )

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] Quantized model saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print("[FAIL] Quantized model not produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
