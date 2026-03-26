#!/usr/bin/env python3
"""
Hybrid Whisper transcription:
- encoder on AMD NPU via native C runner + VitisAI EP
- decoder on CPU via generic Python onnxruntime

Intended to run inside the Ubuntu 22.04 helper container.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

HUB_DIR = os.environ.get("WHISPER_HUB_DIR", "/work/hub")
AMD_LIB = "/work/amd-rai-linux/ryzen14"
MODELS_DIR = os.path.join(HUB_DIR, "runtime", "whisper", "models")
CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", os.path.join(HUB_DIR, "runtime", "whisper", "cache"))
TMP_DIR = os.environ.get("WHISPER_TMP_DIR", os.path.join(HUB_DIR, "runtime", "whisper", "tmp"))
CONFIG_FILE = os.path.join(AMD_LIB, "vaip_config_npu_2_3.json")

ENCODER_MODEL = os.path.join(MODELS_DIR, "tiny_en_encoder_xint8.onnx")
DECODER_MODEL = os.path.join(MODELS_DIR, "tiny_en_decoder.onnx")
ENCODER_BIN = os.environ.get("WHISPER_ENCODER_BIN", os.path.join(HUB_DIR, "runtime", "whisper", "whisper_encode_dump"))

SAMPLE_RATE = 16000
ENCODER_OUTPUT_SHAPE = (1, 1500, 384)
MAX_DECODE_LEN = 448

EOT_TOKEN = 50256


def load_audio(path: str) -> np.ndarray:
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(frames)
        if sample_width != 2:
            raise RuntimeError(f"unsupported WAV sample width: {sample_width}")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        audio = audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac
        audio = audio.astype(np.float32)

    return audio


def build_test_audio() -> np.ndarray:
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


def compute_features(audio: np.ndarray) -> np.ndarray:
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    features = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    return np.asarray(features["input_features"], dtype=np.float32)


def run_encoder(features: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, float]:
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "hybrid_encoder"), exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=".bin", delete=False) as fin:
        features.tofile(fin)
        input_path = fin.name
    with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=".bin", delete=False) as fout:
        output_path = fout.name

    cmd = [
        ENCODER_BIN,
        ENCODER_MODEL,
        input_path,
        output_path,
        f"config_file={CONFIG_FILE}",
        f"cache_dir={os.path.join(CACHE_DIR, 'hybrid_encoder')}",
        "cache_key=whisper_encoder_hybrid",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{AMD_LIB}:{os.path.join(HUB_DIR, 'runtime', 'ubuntu22', 'lib')}:" + env.get(
        "LD_LIBRARY_PATH", ""
    )

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    encode_time = time.perf_counter() - t0
    if verbose:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"encoder runner failed:\n{result.stdout}\n{result.stderr}")

    encoder_out = np.fromfile(output_path, dtype=np.float32).reshape(ENCODER_OUTPUT_SHAPE)
    os.unlink(input_path)
    os.unlink(output_path)
    return encoder_out, encode_time


def run_decoder(encoder_out: np.ndarray, verbose: bool = False) -> tuple[str, dict]:
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    session = ort.InferenceSession(DECODER_MODEL, providers=["CPUExecutionProvider"])

    # `whisper-tiny.en` defaults to the compact English-only prefix:
    # <|startoftranscript|><|notimestamps|>
    tokens = [int(token_id) for token_id in tokenizer.prefix_tokens]
    prompt_len = len(tokens)
    t0 = time.perf_counter()
    ttft = None

    for _ in range(MAX_DECODE_LEN - len(tokens)):
        decoder_input = np.full((1, MAX_DECODE_LEN), EOT_TOKEN, dtype=np.int64)
        decoder_input[0, : len(tokens)] = tokens
        logits = session.run(None, {"x": decoder_input, "xa": encoder_out})[0]
        next_token = int(np.argmax(logits[0, len(tokens) - 1]))
        if ttft is None:
            ttft = time.perf_counter() - t0
        if next_token == EOT_TOKEN:
            break
        tokens.append(next_token)

    decode_time = time.perf_counter() - t0
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    info = {
        "ttft_s": ttft,
        "decode_s": decode_time,
        "n_tokens": max(0, len(tokens) - prompt_len),
        "prompt_tokens": tokens[:prompt_len],
    }
    if verbose:
        print(f"[INFO] decoder prompt: {tokens[:prompt_len]}")
        print(f"[INFO] decoder tokens: {tokens[:24]}")
    return text, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid Whisper transcription")
    parser.add_argument("--audio", help="Path to WAV file inside container")
    parser.add_argument("--test", action="store_true", help="Use 1s silence synthetic audio")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.audio and not args.test:
        parser.error("Provide --audio FILE.wav or --test")

    if not os.path.exists(ENCODER_BIN):
        print(f"[ERROR] encoder runner not found: {ENCODER_BIN}", file=sys.stderr)
        return 2

    if args.test:
        audio = build_test_audio()
        audio_desc = "synthetic 1s silence"
    else:
        audio = load_audio(args.audio)
        audio_desc = args.audio

    print(f"[INFO] audio: {audio_desc}")
    t0 = time.perf_counter()
    features = compute_features(audio)
    preprocess_time = time.perf_counter() - t0
    print(f"[INFO] preprocess_s={preprocess_time:.3f}")

    encoder_out, encode_time = run_encoder(features, verbose=args.verbose)
    print(f"[INFO] encode_npu_s={encode_time:.3f}")

    text, dec_info = run_decoder(encoder_out, verbose=args.verbose)
    total = preprocess_time + encode_time + dec_info["decode_s"]

    print(f"[INFO] decode_cpu_s={dec_info['decode_s']:.3f}")
    if dec_info["ttft_s"] is not None:
        print(f"[INFO] ttft_s={dec_info['ttft_s']:.3f}")
    print(f"[INFO] generated_tokens={dec_info['n_tokens']}")
    print(f"[INFO] total_s={total:.3f}")
    print(f"[INFO] transcription={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
