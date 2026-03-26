#!/usr/bin/env python3
"""
Whisper transcription on AMD NPU via VitisAI EP.
Uses XINT8-quantized encoder and decoder with DPU/DD config.

Usage:
    python3 run_whisper_npu_transcribe.py --audio FILE.wav [--device npu|cpu]
    python3 run_whisper_npu_transcribe.py --test  # quick self-test with synthetic audio
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Paths
HUB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMD_LIB = "/var/home/mariostjr/amd-rai-linux/ryzen14"
MODELS_DIR = os.path.join(HUB_DIR, "runtime", "whisper", "models")
CACHE_DIR = os.path.join(HUB_DIR, "runtime", "whisper", "cache")
CONFIG_FILE = os.path.join(AMD_LIB, "vaip_config_npu_2_3.json")

# Model files
ENCODER_XINT8 = os.path.join(MODELS_DIR, "tiny_en_encoder_xint8.onnx")
DECODER_XINT8 = os.path.join(MODELS_DIR, "tiny_en_decoder_xint8.onnx")
ENCODER_FP32 = os.path.join(MODELS_DIR, "tiny_en_encoder.onnx")
DECODER_FP32 = os.path.join(MODELS_DIR, "tiny_en_decoder.onnx")

SAMPLE_RATE = 16000
MAX_DECODE_LEN = 448

# Whisper special tokens (for tiny.en)
SOT_TOKEN = 50257  # <|startoftranscript|>
EOT_TOKEN = 50256  # <|endoftext|>
TRANSCRIBE_TOKEN = 50358  # <|transcribe|>
NOTIMESTAMPS_TOKEN = 50362  # <|notimestamps|>
EN_TOKEN = 50258   # <|en|>


def get_provider_options(device):
    """Return ORT provider config for cpu or npu."""
    if device == "cpu":
        return ["CPUExecutionProvider"], ["CPUExecutionProvider"]

    # NPU via VitisAI EP
    provider = "VitisAIExecutionProvider"
    enc_opts = {
        "config_file": CONFIG_FILE,
        "cache_dir": os.path.join(CACHE_DIR, "xint8_encoder"),
        "cache_key": "whisper_enc_xint8",
    }
    dec_opts = {
        "config_file": CONFIG_FILE,
        "cache_dir": os.path.join(CACHE_DIR, "xint8_decoder"),
        "cache_key": "whisper_dec_xint8",
    }
    return [(provider, enc_opts)], [(provider, dec_opts)]


def load_audio(path):
    """Load WAV file and return float32 mono audio at 16kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except ImportError:
        # Fallback: use wave module for PCM WAV
        import wave
        import struct

        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            raw = wf.readframes(n_frames)
            dtype_map = {1: "b", 2: "h", 4: "i"}
            sw = wf.getsampwidth()
            fmt = f"<{n_frames * n_channels}{dtype_map[sw]}"
            samples = struct.unpack(fmt, raw)
            audio = np.array(samples, dtype=np.float32)
            if sw == 2:
                audio /= 32768.0
            elif sw == 4:
                audio /= 2147483648.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != SAMPLE_RATE:
        # Simple linear resample
        ratio = SAMPLE_RATE / sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        audio = audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac
        audio = audio.astype(np.float32)

    return audio


def log_mel_spectrogram(audio, n_mels=80, n_fft=400, hop_length=160):
    """Compute log-mel spectrogram matching Whisper's preprocessing."""
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
        features = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return features["input_features"]
    except ImportError:
        pass

    # Minimal fallback: pad to 30s, compute STFT, apply mel filterbank
    target_len = SAMPLE_RATE * 30  # 30 seconds
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Simple STFT
    n_frames_stft = 1 + (len(audio) - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float32)
    stft = np.zeros((n_fft // 2 + 1, n_frames_stft), dtype=np.float32)
    for i in range(n_frames_stft):
        frame = audio[i * hop_length : i * hop_length + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = np.abs(spectrum) ** 2

    # Simple mel filterbank
    mel_min = 0.0
    mel_max = 2595.0 * np.log10(1 + (SAMPLE_RATE / 2) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / SAMPLE_RATE).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if center > left:
                filterbank[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                filterbank[m, k] = (right - k) / (right - center)

    mel_spec = filterbank @ stft
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    # Ensure shape [1, 80, 3000]
    if log_mel.shape[1] > 3000:
        log_mel = log_mel[:, :3000]
    elif log_mel.shape[1] < 3000:
        log_mel = np.pad(log_mel, ((0, 0), (0, 3000 - log_mel.shape[1])))

    return log_mel[np.newaxis, :, :].astype(np.float32)


def decode_tokens(tokens):
    """Decode token IDs to text."""
    try:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except ImportError:
        return f"[token IDs: {tokens[:20]}...]"


def run_transcription(audio, encoder_session, decoder_session, verbose=False):
    """Run full Whisper encode-decode pipeline."""
    # Encode
    t0 = time.perf_counter()
    input_features = log_mel_spectrogram(audio)
    t_preprocess = time.perf_counter() - t0

    t0 = time.perf_counter()
    encoder_out = encoder_session.run(None, {"x": input_features})[0]
    t_encode = time.perf_counter() - t0

    if verbose:
        print(f"  Preprocess: {t_preprocess:.3f}s")
        print(f"  Encode:     {t_encode:.3f}s")

    # Decode (greedy)
    tokens = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NOTIMESTAMPS_TOKEN]
    t0 = time.perf_counter()
    t_first_token = None

    for step in range(MAX_DECODE_LEN - len(tokens)):
        # Pad to max length
        decoder_input = np.full((1, MAX_DECODE_LEN), EOT_TOKEN, dtype=np.int64)
        decoder_input[0, :len(tokens)] = tokens

        logits = decoder_session.run(None, {
            "x": decoder_input,
            "xa": encoder_out,
        })[0]

        next_token = int(np.argmax(logits[0, len(tokens) - 1]))

        if t_first_token is None:
            t_first_token = time.perf_counter() - t0

        if next_token == EOT_TOKEN:
            break
        tokens.append(next_token)

    t_decode = time.perf_counter() - t0
    n_generated = len(tokens) - 4  # subtract prompt tokens

    if verbose:
        print(f"  Decode:     {t_decode:.3f}s ({n_generated} tokens)")
        if t_first_token:
            print(f"  TTFT:       {t_first_token:.3f}s")
        if n_generated > 0 and t_decode > 0:
            print(f"  Tokens/s:   {n_generated / t_decode:.1f}")

    text = decode_tokens(tokens)
    return text, {
        "preprocess_s": t_preprocess,
        "encode_s": t_encode,
        "decode_s": t_decode,
        "ttft_s": t_first_token,
        "n_tokens": n_generated,
        "total_s": t_preprocess + t_encode + t_decode,
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription on AMD NPU")
    parser.add_argument("--audio", help="Path to WAV file")
    parser.add_argument("--test", action="store_true", help="Quick test with synthetic audio")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 models instead of XINT8")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.audio and not args.test:
        parser.error("Provide --audio FILE.wav or --test")

    # Setup LD paths
    import ctypes
    ubuntu_lib = os.path.join(HUB_DIR, "runtime", "ubuntu22", "lib")
    for lib_dir in [AMD_LIB, ubuntu_lib]:
        if os.path.isdir(lib_dir):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available EPs: {ort.get_available_providers()}")

    # Select models
    enc_path = ENCODER_FP32 if args.fp32 else ENCODER_XINT8
    dec_path = DECODER_FP32 if args.fp32 else DECODER_XINT8
    print(f"Encoder: {os.path.basename(enc_path)}")
    print(f"Decoder: {os.path.basename(dec_path)}")
    print(f"Device:  {args.device}")

    # Check models exist
    for p in [enc_path, dec_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Model not found: {p}")
            sys.exit(1)

    # Create sessions
    enc_providers, dec_providers = get_provider_options(args.device)
    print("\nLoading encoder...")
    encoder = ort.InferenceSession(enc_path, providers=enc_providers)
    print("Loading decoder...")
    decoder = ort.InferenceSession(dec_path, providers=dec_providers)
    print("Sessions ready.\n")

    # Load or generate audio
    if args.test:
        print("=== Synthetic audio test (1s silence) ===")
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    else:
        print(f"=== Transcribing: {args.audio} ===")
        audio = load_audio(args.audio)
        duration = len(audio) / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s")

    print()
    text, timing = run_transcription(audio, encoder, decoder, verbose=True)
    print(f"\nTranscription: {text}")
    print(f"\nTotal time: {timing['total_s']:.3f}s")


if __name__ == "__main__":
    main()
