#!/usr/bin/env python3
"""Extract the Whisper decoder body without token embedding and lm_head."""

import argparse
import os
import sys

import onnx.utils


DEFAULT_INPUT = "/work/hub/runtime/whisper/models/tiny_en_decoder_xint8.onnx"
DEFAULT_OUTPUT = "/work/hub/runtime/whisper/models/tiny_en_decoder_body_xint8.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Whisper decoder body submodel")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input decoder model")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output body model")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] input model not found: {args.input}", file=sys.stderr)
        return 1

    onnx.utils.extract_model(
        args.input,
        args.output,
        ["add_DequantizeLinear_Output", "xa"],
        ["layer_norm_12_DequantizeLinear_Output"],
    )

    print(f"[OK] wrote {args.output}")
    print("[OK] inputs: add_DequantizeLinear_Output, xa")
    print("[OK] output: layer_norm_12_DequantizeLinear_Output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
