#!/usr/bin/env python3

import argparse

import numpy as np
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config
from torchvision.models import resnet18


class RandomImageReader(CalibrationDataReader):
    def __init__(self, input_name: str, sample_count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._batches = iter(
            [{input_name: rng.random((1, 3, 224, 224)).astype(np.float32)} for _ in range(sample_count)]
        )

    def get_next(self):
        return next(self._batches, None)


def export_fp32_model(path: str, seed: int) -> None:
    torch.manual_seed(seed)
    model = resnet18(weights=None)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )


def quantize_xint8(input_model: str, output_model: str, sample_count: int, seed: int) -> None:
    config = Config(global_quant_config=get_default_config("XINT8"))
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        input_model,
        output_model,
        calibration_data_reader=RandomImageReader("input", sample_count, seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ResNet18 and quantize it with Quark XINT8.")
    parser.add_argument("fp32_out", help="Path for the exported FP32 ONNX model.")
    parser.add_argument("xint8_out", help="Path for the Quark XINT8 ONNX model.")
    parser.add_argument("--sample-count", type=int, default=32, help="Number of random calibration samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for export and calibration reproducibility.")
    args = parser.parse_args()

    export_fp32_model(args.fp32_out, args.seed)
    quantize_xint8(args.fp32_out, args.xint8_out, args.sample_count, args.seed)

    print(args.fp32_out)
    print(args.xint8_out)


if __name__ == "__main__":
    main()
