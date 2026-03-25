#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def randn(shape, rng, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def build_model():
    rng = np.random.default_rng(0)

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    initializers = [
        numpy_helper.from_array(randn((8, 3, 3, 3), rng), "conv1_w"),
        numpy_helper.from_array(randn((8,), rng), "conv1_b"),
        numpy_helper.from_array(randn((16, 8, 3, 3), rng), "conv2_w"),
        numpy_helper.from_array(randn((16,), rng), "conv2_b"),
        numpy_helper.from_array(randn((16, 10), rng), "fc_w"),
        numpy_helper.from_array(randn((10,), rng), "fc_b"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1_w", "conv1_b"],
            ["conv1"],
            name="conv1",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv1"], ["relu1"], name="relu1"),
        helper.make_node(
            "AveragePool",
            ["relu1"],
            ["pool1"],
            name="pool1",
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node(
            "Conv",
            ["pool1", "conv2_w", "conv2_b"],
            ["conv2"],
            name="conv2",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv2"], ["relu2"], name="relu2"),
        helper.make_node("GlobalAveragePool", ["relu2"], ["gap"], name="gap"),
        helper.make_node("Flatten", ["gap"], ["flat"], name="flat", axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["output"], name="fc"),
    ]

    graph = helper.make_graph(
        nodes,
        "probe_small_cnn",
        [input_info],
        [output_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="npuamd-probe",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main():
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} OUTPUT_PATH", file=sys.stderr)
        return 2

    output_path = Path(sys.argv[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
