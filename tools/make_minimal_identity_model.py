import sys

import onnx
from onnx import TensorProto, helper


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/minimal_identity_ir10.onnx"
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "id_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
