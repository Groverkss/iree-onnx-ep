#!/usr/bin/env python3
"""Test that the IREE ONNX Runtime EP loads and runs correctly."""

import pathlib
import sys

import numpy as np
from onnx import TensorProto, helper

import test_utils


def test_ep_load():
    """Test that the IREE EP plugin loads and runs inference correctly."""

    device = test_utils.get_iree_device("local-task")
    if not device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    print(f"IREE EP available (vendor={device.ep_vendor})")
    print(f"  Device metadata: {device.device.metadata}")
    print(f"  Device type: {device.device.type}")

    # Create a simple Add model: C = A + D (constant).
    a = np.float32(np.random.rand(64, 64))
    b = np.float32(np.random.rand(64, 64))

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a.shape))
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, list(b.shape))
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=list(b.shape),
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])

    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph, producer_name="iree_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    model_path = test_utils.save_model(model)
    try:
        session = test_utils.create_session(model_path, device, {"target_arch": "host"})

        expected = a + b
        outputs = session.run(None, {"A": a})
        result = outputs[0]

        if result.shape != expected.shape:
            print(f"FAIL: Shape mismatch: {result.shape} vs {expected.shape}")
            return False

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            return False

        print("Inference output verified!")
        print(f"  Input A: {a.flatten()[:4]}...")
        print(f"  Input B: {b.flatten()[:4]}...")
        print(f"  Output:  {result.flatten()[:4]}...")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== Test PASSED ===")
    return True


if __name__ == "__main__":
    test_utils.register_ep()
    success = test_ep_load()
    sys.exit(0 if success else 1)
