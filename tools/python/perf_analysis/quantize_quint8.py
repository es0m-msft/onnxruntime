#!/usr/bin/env python3
"""
Quantize Florence model to QUInt8×QUInt8 for comparison.
"""

import onnx
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np
import os

class DummyCalibrationDataReader(CalibrationDataReader):
    """Dummy calibration data reader for testing."""

    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.current_sample = 0

    def get_next(self):
        """Generate next calibration sample."""
        if self.current_sample >= self.num_samples:
            return None

        self.current_sample += 1

        # Generate dummy inputs matching model signature
        return {
            '/lang_encoder/embeddings/Add_1_output_0': np.random.randn(1, 77, 768).astype(np.float32),
            'attention_mask': np.ones((1, 77), dtype=np.float32)
        }

    def rewind(self):
        """Reset to beginning."""
        self.current_sample = 0

def quantize_florence_quint8(input_model_path, output_model_path):
    """Quantize Florence model to QUInt8×QUInt8."""

    print(f"Loading model: {input_model_path}")

    # Create calibration data reader
    calibration_data_reader = DummyCalibrationDataReader(num_samples=10)

    print("Quantizing model to QUInt8×QUInt8...")
    print("  Format: QDQ (Quantize-Dequantize)")
    print("  Activation type: QUInt8")
    print("  Weight type: QUInt8")
    print("  Per-channel: False")

    # Quantize with QUInt8 settings
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # Use QDQ format for fusion
        activation_type=QuantType.QUInt8,  # 8-bit activations
        weight_type=QuantType.QUInt8,  # 8-bit weights
        per_channel=False,  # Per-tensor quantization
        reduce_range=False,  # Use full uint8 range
        extra_options={
            'DedicatedQDQPair': False,  # Allow fusion
            'ForceQuantizeNoInputCheck': True,
            'WeightSymmetric': False,
            'ActivationSymmetric': False,
        }
    )

    print(f"\nQuantized model saved to: {output_model_path}")

    # Verify the quantized model
    print("\nVerifying quantized model...")
    model = onnx.load(output_model_path)

    # Count node types
    node_types = {}
    for node in model.graph.node:
        op_type = node.op_type
        node_types[op_type] = node_types.get(op_type, 0) + 1

    q_count = node_types.get('QuantizeLinear', 0)
    dq_count = node_types.get('DequantizeLinear', 0)
    qlinear_count = sum(count for op, count in node_types.items() if op.startswith('QLinear'))
    matmul_count = node_types.get('MatMul', 0)

    total_nodes = len(model.graph.node)

    print(f"  Total nodes: {total_nodes}")
    print(f"  QuantizeLinear: {q_count}")
    print(f"  DequantizeLinear: {dq_count}")
    print(f"  MatMul (unfused): {matmul_count}")
    print(f"  QLinear* operations: {qlinear_count}")

    model_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    # Input: Full original model
    input_model_path = r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\model\florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx"

    # Output path
    output_dir = r"C:\d\onnxruntime\quantized_models"
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "florence_quint8_qdq.onnx")

    # Check if input model exists
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model not found: {input_model_path}")
    else:
        quantize_florence_quint8(input_model_path, output_model_path)
        print("\n[OK] QUInt8 quantization complete!")
