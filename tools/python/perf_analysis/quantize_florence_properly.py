#!/usr/bin/env python3
"""
Properly quantize Florence model to QUInt16×QUInt8 with QLinearMatMul fusion.
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
        # Model starts from embeddings output, not input_ids
        return {
            '/lang_encoder/embeddings/Add_1_output_0': np.random.randn(1, 77, 768).astype(np.float32),
            'attention_mask': np.ones((1, 77), dtype=np.float32)
        }

    def rewind(self):
        """Reset to beginning."""
        self.current_sample = 0

def quantize_florence_model(input_model_path, output_model_path):
    """Quantize Florence model properly."""

    print(f"Loading model: {input_model_path}")

    # Create calibration data reader
    calibration_data_reader = DummyCalibrationDataReader(num_samples=10)

    print("Quantizing model...")
    print("  Format: QDQ (Quantize-Dequantize)")
    print("  Activation type: QUInt16")
    print("  Weight type: QUInt8")
    print("  Per-channel: False")

    # Quantize with proper settings
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # Use QDQ format for fusion
        activation_type=QuantType.QUInt16,  # 16-bit activations
        weight_type=QuantType.QUInt8,  # 8-bit weights
        per_channel=False,  # Per-tensor quantization
        reduce_range=False,  # Use full uint16/uint8 range
        extra_options={
            'DedicatedQDQPair': False,  # Allow fusion
            'ForceQuantizeNoInputCheck': True,  # Force quantization everywhere
            'WeightSymmetric': False,  # Unsigned weights
            'ActivationSymmetric': False,  # Unsigned activations
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

    print(f"  Total nodes: {len(model.graph.node)}")
    print(f"  QuantizeLinear: {q_count}")
    print(f"  DequantizeLinear: {dq_count}")
    print(f"  MatMul (unfused): {matmul_count}")
    print(f"  QLinear* operations: {qlinear_count}")

    if matmul_count > 0 and qlinear_count == 0:
        print("\n⚠ WARNING: Model has MatMul operations but no QLinearMatMul!")
        print("  This means DQ→MatMul→Q patterns were not fused.")
        print("  Graph optimization at runtime should fuse these.")

if __name__ == "__main__":
    # Input: Full original model (before any slicing)
    input_model_path = r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\model\florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx"

    # Output path
    output_dir = r"C:\d\onnxruntime\quantized_models"
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "florence_quint16_qdq_proper.onnx")

    # Check if input model exists
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model not found: {input_model_path}")
        print("\nLooking for alternative models...")

        # Try to find the full model
        model_dir = os.path.dirname(input_model_path)
        print(f"Model directory: {model_dir}")
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            print(f"Available models: {models}")
    else:
        quantize_florence_model(input_model_path, output_model_path)
