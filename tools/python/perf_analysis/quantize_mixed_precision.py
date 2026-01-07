#!/usr/bin/env python3
"""
Mixed-Precision Quantization: QUInt16 MatMul Activations + QUInt8 Everything Else

Strategy:
1. Start with QUInt8 quantization (gets all the QLinear* operator fusion)
2. Selectively upgrade MatMul input activations to QUInt16
3. Keep QLinearAdd/Mul/Softmax/etc in QUInt8 for speed
4. Result: Best of both worlds - accuracy where it matters, speed everywhere else
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class DummyCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader that generates dummy data based on model inputs."""

    def __init__(self, model_path: str, num_samples: int = 100, seed: int = 42, model_description: Dict = None):
        self.model_path = model_path
        self.num_samples = num_samples
        self.seed = seed
        self.model_description = model_description
        self.samples = None
        self.iterator = None
        self._generate_samples()

    def _generate_samples(self):
        """Generate dummy calibration samples based on model input shapes."""
        np.random.seed(self.seed)
        self.samples = []

        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()

        for i in range(self.num_samples):
            sample = {}
            for input_tensor in inputs:
                shape = input_tensor.shape
                dtype = input_tensor.type

                # Handle dynamic dimensions
                concrete_shape = []
                for dim in shape:
                    if isinstance(dim, str) or dim is None or dim == -1:
                        concrete_shape.append(1)
                    else:
                        concrete_shape.append(dim)

                # Generate random data based on type
                if "float" in dtype:
                    if "mask" in input_tensor.name.lower():
                        data = np.random.randint(0, 2, size=concrete_shape).astype(np.float32)
                    else:
                        data = np.random.randn(*concrete_shape).astype(np.float32)
                elif "int64" in dtype:
                    data = np.random.randint(0, 100, size=concrete_shape).astype(np.int64)
                elif "int32" in dtype:
                    data = np.random.randint(0, 100, size=concrete_shape).astype(np.int32)
                else:
                    data = np.random.randn(*concrete_shape).astype(np.float32)

                sample[input_tensor.name] = data

            self.samples.append(sample)

        self.iterator = iter(self.samples)

    def get_next(self) -> Dict[str, np.ndarray]:
        """Return next calibration sample or None when done."""
        return next(self.iterator, None)

    def rewind(self):
        """Reset iterator to beginning."""
        self.iterator = iter(self.samples)


def quantize_to_quint8(model_path: str, output_path: str, calibration_reader: CalibrationDataReader, config: Dict) -> str:
    """Quantize entire model to QUInt8 to get good operator fusion."""
    print("\n" + "=" * 80)
    print("STEP 1: Quantizing to QUInt8 (for operator fusion)")
    print("=" * 80)

    calibration_reader.rewind()

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=config.get("per_channel", False),
        reduce_range=config.get("reduce_range", False),
        op_types_to_quantize=config.get("op_types_to_quantize", None),
        extra_options=config.get("extra_options", {}),
    )

    print(f"[OK] QUInt8 model saved to: {output_path}")
    return output_path


def find_matmul_activation_tensors(model: onnx.ModelProto) -> Set[str]:
    """Find all activation tensors that feed into MatMul/QLinearMatMul operations."""
    matmul_activations = set()

    for node in model.graph.node:
        # Check for MatMul, QLinearMatMul, Gemm
        if node.op_type in ['MatMul', 'QLinearMatMul', 'Gemm']:
            # First input is the activation (second is weight)
            activation_input = node.input[0]
            matmul_activations.add(activation_input)
            print(f"  Found MatMul activation: {activation_input} (node: {node.name or node.op_type})")

        # Also check for DequantizeLinear that feeds MatMul
        elif node.op_type == 'DequantizeLinear':
            # Check if output goes to MatMul
            dq_output = node.output[0]
            for other_node in model.graph.node:
                if other_node.op_type in ['MatMul', 'QLinearMatMul', 'Gemm']:
                    if dq_output in other_node.input:
                        # This DQ feeds a MatMul, mark its input
                        matmul_activations.add(node.input[0])
                        print(f"  Found MatMul activation (via DQ): {node.input[0]} -> {dq_output}")

    return matmul_activations


def upgrade_quantization_to_uint16(model: onnx.ModelProto, tensor_name: str) -> bool:
    """Upgrade a specific tensor's quantization from uint8 to uint16."""
    modified = False

    # Find QuantizeLinear nodes that produce this tensor
    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear' and node.output[0] == tensor_name:
            print(f"    Upgrading QuantizeLinear: {node.output[0]}")

            # Change scale and zero_point to support uint16 range
            # Note: We need to adjust the quantization parameters
            # uint8: [0, 255]
            # uint16: [0, 65535]
            # Scale needs to be adjusted: new_scale = old_scale * 255 / 65535

            # Find the scale and zero_point tensors
            scale_name = node.input[1] if len(node.input) > 1 else None
            zp_name = node.input[2] if len(node.input) > 2 else None

            if scale_name:
                # Find and modify the scale initializer
                for init in model.graph.initializer:
                    if init.name == scale_name:
                        # Adjust scale for uint16 range
                        scale_data = numpy_helper.to_array(init)
                        new_scale_data = scale_data * (255.0 / 65535.0)
                        new_init = numpy_helper.from_array(new_scale_data, init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        # Handle scalar (0-dimensional) arrays
                        if scale_data.ndim == 0:
                            print(f"      Adjusted scale: {float(scale_data):.6f} -> {float(new_scale_data):.6f}")
                        else:
                            print(f"      Adjusted scale: {scale_data[0]:.6f} -> {new_scale_data[0]:.6f}")
                        break

            # Keep zero_point as uint8 (don't convert to uint16)
            # ONNX Runtime CPU provider supports uint16 tensors with uint8 zero_points
            if zp_name:
                # Just print the existing zero_point, don't modify it
                for init in model.graph.initializer:
                    if init.name == zp_name:
                        zp_data = numpy_helper.to_array(init)
                        if zp_data.ndim == 0:
                            print(f"      Keeping zero_point as uint8: {int(zp_data)}")
                        else:
                            print(f"      Keeping zero_point as uint8: {zp_data[0]}")
                        break

            # Update the tensor type in value_info
            for value_info in model.graph.value_info:
                if value_info.name == tensor_name:
                    value_info.type.tensor_type.elem_type = TensorProto.UINT16
                    print(f"      Changed tensor type to UINT16")
                    break

            modified = True

    # Find DequantizeLinear nodes that consume this tensor
    for node in model.graph.node:
        if node.op_type == 'DequantizeLinear' and node.input[0] == tensor_name:
            print(f"    Upgrading DequantizeLinear consuming: {node.input[0]}")

            # Similar adjustments for DQ scale and zero_point
            scale_name = node.input[1] if len(node.input) > 1 else None
            zp_name = node.input[2] if len(node.input) > 2 else None

            if scale_name:
                for init in model.graph.initializer:
                    if init.name == scale_name:
                        scale_data = numpy_helper.to_array(init)
                        new_scale_data = scale_data * (255.0 / 65535.0)
                        new_init = numpy_helper.from_array(new_scale_data, init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        # Handle scalar (0-dimensional) arrays
                        if scale_data.ndim == 0:
                            print(f"      Adjusted scale: {float(scale_data):.6f} -> {float(new_scale_data):.6f}")
                        else:
                            print(f"      Adjusted scale: {scale_data[0]:.6f} -> {new_scale_data[0]:.6f}")
                        break

            # Keep zero_point as uint8 (don't convert to uint16)
            # ONNX Runtime CPU provider supports uint16 tensors with uint8 zero_points
            if zp_name:
                # Just print the existing zero_point, don't modify it
                for init in model.graph.initializer:
                    if init.name == zp_name:
                        zp_data = numpy_helper.to_array(init)
                        if zp_data.ndim == 0:
                            print(f"      Keeping zero_point as uint8: {int(zp_data)}")
                        else:
                            print(f"      Keeping zero_point as uint8: {zp_data[0]}")
                        break

            modified = True

    return modified


def create_uint16_to_uint8_lookup_table(scale_u16: float, zp_u16: int, scale_u8: float, zp_u8: int) -> np.ndarray:
    """
    Create a 64K lookup table for direct uint16->uint8 conversion.

    This replaces the pattern:
        uint16 -> DequantizeLinear -> FP32 -> QuantizeLinear -> uint8

    With a direct lookup:
        uint16 -> Gather(LookupTable) -> uint8

    Args:
        scale_u16: Scale factor for uint16 quantization
        zp_u16: Zero point for uint16 quantization
        scale_u8: Scale factor for uint8 quantization
        zp_u8: Zero point for uint8 quantization

    Returns:
        64K lookup table mapping uint16 values to uint8 values
    """
    lookup = np.zeros(65536, dtype=np.uint8)

    for i in range(65536):
        # Dequantize uint16 to FP32
        fp32_value = (i - zp_u16) * scale_u16

        # Quantize FP32 to uint8
        quant_value = round(fp32_value / scale_u8) + zp_u8

        # Clamp to uint8 range
        lookup[i] = np.clip(quant_value, 0, 255).astype(np.uint8)

    return lookup


def insert_lookup_table_conversions(model: onnx.ModelProto, matmul_activations: Set[str]) -> int:
    """
    Insert lookup table conversions for uint16->uint8 transitions.

    This replaces:
        [uint16 tensor] -> DequantizeLinear -> [fp32] -> QuantizeLinear -> [uint8]

    With:
        [uint16 tensor] -> Gather(LookupTable) -> [uint8]

    Returns:
        Number of conversions inserted
    """
    conversions_inserted = 0
    nodes_to_add = []
    nodes_to_remove = []
    initializers_to_add = []

    # Find patterns: uint16_dq -> fp32 -> uint8_q
    for i, node in enumerate(model.graph.node):
        if node.op_type != 'DequantizeLinear':
            continue

        # Check if this DQ operates on a uint16 tensor (matmul activation)
        dq_input = node.input[0]
        if dq_input not in matmul_activations:
            continue

        dq_output = node.output[0]

        # Find the corresponding QuantizeLinear that consumes this DQ output
        for j, other_node in enumerate(model.graph.node):
            if other_node.op_type == 'QuantizeLinear' and dq_output in other_node.input:
                # Found the pattern! Extract quantization parameters
                print(f"    Found conversion pattern: {dq_input} -> {dq_output} -> {other_node.output[0]}")

                # Get uint16 DQ parameters
                dq_scale_name = node.input[1] if len(node.input) > 1 else None
                dq_zp_name = node.input[2] if len(node.input) > 2 else None

                # Get uint8 Q parameters
                q_scale_name = other_node.input[1] if len(other_node.input) > 1 else None
                q_zp_name = other_node.input[2] if len(other_node.input) > 2 else None

                # Extract scale and zero_point values
                dq_scale = 1.0
                dq_zp = 0
                q_scale = 1.0
                q_zp = 0

                for init in model.graph.initializer:
                    if init.name == dq_scale_name:
                        dq_scale_data = numpy_helper.to_array(init)
                        dq_scale = float(dq_scale_data) if dq_scale_data.ndim == 0 else float(dq_scale_data[0])
                    elif init.name == dq_zp_name:
                        dq_zp_data = numpy_helper.to_array(init)
                        dq_zp = int(dq_zp_data) if dq_zp_data.ndim == 0 else int(dq_zp_data[0])
                    elif init.name == q_scale_name:
                        q_scale_data = numpy_helper.to_array(init)
                        q_scale = float(q_scale_data) if q_scale_data.ndim == 0 else float(q_scale_data[0])
                    elif init.name == q_zp_name:
                        q_zp_data = numpy_helper.to_array(init)
                        q_zp = int(q_zp_data) if q_zp_data.ndim == 0 else int(q_zp_data[0])

                print(f"      uint16: scale={dq_scale:.6f}, zp={dq_zp}")
                print(f"      uint8:  scale={q_scale:.6f}, zp={q_zp}")

                # Create the lookup table
                lookup_table = create_uint16_to_uint8_lookup_table(dq_scale, dq_zp, q_scale, q_zp)

                # Create lookup table initializer
                lut_name = f"lut_u16_to_u8_{conversions_inserted}"
                lut_init = numpy_helper.from_array(lookup_table, lut_name)
                initializers_to_add.append(lut_init)

                # Create Gather node to perform lookup
                # Gather(data=lookup_table[65536], indices=uint16_tensor, axis=0)
                gather_node = helper.make_node(
                    'Gather',
                    inputs=[lut_name, dq_input],
                    outputs=[other_node.output[0]],
                    name=f"gather_lut_{conversions_inserted}",
                    axis=0
                )
                nodes_to_add.append(gather_node)

                # Mark old nodes for removal
                nodes_to_remove.append(node)  # Remove DQ
                nodes_to_remove.append(other_node)  # Remove Q

                conversions_inserted += 1
                print(f"      Created lookup table with {len(lookup_table)} entries")
                break

    # Apply changes
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    for node in nodes_to_add:
        model.graph.node.append(node)

    for init in initializers_to_add:
        model.graph.initializer.append(init)

    return conversions_inserted


def create_mixed_precision_model(quint8_model_path: str, output_path: str, use_lookup_table: bool = False) -> str:
    """Convert QUInt8 model to mixed precision: QUInt16 MatMul activations, QUInt8 everything else."""
    print("\n" + "=" * 80)
    print("STEP 2: Upgrading MatMul activations to QUInt16")
    print("=" * 80)

    # Load the QUInt8 model
    model = onnx.load(quint8_model_path)

    # Find all MatMul activation tensors
    print("\nScanning for MatMul operations...")
    matmul_activations = find_matmul_activation_tensors(model)
    print(f"\nFound {len(matmul_activations)} MatMul activation tensors")

    if not matmul_activations:
        print("WARNING: No MatMul activations found. Model may already be optimized.")
        return quint8_model_path

    # Upgrade each MatMul activation from uint8 to uint16
    print("\nUpgrading quantization precision...")
    upgraded_count = 0
    for tensor_name in matmul_activations:
        print(f"\n  Processing: {tensor_name}")
        if upgrade_quantization_to_uint16(model, tensor_name):
            upgraded_count += 1

    print(f"\n[OK] Upgraded {upgraded_count} tensors to QUInt16")

    # Optionally insert lookup table conversions
    if use_lookup_table:
        print("\n" + "=" * 80)
        print("STEP 2b: Inserting lookup table conversions (uint16->uint8)")
        print("=" * 80)
        print("Replacing patterns: uint16 -> DQ -> FP32 -> Q -> uint8")
        print("With direct lookup:  uint16 -> Gather(LUT) -> uint8")
        lut_count = insert_lookup_table_conversions(model, matmul_activations)
        print(f"\n[OK] Inserted {lut_count} lookup table conversions")
        print(f"Total lookup table memory: {lut_count * 65536} bytes ({lut_count * 64} KB)")

    # Save the mixed-precision model
    onnx.save(model, output_path)
    print(f"[OK] Mixed-precision model saved to: {output_path}")

    return output_path


def verify_model(model_path: str):
    """Verify the model structure and quantization types."""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying mixed-precision model")
    print("=" * 80)

    model = onnx.load(model_path)

    # Count different operator types
    op_counts = {}
    quant_info = {'uint8_q': 0, 'uint16_q': 0, 'uint8_dq': 0, 'uint16_dq': 0}

    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        # Check quantization types
        if node.op_type == 'QuantizeLinear':
            # Check output tensor type
            output_name = node.output[0]
            for value_info in model.graph.value_info:
                if value_info.name == output_name:
                    elem_type = value_info.type.tensor_type.elem_type
                    if elem_type == TensorProto.UINT16:
                        quant_info['uint16_q'] += 1
                    elif elem_type == TensorProto.UINT8:
                        quant_info['uint8_q'] += 1
                    break

        elif node.op_type == 'DequantizeLinear':
            # Check input tensor type
            input_name = node.input[0]
            for value_info in model.graph.value_info:
                if value_info.name == input_name:
                    elem_type = value_info.type.tensor_type.elem_type
                    if elem_type == TensorProto.UINT16:
                        quant_info['uint16_dq'] += 1
                    elif elem_type == TensorProto.UINT8:
                        quant_info['uint8_dq'] += 1
                    break

    print("\nOperator counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {op:30} {count:6} nodes")

    print("\nQuantization distribution:")
    print(f"  QUInt8 QuantizeLinear:      {quant_info['uint8_q']:6} nodes")
    print(f"  QUInt16 QuantizeLinear:     {quant_info['uint16_q']:6} nodes")
    print(f"  QUInt8 DequantizeLinear:    {quant_info['uint8_dq']:6} nodes")
    print(f"  QUInt16 DequantizeLinear:   {quant_info['uint16_dq']:6} nodes")

    print("\nKey operators for mixed precision:")
    print(f"  QLinearMatMul:              {op_counts.get('QLinearMatMul', 0):6} nodes  <- Should use QUInt16 activations")
    print(f"  QLinearAdd:                 {op_counts.get('QLinearAdd', 0):6} nodes  <- Should stay QUInt8")
    print(f"  QLinearMul:                 {op_counts.get('QLinearMul', 0):6} nodes  <- Should stay QUInt8")
    print(f"  QLinearSoftmax:             {op_counts.get('QLinearSoftmax', 0):6} nodes  <- Should stay QUInt8")

    print("\n[OK] Model verification complete")


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-Precision Quantization: QUInt16 MatMul activations + QUInt8 everything else",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python quantize_mixed_precision.py --model model.onnx --output model_mixed.onnx

  # With custom config
  python quantize_mixed_precision.py --model model.onnx --config config.json --output model_mixed.onnx

Strategy:
  1. Quantize entire model to QUInt8 (gets QLinearAdd/Mul/Softmax fusion)
  2. Selectively upgrade MatMul input activations to QUInt16 (better accuracy)
  3. Result: Fast QUInt8 ops everywhere except MatMul which uses QUInt16
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (optional)")
    parser.add_argument("--output", type=str, required=True, help="Path for output mixed-precision model")
    parser.add_argument("--num-calibration-samples", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-lookup-table", action="store_true",
                        help="Use 64K lookup tables for uint16->uint8 conversion (avoids FP32 conversions)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            full_config = json.load(f)
            config = full_config.get("quantize_options", {}).get("Onnx", {})
            if not config:
                config = full_config

    # Set default config values
    config.setdefault("per_channel", False)
    config.setdefault("reduce_range", False)
    config.setdefault("extra_options", {
        "DedicatedQDQPair": False,
        "ForceQuantizeNoInputCheck": True,
        "WeightSymmetric": False
    })

    print("=" * 80)
    print("MIXED-PRECISION QUANTIZATION")
    print("=" * 80)
    print(f"Input model:  {args.model}")
    print(f"Output model: {args.output}")
    print(f"Strategy:     QUInt16 MatMul activations + QUInt8 everything else")
    print(f"Lookup table: {'ENABLED (64K LUT for uint16->uint8)' if args.use_lookup_table else 'DISABLED (FP32 conversions)'}")

    # Generate calibration data
    print("\n" + "=" * 80)
    print("GENERATING CALIBRATION DATA")
    print("=" * 80)
    calibration_reader = DummyCalibrationDataReader(
        args.model,
        num_samples=args.num_calibration_samples,
        seed=args.seed
    )
    print(f"[OK] Generated {args.num_calibration_samples} calibration samples")

    # Create temporary path for intermediate QUInt8 model
    output_path = Path(args.output)
    quint8_temp = output_path.parent / (output_path.stem + "_quint8_temp.onnx")

    try:
        # Step 1: Quantize to QUInt8
        quantize_to_quint8(args.model, str(quint8_temp), calibration_reader, config)

        # Step 2: Upgrade MatMul activations to QUInt16
        create_mixed_precision_model(str(quint8_temp), args.output, use_lookup_table=args.use_lookup_table)

        # Step 3: Verify the result
        verify_model(args.output)

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Mixed-precision model created: {args.output}")
        print("\nExpected benefits:")
        print("  - QLinearMatMul with QUInt16 activations (high accuracy)")
        print("  - QLinearAdd/Mul/Softmax with QUInt8 (fast, good fusion)")
        print("  - Reduced Q/DQ overhead compared to pure QUInt16")
        print("  - Better performance than pure QUInt16 (fewer conversions)")

    finally:
        # Clean up temporary file
        if quint8_temp.exists():
            quint8_temp.unlink()
            print(f"\n[OK] Cleaned up temporary file: {quint8_temp}")

    return 0


if __name__ == "__main__":
    exit(main())
