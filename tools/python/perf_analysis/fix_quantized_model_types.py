#!/usr/bin/env python3
"""
Fix ONNX quantized models with undefined output types.

This script repairs ONNX models where quantization outputs have undefined types,
which causes ONNX validation to fail (but ONNX Runtime can still infer types).

Usage:
    python fix_quantized_model_types.py input_model.onnx output_model.onnx
"""

import onnx
from onnx import TensorProto
import sys
import argparse


def infer_output_type_from_name(output_name: str, model: onnx.ModelProto) -> int:
    """
    Infer the tensor type based on output name patterns.

    Common patterns:
    - *_QuantizeLinear_Output -> uint8/uint16 (check for scale/zp to determine)
    - *_scale_* -> float
    - *_zero_point_* -> uint8/uint16
    """
    name_lower = output_name.lower()

    # Check if it's a scale parameter
    if 'scale' in name_lower:
        return TensorProto.FLOAT

    # Check if it's a zero point parameter
    if 'zero_point' in name_lower or 'zeropoint' in name_lower:
        # Need to determine if uint8 or uint16
        # For now, assume uint8 (most common)
        return TensorProto.UINT8

    # Check if it's a quantized output
    if 'quantize' in name_lower or 'quant' in name_lower:
        # Default to uint8 for quantized outputs
        return TensorProto.UINT8

    # Default to float for unknown
    return TensorProto.FLOAT


def infer_output_shape_from_initializers(output_name: str, model: onnx.ModelProto) -> list:
    """
    Try to infer shape from initializers with similar names.
    """
    # Look for scale/zero_point initializers that might hint at the shape
    if 'scale' in output_name.lower():
        # Scale outputs are typically scalar or [1]
        return [1]

    if 'zero_point' in output_name.lower() or 'zeropoint' in output_name.lower():
        # Zero point outputs are typically scalar or [1]
        return [1]

    # For quantized outputs, try to find corresponding nodes
    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear' and output_name in node.output:
            # Look at input shape to infer output shape
            # For now, return empty (dynamic)
            return []

    return []


def fix_undefined_output_types(model_path: str, output_path: str) -> None:
    """
    Fix undefined output types in ONNX model.
    """
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)

    print(f"Model loaded. IR version: {model.ir_version}, Opset: {model.opset_import[0].version if model.opset_import else 'N/A'}")
    print(f"Graph: {model.graph.name}, Nodes: {len(model.graph.node)}")

    fixed_count = 0

    print("\n=== Checking outputs ===")
    for output in model.graph.output:
        # Check if type is undefined (value_case == 0)
        if not output.type.HasField("tensor_type"):
            print(f"\n[FIX] Output '{output.name}' has undefined type")

            # Infer type from name
            inferred_type = infer_output_type_from_name(output.name, model)
            inferred_shape = infer_output_shape_from_initializers(output.name, model)

            print(f"  Inferred type: {TensorProto.DataType.Name(inferred_type)}")
            print(f"  Inferred shape: {inferred_shape if inferred_shape else '[dynamic]'}")

            # Set the tensor type
            output.type.tensor_type.elem_type = inferred_type

            # Set shape if inferred
            if inferred_shape:
                for dim_size in inferred_shape:
                    dim = output.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_size

            fixed_count += 1
        else:
            elem_type = output.type.tensor_type.elem_type
            print(f"[OK] Output '{output.name}' type: {TensorProto.DataType.Name(elem_type)}")

    if fixed_count > 0:
        print(f"\n=== Fixed {fixed_count} outputs ===")

        # Validate the fixed model
        print("Validating fixed model...")
        try:
            onnx.checker.check_model(model)
            print("[OK] Model validation passed!")
        except Exception as e:
            print(f"[WARNING] Validation still has issues: {e}")
            print("The model may still be usable with ONNX Runtime.")

        # Save the fixed model
        print(f"\nSaving fixed model to: {output_path}")
        onnx.save(model, output_path)
        print("[OK] Fixed model saved successfully!")
    else:
        print("\n[OK] No undefined outputs found. Model is already valid.")


def main():
    parser = argparse.ArgumentParser(
        description="Fix ONNX models with undefined output types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fix_quantized_model_types.py input.onnx output.onnx
    python fix_quantized_model_types.py embimg_c.quant.onnx embimg_c.quant.fixed.onnx
        """
    )

    parser.add_argument("input_model", help="Input ONNX model path")
    parser.add_argument("output_model", help="Output ONNX model path")

    args = parser.parse_args()

    try:
        fix_undefined_output_types(args.input_model, args.output_model)
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
