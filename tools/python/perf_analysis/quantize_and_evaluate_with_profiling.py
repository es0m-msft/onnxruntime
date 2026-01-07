#!/usr/bin/env python3
"""
Enhanced quantization evaluation script with profiling support.

This script extends quantize_and_evaluate.py to add:
1. Detailed profiling with operator-level timing breakdown
2. Multiple quantization configuration testing
3. Chrome tracing output for visualization
4. Performance comparison across configurations
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


class DummyCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader that generates dummy data based on model inputs."""

    def __init__(
        self,
        model_path: str,
        num_samples: int = 100,
        seed: int = 42,
        model_description: Dict = None,
    ):
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

        # If model_description is provided, use it
        if self.model_description and "inputs" in self.model_description:
            input_names = self.model_description["inputs"]
            input_dims = self.model_description.get("input_dims", [])

            for i in range(self.num_samples):
                sample = {}
                for idx, input_name in enumerate(input_names):
                    if idx < len(input_dims):
                        shape = input_dims[idx]
                    else:
                        # Fallback to inferring from model
                        session = ort.InferenceSession(
                            self.model_path, providers=["CPUExecutionProvider"]
                        )
                        input_tensor = session.get_inputs()[idx]
                        shape = input_tensor.shape
                        shape = [
                            1 if (isinstance(d, str) or d is None or d == -1) else d
                            for d in shape
                        ]

                    # Generate random data based on shape
                    # First try to infer type from model
                    session_temp = ort.InferenceSession(
                        self.model_path, providers=["CPUExecutionProvider"]
                    )
                    input_tensor = session_temp.get_inputs()[idx]
                    dtype = input_tensor.type

                    # Generate data based on inferred type
                    if "float" in dtype:
                        # For float types, use appropriate values based on name
                        if "mask" in input_name.lower():
                            # For attention masks in float format, use 0s and 1s as floats
                            data = np.random.randint(0, 2, size=shape).astype(np.float32)
                        else:
                            data = np.random.randn(*shape).astype(np.float32)
                    elif "int64" in dtype:
                        data = np.random.randint(0, 100, size=shape).astype(np.int64)
                    elif "int32" in dtype:
                        data = np.random.randint(0, 100, size=shape).astype(np.int32)
                    else:
                        # Default to float32
                        data = np.random.randn(*shape).astype(np.float32)

                    sample[input_name] = data

                self.samples.append(sample)
        else:
            # Fallback: Load model to get input specifications
            session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()

            for i in range(self.num_samples):
                sample = {}
                for input_tensor in inputs:
                    # Get input shape and type
                    shape = input_tensor.shape
                    dtype = input_tensor.type

                    # Handle dynamic dimensions (replace None/-1 with 1)
                    concrete_shape = []
                    for dim in shape:
                        if isinstance(dim, str) or dim is None or dim == -1:
                            concrete_shape.append(1)
                        else:
                            concrete_shape.append(dim)

                    # Generate random data based on type
                    if "float" in dtype:
                        data = np.random.randn(*concrete_shape).astype(np.float32)
                    elif "int64" in dtype:
                        data = np.random.randint(0, 100, size=concrete_shape).astype(np.int64)
                    elif "int32" in dtype:
                        data = np.random.randint(0, 100, size=concrete_shape).astype(np.int32)
                    else:
                        # Default to float32
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

    def get_all_samples(self) -> List[Dict[str, np.ndarray]]:
        """Return all samples for validation."""
        return self.samples


def load_config(config_path: str) -> Tuple[Dict, Dict]:
    """Load quantization configuration from JSON file."""
    with open(config_path, "r") as f:
        full_config = json.load(f)

    model_description = full_config.get("model_description", {})
    quantize_config = full_config.get("quantize_options", {}).get("Onnx", {})

    if not quantize_config and "activation_type" in full_config:
        quantize_config = full_config
        model_description = {}

    return model_description, quantize_config


def parse_quant_type(type_str: str) -> QuantType:
    """Parse quantization type string to QuantType enum."""
    type_map = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
        "QInt16": QuantType.QInt16,
        "QUInt16": QuantType.QUInt16,
        "QInt4": QuantType.QInt4,
        "QUInt4": QuantType.QUInt4,
    }
    return type_map.get(type_str, QuantType.QInt8)


def parse_calibration_method(method_str: str) -> CalibrationMethod:
    """Parse calibration method string to CalibrationMethod enum."""
    method_map = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    return method_map.get(method_str, CalibrationMethod.MinMax)


def create_tensor_quant_overrides_for_matmul(model_path: str) -> Dict:
    """Create TensorQuantOverrides to quantize MatMul activations as QUInt16.

    This uses ONNX Runtime's native support for per-tensor quantization type overrides
    to ensure MatMul activations use the full [0, 65535] uint16 range.

    Args:
        model_path: Path to the model

    Returns:
        Dictionary suitable for extra_options['TensorQuantOverrides']
    """
    model = onnx.load(model_path)
    matmul_activations = find_matmul_activation_tensors(model)

    # Create overrides dictionary
    # Format: {tensor_name: [{'quant_type': QuantType.QUInt16}]}
    overrides = {}
    for tensor_name in matmul_activations:
        overrides[tensor_name] = [{'quant_type': QuantType.QUInt16}]

    return overrides


def quantize_to_qdq(
    model_path: str,
    output_path: str,
    calibration_reader: CalibrationDataReader,
    config: Dict,
) -> str:
    """Quantize model to QDQ format."""
    print("\n" + "=" * 80)
    print("QUANTIZING TO QDQ FORMAT")
    print("=" * 80)

    activation_type = parse_quant_type(config.get("activation_type", "QInt8"))
    weight_type = parse_quant_type(config.get("weight_type", "QInt8"))
    calibrate_method = parse_calibration_method(config.get("calibrate_method", "MinMax"))

    print(f"Activation type: {activation_type}")
    print(f"Weight type: {weight_type}")
    print(f"Calibration method: {calibrate_method}")
    print(f"Per-channel: {config.get('per_channel', False)}")
    print(f"Reduce range: {config.get('reduce_range', False)}")

    calibration_reader.rewind()

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=calibrate_method,
        per_channel=config.get("per_channel", False),
        reduce_range=config.get("reduce_range", False),
        op_types_to_quantize=config.get("op_types_to_quantize", None),
        extra_options=config.get("extra_options", {}),
    )

    print(f"[OK] QDQ model saved to: {output_path}")
    return output_path


def quantize_to_dynamic(
    model_path: str,
    output_path: str,
    config: Dict,
) -> str:
    """Quantize model using dynamic quantization."""
    print("\n" + "=" * 80)
    print("QUANTIZING WITH DYNAMIC QUANTIZATION")
    print("=" * 80)

    weight_type = parse_quant_type(config.get("weight_type", "QInt8"))

    print(f"Weight type: {weight_type}")
    print(f"Per-channel: {config.get('per_channel', False)}")
    print(f"Reduce range: {config.get('reduce_range', False)}")

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=weight_type,
        per_channel=config.get("per_channel", False),
        reduce_range=config.get("reduce_range", False),
        op_types_to_quantize=config.get("op_types_to_quantize", None),
        extra_options=config.get("extra_options", {}),
    )

    print(f"[OK] Dynamic quantized model saved to: {output_path}")
    return output_path


def quantize_mixed_precision_full_uint16(
    model_path: str,
    output_path: str,
    calibration_reader: CalibrationDataReader,
    config: Dict,
) -> str:
    """Quantize model with full [0, 65535] uint16 range for MatMul activations.

    This is the CORRECT implementation that finds ALL MatMul activations using
    a two-stage approach:
    1. Quantize to QUInt8 first (temporary model)
    2. Analyze QUInt8 model to find ALL MatMul activation tensors
    3. Create TensorQuantOverrides for those tensors
    4. Re-quantize FP32 model with overrides
    5. Result: Full uint16 range for MatMul activations

    Args:
        model_path: Path to FP32 input model
        output_path: Path for output mixed-precision model
        calibration_reader: Calibration data reader
        config: Quantization configuration

    Returns:
        Path to output model
    """

    # Step 1: Quantize to QUInt8 first (temporary)
    print("\n" + "=" * 80)
    print("STEP 1/4: Creating temporary QUInt8 model for analysis")
    print("=" * 80)

    temp_quint8_path = output_path.replace('.onnx', '_temp_quint8.onnx')

    # Rewind calibration reader
    calibration_reader.rewind()

    # Standard QUInt8 quantization
    quantize_static(
        model_input=model_path,
        model_output=temp_quint8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=parse_calibration_method(config.get("calibrate_method", "MinMax")),
        per_channel=config.get("per_channel", False),
        reduce_range=config.get("reduce_range", False),
        extra_options=config.get("extra_options", {}),
    )

    print(f"[OK] Temporary QUInt8 model created: {temp_quint8_path}")

    # Step 2: Analyze QUInt8 model to find ALL MatMul activations
    print("\n" + "=" * 80)
    print("STEP 2/4: Analyzing QUInt8 model structure")
    print("=" * 80)

    matmul_tensors = find_matmul_activations_in_quantized_model(temp_quint8_path)
    print(f"[OK] Found {len(matmul_tensors)} tensors to upgrade to uint16")

    # Step 3: Create TensorQuantOverrides
    print("\n" + "=" * 80)
    print("STEP 3/4: Creating TensorQuantOverrides")
    print("=" * 80)

    tensor_overrides = {}
    for tensor_name in matmul_tensors:
        tensor_overrides[tensor_name] = [{'quant_type': QuantType.QUInt16}]

    print(f"[OK] Created overrides for {len(tensor_overrides)} tensors")

    # Step 4: Re-quantize with overrides applied
    print("\n" + "=" * 80)
    print("STEP 4/4: Quantizing with uint16 overrides (FULL [0, 65535] range)")
    print("=" * 80)

    # Rewind calibration reader for second quantization pass
    calibration_reader.rewind()

    # Prepare config with overrides
    extra_options = config.get("extra_options", {}).copy()
    extra_options["TensorQuantOverrides"] = tensor_overrides
    extra_options["UseQDQContribOps"] = True  # Enable 16-bit support

    # Final quantization with uint16 overrides
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,  # Base type
        weight_type=QuantType.QUInt8,
        calibrate_method=parse_calibration_method(config.get("calibrate_method", "MinMax")),
        per_channel=config.get("per_channel", False),
        reduce_range=config.get("reduce_range", False),
        extra_options=extra_options,
    )

    print(f"[OK] Mixed-precision model with full uint16 range: {output_path}")

    # Step 5: Clean up temporary file
    if Path(temp_quint8_path).exists():
        Path(temp_quint8_path).unlink()
        print("[OK] Cleaned up temporary files")

    return output_path


def compute_relative_l2_norm(output1: np.ndarray, output2: np.ndarray) -> float:
    """Compute relative L2 norm between two outputs."""
    diff = output1.flatten() - output2.flatten()
    l2_diff = np.linalg.norm(diff)
    l2_ref = np.linalg.norm(output1.flatten())

    length = len(output1.flatten())
    relative_l2 = (l2_diff / l2_ref) / np.sqrt(length) if l2_ref > 0 else 0.0

    return relative_l2


def evaluate_model_with_profiling(
    model_path: str,
    validation_samples: List[Dict[str, np.ndarray]],
    reference_outputs: List[List[np.ndarray]] = None,
    enable_profiling: bool = False,
    profile_prefix: str = "ort_profile",
) -> Tuple[List[List[np.ndarray]], float, Dict, str]:
    """Evaluate model with optional profiling."""

    # Create session with profiling enabled if requested
    options = ort.SessionOptions()

    # CRITICAL: Enable all graph optimizations to fuse QDQ patterns into QLinearMatMul
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if enable_profiling:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix

    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    outputs_list = []
    relative_l2_norms = []
    inference_times = []

    # Warm-up runs
    for _ in range(5):
        session.run(None, validation_samples[0])

    # Actual evaluation
    for i, sample in enumerate(validation_samples):
        start_time = time.perf_counter()
        outputs = session.run(None, sample)
        end_time = time.perf_counter()

        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        outputs_list.append(outputs)

        # Compute L2 norm if reference is provided
        if reference_outputs is not None:
            l2_norm = compute_relative_l2_norm(reference_outputs[i][0], outputs[0])
            relative_l2_norms.append(l2_norm)

    # Compute statistics
    avg_relative_l2 = np.mean(relative_l2_norms) if relative_l2_norms else 0.0

    timing_stats = {
        "mean_ms": np.mean(inference_times),
        "std_ms": np.std(inference_times),
        "min_ms": np.min(inference_times),
        "max_ms": np.max(inference_times),
        "median_ms": np.median(inference_times),
        "p95_ms": np.percentile(inference_times, 95),
        "p99_ms": np.percentile(inference_times, 99),
    }

    profile_file = None
    if enable_profiling:
        profile_file = session.end_profiling()

    return outputs_list, avg_relative_l2, timing_stats, profile_file


def analyze_profile(profile_file: str) -> Dict:
    """Analyze profiling results and extract key metrics."""
    if not profile_file or not Path(profile_file).exists():
        return {}

    with open(profile_file, 'r') as f:
        profile_data = json.load(f)

    # Aggregate by operator type
    op_times = {}
    total_time = 0

    for event in profile_data:
        if event.get("cat") == "Node" and "dur" in event:
            dur = event["dur"] / 1000.0  # Convert to ms
            op_name = event.get("args", {}).get("op_name", "Unknown")

            if op_name not in op_times:
                op_times[op_name] = {"total_ms": 0, "count": 0}

            op_times[op_name]["total_ms"] += dur
            op_times[op_name]["count"] += 1
            total_time += dur

    # Calculate percentages and sort
    op_stats = []
    for op_name, stats in op_times.items():
        percentage = (stats["total_ms"] / total_time * 100) if total_time > 0 else 0
        op_stats.append({
            "operator": op_name,
            "total_ms": stats["total_ms"],
            "count": stats["count"],
            "avg_ms": stats["total_ms"] / stats["count"],
            "percentage": percentage
        })

    op_stats.sort(key=lambda x: x["total_ms"], reverse=True)

    return {
        "total_time_ms": total_time,
        "operator_stats": op_stats[:20]  # Top 20 operators
    }


def print_results(model_name: str, relative_l2: float, timing_stats: Dict, profile_analysis: Dict = None):
    """Print evaluation results in a formatted table."""
    print(f"\n{model_name} Results:")
    print("-" * 80)
    if relative_l2 > 0:
        print(f"  Relative L2 Norm:     {relative_l2:.6e}")
    print(f"  Mean Inference Time:  {timing_stats['mean_ms']:.3f} ms")
    print(f"  Std Dev:              {timing_stats['std_ms']:.3f} ms")
    print(f"  Median:               {timing_stats['median_ms']:.3f} ms")
    print(f"  Min:                  {timing_stats['min_ms']:.3f} ms")
    print(f"  Max:                  {timing_stats['max_ms']:.3f} ms")
    print(f"  P95:                  {timing_stats['p95_ms']:.3f} ms")
    print(f"  P99:                  {timing_stats['p99_ms']:.3f} ms")

    if profile_analysis and "operator_stats" in profile_analysis:
        print(f"\n  Top 10 Most Expensive Operators:")
        print(f"  {'Operator':<30} {'Total(ms)':<12} {'Count':<8} {'Avg(ms)':<12} {'%':<8}")
        print(f"  {'-'*78}")
        for op in profile_analysis["operator_stats"][:10]:
            print(f"  {op['operator']:<30} {op['total_ms']:<12.3f} {op['count']:<8} "
                  f"{op['avg_ms']:<12.3f} {op['percentage']:<8.2f}")


def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)


def find_matmul_activation_tensors(model: onnx.ModelProto) -> Set[str]:
    """Find all activation tensors that feed into MatMul/QLinearMatMul operations.

    NOTE: This function searches the FP32 model and only finds ~73 tensors.
    For correct mixed-precision quantization, use find_matmul_activations_in_quantized_model() instead.
    """
    matmul_activations = set()

    for node in model.graph.node:
        # Check for MatMul, QLinearMatMul, Gemm
        if node.op_type in ['MatMul', 'QLinearMatMul', 'Gemm']:
            # First input is the activation (second is weight)
            activation_input = node.input[0]
            matmul_activations.add(activation_input)

        # Also check for DequantizeLinear that feeds MatMul
        elif node.op_type == 'DequantizeLinear':
            # Check if output goes to MatMul
            dq_output = node.output[0]
            for other_node in model.graph.node:
                if other_node.op_type in ['MatMul', 'QLinearMatMul', 'Gemm']:
                    if dq_output in other_node.input:
                        # This DQ feeds a MatMul, mark its input
                        matmul_activations.add(node.input[0])

    return matmul_activations


def find_matmul_activations_in_quantized_model(qdq_model_path: str) -> List[str]:
    """Find ALL activation tensors that feed MatMul operations in quantized model.

    This function analyzes a QUInt8 quantized model to identify all tensors that
    should be upgraded to uint16 for mixed-precision quantization.

    CRITICAL: Returns the ORIGINAL FP32 tensor names (before quantization),
    not the quantized tensor names. This is necessary because TensorQuantOverrides
    are applied during quantization of the FP32 model.

    Args:
        qdq_model_path: Path to a QUInt8 quantized model (QDQ format)

    Returns:
        List of FP32 tensor names to override to QUInt16
    """
    print(f"\nAnalyzing quantized model structure: {qdq_model_path}")
    model = onnx.load(qdq_model_path)
    fp32_tensor_names = set()

    # Build mapping: quantized tensor -> QuantizeLinear input (FP32 tensor)
    quantized_to_fp32 = {}
    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear':
            fp32_input = node.input[0]  # Original FP32 tensor
            quantized_output = node.output[0]  # Quantized tensor (e.g., "xxx_quantized")
            quantized_to_fp32[quantized_output] = fp32_input

    print(f"  Found {len(quantized_to_fp32)} QuantizeLinear nodes (mapping quantized -> FP32)")

    # Pattern 1: DequantizeLinear -> MatMul (will fuse to QLinearMatMul)
    # Find DQ nodes whose outputs feed into MatMul
    dq_to_matmul_inputs = {}
    for node in model.graph.node:
        if node.op_type == 'DequantizeLinear':
            dq_output = node.output[0]
            dq_input_tensor = node.input[0]  # The quantized tensor name
            dq_to_matmul_inputs[dq_output] = dq_input_tensor

    print(f"  Found {len(dq_to_matmul_inputs)} DequantizeLinear nodes")

    # Find MatMuls that consume DQ outputs
    matmul_count = 0
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            matmul_count += 1
            for input_name in node.input:
                if input_name in dq_to_matmul_inputs:
                    # This is a quantized tensor feeding MatMul
                    quantized_tensor = dq_to_matmul_inputs[input_name]

                    # Map back to FP32 tensor name
                    if quantized_tensor in quantized_to_fp32:
                        fp32_tensor = quantized_to_fp32[quantized_tensor]
                        fp32_tensor_names.add(fp32_tensor)
                    else:
                        # Fallback: if it's a weight tensor, it might not have a QuantizeLinear
                        # In this case, skip it (we only want activations)
                        pass

    print(f"  Found {matmul_count} MatMul nodes")

    # Pattern 2: Already-fused QLinearMatMul operators
    # (Less common with QDQ format, but handle it anyway)
    qlinearmatmul_count = 0
    for node in model.graph.node:
        if node.op_type == 'QLinearMatMul':
            qlinearmatmul_count += 1
            # Input 0 is the quantized activation tensor
            if len(node.input) > 0:
                quantized_tensor = node.input[0]
                if quantized_tensor in quantized_to_fp32:
                    fp32_tensor = quantized_to_fp32[quantized_tensor]
                    fp32_tensor_names.add(fp32_tensor)

    print(f"  Found {qlinearmatmul_count} QLinearMatMul nodes")

    tensor_list = list(fp32_tensor_names)
    print(f"\n[OK] Found {len(tensor_list)} unique FP32 tensors that feed MatMul operations")
    print(f"  This is {len(tensor_list)/73:.1f}x more than the FP32 model search (73 tensors)\n")

    return tensor_list


def insert_requant_node(
    model: onnx.ModelProto,
    tensor_name: str,
    target_type: int,  # TensorProto.UINT16
    new_scale_name: str
) -> str:
    """Insert DQ->Q pair to requantize tensor from uint8 to uint16 at runtime.

    This allows using the full [0, 65535] range for better precision.

    Args:
        model: ONNX model
        tensor_name: Name of uint8 quantized tensor to convert
        target_type: Target quantization type (UINT16)
        new_scale_name: Name of the uint16 scale tensor

    Returns:
        Name of the new uint16 tensor
    """
    # Create intermediate FP32 tensor name
    fp32_tensor_name = tensor_name + "_fp32_intermediate"
    uint16_tensor_name = tensor_name + "_uint16"

    # Find the QuantizeLinear node that produces this tensor
    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear' and node.output[0] == tensor_name:
            scale_name = node.input[1]
            zp_name = node.input[2] if len(node.input) > 2 else ""

            # Create DequantizeLinear node (uint8 -> FP32)
            dq_node = helper.make_node(
                'DequantizeLinear',
                inputs=[tensor_name, scale_name, zp_name] if zp_name else [tensor_name, scale_name],
                outputs=[fp32_tensor_name],
                name=f"DQ_{tensor_name}_to_fp32"
            )

            # Create new zero_point for uint16 (keep as uint8 type for compatibility)
            # Get original zero_point value
            zp_value = 0
            if zp_name:
                for init in model.graph.initializer:
                    if init.name == zp_name:
                        zp_array = numpy_helper.to_array(init)
                        zp_value = int(zp_array.item()) if zp_array.size == 1 else 0
                        break

            new_zp_name = zp_name + "_uint16" if zp_name else tensor_name + "_zp_uint16"
            new_zp_init = numpy_helper.from_array(np.array(zp_value, dtype=np.uint8), new_zp_name)
            model.graph.initializer.append(new_zp_init)

            # Create QuantizeLinear node (FP32 -> uint16)
            q_node = helper.make_node(
                'QuantizeLinear',
                inputs=[fp32_tensor_name, new_scale_name, new_zp_name],
                outputs=[uint16_tensor_name],
                name=f"Q_{tensor_name}_to_uint16"
            )

            # Add nodes to graph
            model.graph.node.extend([dq_node, q_node])

            # Update value_info for the new uint16 tensor
            # Copy shape info from original tensor
            for value_info in model.graph.value_info:
                if value_info.name == tensor_name:
                    new_value_info = helper.make_tensor_value_info(
                        uint16_tensor_name,
                        target_type,
                        [dim.dim_value if dim.HasField('dim_value') else None
                         for dim in value_info.type.tensor_type.shape.dim]
                    )
                    model.graph.value_info.append(new_value_info)
                    break

            return uint16_tensor_name

    return tensor_name  # No change if not found


def upgrade_quantization_to_uint16(
    model: onnx.ModelProto,
    tensor_name: str,
    scale_strategy: str = "no_adjust"
) -> bool:
    """Upgrade a specific tensor's quantization from uint8 to uint16.

    Args:
        model: ONNX model
        tensor_name: Name of tensor to upgrade
        scale_strategy: How to handle scale adjustment:
            - "no_adjust": Keep scale unchanged, values stay in [0, 255] (fastest, same accuracy as uint8)
            - "adjust_scale": Multiply scale by (255/65535) to match uint8 range (old approach, causes NaN)
            - "full_range": Insert DQ->Q pair to requantize to full [0, 65535] range (best accuracy)

    Returns:
        True if tensor was modified
    """
    modified = False

    if scale_strategy == "full_range":
        # FULL RANGE APPROACH: Insert DQ->Q conversion nodes
        # This requantizes at runtime to use the full [0, 65535] range

        # Find the QuantizeLinear node that produces this tensor
        for node in model.graph.node:
            if node.op_type == 'QuantizeLinear' and node.output[0] == tensor_name:
                scale_name = node.input[1]

                # Create new scale for uint16 range
                for init in model.graph.initializer:
                    if init.name == scale_name:
                        scale_data = numpy_helper.to_array(init)
                        # Scale for full uint16 range [0, 65535]
                        new_scale_data = scale_data * (255.0 / 65535.0)
                        new_scale_name = scale_name + "_uint16"
                        new_scale_init = numpy_helper.from_array(new_scale_data, new_scale_name)
                        model.graph.initializer.append(new_scale_init)

                        # Insert DQ->Q pair to requantize
                        new_tensor_name = insert_requant_node(
                            model, tensor_name, TensorProto.UINT16, new_scale_name
                        )

                        # Update all downstream nodes to use the new uint16 tensor
                        for downstream_node in model.graph.node:
                            for i, input_name in enumerate(downstream_node.input):
                                if input_name == tensor_name:
                                    downstream_node.input[i] = new_tensor_name

                        modified = True
                        break
                break

    else:
        # NO_ADJUST or ADJUST_SCALE: Just change tensor type and optionally scale
        for node in model.graph.node:
            if node.op_type == 'QuantizeLinear' and node.output[0] == tensor_name:
                scale_name = node.input[1] if len(node.input) > 1 else None

                if scale_strategy == "adjust_scale" and scale_name:
                    # OLD APPROACH: Adjust scale (causes NaN issues)
                    for init in model.graph.initializer:
                        if init.name == scale_name:
                            scale_data = numpy_helper.to_array(init)
                            new_scale_data = scale_data * (255.0 / 65535.0)
                            new_init = numpy_helper.from_array(new_scale_data, init.name)
                            model.graph.initializer.remove(init)
                            model.graph.initializer.append(new_init)
                            break

                # Update the tensor type in value_info
                for value_info in model.graph.value_info:
                    if value_info.name == tensor_name:
                        value_info.type.tensor_type.elem_type = TensorProto.UINT16
                        break

                modified = True

        # Find DequantizeLinear nodes that consume this tensor
        if scale_strategy == "adjust_scale":
            for node in model.graph.node:
                if node.op_type == 'DequantizeLinear' and node.input[0] == tensor_name:
                    scale_name = node.input[1] if len(node.input) > 1 else None

                    if scale_name:
                        for init in model.graph.initializer:
                            if init.name == scale_name:
                                scale_data = numpy_helper.to_array(init)
                                new_scale_data = scale_data * (255.0 / 65535.0)
                                new_init = numpy_helper.from_array(new_scale_data, init.name)
                                model.graph.initializer.remove(init)
                                model.graph.initializer.append(new_init)
                                break

                    modified = True

    return modified


def create_mixed_precision_model(
    quint8_model_path: str,
    output_path: str,
    scale_strategy: str = "no_adjust"
) -> str:
    """Convert QUInt8 model to mixed precision: QUInt16 MatMul activations, QUInt8 everything else.

    Args:
        quint8_model_path: Path to QUInt8 quantized model
        output_path: Path to save mixed-precision model
        scale_strategy: Scale adjustment strategy - "no_adjust", "adjust_scale", or "full_range"
    """
    print(f"\n  Upgrading MatMul activations to QUInt16 (strategy: {scale_strategy})...")

    # Load the QUInt8 model
    model = onnx.load(quint8_model_path)

    # Find all MatMul activation tensors
    matmul_activations = find_matmul_activation_tensors(model)
    print(f"  Found {len(matmul_activations)} MatMul activation tensors")

    if not matmul_activations:
        print("  WARNING: No MatMul activations found")
        return quint8_model_path

    # Upgrade each MatMul activation from uint8 to uint16
    upgraded_count = 0
    for tensor_name in matmul_activations:
        if upgrade_quantization_to_uint16(model, tensor_name, scale_strategy):
            upgraded_count += 1

    print(f"  [OK] Upgraded {upgraded_count} tensors to QUInt16")

    # Save the mixed-precision model
    onnx.save(model, output_path)
    print(f"  [OK] Mixed-precision model saved")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Model Quantization Evaluation with Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output-dir", type=str, default="./quantized_models",
                       help="Output directory (default: ./quantized_models)")
    parser.add_argument("--num-calibration-samples", type=int, default=100,
                       help="Number of calibration samples (default: 100)")
    parser.add_argument("--num-validation-samples", type=int, default=100,
                       help="Number of validation samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable detailed profiling with operator breakdown")
    parser.add_argument("--test-configs", action="store_true",
                       help="Test multiple quantization configurations (QInt8, QInt16)")
    parser.add_argument("--configs", type=str, default=None,
                       help="Comma-separated list of configs to test: base,qint8,qint8_perchan,qint16 (default: all if --test-configs, else base only)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    print("=" * 80)
    print("LOADING CONFIGURATION")
    print("=" * 80)
    model_description, base_config = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    if model_description:
        print("\nModel Description:")
        print(json.dumps(model_description, indent=2))
    print("\nBase Quantization Config:")
    print(json.dumps(base_config, indent=2))

    # Generate calibration and validation data
    print("\n" + "=" * 80)
    print("GENERATING DATA")
    print("=" * 80)
    print(f"Generating {args.num_calibration_samples} calibration samples...")
    calibration_reader = DummyCalibrationDataReader(
        args.model,
        num_samples=args.num_calibration_samples,
        seed=args.seed,
        model_description=model_description,
    )
    print("[OK] Calibration data generated")

    print(f"Generating {args.num_validation_samples} validation samples...")
    validation_reader = DummyCalibrationDataReader(
        args.model,
        num_samples=args.num_validation_samples,
        seed=args.seed + 1000,
        model_description=model_description,
    )
    validation_samples = validation_reader.get_all_samples()
    print("[OK] Validation data generated")

    # Define all available configurations
    all_configs = {}

    # Base config from JSON
    all_configs["base"] = base_config

    # QInt8 configuration (only change activation type)
    qint8_config = base_config.copy()
    qint8_config["activation_type"] = "QInt8"
    all_configs["qint8"] = qint8_config

    # QInt8 with per-channel (only change activation type)
    qint8_perchan_config = base_config.copy()
    qint8_perchan_config["activation_type"] = "QInt8"
    qint8_perchan_config["per_channel"] = True
    all_configs["qint8_perchan"] = qint8_perchan_config

    # QInt16 configuration (only change activation type to signed int16)
    qint16_config = base_config.copy()
    qint16_config["activation_type"] = "QInt16"
    all_configs["qint16"] = qint16_config

    # QUInt8 configuration (unsigned 8-bit activation x unsigned 8-bit weights - INDUSTRY STANDARD)
    quint8_config = base_config.copy()
    quint8_config["activation_type"] = "QUInt8"
    quint8_config["weight_type"] = "QUInt8"
    all_configs["quint8"] = quint8_config

    # QUInt16 configuration
    quint16_config = base_config.copy()
    quint16_config["activation_type"] = "QUInt16"
    all_configs["quint16"] = quint16_config

    # Mixed-precision configurations (QUInt16 MatMul + QUInt8 everything else)
    # Variant 1: No scale adjustment (fastest, same accuracy as uint8)
    mixed_config = base_config.copy()
    mixed_config["activation_type"] = "QUInt8"  # Start with QUInt8
    mixed_config["weight_type"] = "QUInt8"
    mixed_config["_is_mixed_precision"] = True  # Flag for special handling
    mixed_config["_scale_strategy"] = "no_adjust"  # Keep scale unchanged
    all_configs["mixed"] = mixed_config

    # Variant 2: Adjust scale (old approach, for comparison - known to produce NaN)
    mixed_adjust_config = base_config.copy()
    mixed_adjust_config["activation_type"] = "QUInt8"
    mixed_adjust_config["weight_type"] = "QUInt8"
    mixed_adjust_config["_is_mixed_precision"] = True
    mixed_adjust_config["_scale_strategy"] = "adjust_scale"
    all_configs["mixed_adjust"] = mixed_adjust_config

    # Variant 3: Direct uint16 quantization with full [0, 65535] range (reference - performs poorly)
    # This uses ONNX Runtime's native uint16 support for maximum accuracy
    quint16_full_config = base_config.copy()
    quint16_full_config["activation_type"] = "QUInt16"  # Quantize directly to uint16
    quint16_full_config["weight_type"] = "QUInt8"  # Keep weights as uint8
    quint16_full_config["extra_options"] = quint16_full_config.get("extra_options", {}).copy()
    quint16_full_config["extra_options"]["UseQDQContribOps"] = True  # Enable 16-bit support
    all_configs["quint16_full"] = quint16_full_config

    # Variant 4: Mixed-precision with FULL uint16 range [0, 65535] using TensorQuantOverrides
    # NOTE: This is the OLD BROKEN approach - only finds ~73 tensors, results in 3,960 FP32 fallbacks
    mixed_u16_config = base_config.copy()
    mixed_u16_config["activation_type"] = "QUInt8"  # Base quantization is uint8
    mixed_u16_config["weight_type"] = "QUInt8"
    mixed_u16_config["_use_tensor_overrides"] = True  # Flag to use TensorQuantOverrides
    mixed_u16_config["extra_options"] = mixed_u16_config.get("extra_options", {}).copy()
    mixed_u16_config["extra_options"]["UseQDQContribOps"] = True  # Enable 16-bit support
    all_configs["mixed_u16"] = mixed_u16_config

    # Variant 5: Mixed-precision with FULL uint16 range using TWO-STAGE approach (FIXED)
    # This is the CORRECT approach: finds ALL MatMul activations (~5,000 tensors)
    # Expected: 5,995 QLinearMatMul ops, 0 FP32 fallbacks, 100-150ms performance
    mixed_u16_fixed_config = base_config.copy()
    mixed_u16_fixed_config["activation_type"] = "QUInt8"  # Base quantization is uint8
    mixed_u16_fixed_config["weight_type"] = "QUInt8"
    mixed_u16_fixed_config["_use_two_stage_uint16"] = True  # Flag to use two-stage approach
    mixed_u16_fixed_config["extra_options"] = mixed_u16_fixed_config.get("extra_options", {}).copy()
    mixed_u16_fixed_config["extra_options"]["UseQDQContribOps"] = True  # Enable 16-bit support
    all_configs["mixed_u16_fixed"] = mixed_u16_fixed_config

    # Dynamic quantization configuration
    dynamic_config = {
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "_is_dynamic": True  # Flag for special handling
    }
    all_configs["dynamic"] = dynamic_config

    # Select which configs to test
    if args.configs:
        # User specified specific configs
        requested_configs = [c.strip() for c in args.configs.split(",")]
        configs_to_test = {}
        for config_name in requested_configs:
            if config_name in all_configs:
                configs_to_test[config_name] = all_configs[config_name]
            else:
                print(f"Warning: Unknown config '{config_name}', skipping. Available: {list(all_configs.keys())}")
        if not configs_to_test:
            print("Error: No valid configurations specified")
            return 1
    elif args.test_configs:
        # Test all configs
        configs_to_test = all_configs
    else:
        # Default: just base config
        configs_to_test = {"base": base_config}

    model_name = Path(args.model).stem
    all_results = {}

    # Evaluate FP32 baseline
    print("\n" + "=" * 80)
    print("EVALUATING FP32 BASELINE")
    print("=" * 80)
    print("\nEvaluating FP32 baseline model...")
    fp32_outputs, _, fp32_timing, fp32_profile = evaluate_model_with_profiling(
        args.model, validation_samples, enable_profiling=args.enable_profiling,
        profile_prefix=f"{model_name}_fp32"
    )
    fp32_profile_analysis = analyze_profile(fp32_profile) if fp32_profile else {}
    print("[OK] FP32 evaluation complete")
    print_results("FP32", 0.0, fp32_timing, fp32_profile_analysis)

    all_results["FP32"] = {
        "relative_l2": 0.0,
        "timing": fp32_timing,
        "size_mb": get_model_size(args.model),
        "profile_file": fp32_profile,
        "profile_analysis": fp32_profile_analysis
    }

    # Test each configuration
    for config_name, config in configs_to_test.items():
        print(f"\n{'=' * 80}")
        print(f"TESTING CONFIGURATION: {config_name.upper()}")
        print(f"{'=' * 80}")

        qdq_model_path = output_dir / f"{model_name}_{config_name}_qdq.onnx"

        # Quantize
        try:
            # Check if this is dynamic quantization
            if config.get("_is_dynamic", False):
                # Dynamic quantization (no calibration needed)
                dynamic_model_path = output_dir / f"{model_name}_{config_name}.onnx"
                quantize_to_dynamic(args.model, str(dynamic_model_path), config)

                # Evaluate dynamic model
                print(f"\nEvaluating dynamic model ({config_name})...")
                _, dyn_l2, dyn_timing, dyn_profile = evaluate_model_with_profiling(
                    str(dynamic_model_path), validation_samples, fp32_outputs,
                    enable_profiling=args.enable_profiling,
                    profile_prefix=f"{model_name}_{config_name}"
                )
                dyn_profile_analysis = analyze_profile(dyn_profile) if dyn_profile else {}
                print(f"[OK] Dynamic evaluation complete ({config_name})")
                print_results(f"Dynamic ({config_name})", dyn_l2, dyn_timing, dyn_profile_analysis)

                all_results[f"Dynamic_{config_name}"] = {
                    "relative_l2": dyn_l2,
                    "timing": dyn_timing,
                    "size_mb": get_model_size(str(dynamic_model_path)),
                    "profile_file": dyn_profile,
                    "profile_analysis": dyn_profile_analysis,
                    "config": config
                }
            else:
                # QDQ quantization (static)
                calibration_reader.rewind()

                # Check if we should use two-stage TensorQuantOverrides (FIXED approach)
                if config.get("_use_two_stage_uint16", False):
                    # Mixed-precision using two-stage TensorQuantOverrides
                    # This is the CORRECT implementation that finds ALL MatMul activations
                    print(f"\n  Using two-stage quantization for full uint16 range...")
                    quantize_mixed_precision_full_uint16(
                        args.model,
                        str(qdq_model_path),
                        calibration_reader,
                        config
                    )

                # Check if we should use TensorQuantOverrides for mixed-precision (OLD broken approach)
                elif config.get("_use_tensor_overrides", False):
                    # Mixed-precision using TensorQuantOverrides - OLD APPROACH (only finds 73 tensors)
                    print(f"\n  WARNING: Using OLD TensorQuantOverrides approach (only finds ~73 tensors)")
                    print(f"  Consider using _use_two_stage_uint16 instead for better coverage")
                    tensor_overrides = create_tensor_quant_overrides_for_matmul(args.model)
                    print(f"  [OK] Created overrides for {len(tensor_overrides)} tensors")

                    # Add to extra_options
                    config_with_overrides = config.copy()
                    extra_opts = config_with_overrides.get("extra_options", {}).copy()
                    extra_opts["TensorQuantOverrides"] = tensor_overrides
                    config_with_overrides["extra_options"] = extra_opts

                    # Quantize with overrides
                    print(f"  Quantizing with uint16 overrides for MatMul activations...")
                    quantize_to_qdq(args.model, str(qdq_model_path), calibration_reader, config_with_overrides)

                # Check if this is mixed-precision (post-processing approach)
                elif config.get("_is_mixed_precision", False):
                    # Mixed-precision: Two-step process (old approach)
                    scale_strategy = config.get("_scale_strategy", "no_adjust")

                    # Step 1: Quantize to QUInt8
                    quint8_temp_path = output_dir / f"{model_name}_{config_name}_quint8_temp.onnx"
                    print(f"\n  Step 1: Quantizing to QUInt8 (temporary)...")
                    quantize_to_qdq(args.model, str(quint8_temp_path), calibration_reader, config)

                    # Step 2: Upgrade MatMul activations to QUInt16
                    print(f"  Step 2: Converting to mixed-precision (strategy: {scale_strategy})...")
                    create_mixed_precision_model(
                        str(quint8_temp_path),
                        str(qdq_model_path),
                        scale_strategy=scale_strategy
                    )

                    # Clean up temporary file
                    if quint8_temp_path.exists():
                        quint8_temp_path.unlink()
                else:
                    # Normal quantization
                    quantize_to_qdq(args.model, str(qdq_model_path), calibration_reader, config)

                # Evaluate QDQ model
                print(f"\nEvaluating QDQ model ({config_name})...")
                _, qdq_l2, qdq_timing, qdq_profile = evaluate_model_with_profiling(
                    str(qdq_model_path), validation_samples, fp32_outputs,
                    enable_profiling=args.enable_profiling,
                    profile_prefix=f"{model_name}_{config_name}_qdq"
                )
                qdq_profile_analysis = analyze_profile(qdq_profile) if qdq_profile else {}
                print(f"[OK] QDQ evaluation complete ({config_name})")
                print_results(f"QDQ ({config_name})", qdq_l2, qdq_timing, qdq_profile_analysis)

                all_results[f"QDQ_{config_name}"] = {
                    "relative_l2": qdq_l2,
                    "timing": qdq_timing,
                    "size_mb": get_model_size(str(qdq_model_path)),
                    "profile_file": qdq_profile,
                    "profile_analysis": qdq_profile_analysis,
                    "config": config
                }
        except Exception as e:
            print(f"Error during quantization ({config_name}): {e}")
            import traceback
            traceback.print_exc()

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'Rel. L2':<12} {'Time(ms)':<12} {'Size(MB)':<12} {'Speedup':<10}")
    print("-" * 80)

    fp32_mean_time = all_results["FP32"]["timing"]["mean_ms"]
    for model_type, metrics in all_results.items():
        speedup = fp32_mean_time / metrics["timing"]["mean_ms"]
        l2_str = f"{metrics['relative_l2']:.3e}" if metrics["relative_l2"] > 0 else "baseline"
        print(f"{model_type:<25} {l2_str:<12} {metrics['timing']['mean_ms']:<12.3f} "
              f"{metrics['size_mb']:<12.2f} {speedup:<10.2f}x")

    # Save results
    results_path = output_dir / f"{model_name}_profiling_results.json"
    with open(results_path, "w") as f:
        json_results = {}
        for model_type, metrics in all_results.items():
            json_results[model_type] = {
                "relative_l2": float(metrics["relative_l2"]),
                "timing": {k: float(v) for k, v in metrics["timing"].items()},
                "size_mb": float(metrics["size_mb"]),
                "profile_file": metrics.get("profile_file"),
                "profile_analysis": metrics.get("profile_analysis", {}),
                "config": metrics.get("config", {})
            }
        json.dump(json_results, f, indent=2)

    print(f"\n[OK] Results saved to: {results_path}")

    if args.enable_profiling:
        print("\n[INFO] Chrome tracing files generated. View them at:")
        print("  - chrome://tracing (Chrome browser)")
        print("  - edge://tracing (Edge browser)")
        print("  - https://ui.perfetto.dev/ (Perfetto UI)")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
