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
from typing import Dict, List, Tuple
import subprocess
import sys

import numpy as np
import onnx
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

    # QUInt16 configuration (unsigned 16-bit activation, typically with QUInt8 weights)
    quint16_config = base_config.copy()
    quint16_config["activation_type"] = "QUInt16"
    all_configs["quint16"] = quint16_config

    # QInt16 configuration (only change activation type to signed int16)
    qint16_config = base_config.copy()
    qint16_config["activation_type"] = "QInt16"
    all_configs["qint16"] = qint16_config

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

        # Quantize to QDQ
        try:
            calibration_reader.rewind()
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
            print(f"Error during QDQ quantization ({config_name}): {e}")

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
