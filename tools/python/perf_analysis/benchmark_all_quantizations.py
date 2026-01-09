#!/usr/bin/env python3
"""
Comprehensive benchmark comparing FP32 vs all quantization approaches.
Tests: FP32, Dynamic QUInt8, Static QUInt8, Static QUInt16
"""

import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path

def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)
    return 0

def benchmark_model(model_path, model_name, num_runs=20, warmup_runs=5,
                   graph_optimization=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                   enable_qdq_cleanup=True):
    """Benchmark a single model."""

    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found!")
        return None

    model_size = get_model_size_mb(model_path)
    print(f"Model size: {model_size:.2f} MB")

    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = graph_optimization
    sess_options.log_severity_level = 3  # ERROR only (reduce log spam)

    # Enable Q/DQ cleanup for quantized models
    if enable_qdq_cleanup and "quint" in model_path.lower():
        sess_options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
        print("Q/DQ cleanup: ENABLED")

    try:
        # Create session
        session = ort.InferenceSession(model_path, sess_options,
                                      providers=['CPUExecutionProvider'])

        # Get input info
        inputs = session.get_inputs()
        print(f"\nModel inputs:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape} ({inp.type})")

        # Create dummy input data
        feed_dict = {}
        for inp in inputs:
            if 'attention_mask' in inp.name:
                feed_dict[inp.name] = np.ones((1, 77), dtype=np.float32)
            else:
                shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                feed_dict[inp.name] = np.random.randn(*shape).astype(np.float32)

        # Warmup runs
        print(f"\nWarmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = session.run(None, feed_dict)

        # Benchmark runs
        print(f"Benchmarking ({num_runs} runs)...")
        times = []

        for i in range(num_runs):
            start = time.perf_counter()
            outputs = session.run(None, feed_dict)
            end = time.perf_counter()
            elapsed = (end - start) * 1000  # Convert to ms
            times.append(elapsed)
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_runs} runs complete")

        # Calculate statistics
        times_array = np.array(times)
        stats = {
            'model_name': model_name,
            'model_path': model_path,
            'model_size_mb': model_size,
            'min_ms': np.min(times_array),
            'mean_ms': np.mean(times_array),
            'median_ms': np.median(times_array),
            'max_ms': np.max(times_array),
            'std_ms': np.std(times_array),
            'p95_ms': np.percentile(times_array, 95),
            'p99_ms': np.percentile(times_array, 99),
            'num_runs': num_runs,
            'output_shape': outputs[0].shape if outputs else None,
            'all_times': times_array.tolist()
        }

        print(f"\nStatistics:")
        print(f"  Min:    {stats['min_ms']:.2f}ms")
        print(f"  Mean:   {stats['mean_ms']:.2f}ms")
        print(f"  Median: {stats['median_ms']:.2f}ms")
        print(f"  Max:    {stats['max_ms']:.2f}ms")
        print(f"  Std:    {stats['std_ms']:.2f}ms")
        print(f"  P95:    {stats['p95_ms']:.2f}ms")
        print(f"  P99:    {stats['p99_ms']:.2f}ms")

        return stats

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_outputs(fp32_model_path, quant_model_path):
    """Compare outputs between FP32 and quantized model for accuracy."""

    print(f"\n{'='*80}")
    print(f"Accuracy Comparison: FP32 vs Quantized")
    print(f"{'='*80}")

    try:
        # Load both models
        fp32_session = ort.InferenceSession(fp32_model_path,
                                           providers=['CPUExecutionProvider'])

        quant_options = ort.SessionOptions()
        quant_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        quant_options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
        quant_session = ort.InferenceSession(quant_model_path, quant_options,
                                            providers=['CPUExecutionProvider'])

        # Create test input
        inputs = fp32_session.get_inputs()
        feed_dict = {}
        for inp in inputs:
            if 'attention_mask' in inp.name:
                feed_dict[inp.name] = np.ones((1, 77), dtype=np.float32)
            else:
                shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                # Use fixed random seed for reproducibility
                np.random.seed(42)
                feed_dict[inp.name] = np.random.randn(*shape).astype(np.float32)

        # Run inference
        fp32_output = fp32_session.run(None, feed_dict)[0]
        quant_output = quant_session.run(None, feed_dict)[0]

        # Calculate accuracy metrics
        diff = fp32_output - quant_output
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / (np.abs(fp32_output) + 1e-8)

        mae = np.mean(abs_diff)
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        max_abs_error = np.max(abs_diff)
        mean_rel_error = np.mean(rel_diff) * 100

        # Cosine similarity
        fp32_flat = fp32_output.flatten()
        quant_flat = quant_output.flatten()
        cosine_sim = np.dot(fp32_flat, quant_flat) / (np.linalg.norm(fp32_flat) * np.linalg.norm(quant_flat))

        print(f"\nAccuracy Metrics:")
        print(f"  Mean Absolute Error (MAE):  {mae:.6e}")
        print(f"  Root Mean Squared Error:    {rmse:.6e}")
        print(f"  Max Absolute Error:         {max_abs_error:.6e}")
        print(f"  Mean Relative Error:        {mean_rel_error:.4f}%")
        print(f"  Cosine Similarity:          {cosine_sim:.6f}")

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'max_abs_error': float(max_abs_error),
            'mean_rel_error_pct': float(mean_rel_error),
            'cosine_similarity': float(cosine_sim)
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def print_comparison_table(results):
    """Print comprehensive comparison table."""

    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE BENCHMARK COMPARISON")
    print(f"{'='*100}")

    if not results:
        print("No results to display")
        return

    # Performance comparison
    print(f"\nPERFORMANCE COMPARISON")
    print(f"{'-'*100}")
    print(f"{'Model':<30} {'Size (MB)':<12} {'Min (ms)':<12} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std (ms)':<12}")
    print(f"{'-'*100}")

    fp32_mean = None
    for result in results:
        if result:
            name = result['model_name']
            size = result['model_size_mb']
            min_t = result['min_ms']
            mean_t = result['mean_ms']
            median_t = result['median_ms']
            std_t = result['std_ms']

            print(f"{name:<30} {size:<12.2f} {min_t:<12.2f} {mean_t:<12.2f} {median_t:<12.2f} {std_t:<12.2f}")

            if 'FP32' in name:
                fp32_mean = mean_t

    # Speedup comparison
    if fp32_mean:
        print(f"\nSPEEDUP vs FP32 BASELINE")
        print(f"{'-'*100}")
        print(f"{'Model':<30} {'Mean Time':<15} {'Speedup':<15} {'Status':<20}")
        print(f"{'-'*100}")

        for result in results:
            if result:
                name = result['model_name']
                mean_t = result['mean_ms']
                speedup = fp32_mean / mean_t

                if speedup > 1.0:
                    status = f"[OK] {speedup:.2f}x FASTER"
                elif speedup < 1.0:
                    status = f"[SLOW] {1/speedup:.2f}x SLOWER"
                else:
                    status = "SAME"

                print(f"{name:<30} {mean_t:<15.2f} {speedup:<15.2f} {status:<20}")

    # Model size comparison
    print(f"\nMODEL SIZE COMPARISON")
    print(f"{'-'*100}")
    print(f"{'Model':<30} {'Size (MB)':<15} {'vs FP32':<15} {'Compression':<20}")
    print(f"{'-'*100}")

    fp32_size = None
    for result in results:
        if result and 'FP32' in result['model_name']:
            fp32_size = result['model_size_mb']
            break

    for result in results:
        if result:
            name = result['model_name']
            size = result['model_size_mb']

            if fp32_size:
                compression = fp32_size / size if size > 0 else 0
                size_reduction = ((fp32_size - size) / fp32_size * 100) if fp32_size > 0 else 0
                print(f"{name:<30} {size:<15.2f} {compression:<15.2f}x {size_reduction:<19.1f}% smaller")
            else:
                print(f"{name:<30} {size:<15.2f} {'N/A':<15} {'N/A':<20}")

def main():
    """Main benchmark runner."""

    print(f"{'='*100}")
    print(f"COMPREHENSIVE QUANTIZATION BENCHMARK")
    print(f"Florence v1.6.2 Text Transformer Model")
    print(f"{'='*100}")

    # Model paths
    base_dir = Path(r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer")
    quantized_dir = Path(r"C:\d\onnxruntime\quantized_models")

    models = [
        {
            'name': 'FP32 Baseline',
            'path': base_dir / "model" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx",
            'enable_qdq_cleanup': False
        },
        {
            'name': 'QUInt16×QUInt8 (Proper)',
            'path': quantized_dir / "florence_quint16_qdq_proper.onnx",
            'enable_qdq_cleanup': True
        },
        {
            'name': 'QUInt16×QUInt8 (Old)',
            'path': quantized_dir / "florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_qdq.onnx",
            'enable_qdq_cleanup': False  # Test the old broken one for comparison
        },
    ]

    # Run benchmarks
    results = []

    for model_info in models:
        result = benchmark_model(
            model_path=str(model_info['path']),
            model_name=model_info['name'],
            num_runs=20,
            warmup_runs=5,
            graph_optimization=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            enable_qdq_cleanup=model_info['enable_qdq_cleanup']
        )
        results.append(result)

        # Sleep briefly between benchmarks to let system settle
        time.sleep(2)

    # Compare accuracy for quantized models
    fp32_path = str(base_dir / "model" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx")
    quint16_proper_path = str(quantized_dir / "florence_quint16_qdq_proper.onnx")

    if os.path.exists(fp32_path) and os.path.exists(quint16_proper_path):
        accuracy_metrics = compare_outputs(fp32_path, quint16_proper_path)
        if accuracy_metrics:
            # Add accuracy to results
            for result in results:
                if result and 'Proper' in result['model_name']:
                    result.update(accuracy_metrics)

    # Print comparison table
    print_comparison_table(results)

    # Save results to JSON
    import json
    output_file = Path(r"C:\d\onnxruntime\runtime_analysis") / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print(f"\n{'='*100}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
