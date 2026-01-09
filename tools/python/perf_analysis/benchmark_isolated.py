#!/usr/bin/env python3
"""
Thermally-controlled benchmark - runs each model in isolation to avoid thermal effects.
Usage: python benchmark_isolated.py <model_name> <model_path>
"""

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import json
from pathlib import Path

def benchmark_single_model(model_path, model_name, num_runs=30, warmup_runs=10,
                          enable_qdq_cleanup=True):
    """Benchmark a single model in isolation."""

    print(f"\n{'='*80}")
    print(f"ISOLATED BENCHMARK: {model_name}")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Runs: {num_runs} (after {warmup_runs} warmup)")
    print(f"Optimization: ORT_ENABLE_EXTENDED")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found!")
        return None

    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Size: {model_size:.2f} MB")

    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.log_severity_level = 3  # ERROR only

    # Enable Q/DQ cleanup for quantized models
    if enable_qdq_cleanup and "quint" in model_path.lower():
        sess_options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
        print(f"Q/DQ cleanup: ENABLED")
    else:
        print(f"Q/DQ cleanup: N/A")

    try:
        # Create session
        print(f"\nLoading model...")
        session = ort.InferenceSession(model_path, sess_options,
                                      providers=['CPUExecutionProvider'])

        # Get input info
        inputs = session.get_inputs()
        print(f"\nInputs:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape}")

        # Create dummy input data (fixed seed for reproducibility)
        np.random.seed(42)
        feed_dict = {}
        for inp in inputs:
            if 'attention_mask' in inp.name:
                feed_dict[inp.name] = np.ones((1, 77), dtype=np.float32)
            else:
                shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                feed_dict[inp.name] = np.random.randn(*shape).astype(np.float32)

        # Initial warmup phase
        print(f"\nWarmup phase ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            _ = session.run(None, feed_dict)
            if (i + 1) % 5 == 0:
                print(f"  Warmup: {i+1}/{warmup_runs}")

        # Cool down before actual benchmark
        print(f"\nCooling down (5 seconds)...")
        time.sleep(5)

        # Benchmark runs
        print(f"\nBenchmark phase ({num_runs} runs)...")
        times = []

        for i in range(num_runs):
            start = time.perf_counter()
            outputs = session.run(None, feed_dict)
            end = time.perf_counter()
            elapsed = (end - start) * 1000  # Convert to ms
            times.append(elapsed)

            if (i + 1) % 10 == 0:
                current_mean = np.mean(times)
                current_std = np.std(times)
                print(f"  Progress: {i+1}/{num_runs} - Mean: {current_mean:.2f}ms, Std: {current_std:.2f}ms")

        # Calculate statistics
        times_array = np.array(times)

        # Remove outliers (> 3 std deviations) for more stable mean
        mean = np.mean(times_array)
        std = np.std(times_array)
        mask = np.abs(times_array - mean) < 3 * std
        times_filtered = times_array[mask]

        stats = {
            'model_name': model_name,
            'model_path': model_path,
            'model_size_mb': float(model_size),
            'num_runs': num_runs,
            'num_outliers_removed': int(num_runs - len(times_filtered)),

            # Raw statistics (all data)
            'raw_min_ms': float(np.min(times_array)),
            'raw_max_ms': float(np.max(times_array)),
            'raw_mean_ms': float(np.mean(times_array)),
            'raw_median_ms': float(np.median(times_array)),
            'raw_std_ms': float(np.std(times_array)),

            # Filtered statistics (outliers removed)
            'min_ms': float(np.min(times_filtered)),
            'max_ms': float(np.max(times_filtered)),
            'mean_ms': float(np.mean(times_filtered)),
            'median_ms': float(np.median(times_filtered)),
            'std_ms': float(np.std(times_filtered)),
            'p95_ms': float(np.percentile(times_filtered, 95)),
            'p99_ms': float(np.percentile(times_filtered, 99)),

            # All timing data
            'all_times_ms': [float(t) for t in times_array.tolist()],
        }

        print(f"\n{'='*80}")
        print(f"RESULTS (outliers removed: {stats['num_outliers_removed']})")
        print(f"{'='*80}")
        print(f"  Min:     {stats['min_ms']:.2f}ms")
        print(f"  Mean:    {stats['mean_ms']:.2f}ms")
        print(f"  Median:  {stats['median_ms']:.2f}ms")
        print(f"  Max:     {stats['max_ms']:.2f}ms")
        print(f"  Std:     {stats['std_ms']:.2f}ms")
        print(f"  P95:     {stats['p95_ms']:.2f}ms")
        print(f"  P99:     {stats['p99_ms']:.2f}ms")

        print(f"\nRaw stats (all data):")
        print(f"  Mean:    {stats['raw_mean_ms']:.2f}ms")
        print(f"  Range:   {stats['raw_min_ms']:.2f}ms - {stats['raw_max_ms']:.2f}ms")
        print(f"  Std:     {stats['raw_std_ms']:.2f}ms")

        # Save results
        output_dir = Path(r"C:\d\onnxruntime\runtime_analysis\isolated_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{model_name.replace(' ', '_')}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nResults saved: {output_file}")

        return stats

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point."""

    if len(sys.argv) < 3:
        print("Usage: python benchmark_isolated.py <model_name> <model_path> [enable_qdq_cleanup]")
        print("\nExamples:")
        print("  python benchmark_isolated.py FP32 model/florence.onnx")
        print("  python benchmark_isolated.py QUInt16 quantized_models/florence_quint16.onnx 1")
        sys.exit(1)

    model_name = sys.argv[1]
    model_path = sys.argv[2]
    enable_qdq_cleanup = (sys.argv[3].lower() in ['1', 'true', 'yes']) if len(sys.argv) > 3 else True

    print(f"{'='*80}")
    print(f"THERMALLY-CONTROLLED ISOLATED BENCHMARK")
    print(f"{'='*80}")
    print(f"Process: Isolated (thermal effects minimized)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    result = benchmark_single_model(model_path, model_name,
                                   num_runs=30, warmup_runs=10,
                                   enable_qdq_cleanup=enable_qdq_cleanup)

    if result:
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'='*80}")
    else:
        print(f"\n[ERROR] Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
