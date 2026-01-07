#!/usr/bin/env python3
"""
Diagnostic script to investigate uint16×uint8 kernel usage and performance.

This script performs three key diagnostic steps:
1. Verify uint16×uint8 NEON kernel registration
2. Enable verbose kernel selection logging
3. Profile and compare kernel execution patterns

Mission: Identify why vanilla QUInt16×QUInt8 quantization is 4.87x slower than FP32
when it should be faster, without modifying the quantized model.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import onnx
import onnxruntime as ort
from collections import defaultdict


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def check_environment():
    """Check the execution environment and available features."""
    print_section("ENVIRONMENT CHECK")

    print(f"Python version: {sys.version}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"ONNX version: {onnx.__version__}")

    # Check available providers
    providers = ort.get_available_providers()
    print(f"\nAvailable Execution Providers: {providers}")

    # Check if we're on ARM64
    import platform
    print(f"\nPlatform: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # Check ONNX Runtime build info
    try:
        build_info = ort.get_build_info()
        print(f"\nONNX Runtime Build Info:")
        print(f"  Build: {build_info}")
    except Exception as e:
        print(f"\nCould not get build info: {e}")

    return providers


def analyze_model_structure(model_path: str) -> Dict:
    """Analyze the quantized model structure."""
    print_section(f"MODEL STRUCTURE ANALYSIS: {Path(model_path).name}")

    model = onnx.load(model_path)

    # Count operator types
    op_counts = defaultdict(int)
    for node in model.graph.node:
        op_counts[node.op_type] += 1

    print(f"\nOperator counts:")
    for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {op_type}: {count}")

    # Analyze QLinearMatMul nodes
    qlinear_matmul_nodes = [node for node in model.graph.node if node.op_type == 'QLinearMatMul']
    print(f"\nQLinearMatMul analysis:")
    print(f"  Total QLinearMatMul nodes: {len(qlinear_matmul_nodes)}")

    if qlinear_matmul_nodes:
        # Check input/output types of first QLinearMatMul
        sample_node = qlinear_matmul_nodes[0]
        print(f"  Sample node: {sample_node.name}")
        print(f"  Inputs: {list(sample_node.input)[:3]}...")  # First 3 inputs
        print(f"  Outputs: {list(sample_node.output)}")

        # Try to find the actual data types
        value_info_map = {vi.name: vi for vi in model.graph.value_info}
        value_info_map.update({vi.name: vi for vi in model.graph.input})
        value_info_map.update({vi.name: vi for vi in model.graph.output})

        if sample_node.input[0] in value_info_map:
            input_type = value_info_map[sample_node.input[0]].type.tensor_type.elem_type
            print(f"  First input type: {onnx.TensorProto.DataType.Name(input_type)}")

    # Analyze data types used
    print(f"\nTensor data types in use:")
    type_counts = defaultdict(int)
    for init in model.graph.initializer:
        type_name = onnx.TensorProto.DataType.Name(init.data_type)
        type_counts[type_name] += 1

    for dtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} tensors")

    return {
        'op_counts': dict(op_counts),
        'qlinear_matmul_count': len(qlinear_matmul_nodes),
        'type_counts': dict(type_counts)
    }


def test_session_configurations(model_path: str, input_data: Dict[str, np.ndarray]) -> Dict:
    """Test different session configurations to find optimal settings."""
    print_section("SESSION CONFIGURATION TESTING")

    configs = [
        {
            'name': 'Default',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'options': {}
        },
        {
            'name': 'Verbose Logging',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'options': {
                'session.log_severity_level': 0,  # Verbose
                'session.log_verbosity_level': 1
            }
        },
        {
            'name': 'Extended Optimizations',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            'options': {}
        },
        {
            'name': 'Basic Optimizations',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            'options': {}
        },
        {
            'name': 'No Optimizations',
            'graph_opt': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            'options': {}
        },
        {
            'name': 'Single Thread',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'options': {
                'intra_op_num_threads': 1,
                'inter_op_num_threads': 1
            }
        },
        {
            'name': 'Max Threads',
            'graph_opt': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'options': {
                'intra_op_num_threads': os.cpu_count(),
                'inter_op_num_threads': 1
            }
        },
    ]

    results = {}

    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"  Graph optimization level: {config['graph_opt']}")
        if config['options']:
            print(f"  Options: {config['options']}")

        try:
            # Create session with configuration
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = config['graph_opt']

            for key, value in config['options'].items():
                if key.startswith('session.'):
                    # Session-level option
                    if key == 'session.log_severity_level':
                        sess_options.log_severity_level = value
                    elif key == 'session.log_verbosity_level':
                        sess_options.log_verbosity_level = value
                else:
                    # Other options
                    if key == 'intra_op_num_threads':
                        sess_options.intra_op_num_threads = value
                    elif key == 'inter_op_num_threads':
                        sess_options.inter_op_num_threads = value

            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )

            # Warmup
            for _ in range(3):
                session.run(None, input_data)

            # Benchmark
            num_runs = 10
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run(None, input_data)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = np.mean(times)
            std_time = np.std(times)

            print(f"  Mean inference time: {mean_time:.2f}ms (std: {std_time:.2f}ms)")

            results[config['name']] = {
                'mean_ms': mean_time,
                'std_ms': std_time,
                'times': times
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[config['name']] = {'error': str(e)}

    return results


def profile_kernel_execution(model_path: str, input_data: Dict[str, np.ndarray],
                             output_prefix: str = "kernel_profile") -> str:
    """Profile kernel execution with detailed operator timing."""
    print_section("KERNEL EXECUTION PROFILING")

    # Create session with profiling enabled
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_profiling = True

    profile_file = f"{output_prefix}_uint16.json"
    sess_options.profile_file_prefix = output_prefix + "_uint16"

    print(f"Creating profiling session...")
    print(f"  Profile output: {profile_file}")

    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # Run inference
    print(f"\nRunning inference with profiling...")
    output = session.run(None, input_data)

    # End profiling
    profile_path = session.end_profiling()
    print(f"Profile saved to: {profile_path}")

    # Analyze profile
    print(f"\nAnalyzing profile data...")
    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        # Extract operator timings
        op_timings = defaultdict(list)
        kernel_info = []

        for event in profile_data:
            if isinstance(event, dict):
                name = event.get('name', '')
                cat = event.get('cat', '')
                dur = event.get('dur', 0)
                args = event.get('args', {})

                # Look for operator execution events
                if cat == 'Node' and 'op_name' in args:
                    op_name = args['op_name']
                    op_timings[op_name].append(dur / 1000)  # Convert to ms

                # Look for kernel-specific information
                if 'kernel' in name.lower() or 'kernel' in str(args).lower():
                    kernel_info.append({
                        'name': name,
                        'category': cat,
                        'duration_us': dur,
                        'args': args
                    })

        # Print operator timing summary
        print(f"\nOperator timing summary (top 15):")
        print(f"{'Operator':<30} {'Count':<8} {'Total(ms)':<12} {'Avg(ms)':<12}")
        print("-" * 70)

        op_summary = []
        for op_name, timings in op_timings.items():
            total_ms = sum(timings)
            count = len(timings)
            avg_ms = total_ms / count if count > 0 else 0
            op_summary.append((op_name, count, total_ms, avg_ms))

        # Sort by total time
        op_summary.sort(key=lambda x: -x[2])

        for op_name, count, total_ms, avg_ms in op_summary[:15]:
            print(f"{op_name:<30} {count:<8} {total_ms:<12.3f} {avg_ms:<12.6f}")

        # Print kernel-specific information if found
        if kernel_info:
            print(f"\n\nKernel execution details found: {len(kernel_info)} events")
            for i, kinfo in enumerate(kernel_info[:10]):  # Show first 10
                print(f"\n  Kernel {i+1}:")
                print(f"    Name: {kinfo['name']}")
                print(f"    Category: {kinfo['category']}")
                print(f"    Duration: {kinfo['duration_us']/1000:.3f}ms")
                if kinfo['args']:
                    print(f"    Args: {kinfo['args']}")
        else:
            print(f"\n\nNo specific kernel execution details found in profile.")
            print(f"This may indicate kernel selection info is not exposed in profiling output.")

    except Exception as e:
        print(f"Error analyzing profile: {e}")
        import traceback
        traceback.print_exc()

    return profile_path


def compare_with_fp32(fp32_model_path: str, quint16_model_path: str, input_data: Dict[str, np.ndarray]):
    """Compare FP32 and QUInt16 execution patterns."""
    print_section("FP32 vs QUInt16 COMPARISON")

    configs = [
        ('FP32', fp32_model_path),
        ('QUInt16', quint16_model_path)
    ]

    for name, model_path in configs:
        if not Path(model_path).exists():
            print(f"\nModel not found: {model_path}")
            continue

        print(f"\n{name} Model: {Path(model_path).name}")

        # Create session with profiling
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = f"compare_{name.lower()}"

        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Warmup
        for _ in range(3):
            session.run(None, input_data)

        # Benchmark
        num_runs = 10
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, input_data)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        print(f"  Mean inference time: {mean_time:.2f}ms")

        profile_path = session.end_profiling()
        print(f"  Profile saved: {profile_path}")


def main():
    """Main diagnostic workflow."""
    print_section("UINT16×UINT8 KERNEL DIAGNOSTIC SCRIPT")
    print("Mission: Identify why vanilla QUInt16×QUInt8 is 4.87x slower than FP32")

    # Configuration
    model_dir = Path(r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer")
    quint16_model = model_dir / "quantized_models" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx"
    fp32_model = model_dir / "model" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx"

    if not quint16_model.exists():
        print(f"\nERROR: QUInt16 model not found: {quint16_model}")
        print("Please run quantization first:")
        print("  python quantize_and_evaluate_with_profiling.py --configs quint16_full")
        return 1

    # Step 0: Check environment
    providers = check_environment()

    if 'CPUExecutionProvider' not in providers:
        print("\nWARNING: CPUExecutionProvider not available!")
        return 1

    # Step 1: Analyze model structure
    model_info = analyze_model_structure(str(quint16_model))

    # Generate sample input data
    print_section("GENERATING SAMPLE INPUT DATA")
    input_data = {
        '/lang_encoder/embeddings/Add_1_output_0': np.random.randn(1, 77, 768).astype(np.float32),
        'attention_mask': np.random.randint(0, 2, (1, 77)).astype(np.int64)
    }
    print("Sample input data generated")
    print(f"  Input 1 shape: {input_data['/lang_encoder/embeddings/Add_1_output_0'].shape}")
    print(f"  Input 2 shape: {input_data['attention_mask'].shape}")

    # Step 2: Test different session configurations
    config_results = test_session_configurations(str(quint16_model), input_data)

    # Step 3: Profile kernel execution
    profile_path = profile_kernel_execution(str(quint16_model), input_data,
                                           output_prefix="diagnostic_profile")

    # Step 4: Compare with FP32 if available
    if fp32_model.exists():
        compare_with_fp32(str(fp32_model), str(quint16_model), input_data)

    # Summary
    print_section("DIAGNOSTIC SUMMARY")

    print("\n1. Model Structure:")
    print(f"   - QLinearMatMul operators: {model_info['qlinear_matmul_count']}")
    print(f"   - Total operators: {sum(model_info['op_counts'].values())}")

    print("\n2. Session Configuration Results:")
    best_config = min(config_results.items(),
                     key=lambda x: x[1].get('mean_ms', float('inf')))
    print(f"   - Best configuration: {best_config[0]}")
    print(f"   - Best time: {best_config[1]['mean_ms']:.2f}ms")

    print("\n3. Profile Data:")
    print(f"   - Profile file: {profile_path}")
    print(f"   - Use Chrome tracing viewer to analyze: chrome://tracing")

    print("\n4. Next Steps:")
    print("   - Review profile data for kernel selection details")
    print("   - Check if QLinearMatMul shows uint16 kernel invocation")
    print("   - Compare operator timings with FP32 baseline")
    print("   - If kernels are slow, investigate MLAS implementation")

    print("\nDiagnostic complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
