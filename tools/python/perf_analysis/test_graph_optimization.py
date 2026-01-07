#!/usr/bin/env python3
"""
Test script to verify whether QDQ fusion is happening at runtime.

This script loads the vanilla QUInt16 model and checks:
1. Whether QLinearMatMul nodes are created after graph optimization
2. What graph optimization level is being used
3. Why fusion might not be happening

Expected: 109 MatMul ops should be fused to QLinearMatMul during graph optimization
Actual: Need to verify by inspecting the optimized graph
"""

import os
import sys
from pathlib import Path
import onnxruntime as ort
import numpy as np


def test_optimization_level(model_path: str, opt_level, level_name: str):
    """Test a specific optimization level and check the result."""
    print(f"\n{'='*80}")
    print(f"Testing optimization level: {level_name}")
    print(f"{'='*80}")

    try:
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = opt_level
        sess_options.optimized_model_filepath = f"optimized_{level_name}.onnx"

        # Enable verbose logging
        sess_options.log_severity_level = 1  # Warning level
        # sess_options.log_verbosity_level = 1  # Not available in all versions

        print(f"Creating session with {level_name}...")
        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        print(f"Optimized graph saved to: optimized_{level_name}.onnx")

        # Load the optimized graph to analyze it
        import onnx
        optimized_model = onnx.load(f"optimized_{level_name}.onnx")

        # Count operator types
        from collections import defaultdict
        op_counts = defaultdict(int)
        for node in optimized_model.graph.node:
            op_counts[node.op_type] += 1

        print(f"\nOperator counts in optimized graph:")
        print(f"  MatMul: {op_counts['MatMul']}")
        print(f"  QLinearMatMul: {op_counts['QLinearMatMul']}")
        print(f"  QuantizeLinear: {op_counts['QuantizeLinear']}")
        print(f"  DequantizeLinear: {op_counts['DequantizeLinear']}")
        print(f"  Total nodes: {len(optimized_model.graph.node)}")

        if op_counts['MatMul'] > 0:
            print(f"\n[!] WARNING: {op_counts['MatMul']} MatMul ops still present (not fused)")
        else:
            print(f"\n[OK] SUCCESS: All MatMul ops were fused!")

        if op_counts['QLinearMatMul'] > 0:
            print(f"[OK] {op_counts['QLinearMatMul']} QLinearMatMul ops created")
        else:
            print(f"[X] No QLinearMatMul ops found - fusion did NOT happen!")

        return {
            'level': level_name,
            'matmul_count': op_counts['MatMul'],
            'qlinearmatmul_count': op_counts['QLinearMatMul'],
            'success': op_counts['MatMul'] == 0 and op_counts['QLinearMatMul'] > 0
        }

    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'level': level_name,
            'error': str(e),
            'success': False
        }


def main():
    """Main test workflow."""
    print("="*80)
    print("GRAPH OPTIMIZATION FUSION TEST")
    print("="*80)
    print("Mission: Verify whether DQ->MatMul->Q patterns are being fused to QLinearMatMul")

    model_path = Path(r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer") / \
                 "quantized_models" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx"

    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        return 1

    print(f"\nModel: {model_path.name}")
    print(f"Expected: 109 MatMul ops in unoptimized model")
    print(f"Expected after fusion: 0 MatMul ops, >100 QLinearMatMul ops")

    # Test different optimization levels
    results = []

    print("\n" + "="*80)
    print("Testing different optimization levels")
    print("="*80)

    # Disable all optimizations (baseline)
    results.append(test_optimization_level(
        str(model_path),
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "DISABLE_ALL"
    ))

    # Basic optimizations
    results.append(test_optimization_level(
        str(model_path),
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ENABLE_BASIC"
    ))

    # Extended optimizations (QDQ fusion happens here)
    results.append(test_optimization_level(
        str(model_path),
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ENABLE_EXTENDED"
    ))

    # All optimizations
    results.append(test_optimization_level(
        str(model_path),
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "ENABLE_ALL"
    ))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Level':<20} {'MatMul':<10} {'QLinearMatMul':<15} {'Status'}")
    print("-"*80)
    for result in results:
        if 'error' in result:
            print(f"{result['level']:<20} {'ERROR':<10} {'ERROR':<15} [X]")
        else:
            status = "[OK]" if result['success'] else "[X]"
            print(f"{result['level']:<20} {result['matmul_count']:<10} {result['qlinearmatmul_count']:<15} {status}")

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)

    # Check if ANY optimization level achieved fusion
    any_success = any(r.get('success', False) for r in results)

    if any_success:
        print("\n[OK] Fusion IS working at some optimization level!")
        print("   This means the QDQ transformer is correctly fusing DQ->MatMul->Q patterns.")
        print("   The performance issue must be in the QLinearMatMul kernel implementation.")
    else:
        print("\n[X] Fusion is NOT working at ANY optimization level!")
        print("   Root causes to investigate:")
        print("   1. QDQ transformer rules don't recognize uint16 patterns")
        print("   2. Schema validation is blocking fusion (likely!)")
        print("   3. Missing type constraints in fusion selectors")
        print("")
        print("   Next steps:")
        print("   - Check QDQ transformer selectors for uint16 support")
        print("   - Review qdq_selectors.cc and qdq_actions.cc")
        print("   - Enable verbose logging to see why fusion is rejected")

    return 0


if __name__ == '__main__':
    sys.exit(main())
