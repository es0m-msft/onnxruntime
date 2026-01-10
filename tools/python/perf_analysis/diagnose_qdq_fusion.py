#!/usr/bin/env python3
"""
Diagnose QDQ fusion in quantized models.

This script analyzes ONNX models to identify QDQ fusion patterns and
determine whether DQ->MatMul->Q patterns are being fused into QLinearMatMul.

Usage:
    python diagnose_qdq_fusion.py model.onnx
"""

import onnxruntime as ort
import onnx
import sys
from collections import Counter

def analyze_fusion(model_path):
    """Analyze QDQ fusion patterns in the model."""
    print(f"\n=== Analyzing: {model_path} ===\n")

    # Load model
    model = onnx.load(model_path)

    # Count node types
    node_types = Counter(node.op_type for node in model.graph.node)

    print("Node type counts:")
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")

    # Check for DQ->MatMul->Q patterns (unfused)
    unfused_patterns = 0
    for i, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            # Check if preceded by DQ and followed by Q
            has_dq_input = any(n.op_type == "DequantizeLinear"
                              for n in model.graph.node[:i]
                              if any(out == inp for out in n.output for inp in node.input))
            has_q_output = any(n.op_type == "QuantizeLinear"
                              for n in model.graph.node[i+1:]
                              if any(inp == out for inp in n.input for out in node.output))
            if has_dq_input and has_q_output:
                unfused_patterns += 1

    print(f"\nUnfused DQ->MatMul->Q patterns: {unfused_patterns}")
    print(f"Fused QLinearMatMul: {node_types.get('QLinearMatMul', 0)}")

    # Test with ORT optimization
    print("\n=== Testing with ORT optimization ===")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = model_path.replace('.onnx', '.optimized.onnx')

    sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    print(f"Optimized graph saved to: {sess_options.optimized_model_filepath}")

    # Analyze optimized graph
    opt_model = onnx.load(sess_options.optimized_model_filepath)
    opt_node_types = Counter(node.op_type for node in opt_model.graph.node)

    print("\nOptimized node type counts:")
    for op_type, count in sorted(opt_node_types.items()):
        if count != node_types.get(op_type, 0):
            change = count - node_types.get(op_type, 0)
            print(f"  {op_type}: {count} ({change:+d})")

    print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_qdq_fusion.py model.onnx")
        sys.exit(1)

    analyze_fusion(sys.argv[1])
