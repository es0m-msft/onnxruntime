#!/usr/bin/env python3
"""
Diagnostic script to investigate why 109 MatMul operations in the vanilla QUInt16×QUInt8
QDQ model are not being fused to QLinearMatMul during graph optimization.

This script analyzes:
1. Input patterns for each unfused MatMul (are they from DQ nodes?)
2. Output patterns (do they go to Q nodes?)
3. Reasons why fusion might have failed
4. Missing patterns or constraints in QDQ transformer

Mission: Identify why these MatMuls are falling back to FP32 execution and fix it.
Expected impact: 30-50ms performance improvement.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import onnx
from collections import defaultdict


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def get_node_by_output(model: onnx.ModelProto, output_name: str) -> onnx.NodeProto:
    """Find the node that produces a given output tensor."""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def get_nodes_by_input(model: onnx.ModelProto, input_name: str) -> List[onnx.NodeProto]:
    """Find all nodes that consume a given input tensor."""
    consumers = []
    for node in model.graph.node:
        if input_name in node.input:
            consumers.append(node)
    return consumers


def get_tensor_type(model: onnx.ModelProto, tensor_name: str) -> str:
    """Get the data type of a tensor."""
    # Check initializers
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return onnx.TensorProto.DataType.Name(init.data_type)

    # Check value_info
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)

    # Check inputs
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)

    # Check outputs
    for out in model.graph.output:
        if out.name == tensor_name:
            return onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)

    return "unknown"


def analyze_matmul_pattern(model: onnx.ModelProto, matmul_node: onnx.NodeProto) -> Dict:
    """
    Analyze the pattern around a MatMul node to understand why it wasn't fused.

    Returns a dictionary with:
    - has_dq_inputs: Whether inputs come from DequantizeLinear nodes
    - has_q_output: Whether output goes to QuantizeLinear node
    - input_patterns: Description of input patterns
    - output_patterns: Description of output patterns
    - fusion_blocker: Reason why fusion might have failed
    """
    result = {
        'name': matmul_node.name,
        'has_dq_inputs': [False, False],
        'has_q_output': False,
        'input_patterns': [],
        'output_patterns': [],
        'fusion_blocker': None
    }

    # Analyze inputs
    for idx, input_name in enumerate(matmul_node.input[:2]):  # MatMul has 2 inputs
        producer = get_node_by_output(model, input_name)

        if producer is None:
            result['input_patterns'].append(f"Input {idx}: <graph_input or initializer>")
            result['fusion_blocker'] = result['fusion_blocker'] or "Missing producer node"
        elif producer.op_type == "DequantizeLinear":
            result['has_dq_inputs'][idx] = True
            # Check what feeds the DQ node
            dq_input = producer.input[0] if producer.input else "<unknown>"
            dq_input_type = get_tensor_type(model, dq_input)
            result['input_patterns'].append(
                f"Input {idx}: DQ({dq_input_type}) <- {producer.name}"
            )
        else:
            result['input_patterns'].append(
                f"Input {idx}: {producer.op_type} <- {producer.name}"
            )
            result['fusion_blocker'] = result['fusion_blocker'] or f"Input {idx} not from DQ (from {producer.op_type})"

    # Analyze outputs
    if matmul_node.output:
        output_name = matmul_node.output[0]
        consumers = get_nodes_by_input(model, output_name)

        if not consumers:
            result['output_patterns'].append("Output: <graph_output>")
            result['fusion_blocker'] = result['fusion_blocker'] or "Output is graph output"
        else:
            for consumer in consumers:
                if consumer.op_type == "QuantizeLinear":
                    result['has_q_output'] = True
                    q_output_type = get_tensor_type(model, consumer.output[0]) if consumer.output else "unknown"
                    result['output_patterns'].append(
                        f"Output: -> Q({q_output_type}) -> {consumer.name}"
                    )
                else:
                    result['output_patterns'].append(
                        f"Output: -> {consumer.op_type} -> {consumer.name}"
                    )
                    if not result['fusion_blocker']:
                        result['fusion_blocker'] = f"Output consumed by {consumer.op_type} (not Q)"

    # Determine fusion potential
    if result['has_dq_inputs'][0] and result['has_dq_inputs'][1] and result['has_q_output']:
        result['fusion_potential'] = "HIGH - Full DQ->MatMul->Q pattern"
        if not result['fusion_blocker']:
            result['fusion_blocker'] = "Unknown - should have been fused!"
    elif result['has_dq_inputs'][0] or result['has_dq_inputs'][1]:
        result['fusion_potential'] = "MEDIUM - Partial DQ inputs"
    else:
        result['fusion_potential'] = "LOW - No DQ inputs"

    return result


def categorize_fusion_blockers(analyses: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize MatMul nodes by their fusion blocker reasons."""
    categories = defaultdict(list)

    for analysis in analyses:
        blocker = analysis.get('fusion_blocker', 'Unknown')
        categories[blocker].append(analysis)

    return dict(categories)


def main():
    """Main analysis workflow."""
    print_section("UNFUSED MATMUL INVESTIGATION")
    print("Mission: Identify why 109 MatMul operations are not fused to QLinearMatMul")
    print("Expected impact: 30-50ms performance improvement")

    # Configuration
    model_path = Path(r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer") / \
                 "quantized_models" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx"

    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        return 1

    print(f"\nLoading model: {model_path.name}")
    model = onnx.load(str(model_path))

    # Find all MatMul nodes
    print_section("FINDING MATMUL NODES")
    matmul_nodes = [node for node in model.graph.node if node.op_type == "MatMul"]
    print(f"Found {len(matmul_nodes)} MatMul nodes in the model")

    # Analyze each MatMul
    print_section("ANALYZING MATMUL PATTERNS")
    analyses = []
    for i, matmul in enumerate(matmul_nodes, 1):
        print(f"\nAnalyzing MatMul {i}/{len(matmul_nodes)}: {matmul.name}")
        analysis = analyze_matmul_pattern(model, matmul)
        analyses.append(analysis)

        # Print summary
        print(f"  Fusion potential: {analysis['fusion_potential']}")
        print(f"  Input patterns:")
        for pattern in analysis['input_patterns']:
            print(f"    - {pattern}")
        print(f"  Output patterns:")
        for pattern in analysis['output_patterns']:
            print(f"    - {pattern}")
        if analysis['fusion_blocker']:
            print(f"  Fusion blocker: {analysis['fusion_blocker']}")

    # Categorize by fusion blocker
    print_section("FUSION BLOCKER CATEGORIZATION")
    categories = categorize_fusion_blockers(analyses)

    print(f"\nFound {len(categories)} categories of fusion blockers:\n")

    for blocker, nodes in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"Category: {blocker}")
        print(f"  Count: {len(nodes)}")
        print(f"  Percentage: {len(nodes)/len(matmul_nodes)*100:.1f}%")

        # Show first 3 examples
        print(f"  Examples:")
        for node_analysis in nodes[:3]:
            print(f"    - {node_analysis['name']}")
            print(f"      Fusion potential: {node_analysis['fusion_potential']}")
        print()

    # Analyze fusion potential distribution
    print_section("FUSION POTENTIAL DISTRIBUTION")

    potential_counts = defaultdict(int)
    for analysis in analyses:
        potential_counts[analysis['fusion_potential']] += 1

    print("\nFusion potential breakdown:")
    for potential, count in sorted(potential_counts.items(), key=lambda x: -x[1]):
        print(f"  {potential}: {count} nodes ({count/len(matmul_nodes)*100:.1f}%)")

    # Identify high-value optimization targets
    print_section("HIGH-VALUE OPTIMIZATION TARGETS")

    high_potential = [a for a in analyses if a['fusion_potential'] == "HIGH - Full DQ->MatMul->Q pattern"]
    medium_potential = [a for a in analyses if a['fusion_potential'] == "MEDIUM - Partial DQ inputs"]

    print(f"\nHIGH priority (Full DQ->MatMul->Q pattern): {len(high_potential)} nodes")
    if high_potential:
        print("  These SHOULD have been fused but weren't!")
        print("  Blockers:")
        high_blockers = categorize_fusion_blockers(high_potential)
        for blocker, nodes in sorted(high_blockers.items(), key=lambda x: -len(x[1])):
            print(f"    - {blocker}: {len(nodes)} nodes")

    print(f"\nMEDIUM priority (Partial DQ inputs): {len(medium_potential)} nodes")
    if medium_potential:
        print("  These might be fusible with additional patterns")
        medium_blockers = categorize_fusion_blockers(medium_potential)
        for blocker, nodes in sorted(medium_blockers.items(), key=lambda x: -len(x[1])):
            print(f"    - {blocker}: {len(nodes)} nodes")

    # Generate recommendations
    print_section("RECOMMENDATIONS")

    print("\n1. Investigate QDQ Transformer Fusion Rules:")
    print("   Files to check:")
    print("   - onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_actions.cc")
    print("   - onnxruntime/core/optimizer/qdq_transformer/selectors_actions/shared/utils.cc")
    print("   - onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.cc")

    print("\n2. Enable verbose graph optimization logging:")
    print("   Set session options:")
    print("     - log_severity_level = 0 (Verbose)")
    print("     - log_verbosity_level = 1")
    print("   This will show why fusion is failing for each pattern")

    print("\n3. Check graph optimization level:")
    print("   Ensure using ORT_ENABLE_EXTENDED or ORT_ENABLE_ALL")
    print("   QDQ fusion requires extended optimizations")

    if high_potential:
        print(f"\n4. URGENT: Fix high-priority fusion blockers")
        print(f"   {len(high_potential)} nodes have full DQ->MatMul->Q pattern but aren't fused")
        print(f"   Expected impact: 30-50ms improvement")

    # Save detailed report
    output_path = model_path.parent / "unfused_matmul_analysis.txt"
    print(f"\n\nSaving detailed report to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("UNFUSED MATMUL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Total MatMul nodes: {len(matmul_nodes)}\n\n")

        f.write("FUSION BLOCKER CATEGORIES\n")
        f.write("-" * 80 + "\n\n")
        for blocker, nodes in sorted(categories.items(), key=lambda x: -len(x[1])):
            f.write(f"{blocker}: {len(nodes)} nodes\n")
            for node_analysis in nodes:
                f.write(f"  - {node_analysis['name']}\n")
                f.write(f"    Potential: {node_analysis['fusion_potential']}\n")
                for pattern in node_analysis['input_patterns']:
                    f.write(f"    Input: {pattern}\n")
                for pattern in node_analysis['output_patterns']:
                    f.write(f"    Output: {pattern}\n")
            f.write("\n")

    print_section("ANALYSIS COMPLETE")
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review fusion blocker categories")
    print("  2. Enable verbose logging to see fusion decisions")
    print("  3. Modify QDQ transformer to handle missing patterns")
    print(f"  4. Target high-priority nodes first ({len(high_potential)} nodes)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
