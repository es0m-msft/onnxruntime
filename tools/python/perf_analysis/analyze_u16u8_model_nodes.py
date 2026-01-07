#!/usr/bin/env python3
"""
Analyze vanilla QUInt16×QUInt8 QDQ model to identify optimization opportunities.

This script performs comprehensive analysis of the quantized model:
1. Maps all nodes with their input/output types and counts occurrences
2. Identifies operations that would benefit most from optimization
3. Detects Q/DQ and DQ/Q roundtrip patterns that could be elided
4. Analyzes accuracy-limiting factors

Mission: Enable vanilla u16u8 QDQ model to meet or outperform FP32 performance on CPU.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import onnx
from onnx import numpy_helper
import numpy as np


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def get_tensor_type_name(tensor_type: int) -> str:
    """Convert ONNX tensor type to readable name."""
    type_map = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        10: "float16",
        11: "float64",
    }
    return type_map.get(tensor_type, f"unknown({tensor_type})")


def analyze_tensor_types(model: onnx.ModelProto) -> Dict[str, str]:
    """Build a map of tensor names to their data types."""
    tensor_types = {}

    # Get types from graph inputs
    for inp in model.graph.input:
        if inp.type.HasField('tensor_type'):
            tensor_types[inp.name] = get_tensor_type_name(inp.type.tensor_type.elem_type)

    # Get types from graph outputs
    for out in model.graph.output:
        if out.type.HasField('tensor_type'):
            tensor_types[out.name] = get_tensor_type_name(out.type.tensor_type.elem_type)

    # Get types from value_info
    for vi in model.graph.value_info:
        if vi.type.HasField('tensor_type'):
            tensor_types[vi.name] = get_tensor_type_name(vi.type.tensor_type.elem_type)

    # Get types from initializers
    for init in model.graph.initializer:
        tensor_types[init.name] = get_tensor_type_name(init.data_type)

    return tensor_types


def get_node_signature(node: onnx.NodeProto, tensor_types: Dict[str, str]) -> str:
    """
    Create a signature for a node based on its type and input/output types.
    Format: "OpType(input_types -> output_types)"
    """
    input_types = []
    for inp in node.input:
        if inp:  # Skip empty inputs
            input_types.append(tensor_types.get(inp, "unknown"))

    output_types = []
    for out in node.output:
        if out:  # Skip empty outputs
            output_types.append(tensor_types.get(out, "unknown"))

    input_str = ",".join(input_types) if input_types else "none"
    output_str = ",".join(output_types) if output_types else "none"

    return f"{node.op_type}({input_str} -> {output_str})"


def analyze_node_configurations(model: onnx.ModelProto) -> Tuple[Counter, Dict]:
    """
    Analyze all nodes in the model and count their type configurations.
    Returns: (node_signature_counts, node_details)
    """
    print_section("ANALYZING NODE CONFIGURATIONS")

    tensor_types = analyze_tensor_types(model)
    node_signatures = Counter()
    node_details = defaultdict(list)

    for node in model.graph.node:
        signature = get_node_signature(node, tensor_types)
        node_signatures[signature] += 1
        node_details[signature].append({
            'name': node.name,
            'inputs': list(node.input),
            'outputs': list(node.output),
        })

    return node_signatures, node_details


def identify_optimization_opportunities(node_signatures: Counter) -> List[Tuple[str, int, str]]:
    """
    Identify nodes that would benefit most from optimization.
    Returns list of (signature, count, priority_reason) tuples.
    """
    print_section("OPTIMIZATION OPPORTUNITIES")

    opportunities = []

    # High-priority patterns (uint16 operations without optimized kernels)
    uint16_patterns = [
        ("ReduceMean", "uint16", "No optimized uint16 kernel - 20% of runtime"),
        ("Add", "uint16", "May use reference implementation"),
        ("Mul", "uint16", "May use reference implementation"),
        ("Div", "uint16", "May use reference implementation"),
        ("Sqrt", "uint16", "May use reference implementation"),
        ("Pow", "uint16", "May use reference implementation"),
        ("Softmax", "uint16", "May use reference implementation"),
        ("LayerNormalization", "uint16", "May use reference implementation"),
    ]

    # Q/DQ operations (overhead)
    qdq_patterns = [
        ("QuantizeLinear", "High overhead - 11.6% of runtime"),
        ("DequantizeLinear", "High overhead - 6.3% of runtime"),
    ]

    for signature, count in node_signatures.most_common():
        op_type = signature.split('(')[0]

        # Check for uint16 operations
        if 'uint16' in signature:
            for pattern_op, pattern_type, reason in uint16_patterns:
                if op_type == pattern_op:
                    opportunities.append((signature, count, f"HIGH: {reason}"))
                    break

        # Check for Q/DQ operations
        for pattern_op, reason in qdq_patterns:
            if op_type == pattern_op:
                opportunities.append((signature, count, f"MEDIUM: {reason}"))
                break

    return opportunities


def detect_qdq_roundtrips(model: onnx.ModelProto, tensor_types: Dict[str, str]) -> List[Dict]:
    """
    Detect Q/DQ and DQ/Q roundtrip patterns that could be elided.
    These represent unnecessary quantization/dequantization cycles.
    """
    print_section("Q/DQ ROUNDTRIP DETECTION")

    # Build a map of tensor producers
    tensor_producer = {}
    for node in model.graph.node:
        for output in node.output:
            tensor_producer[output] = node

    roundtrips = []

    for node in model.graph.node:
        # Pattern 1: DQ -> Q (dequantize then immediately quantize)
        if node.op_type == "QuantizeLinear":
            for inp in node.input:
                if inp in tensor_producer:
                    producer = tensor_producer[inp]
                    if producer.op_type == "DequantizeLinear":
                        roundtrips.append({
                            'pattern': 'DQ -> Q',
                            'producer': producer.name,
                            'consumer': node.name,
                            'tensor': inp,
                            'opportunity': 'Could use direct uint->uint cast or type conversion',
                            'potential_saving': 'Eliminate both Q and DQ operations'
                        })

        # Pattern 2: Q -> DQ (quantize then immediately dequantize)
        elif node.op_type == "DequantizeLinear":
            for inp in node.input:
                if inp in tensor_producer:
                    producer = tensor_producer[inp]
                    if producer.op_type == "QuantizeLinear":
                        roundtrips.append({
                            'pattern': 'Q -> DQ',
                            'producer': producer.name,
                            'consumer': node.name,
                            'tensor': inp,
                            'opportunity': 'Could use direct type conversion or bypass',
                            'potential_saving': 'Eliminate both Q and DQ operations'
                        })

        # Pattern 3: DQ -> Op -> Q (common pattern for unsupported quantized ops)
        elif node.op_type == "QuantizeLinear":
            for inp in node.input:
                if inp in tensor_producer:
                    producer = tensor_producer[inp]
                    # Check if producer has DQ inputs
                    has_dq_input = False
                    for producer_inp in producer.input:
                        if producer_inp in tensor_producer:
                            grand_producer = tensor_producer[producer_inp]
                            if grand_producer.op_type == "DequantizeLinear":
                                has_dq_input = True
                                break

                    if has_dq_input:
                        roundtrips.append({
                            'pattern': f'DQ -> {producer.op_type} -> Q',
                            'middle_op': producer.op_type,
                            'producer': producer.name,
                            'consumer': node.name,
                            'opportunity': f'Implement quantized {producer.op_type} kernel',
                            'potential_saving': f'Eliminate 2 DQ + 1 Q operation per {producer.op_type}'
                        })

    return roundtrips


def analyze_accuracy_factors(model: onnx.ModelProto, tensor_types: Dict[str, str]) -> Dict:
    """
    Analyze factors that may be limiting accuracy in the quantized model.
    """
    print_section("ACCURACY ANALYSIS")

    factors = {
        'uint16_range_usage': [],
        'quantization_parameters': [],
        'mixed_precision_opportunities': [],
        'potential_issues': [],
    }

    # Analyze quantization parameters
    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            # Check scale values
            if len(node.input) >= 2:
                scale_name = node.input[1]
                factors['quantization_parameters'].append({
                    'node': node.name,
                    'type': node.op_type,
                    'scale_tensor': scale_name,
                })

        # Check for operations that might benefit from higher precision
        if node.op_type in ["MatMul", "Gemm", "Conv"]:
            input_types = [tensor_types.get(inp, "unknown") for inp in node.input if inp]
            if all(t in ["uint8", "int8"] for t in input_types):
                factors['mixed_precision_opportunities'].append({
                    'node': node.name,
                    'op_type': node.op_type,
                    'current': 'uint8',
                    'recommendation': 'Consider uint16 for activations',
                })

    # Check for potential accuracy issues
    node_counts = Counter(node.op_type for node in model.graph.node)
    if node_counts.get('QuantizeLinear', 0) + node_counts.get('DequantizeLinear', 0) > 100:
        factors['potential_issues'].append({
            'issue': 'High Q/DQ overhead',
            'count': node_counts.get('QuantizeLinear', 0) + node_counts.get('DequantizeLinear', 0),
            'impact': 'Accumulation of quantization errors',
            'recommendation': 'Reduce Q/DQ pairs through operator fusion',
        })

    return factors


def generate_optimization_plan(
    opportunities: List[Tuple[str, int, str]],
    roundtrips: List[Dict],
    accuracy_factors: Dict
) -> List[Dict]:
    """Generate a prioritized optimization plan."""
    print_section("OPTIMIZATION PLAN")

    plan = []

    # Priority 1: High-impact performance optimizations
    plan.append({
        'priority': 1,
        'category': 'Performance',
        'title': 'Implement NEON-optimized ReduceMean for uint16',
        'rationale': 'Accounts for 20% of runtime (8.6s), extremely slow',
        'estimated_impact': '~30ms improvement (15% of current runtime)',
        'complexity': 'Medium',
        'files': ['onnxruntime/core/providers/cpu/reduction/reduction_ops.cc'],
    })

    plan.append({
        'priority': 2,
        'category': 'Performance',
        'title': 'Optimize column sum computation in QLinearMatMul dispatcher',
        'rationale': 'Accounts for ~100ms (50% of runtime), strided memory access',
        'estimated_impact': '~50-70ms improvement through matrix transpose or kernel integration',
        'complexity': 'High',
        'files': ['onnxruntime/core/mlas/lib/qgemm_u16u8.cpp'],
    })

    plan.append({
        'priority': 3,
        'category': 'Performance',
        'title': 'Implement NEON-optimized Q/DQ kernels for uint16',
        'rationale': 'Combined 18% of runtime, using reference implementations',
        'estimated_impact': '~8-15ms improvement',
        'complexity': 'Low-Medium',
        'files': ['onnxruntime/core/providers/cpu/math/quantize_linear.cc'],
    })

    # Priority 4: Operator fusion to reduce Q/DQ roundtrips
    if roundtrips:
        unique_patterns = set(rt['pattern'] for rt in roundtrips)
        plan.append({
            'priority': 4,
            'category': 'Graph Optimization',
            'title': 'Implement graph transformations to elide Q/DQ roundtrips',
            'rationale': f'Found {len(roundtrips)} roundtrip patterns: {", ".join(unique_patterns)}',
            'estimated_impact': f'Eliminate {len(roundtrips)} redundant Q/DQ pairs',
            'complexity': 'Medium',
            'files': ['onnxruntime/core/optimizer/qdq_transformer/*'],
        })

    # Priority 5: Accuracy improvements
    plan.append({
        'priority': 5,
        'category': 'Accuracy',
        'title': 'Analyze and optimize quantization parameters',
        'rationale': 'Current L2 error 5.06e-02 is 11.6x worse than QUInt8',
        'estimated_impact': 'Reduce quantization error through better calibration',
        'complexity': 'Medium',
        'files': ['quantization scripts'],
    })

    plan.append({
        'priority': 6,
        'category': 'Accuracy',
        'title': 'Implement per-channel quantization for weights',
        'rationale': 'Can significantly improve accuracy with minimal performance cost',
        'estimated_impact': '2-5x improvement in accuracy metrics',
        'complexity': 'Medium',
        'files': ['quantization configuration'],
    })

    return plan


def main():
    """Main analysis workflow."""
    print_section("VANILLA UINT16×UINT8 QDQ MODEL ANALYSIS")
    print("Mission: Enable vanilla u16u8 QDQ model to meet or outperform FP32 performance")

    # Model path
    model_dir = Path(r"C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer")
    model_path = model_dir / "quantized_models" / "florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx"

    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        print("Please run quantization first:")
        print("  python quantize_and_evaluate_with_profiling.py --configs quint16_full")
        return 1

    print(f"\nLoading model: {model_path.name}")
    model = onnx.load(str(model_path))

    # Step 1: Analyze node configurations
    node_signatures, node_details = analyze_node_configurations(model)

    print(f"\nTotal nodes: {len(model.graph.node)}")
    print(f"Unique node configurations: {len(node_signatures)}")
    print("\nTop 20 node configurations by count:")
    print(f"{'Configuration':<80} {'Count':>8}")
    print("-" * 90)

    for signature, count in node_signatures.most_common(20):
        print(f"{signature:<80} {count:>8}")

    # Step 2: Identify optimization opportunities
    tensor_types = analyze_tensor_types(model)
    opportunities = identify_optimization_opportunities(node_signatures)

    if opportunities:
        print(f"\nFound {len(opportunities)} optimization opportunities:")
        print(f"{'Configuration':<70} {'Count':>6} {'Priority & Reason':<50}")
        print("-" * 128)

        for signature, count, reason in opportunities[:15]:
            print(f"{signature:<70} {count:>6} {reason:<50}")

    # Step 3: Detect Q/DQ roundtrips
    roundtrips = detect_qdq_roundtrips(model, tensor_types)

    if roundtrips:
        print(f"\nFound {len(roundtrips)} Q/DQ roundtrip patterns:")

        # Group by pattern type
        pattern_groups = defaultdict(list)
        for rt in roundtrips:
            pattern_groups[rt['pattern']].append(rt)

        for pattern, instances in pattern_groups.items():
            print(f"\n  Pattern: {pattern}")
            print(f"    Count: {len(instances)}")
            print(f"    Opportunity: {instances[0]['opportunity']}")
            print(f"    Potential saving: {instances[0]['potential_saving']}")

            # Show a few examples
            for i, instance in enumerate(instances[:3]):
                print(f"      Example {i+1}: {instance.get('producer', 'N/A')} -> {instance.get('consumer', 'N/A')}")
    else:
        print("\nNo Q/DQ roundtrips detected (model is well optimized)")

    # Step 4: Analyze accuracy factors
    accuracy_factors = analyze_accuracy_factors(model, tensor_types)

    if accuracy_factors['potential_issues']:
        print(f"\nIdentified {len(accuracy_factors['potential_issues'])} potential accuracy issues:")
        for issue in accuracy_factors['potential_issues']:
            print(f"  - {issue['issue']}: {issue['impact']}")
            print(f"    Recommendation: {issue['recommendation']}")

    if accuracy_factors['mixed_precision_opportunities']:
        print(f"\nFound {len(accuracy_factors['mixed_precision_opportunities'])} mixed-precision opportunities")

    # Step 5: Generate optimization plan
    plan = generate_optimization_plan(opportunities, roundtrips, accuracy_factors)

    print(f"\nGenerated {len(plan)} optimization targets:")
    for item in plan:
        print(f"\n  Priority {item['priority']}: {item['title']}")
        print(f"    Category: {item['category']}")
        print(f"    Rationale: {item['rationale']}")
        print(f"    Estimated Impact: {item['estimated_impact']}")
        print(f"    Complexity: {item['complexity']}")

    # Save detailed results
    output_file = model_dir / "quantized_models" / "u16u8_model_analysis.txt"
    print(f"\n\nSaving detailed analysis to: {output_file}")

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VANILLA UINT16×UINT8 QDQ MODEL ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {model_path.name}\n")
        f.write(f"Total nodes: {len(model.graph.node)}\n")
        f.write(f"Unique configurations: {len(node_signatures)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ALL NODE CONFIGURATIONS (sorted by count)\n")
        f.write("=" * 80 + "\n\n")

        for signature, count in node_signatures.most_common():
            f.write(f"{signature}: {count}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION OPPORTUNITIES\n")
        f.write("=" * 80 + "\n\n")

        for signature, count, reason in opportunities:
            f.write(f"{signature}\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  {reason}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Q/DQ ROUNDTRIP PATTERNS\n")
        f.write("=" * 80 + "\n\n")

        for pattern, instances in pattern_groups.items():
            f.write(f"Pattern: {pattern}\n")
            f.write(f"  Count: {len(instances)}\n")
            f.write(f"  Opportunity: {instances[0]['opportunity']}\n")
            f.write(f"  Potential saving: {instances[0]['potential_saving']}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION PLAN\n")
        f.write("=" * 80 + "\n\n")

        for item in plan:
            f.write(f"Priority {item['priority']}: {item['title']}\n")
            f.write(f"  Category: {item['category']}\n")
            f.write(f"  Rationale: {item['rationale']}\n")
            f.write(f"  Estimated Impact: {item['estimated_impact']}\n")
            f.write(f"  Complexity: {item['complexity']}\n")
            f.write(f"  Files: {', '.join(item['files'])}\n\n")

    print_section("ANALYSIS COMPLETE")
    print(f"\nResults saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Review optimization plan priorities")
    print("  2. Implement high-priority optimizations (ReduceMean, column sums)")
    print("  3. Implement graph transformations to elide Q/DQ roundtrips")
    print("  4. Improve accuracy through better quantization parameters")
    print("\nGoal: Meet or outperform FP32 performance (54.9ms) with current 198ms")

    return 0


if __name__ == '__main__':
    sys.exit(main())
