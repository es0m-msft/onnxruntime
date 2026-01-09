#!/usr/bin/env python3
"""
Profile the actual runtime graph after ORT optimizations.

This script:
1. Captures the optimized graph that ORT actually executes
2. Profiles execution to get per-node timing
3. Analyzes node types and data types
4. Creates a priority list of nodes to optimize
5. Identifies Q/DQ patterns that can be eliminated
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort


def get_onnx_dtype_name(dtype_int: int) -> str:
    """Convert ONNX TensorProto dtype to readable name."""
    dtype_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.UINT16: "uint16",
        TensorProto.INT16: "int16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.BOOL: "bool",
        TensorProto.FLOAT16: "float16",
        TensorProto.DOUBLE: "float64",
    }
    return dtype_map.get(dtype_int, f"unknown({dtype_int})")


def create_optimized_session(model_path: str, output_dir: Path) -> Tuple[ort.InferenceSession, str]:
    """Create session and save optimized graph."""
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Enable Q/DQ cleanup to eliminate redundant roundtrips
    sess_options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")

    # Save optimized model
    model_name = Path(model_path).stem
    optimized_path = output_dir / f"{model_name}_optimized.onnx"
    sess_options.optimized_model_filepath = str(optimized_path)

    # Enable profiling
    sess_options.enable_profiling = True

    # Create session (this triggers optimization and saves the graph)
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=["CPUExecutionProvider"]
    )

    print(f"[OK] Optimized graph saved to: {optimized_path}")
    return session, str(optimized_path)


def analyze_graph_structure(model_path: str) -> Dict:
    """Analyze the structure of an ONNX graph."""
    model = onnx.load(model_path)
    graph = model.graph

    # Collect node information
    nodes = []
    tensor_types = {}  # tensor_name -> dtype

    # Build tensor type map from inputs, outputs, and value_info
    for input_tensor in graph.input:
        if input_tensor.type.tensor_type.elem_type:
            tensor_types[input_tensor.name] = input_tensor.type.tensor_type.elem_type

    for output_tensor in graph.output:
        if output_tensor.type.tensor_type.elem_type:
            tensor_types[output_tensor.name] = output_tensor.type.tensor_type.elem_type

    for value_info in graph.value_info:
        if value_info.type.tensor_type.elem_type:
            tensor_types[value_info.name] = value_info.type.tensor_type.elem_type

    # Analyze each node
    node_type_counts = Counter()
    for node in graph.node:
        node_type_counts[node.op_type] += 1

        # Get input/output types
        input_types = []
        for inp in node.input:
            if inp in tensor_types:
                input_types.append(get_onnx_dtype_name(tensor_types[inp]))
            else:
                input_types.append("unknown")

        output_types = []
        for out in node.output:
            if out in tensor_types:
                output_types.append(get_onnx_dtype_name(tensor_types[out]))
            else:
                output_types.append("unknown")

        nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "input_names": list(node.input),
            "output_names": list(node.output),
            "input_types": input_types,
            "output_types": output_types,
            "domain": node.domain,
        })

    return {
        "node_count": len(graph.node),
        "node_type_counts": dict(node_type_counts),
        "nodes": nodes,
        "tensor_types": {k: get_onnx_dtype_name(v) for k, v in tensor_types.items()}
    }


def parse_profile_json(profile_path: str) -> Dict:
    """Parse ORT profiling JSON to extract operator timing."""
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)

    # Group by operator type
    op_timings = defaultdict(list)

    for event in profile_data:
        if event.get("cat") == "Node" and "dur" in event:
            # Event name format: "node_name (op_type)"
            name = event.get("name", "")
            if "(" in name and ")" in name:
                op_type = name.split("(")[1].split(")")[0]
            else:
                op_type = name

            duration_us = event["dur"]
            op_timings[op_type].append(duration_us)

    # Aggregate statistics
    op_stats = {}
    for op_type, timings in op_timings.items():
        op_stats[op_type] = {
            "count": len(timings),
            "total_us": sum(timings),
            "mean_us": np.mean(timings),
            "median_us": np.median(timings),
            "std_us": np.std(timings),
            "min_us": min(timings),
            "max_us": max(timings),
        }

    # Sort by total time
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

    return {
        "operators": dict(sorted_ops),
        "total_time_us": sum(s["total_us"] for s in op_stats.values())
    }


def identify_qdq_patterns(nodes: List[Dict]) -> Dict:
    """Identify Q/DQ patterns in the graph."""
    patterns = {
        "q_dq_roundtrips": [],  # Q immediately followed by DQ
        "dq_op_q": [],  # DQ -> Op -> Q (could be fused)
        "isolated_q": [],  # Q not followed by DQ
        "isolated_dq": [],  # DQ not preceded by Q
        "uint16_uint8_conversions": [],  # Type conversions
    }

    # Build adjacency map
    output_to_node = {}
    for node in nodes:
        for out in node["output_names"]:
            output_to_node[out] = node

    input_to_nodes = defaultdict(list)
    for node in nodes:
        for inp in node["input_names"]:
            input_to_nodes[inp].append(node)

    # Identify patterns
    for i, node in enumerate(nodes):
        if node["op_type"] == "QuantizeLinear":
            # Check if followed by DequantizeLinear
            output = node["output_names"][0] if node["output_names"] else None
            if output:
                consumers = input_to_nodes.get(output, [])
                dq_consumers = [n for n in consumers if n["op_type"] == "DequantizeLinear"]
                if len(dq_consumers) == len(consumers) and len(dq_consumers) > 0:
                    # All consumers are DQ - this is a roundtrip
                    patterns["q_dq_roundtrips"].append({
                        "q_node": node["name"],
                        "dq_nodes": [n["name"] for n in dq_consumers],
                        "tensor": output
                    })

        elif node["op_type"] == "DequantizeLinear":
            # Check for DQ -> Op -> Q pattern
            output = node["output_names"][0] if node["output_names"] else None
            if output:
                consumers = input_to_nodes.get(output, [])
                for consumer in consumers:
                    # Check if consumer's output goes to QuantizeLinear
                    consumer_output = consumer["output_names"][0] if consumer["output_names"] else None
                    if consumer_output:
                        quant_consumers = input_to_nodes.get(consumer_output, [])
                        if any(n["op_type"] == "QuantizeLinear" for n in quant_consumers):
                            patterns["dq_op_q"].append({
                                "dq_node": node["name"],
                                "op_node": consumer["name"],
                                "op_type": consumer["op_type"],
                                "q_nodes": [n["name"] for n in quant_consumers if n["op_type"] == "QuantizeLinear"]
                            })

        # Check for uint16/uint8 conversions
        if "uint16" in node["input_types"] or "uint16" in node["output_types"]:
            if "uint8" in node["input_types"] or "uint8" in node["output_types"]:
                patterns["uint16_uint8_conversions"].append({
                    "node": node["name"],
                    "op_type": node["op_type"],
                    "input_types": node["input_types"],
                    "output_types": node["output_types"]
                })

    return patterns


def analyze_uint8_compatibility(nodes: List[Dict], tensor_types: Dict) -> Dict:
    """Analyze which operations could use uint8 instead of uint16."""
    compatible_ops = []
    incompatible_ops = []

    # Operations that typically need higher precision for accumulation
    high_precision_ops = {
        "MatMul", "Gemm", "QLinearMatMul", "Conv", "QLinearConv",
        "ReduceMean", "ReduceSum", "ReduceProd",  # Accumulation
        "LayerNormalization", "SkipLayerNormalization",  # Variance calculation
    }

    # Operations that can typically work with uint8
    uint8_safe_ops = {
        "Add", "Sub", "Mul", "Div",  # Element-wise
        "Relu", "Sigmoid", "Tanh", "Gelu", "Erf",  # Activations
        "MaxPool", "AveragePool",  # Pooling
        "Concat", "Split", "Reshape", "Transpose",  # Shape ops
        "Softmax",  # Can work with uint8 with proper scaling
    }

    for node in nodes:
        op_type = node["op_type"]

        # Check if using uint16
        using_uint16 = ("uint16" in node["input_types"] or
                       "uint16" in node["output_types"])

        if not using_uint16:
            continue

        # Determine if uint8 would be sufficient
        if op_type in high_precision_ops:
            incompatible_ops.append({
                "node": node["name"],
                "op_type": op_type,
                "reason": "Requires high precision for accumulation",
                "input_types": node["input_types"],
                "output_types": node["output_types"]
            })
        elif op_type in uint8_safe_ops:
            compatible_ops.append({
                "node": node["name"],
                "op_type": op_type,
                "reason": "Element-wise or shape operation",
                "input_types": node["input_types"],
                "output_types": node["output_types"]
            })
        else:
            # Unknown - need manual review
            compatible_ops.append({
                "node": node["name"],
                "op_type": op_type,
                "reason": "Unknown - requires manual review",
                "input_types": node["input_types"],
                "output_types": node["output_types"]
            })

    return {
        "uint8_compatible": compatible_ops,
        "needs_uint16": incompatible_ops,
        "potential_savings": len(compatible_ops)
    }


def create_optimization_priority_list(
    profile_stats: Dict,
    graph_analysis: Dict,
    qdq_patterns: Dict
) -> List[Dict]:
    """Create prioritized list of optimization targets."""
    priorities = []

    operators = profile_stats.get("operators", {})
    total_time = profile_stats.get("total_time_us", 1)

    for op_type, stats in operators.items():
        percentage = (stats["total_us"] / total_time) * 100

        # Find nodes of this type in graph
        matching_nodes = [n for n in graph_analysis["nodes"] if n["op_type"] == op_type]

        # Get type signatures
        type_signatures = set()
        for node in matching_nodes:
            inputs = tuple(node["input_types"])
            outputs = tuple(node["output_types"])
            type_signatures.add((inputs, outputs))

        priorities.append({
            "operator": op_type,
            "percentage": percentage,
            "total_time_ms": stats["total_us"] / 1000,
            "count": stats["count"],
            "avg_time_us": stats["mean_us"],
            "type_signatures": [{"inputs": list(sig[0]), "outputs": list(sig[1])}
                              for sig in type_signatures],
            "optimization_potential": categorize_optimization(op_type, matching_nodes, qdq_patterns)
        })

    # Sort by percentage of runtime
    priorities.sort(key=lambda x: x["percentage"], reverse=True)

    return priorities


def categorize_optimization(op_type: str, nodes: List[Dict], qdq_patterns: Dict) -> str:
    """Categorize the optimization approach for an operator."""
    # Check if involved in Q/DQ patterns
    involved_in_roundtrip = any(
        any(n["name"] in p["q_node"] or n["name"] in p.get("dq_nodes", [])
            for p in qdq_patterns["q_dq_roundtrips"])
        for n in nodes
    )

    involved_in_dq_op_q = any(
        any(n["name"] in p.get("op_node", "") for p in qdq_patterns["dq_op_q"])
        for n in nodes
    )

    if involved_in_roundtrip:
        return "ELIMINATE: Q/DQ roundtrip"
    elif involved_in_dq_op_q:
        return "FUSE: DQ->Op->Q pattern"
    elif op_type in ["QLinearMatMul", "QLinearConv"]:
        return "OPTIMIZE: Kernel performance"
    elif op_type in ["QuantizeLinear", "DequantizeLinear"]:
        return "OPTIMIZE: NEON kernel or eliminate"
    elif op_type in ["ReduceMean", "ReduceSum"]:
        # Check types
        using_uint16 = any("uint16" in n["input_types"] for n in nodes)
        if using_uint16:
            return "OPTIMIZE: Implement uint16 NEON kernel"
        else:
            return "OPTIMIZE: Check if using SIMD"
    else:
        return "REVIEW: Check if optimization needed"


def generate_report(
    original_analysis: Dict,
    optimized_analysis: Dict,
    profile_stats: Dict,
    qdq_patterns: Dict,
    uint8_compat: Dict,
    priorities: List[Dict],
    output_dir: Path
) -> str:
    """Generate comprehensive analysis report."""
    report = []

    report.append("=" * 80)
    report.append("RUNTIME GRAPH ANALYSIS REPORT")
    report.append("=" * 80)

    # Graph transformation summary
    report.append("\n" + "=" * 80)
    report.append("GRAPH OPTIMIZATION SUMMARY")
    report.append("=" * 80)
    report.append(f"Original node count:   {original_analysis['node_count']}")
    report.append(f"Optimized node count:  {optimized_analysis['node_count']}")
    report.append(f"Nodes eliminated:      {original_analysis['node_count'] - optimized_analysis['node_count']}")

    report.append("\nOriginal node types:")
    for op_type, count in sorted(original_analysis["node_type_counts"].items(),
                                  key=lambda x: x[1], reverse=True)[:20]:
        report.append(f"  {op_type:30s} {count:6d}")

    report.append("\nOptimized node types:")
    for op_type, count in sorted(optimized_analysis["node_type_counts"].items(),
                                  key=lambda x: x[1], reverse=True)[:20]:
        report.append(f"  {op_type:30s} {count:6d}")

    # Q/DQ Pattern Analysis
    report.append("\n" + "=" * 80)
    report.append("Q/DQ PATTERN ANALYSIS")
    report.append("=" * 80)
    report.append(f"Q->DQ roundtrips:           {len(qdq_patterns['q_dq_roundtrips'])}")
    report.append(f"DQ->Op->Q patterns:          {len(qdq_patterns['dq_op_q'])}")
    report.append(f"uint16/uint8 conversions:  {len(qdq_patterns['uint16_uint8_conversions'])}")

    if qdq_patterns["q_dq_roundtrips"]:
        report.append("\nTop Q->DQ roundtrips (should be eliminated):")
        for pattern in qdq_patterns["q_dq_roundtrips"][:10]:
            report.append(f"  {pattern['q_node']} -> {pattern['dq_nodes'][0]}")

    if qdq_patterns["dq_op_q"]:
        report.append("\nTop DQ->Op->Q patterns (fusion opportunities):")
        dq_op_q_by_type = defaultdict(int)
        for pattern in qdq_patterns["dq_op_q"]:
            dq_op_q_by_type[pattern["op_type"]] += 1
        for op_type, count in sorted(dq_op_q_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  {op_type:30s} {count:6d} instances")

    # uint8 Compatibility Analysis
    report.append("\n" + "=" * 80)
    report.append("UINT8 COMPATIBILITY ANALYSIS")
    report.append("=" * 80)
    report.append(f"Operations using uint16 that could use uint8:  {len(uint8_compat['uint8_compatible'])}")
    report.append(f"Operations that legitimately need uint16:      {len(uint8_compat['needs_uint16'])}")

    if uint8_compat["uint8_compatible"]:
        report.append("\nOperations that could be downgraded to uint8:")
        op_type_counts = Counter(op["op_type"] for op in uint8_compat["uint8_compatible"])
        for op_type, count in op_type_counts.most_common(10):
            report.append(f"  {op_type:30s} {count:6d} instances")

    # Optimization Priority List
    report.append("\n" + "=" * 80)
    report.append("OPTIMIZATION PRIORITY LIST")
    report.append("=" * 80)
    report.append(f"{'Rank':<6} {'Operator':<25} {'% Time':<8} {'Count':<8} {'Avg (μs)':<12} {'Strategy'}")
    report.append("-" * 100)

    for i, priority in enumerate(priorities[:20], 1):
        report.append(f"{i:<6} {priority['operator']:<25} {priority['percentage']:6.2f}%  "
                     f"{priority['count']:<8} {priority['avg_time_us']:10.1f}  "
                     f"{priority['optimization_potential']}")

    # Detailed recommendations
    report.append("\n" + "=" * 80)
    report.append("SPECIFIC RECOMMENDATIONS")
    report.append("=" * 80)

    # Priority 1: Q/DQ elimination
    if qdq_patterns["q_dq_roundtrips"]:
        report.append("\n[PRIORITY 1] Eliminate Q->DQ Roundtrips")
        report.append(f"  Target: {len(qdq_patterns['q_dq_roundtrips'])} roundtrips")
        report.append("  Action: Implement graph optimization pass to remove redundant Q/DQ pairs")
        report.append("  File: onnxruntime/core/optimizer/qdq_transformer/")
        report.append("  Expected gain: 5-10% performance improvement")

    # Priority 2: Top expensive operators
    top_op = priorities[0] if priorities else None
    if top_op:
        report.append(f"\n[PRIORITY 2] Optimize {top_op['operator']}")
        report.append(f"  Current: {top_op['percentage']:.1f}% of runtime")
        report.append(f"  Strategy: {top_op['optimization_potential']}")

        if top_op['operator'] == 'ReduceMean':
            report.append("  Action: Implement NEON-optimized uint16 ReduceMean kernel")
            report.append("  File: onnxruntime/core/providers/cpu/reduction/reduction_ops.cc")
            report.append("  Expected gain: 10-20% overall performance improvement")
        elif top_op['operator'] == 'QLinearMatMul':
            report.append("  Action: Optimize column sum computation in GEMM kernel")
            report.append("  File: onnxruntime/core/mlas/lib/qgemm_u16u8.cpp")
            report.append("  Expected gain: 20-30% overall performance improvement")

    # Priority 3: uint8 downgrade opportunities
    if len(uint8_compat["uint8_compatible"]) > 0:
        report.append("\n[PRIORITY 3] Downgrade Operations to uint8")
        report.append(f"  Target: {len(uint8_compat['uint8_compatible'])} operations currently using uint16")
        report.append("  Action: Implement selective quantization strategy")
        report.append("  - Keep MatMul/Conv activations at uint16")
        report.append("  - Use uint8 for element-wise and shape operations")
        report.append("  Expected gain: Faster Q/DQ, better cache utilization")

    report_text = "\n".join(report)

    # Save to file
    report_path = output_dir / "runtime_graph_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"[OK] Report saved to: {report_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ONNX Runtime execution graph and identify optimization opportunities"
    )
    parser.add_argument("--model", required=True, help="Path to quantized ONNX model")
    parser.add_argument("--output-dir", default="./runtime_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--config", help="Optional config JSON for test data generation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUNTIME GRAPH PROFILING AND ANALYSIS")
    print("=" * 80)

    # Step 1: Analyze original model
    print("\nStep 1: Analyzing original model structure...")
    original_analysis = analyze_graph_structure(args.model)
    print(f"  Original model: {original_analysis['node_count']} nodes")

    # Step 2: Create optimized session and capture graph
    print("\nStep 2: Creating optimized session and capturing runtime graph...")
    session, optimized_path = create_optimized_session(args.model, output_dir)

    # Step 3: Analyze optimized graph
    print("\nStep 3: Analyzing optimized graph structure...")
    optimized_analysis = analyze_graph_structure(optimized_path)
    print(f"  Optimized model: {optimized_analysis['node_count']} nodes")
    print(f"  Nodes eliminated by optimization: {original_analysis['node_count'] - optimized_analysis['node_count']}")

    # Step 4: Run inference to generate profile
    print("\nStep 4: Running inference to generate profiling data...")

    # Generate dummy input
    inputs = {}
    for inp in session.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None or d == -1) else d for d in inp.shape]
        if "int" in inp.type:
            inputs[inp.name] = np.random.randint(0, 100, size=shape).astype(np.int64)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Run inference multiple times
    for i in range(10):
        _ = session.run(None, inputs)

    # Get profile file
    profile_file = session.end_profiling()
    print(f"  Profile saved to: {profile_file}")

    # Step 5: Parse profile
    print("\nStep 5: Parsing profiling data...")
    profile_stats = parse_profile_json(profile_file)
    print(f"  Total operators: {len(profile_stats['operators'])}")

    # Step 6: Identify Q/DQ patterns
    print("\nStep 6: Identifying Q/DQ patterns...")
    qdq_patterns = identify_qdq_patterns(optimized_analysis["nodes"])
    print(f"  Q->DQ roundtrips: {len(qdq_patterns['q_dq_roundtrips'])}")
    print(f"  DQ->Op->Q patterns: {len(qdq_patterns['dq_op_q'])}")

    # Step 7: Analyze uint8 compatibility
    print("\nStep 7: Analyzing uint8 compatibility...")
    uint8_compat = analyze_uint8_compatibility(
        optimized_analysis["nodes"],
        optimized_analysis["tensor_types"]
    )
    print(f"  Operations that could use uint8: {len(uint8_compat['uint8_compatible'])}")
    print(f"  Operations that need uint16: {len(uint8_compat['needs_uint16'])}")

    # Step 8: Create priority list
    print("\nStep 8: Creating optimization priority list...")
    priorities = create_optimization_priority_list(
        profile_stats,
        optimized_analysis,
        qdq_patterns
    )

    # Step 9: Generate report
    print("\nStep 9: Generating comprehensive report...")
    report = generate_report(
        original_analysis,
        optimized_analysis,
        profile_stats,
        qdq_patterns,
        uint8_compat,
        priorities,
        output_dir
    )

    # Step 10: Save JSON data
    print("\nStep 10: Saving detailed analysis data...")
    analysis_data = {
        "original_graph": original_analysis,
        "optimized_graph": optimized_analysis,
        "profile_stats": profile_stats,
        "qdq_patterns": qdq_patterns,
        "uint8_compatibility": uint8_compat,
        "optimization_priorities": priorities
    }

    json_path = output_dir / "runtime_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  JSON data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(report)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Optimized graph: {optimized_path}")
    print(f"  - Profile data: {profile_file}")
    print(f"  - Analysis report: {output_dir / 'runtime_graph_analysis_report.txt'}")
    print(f"  - JSON data: {json_path}")


if __name__ == "__main__":
    main()
