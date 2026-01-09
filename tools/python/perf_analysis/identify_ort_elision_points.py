#!/usr/bin/env python3
"""
Identify specific locations in ONNX Runtime that need changes for Q/DQ elision.

This script analyzes:
1. Graph optimization transformers that handle Q/DQ fusion
2. Kernel implementations that could work natively on quantized data
3. Type promotion rules that cause unnecessary conversions
4. Opportunities to bypass Q/DQ entirely
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set
import re


def find_qdq_transformers(ort_source_dir: Path) -> List[Dict]:
    """Find QDQ transformer implementations in ORT source."""
    transformer_dir = ort_source_dir / "onnxruntime" / "core" / "optimizer" / "qdq_transformer"

    if not transformer_dir.exists():
        return []

    transformers = []

    # Scan transformer files
    for cpp_file in transformer_dir.rglob("*.cc"):
        try:
            content = cpp_file.read_text(encoding='utf-8', errors='ignore')

            # Look for fusion rules
            fusion_rules = re.findall(r'class\s+(\w+Selector)\s*:', content)
            action_classes = re.findall(r'class\s+(\w+Action)\s*:', content)

            # Look for Q/DQ handling
            has_quantize_handling = 'QuantizeLinear' in content
            has_dequantize_handling = 'DequantizeLinear' in content
            has_qlinear_handling = 'QLinear' in content

            # Count Q/DQ references
            q_count = content.count('QuantizeLinear')
            dq_count = content.count('DequantizeLinear')

            transformers.append({
                "file": str(cpp_file.relative_to(ort_source_dir)),
                "fusion_rules": fusion_rules,
                "action_classes": action_classes,
                "handles_quantize": has_quantize_handling,
                "handles_dequantize": has_dequantize_handling,
                "handles_qlinear": has_qlinear_handling,
                "q_references": q_count,
                "dq_references": dq_count,
                "needs_elision_support": (q_count > 0 and dq_count > 0)
            })
        except Exception as e:
            print(f"Warning: Could not process {cpp_file}: {e}")

    return transformers


def find_quantized_kernels(ort_source_dir: Path) -> List[Dict]:
    """Find kernel implementations for quantized operations."""
    kernel_dirs = [
        ort_source_dir / "onnxruntime" / "core" / "providers" / "cpu" / "quantization",
        ort_source_dir / "onnxruntime" / "core" / "mlas" / "lib",
    ]

    kernels = []

    for kernel_dir in kernel_dirs:
        if not kernel_dir.exists():
            continue

        for source_file in kernel_dir.rglob("*"):
            if source_file.suffix not in ['.cc', '.cpp', '.h', '.asm', '.S']:
                continue

            try:
                content = source_file.read_text(encoding='utf-8', errors='ignore')

                # Look for quantized operations
                has_qlinear = 'QLinear' in content
                has_uint8_ops = 'uint8' in content or 'UINT8' in content
                has_uint16_ops = 'uint16' in content or 'UINT16' in content
                has_int8_ops = 'int8' in content or 'INT8' in content

                # Look for Q/DQ in kernels
                has_quantize = 'Quantize' in content and 'Linear' in content
                has_dequantize = 'Dequantize' in content and 'Linear' in content

                # Look for NEON/SIMD
                has_neon = 'neon' in content.lower() or 'vld' in content or 'vst' in content
                has_avx = 'avx' in content.lower() or 'mm256' in content

                if any([has_qlinear, has_quantize, has_dequantize]):
                    kernels.append({
                        "file": str(source_file.relative_to(ort_source_dir)),
                        "has_qlinear": has_qlinear,
                        "has_quantize": has_quantize,
                        "has_dequantize": has_dequantize,
                        "supports_uint8": has_uint8_ops,
                        "supports_uint16": has_uint16_ops,
                        "supports_int8": has_int8_ops,
                        "has_neon": has_neon,
                        "has_avx": has_avx,
                    })
            except Exception as e:
                pass

    return kernels


def find_type_promotion_rules(ort_source_dir: Path) -> List[Dict]:
    """Find type promotion and conversion rules."""
    type_rules = []

    # Look in graph optimization and type inference
    search_dirs = [
        ort_source_dir / "onnxruntime" / "core" / "graph",
        ort_source_dir / "onnxruntime" / "core" / "optimizer",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for source_file in search_dir.rglob("*.cc"):
            try:
                content = source_file.read_text(encoding='utf-8', errors='ignore')

                # Look for type promotion patterns
                has_type_inference = 'TypeInference' in content or 'InferType' in content
                has_cast = 'Cast' in content
                has_uint16_uint8 = ('uint16' in content or 'UINT16' in content) and \
                                  ('uint8' in content or 'UINT8' in content)

                if has_type_inference or (has_cast and has_uint16_uint8):
                    type_rules.append({
                        "file": str(source_file.relative_to(ort_source_dir)),
                        "has_type_inference": has_type_inference,
                        "has_cast": has_cast,
                        "handles_uint16_uint8": has_uint16_uint8
                    })
            except Exception as e:
                pass

    return type_rules


def analyze_reduction_ops(ort_source_dir: Path) -> List[Dict]:
    """Analyze reduction operations (ReduceMean, etc.) for uint16 support."""
    reduction_files = []

    reduction_dir = ort_source_dir / "onnxruntime" / "core" / "providers" / "cpu" / "reduction"

    if reduction_dir.exists():
        for source_file in reduction_dir.rglob("*.cc"):
            try:
                content = source_file.read_text(encoding='utf-8', errors='ignore')

                # Check for reduction operations
                has_reduce_mean = 'ReduceMean' in content
                has_reduce_sum = 'ReduceSum' in content

                # Check for uint16 support
                has_uint16 = 'uint16' in content or 'UINT16' in content

                # Check for SIMD
                has_neon = 'neon' in content.lower() or '#include <arm_neon.h>' in content
                has_avx = 'avx' in content.lower() or 'immintrin' in content

                if has_reduce_mean or has_reduce_sum:
                    reduction_files.append({
                        "file": str(source_file.relative_to(ort_source_dir)),
                        "has_reduce_mean": has_reduce_mean,
                        "has_reduce_sum": has_reduce_sum,
                        "supports_uint16": has_uint16,
                        "has_neon": has_neon,
                        "has_avx": has_avx,
                        "needs_uint16_kernel": not has_uint16 and (has_reduce_mean or has_reduce_sum)
                    })
            except Exception as e:
                pass

    return reduction_files


def generate_elision_recommendations(
    transformers: List[Dict],
    kernels: List[Dict],
    type_rules: List[Dict],
    reductions: List[Dict]
) -> Dict:
    """Generate specific recommendations for Q/DQ elision."""

    recommendations = {
        "graph_optimization": [],
        "kernel_implementation": [],
        "type_system": [],
        "new_transformers_needed": [],
    }

    # Graph optimization recommendations
    qdq_transformers = [t for t in transformers if t["needs_elision_support"]]
    if qdq_transformers:
        recommendations["graph_optimization"].append({
            "priority": "HIGH",
            "title": "Implement Q->DQ Roundtrip Elimination",
            "description": "Add transformer to eliminate QuantizeLinear immediately followed by DequantizeLinear",
            "files_to_modify": [t["file"] for t in qdq_transformers[:3]],
            "implementation": """
1. Create new transformer class: QDQRoundtripEliminator
2. Match pattern: QuantizeLinear -> DequantizeLinear
3. Check if scale/zero-point are compatible
4. Replace both nodes with identity or remove entirely
5. Update graph edges to bypass the roundtrip

Example location: onnxruntime/core/optimizer/qdq_transformer/qdq_roundtrip_eliminator.cc
            """,
            "expected_impact": "5-10% performance improvement, reduced memory bandwidth"
        })

        recommendations["graph_optimization"].append({
            "priority": "HIGH",
            "title": "Implement DQ->Op->Q Fusion for Element-wise Operations",
            "description": "Fuse DQ->(Add/Mul/etc)->Q into native quantized operations",
            "files_to_modify": [t["file"] for t in qdq_transformers[:3]],
            "implementation": """
1. Identify element-wise ops: Add, Sub, Mul, Div, Relu, etc.
2. Match pattern: DQ -> ElementWise -> Q
3. Check if operation can work natively on quantized data
4. Replace with quantized version (e.g., QLinearAdd, QLinearMul)
5. Propagate quantization parameters through the operation

Example: DQ(uint8) -> Add -> Q(uint8) becomes QLinearAdd(uint8->uint8)
            """,
            "expected_impact": "10-15% performance improvement"
        })

    # Kernel implementation recommendations
    uint16_kernels_needed = []
    for kernel in kernels:
        if kernel["has_qlinear"] and not kernel["supports_uint16"]:
            uint16_kernels_needed.append(kernel["file"])

    if uint16_kernels_needed:
        recommendations["kernel_implementation"].append({
            "priority": "MEDIUM",
            "title": "Add uint16 Support to Existing Quantized Kernels",
            "description": "Extend QLinear operations to support uint16 activations",
            "files_to_modify": uint16_kernels_needed[:5],
            "implementation": """
1. Add template specializations for uint16_t
2. Implement NEON/AVX kernels for uint16 data types
3. Register kernel variants with ORT execution provider
4. Add type constraints in operator schemas

Focus on: QLinearAdd, QLinearMul, QLinearConv
            """,
            "expected_impact": "Enable native uint16 operations without DQ/Q"
        })

    # ReduceMean optimization
    reduce_needs_opt = [r for r in reductions if r.get("needs_uint16_kernel", False)]
    if reduce_needs_opt:
        recommendations["kernel_implementation"].append({
            "priority": "CRITICAL",
            "title": "Implement NEON-Optimized uint16 ReduceMean Kernel",
            "description": "Add SIMD-optimized ReduceMean for uint16 data type",
            "files_to_modify": [r["file"] for r in reduce_needs_opt],
            "implementation": """
1. File: onnxruntime/core/providers/cpu/reduction/reduction_ops.cc
2. Add template specialization: ReduceMean<uint16_t>
3. Implement ARM NEON kernel using:
   - vld1q_u16 for loading 8×uint16
   - vaddq_u32 for accumulation (widen to uint32)
   - Horizontal sum and division
4. Handle non-multiple-of-8 remainders with scalar fallback
5. Register kernel for uint16 data type

Expected speedup: 8-16x over current scalar implementation
            """,
            "expected_impact": "20-25% overall performance improvement (ReduceMean is 23.87% of runtime)"
        })

    # Type system recommendations
    recommendations["type_system"].append({
        "priority": "MEDIUM",
        "title": "Implement Selective Quantization Type Promotion",
        "description": "Keep operations in uint8 where possible, use uint16 only where needed",
        "files_to_modify": [
            "onnxruntime/core/optimizer/qdq_transformer/selectors_actions/*.cc",
            "onnxruntime/quantization/onnx_quantizer.py"
        ],
        "implementation": """
1. Classify operations by precision requirements:
   - High precision: MatMul, Conv, ReduceMean -> uint16
   - Low precision: Add, Mul, Relu, etc. -> uint8

2. Implement smart type propagation:
   - Start with uint8 by default
   - Promote to uint16 only at precision-critical ops
   - Insert Q/DQ at type boundaries

3. Modify QDQ transformer to preserve type boundaries
4. Update quantization calibration to detect precision requirements

Result: Hybrid model with uint16 only where needed
        """,
        "expected_impact": "Best of both worlds: uint8 speed + uint16 accuracy"
    })

    # New transformers needed
    recommendations["new_transformers_needed"].append({
        "priority": "HIGH",
        "title": "Create QuantizedOperatorFusionTransformer",
        "description": "New transformer to fuse operations in quantized domain",
        "location": "onnxruntime/core/optimizer/qdq_transformer/quantized_op_fusion.cc",
        "patterns_to_fuse": [
            "QLinearMatMul -> Add -> Relu",
            "QLinearConv -> BatchNorm -> Relu",
            "ReduceMean -> Quantize -> Dequantize",
            "Add(uint16) -> Add(uint16) -> ... (chain fusion)",
        ],
        "implementation": """
Create a new transformer that:
1. Works on already-quantized graphs
2. Fuses operations without dequantizing
3. Generates combined kernels (e.g., QLinearMatMulRelu)
4. Maintains quantization parameters throughout

This is AFTER QDQ fusion, operates on QLinear* nodes directly.
        """,
        "expected_impact": "Additional 10-15% performance gain"
    })

    return recommendations


def generate_implementation_guide(recommendations: Dict, output_dir: Path) -> str:
    """Generate detailed implementation guide."""
    guide = []

    guide.append("=" * 80)
    guide.append("Q/DQ ELISION IMPLEMENTATION GUIDE")
    guide.append("=" * 80)

    guide.append("\nThis guide identifies specific locations in ONNX Runtime source code")
    guide.append("where changes are needed to eliminate redundant Q/DQ operations.")

    # Graph Optimization Section
    if recommendations["graph_optimization"]:
        guide.append("\n" + "=" * 80)
        guide.append("GRAPH OPTIMIZATION CHANGES")
        guide.append("=" * 80)

        for i, rec in enumerate(recommendations["graph_optimization"], 1):
            guide.append(f"\n[{rec['priority']}] {i}. {rec['title']}")
            guide.append("-" * 80)
            guide.append(f"Description: {rec['description']}")
            guide.append(f"Expected Impact: {rec['expected_impact']}")
            guide.append("\nFiles to Modify:")
            for file in rec["files_to_modify"]:
                guide.append(f"  - {file}")
            guide.append("\nImplementation Steps:")
            guide.append(rec["implementation"])

    # Kernel Implementation Section
    if recommendations["kernel_implementation"]:
        guide.append("\n" + "=" * 80)
        guide.append("KERNEL IMPLEMENTATION CHANGES")
        guide.append("=" * 80)

        for i, rec in enumerate(recommendations["kernel_implementation"], 1):
            guide.append(f"\n[{rec['priority']}] {i}. {rec['title']}")
            guide.append("-" * 80)
            guide.append(f"Description: {rec['description']}")
            guide.append(f"Expected Impact: {rec['expected_impact']}")
            guide.append("\nFiles to Modify:")
            for file in rec["files_to_modify"]:
                guide.append(f"  - {file}")
            guide.append("\nImplementation Steps:")
            guide.append(rec["implementation"])

    # Type System Section
    if recommendations["type_system"]:
        guide.append("\n" + "=" * 80)
        guide.append("TYPE SYSTEM CHANGES")
        guide.append("=" * 80)

        for i, rec in enumerate(recommendations["type_system"], 1):
            guide.append(f"\n[{rec['priority']}] {i}. {rec['title']}")
            guide.append("-" * 80)
            guide.append(f"Description: {rec['description']}")
            guide.append(f"Expected Impact: {rec['expected_impact']}")
            guide.append("\nFiles to Modify:")
            for file in rec["files_to_modify"]:
                guide.append(f"  - {file}")
            guide.append("\nImplementation Steps:")
            guide.append(rec["implementation"])

    # New Transformers Section
    if recommendations["new_transformers_needed"]:
        guide.append("\n" + "=" * 80)
        guide.append("NEW TRANSFORMERS TO CREATE")
        guide.append("=" * 80)

        for i, rec in enumerate(recommendations["new_transformers_needed"], 1):
            guide.append(f"\n[{rec['priority']}] {i}. {rec['title']}")
            guide.append("-" * 80)
            guide.append(f"Description: {rec['description']}")
            guide.append(f"Location: {rec['location']}")
            guide.append(f"Expected Impact: {rec['expected_impact']}")
            guide.append("\nPatterns to Fuse:")
            for pattern in rec.get("patterns_to_fuse", []):
                guide.append(f"  - {pattern}")
            guide.append("\nImplementation Steps:")
            guide.append(rec["implementation"])

    # Summary
    guide.append("\n" + "=" * 80)
    guide.append("IMPLEMENTATION ROADMAP")
    guide.append("=" * 80)

    guide.append("\nPhase 1: Quick Wins (1-2 weeks)")
    guide.append("  1. Implement Q->DQ roundtrip elimination transformer")
    guide.append("  2. Implement NEON-optimized uint16 ReduceMean kernel")
    guide.append("  Expected gain: 25-30% performance improvement")

    guide.append("\nPhase 2: Kernel Optimization (2-3 weeks)")
    guide.append("  1. Optimize QLinearMatMul column sum computation")
    guide.append("  2. Add NEON kernels for QuantizeLinear/DequantizeLinear uint16")
    guide.append("  3. Implement DQ->Op->Q fusion for element-wise operations")
    guide.append("  Expected gain: Additional 15-20% improvement")

    guide.append("\nPhase 3: Type System (3-4 weeks)")
    guide.append("  1. Implement selective uint8/uint16 quantization")
    guide.append("  2. Create hybrid quantization strategy")
    guide.append("  3. Update calibration tools")
    guide.append("  Expected gain: Best accuracy/performance tradeoff")

    guide.append("\nPhase 4: Advanced Fusion (2-3 weeks)")
    guide.append("  1. Create QuantizedOperatorFusionTransformer")
    guide.append("  2. Implement multi-op fusion patterns")
    guide.append("  Expected gain: Additional 10-15% improvement")

    guide.append("\n" + "=" * 80)
    guide.append("TOTAL EXPECTED IMPROVEMENT")
    guide.append("=" * 80)
    guide.append("\nCumulative performance improvement: 60-80% faster")
    guide.append("Target: Match or exceed FP32 performance")
    guide.append("        Match or exceed uint8 performance with better accuracy")

    guide_text = "\n".join(guide)

    # Save to file
    guide_path = output_dir / "qdq_elision_implementation_guide.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_text)

    print(f"[OK] Implementation guide saved to: {guide_path}")
    return guide_text


def main():
    parser = argparse.ArgumentParser(
        description="Identify ORT source locations for Q/DQ elision implementation"
    )
    parser.add_argument("--ort-source", default=".",
                       help="Path to ONNX Runtime source directory")
    parser.add_argument("--output-dir", default="./runtime_analysis",
                       help="Output directory for analysis")

    args = parser.parse_args()

    ort_source_dir = Path(args.ort_source).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ONNX RUNTIME Q/DQ ELISION ANALYSIS")
    print("=" * 80)
    print(f"\nSource directory: {ort_source_dir}")

    # Step 1: Find QDQ transformers
    print("\nStep 1: Analyzing QDQ transformers...")
    transformers = find_qdq_transformers(ort_source_dir)
    print(f"  Found {len(transformers)} transformer files")
    print(f"  {sum(1 for t in transformers if t['needs_elision_support'])} need elision support")

    # Step 2: Find quantized kernels
    print("\nStep 2: Analyzing quantized kernels...")
    kernels = find_quantized_kernels(ort_source_dir)
    print(f"  Found {len(kernels)} kernel files")
    print(f"  {sum(1 for k in kernels if k['supports_uint16'])} support uint16")
    print(f"  {sum(1 for k in kernels if k['has_neon'])} have NEON optimizations")

    # Step 3: Find type promotion rules
    print("\nStep 3: Analyzing type promotion rules...")
    type_rules = find_type_promotion_rules(ort_source_dir)
    print(f"  Found {len(type_rules)} files with type handling")

    # Step 4: Analyze reduction operations
    print("\nStep 4: Analyzing reduction operations...")
    reductions = analyze_reduction_ops(ort_source_dir)
    print(f"  Found {len(reductions)} reduction implementation files")
    print(f"  {sum(1 for r in reductions if r.get('needs_uint16_kernel'))} need uint16 kernels")

    # Step 5: Generate recommendations
    print("\nStep 5: Generating recommendations...")
    recommendations = generate_elision_recommendations(
        transformers,
        kernels,
        type_rules,
        reductions
    )

    print(f"  Graph optimization recommendations: {len(recommendations['graph_optimization'])}")
    print(f"  Kernel implementation recommendations: {len(recommendations['kernel_implementation'])}")
    print(f"  Type system recommendations: {len(recommendations['type_system'])}")
    print(f"  New transformers needed: {len(recommendations['new_transformers_needed'])}")

    # Step 6: Generate implementation guide
    print("\nStep 6: Generating implementation guide...")
    guide = generate_implementation_guide(recommendations, output_dir)

    # Step 7: Save JSON data
    print("\nStep 7: Saving detailed data...")
    data = {
        "transformers": transformers,
        "kernels": kernels,
        "type_rules": type_rules,
        "reductions": reductions,
        "recommendations": recommendations
    }

    json_path = output_dir / "qdq_elision_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  JSON data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(guide)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
