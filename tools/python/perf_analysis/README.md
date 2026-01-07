# Performance Analysis Tools

Python scripts for analyzing and optimizing quantized model performance on ONNX Runtime.

## Model Analysis Scripts

### `analyze_u16u8_model_nodes.py`
Comprehensive model structure analysis for QDQ quantized models.
- Counts node types and unique configurations
- Identifies Q/DQ roundtrip patterns
- Detects MatMul operations and fusion opportunities

### `investigate_unfused_matmuls.py`
Analyzes MatMul operations in quantized models to identify fusion issues.
- Examines DQ→MatMul→Q patterns
- Categorizes fusion blockers
- Identifies high-priority optimization targets

### `test_graph_optimization.py`
Verifies graph optimization and fusion behavior at different optimization levels.
- Tests ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
- Compares MatMul vs QLinearMatMul node counts
- Validates that QDQ fusion is working correctly

### `diagnose_uint16_kernel_usage.py`
Diagnostic tool for verifying uint16 kernel registration and usage.
- Checks if NEON kernels are being invoked
- Identifies FP32 fallbacks
- Provides detailed operator-level profiling

## Quantization Scripts

### `quantize_and_evaluate.py`
Basic quantization and evaluation script for ONNX models.
- Supports QUInt8 and QUInt16 quantization
- Evaluates accuracy (L2 error) and performance
- Outputs quantized models in QDQ format

### `quantize_and_evaluate_with_profiling.py`
Enhanced quantization script with detailed profiling support.
- Operator-level timing breakdown
- Memory usage analysis
- Exports profiling data to JSON
- Supports multiple quantization types

### `quantize_mixed_precision.py`
Mixed-precision quantization implementation.
- QUInt16 for MatMul activations
- QUInt8 for all other operations
- Optional lookup table optimization
- Best performance-accuracy tradeoff

## Schema Patch Scripts (Deprecated)

### `patch_onnx_schema.py`
Patches ONNX operator schemas to add uint16 support.
- Modifies QLinearMatMul, QuantizeLinear, DequantizeLinear schemas
- **Status**: Deprecated (not needed with proper kernel registration)

### `patch_onnx_schema_old.py`
Earlier version of schema patching script.
- **Status**: Historical reference, not actively used

### `patch_qdq_opset13.py`
Patches opset 13 Q/DQ operator schemas.
- **Status**: Deprecated (handled by conditional compilation)

## Usage Examples

### Analyze Model Structure
```bash
python analyze_u16u8_model_nodes.py --model model.onnx
```

### Test Graph Optimization
```bash
python test_graph_optimization.py --model quantized_model.onnx
```

### Quantize and Evaluate (Basic)
```bash
python quantize_and_evaluate.py --model model.onnx --config config.json
```

### Quantize and Evaluate (With Profiling)
```bash
python quantize_and_evaluate_with_profiling.py --model model.onnx --config config.json --quant-type quint16
```

### Mixed-Precision Quantization
```bash
python quantize_mixed_precision.py --model model.onnx --output mixed_model.onnx
```

### Diagnose Kernel Usage
```bash
python diagnose_uint16_kernel_usage.py --model quantized_model.onnx
```

## Requirements

- ONNX Runtime 1.24.0+ (with uint16 kernel support)
- ONNX
- NumPy
- ARM64 platform (for NEON kernel testing)

## Related Documentation

- [CLAUDE.md](../../../CLAUDE.md) - Project overview and optimization progress
- [FINAL_QUANTIZATION_COMPARISON.md](../../../FINAL_QUANTIZATION_COMPARISON.md) - Detailed performance analysis
- [INVESTIGATION_SUMMARY_2026-01-07.md](../../../INVESTIGATION_SUMMARY_2026-01-07.md) - Investigation findings
