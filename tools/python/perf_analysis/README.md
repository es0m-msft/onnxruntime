# Performance Analysis Tools

Python scripts and documentation for analyzing and optimizing quantized model performance on ONNX Runtime.

---

## 📊 Performance Analysis Documentation

### `OPTIMIZATION_SUMMARY.md`
Complete journey of QUInt16 optimization from broken (598ms) to production-ready (127ms).
- Timeline and performance improvements
- Before/after comparisons
- Key takeaways and lessons learned

### `QUINT8_VS_QUINT16_ANALYSIS.md`
Comprehensive 300+ line root cause analysis explaining why QUInt8 was 17.5x faster than QUInt16.
- Architectural differences (column sum computation strategy)
- Memory access pattern analysis
- Solution options with expected speedups
- Detailed performance impact quantification

### `TILE_SIZE_OPTIMIZATION_RESULTS.md`
Results from implementing Option A (tile size optimization).
- 4.72x speedup achieved (598ms → 127ms)
- CPU utilization improvement (28% → 77%)
- Technical details and why it works
- Comparison with predictions

### `PERF_TEST_RESULTS.md`
Official benchmark results using `onnxruntime_perf_test.exe`.
- FP32, QUInt8, QUInt16 performance comparison
- Detailed latency distribution (P50, P95, P99)
- CPU usage and memory footprint analysis
- Investigation results and next steps

---

## 🔧 Benchmarking Scripts

### PowerShell Scripts

#### `run_perf_test_comparison.ps1`
Automated performance testing using official `onnxruntime_perf_test.exe` tool.
- Tests FP32, QUInt8, and QUInt16 models
- 20 iterations per model
- 30-second cool-down between tests
- Exports results to JSON

**Usage**:
```powershell
.\run_perf_test_comparison.ps1
```

#### `run_thermal_controlled_benchmark.ps1`
Benchmarking with thermal state control.
- Isolated process execution
- Cool-down periods between runs
- Minimizes thermal throttling effects

**Usage**:
```powershell
.\run_thermal_controlled_benchmark.ps1
```

### Python Scripts

#### `benchmark_all_quantizations.py`
Comprehensive benchmark comparing all quantization approaches.
- FP32, dynamic quantization, QUInt8, QUInt16
- Multiple runs with statistical analysis
- CPU usage and memory profiling

**Usage**:
```bash
python benchmark_all_quantizations.py --model model.onnx --runs 20
```

#### `benchmark_isolated.py`
Isolated benchmark execution for accurate measurements.
- Spawns separate processes for each test
- Avoids cross-contamination between runs
- Thermal state management

**Usage**:
```bash
python benchmark_isolated.py --model model.onnx --quant-type quint16
```

---

## 🎯 Quantization Scripts

### `quantize_and_evaluate_with_profiling.py`
Enhanced quantization script with detailed profiling support.
- Operator-level timing breakdown
- Memory usage analysis
- Exports profiling data to JSON
- Supports multiple quantization types (fp32, dynamic, quint8, quint16)

**Usage**:
```bash
python quantize_and_evaluate_with_profiling.py \
    --model model.onnx \
    --config config.json \
    --quant-type quint16
```

### `quantize_and_evaluate.py`
Basic quantization and evaluation script for ONNX models.
- Supports QUInt8 and QUInt16 quantization
- Evaluates accuracy (L2 error) and performance
- Outputs quantized models in QDQ format

**Usage**:
```bash
python quantize_and_evaluate.py --model model.onnx --config config.json
```

### `quantize_mixed_precision.py`
Mixed-precision quantization implementation.
- QUInt16 for MatMul activations (better accuracy)
- QUInt8 for all other operations (better performance)
- Optional lookup table optimization
- Best performance-accuracy tradeoff

**Usage**:
```bash
python quantize_mixed_precision.py \
    --model model.onnx \
    --output mixed_model.onnx \
    --use-lookup-table  # Optional
```

### `quantize_quint8.py`
Simple QUInt8×QUInt8 quantization script.
- Pure 8-bit quantization (best performance)
- DummyCalibrationDataReader for testing
- QDQ format with fusion support

**Usage**:
```bash
python quantize_quint8.py  # Uses hardcoded paths
```

### `quantize_florence_properly.py`
Specialized quantization script for Florence model.
- Configurable quantization parameters
- Proper QDQ cleanup
- Model-specific optimizations

**Usage**:
```bash
python quantize_florence_properly.py
```

---

## 🔍 Model Analysis Scripts

### `analyze_u16u8_model_nodes.py`
Comprehensive model structure analysis for QDQ quantized models.
- Counts node types and unique configurations
- Identifies Q/DQ roundtrip patterns
- Detects MatMul operations and fusion opportunities
- Export analysis to JSON

**Usage**:
```bash
python analyze_u16u8_model_nodes.py --model model.onnx
```

### `investigate_unfused_matmuls.py`
Analyzes MatMul operations in quantized models to identify fusion issues.
- Examines DQ→MatMul→Q patterns
- Categorizes fusion blockers (shared Q/DQ, unsupported types, etc.)
- Identifies high-priority optimization targets
- Generates detailed reports

**Usage**:
```bash
python investigate_unfused_matmuls.py --model quantized_model.onnx
```

### `diagnose_uint16_kernel_usage.py`
Diagnostic tool for verifying uint16 kernel registration and usage.
- Checks if NEON kernels are being invoked
- Identifies FP32 fallbacks
- Provides detailed operator-level profiling
- Verifies input/output data types

**Usage**:
```bash
python diagnose_uint16_kernel_usage.py --model quantized_model.onnx
```

### `profile_runtime_graph.py`
Runtime graph profiling and analysis.
- Execution time per operator
- Memory allocation tracking
- Identifies performance bottlenecks
- Exports detailed profiling data

**Usage**:
```bash
python profile_runtime_graph.py --model model.onnx
```

### `identify_ort_elision_points.py`
Identifies Q/DQ elision opportunities in ONNX Runtime graphs.
- Detects redundant Q/DQ pairs
- Analyzes elision patterns
- Suggests optimization opportunities

**Usage**:
```bash
python identify_ort_elision_points.py --model model.onnx
```

---

## 🧪 Testing Scripts

### `test_graph_optimization.py`
Verifies graph optimization and fusion behavior at different optimization levels.
- Tests ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
- Compares MatMul vs QLinearMatMul node counts
- Validates that QDQ fusion is working correctly
- Identifies optimization blockers

**Usage**:
```bash
python test_graph_optimization.py --model quantized_model.onnx
```

---

## 🔧 Schema Patch Scripts (Deprecated)

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

---

## 📈 Key Findings Summary

### Performance Comparison (Florence Model, ARM64)

| Model | Avg Latency | vs FP32 | CPU Usage | Status |
|-------|-------------|---------|-----------|--------|
| **FP32 Baseline** | 69ms | 1.0x | 75% | Reference |
| **QUInt8 QDQ** | 55ms | **1.24x faster** | 76% | ✅ **Best** |
| **QUInt16 QDQ (OLD)** | 598ms | 8.72x slower | 28% | ❌ Broken |
| **QUInt16 QDQ (NEW)** | 127ms | 1.85x slower | 77% | ✅ **Fixed!** |

**Key Achievement**: 4.72x speedup for QUInt16 with simple tile size optimization!

### Root Cause of QUInt16 Performance Issue

**Problem**: Small tile sizes (4 rows vs QUInt8's 24 rows) caused excessive overhead recomputation.

**Solution**: Increased StrideM from 4 to 24, StrideN from 16 to 128 (7 lines of code changed).

**Result**:
- CPU utilization: 28% → 77% (memory-bound → compute-bound)
- Tile iterations: 20 → 4 (5x reduction)
- Performance: 598ms → 127ms (4.72x faster)

---

## 🚀 Quick Start

### 1. Run Full Performance Comparison
```powershell
cd C:\d\onnxruntime\tools\python\perf_analysis
.\run_perf_test_comparison.ps1
```

### 2. Quantize and Profile a Model
```bash
python quantize_and_evaluate_with_profiling.py \
    --model /path/to/model.onnx \
    --config /path/to/config.json \
    --quant-type quint16
```

### 3. Analyze Quantized Model
```bash
python analyze_u16u8_model_nodes.py --model quantized_model.onnx
python diagnose_uint16_kernel_usage.py --model quantized_model.onnx
```

---

## 📋 Requirements

- **ONNX Runtime**: 1.24.0+ (with uint16 kernel support)
- **ONNX**: Latest version
- **NumPy**: Latest version
- **Platform**: ARM64 (for NEON kernel testing)
- **Python**: 3.8+
- **PowerShell**: 5.1+ (for .ps1 scripts on Windows)

---

## 📚 Related Documentation

### Main Documentation
- [CLAUDE.md](../../../CLAUDE.md) - Project overview and integration guide
- [COMPREHENSIVE_OPTIMIZATION_PLAN.md](../../../runtime_analysis/COMPREHENSIVE_OPTIMIZATION_PLAN.md) - Original optimization strategy

### Performance Analysis (This Directory)
- [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md) - Complete optimization journey
- [QUINT8_VS_QUINT16_ANALYSIS.md](./QUINT8_VS_QUINT16_ANALYSIS.md) - Root cause analysis
- [TILE_SIZE_OPTIMIZATION_RESULTS.md](./TILE_SIZE_OPTIMIZATION_RESULTS.md) - Option A results
- [PERF_TEST_RESULTS.md](./PERF_TEST_RESULTS.md) - Official benchmark results

---

## 🎯 Recommended Workflow

### For New Models

1. **Quantize with profiling**:
   ```bash
   python quantize_and_evaluate_with_profiling.py \
       --model model.onnx \
       --config config.json \
       --quant-type quint16
   ```

2. **Analyze structure**:
   ```bash
   python analyze_u16u8_model_nodes.py --model quantized_model.onnx
   ```

3. **Verify kernel usage**:
   ```bash
   python diagnose_uint16_kernel_usage.py --model quantized_model.onnx
   ```

4. **Run official benchmarks**:
   ```powershell
   .\run_perf_test_comparison.ps1
   ```

### For Performance Investigation

1. Read [QUINT8_VS_QUINT16_ANALYSIS.md](./QUINT8_VS_QUINT16_ANALYSIS.md) for architectural understanding
2. Check [PERF_TEST_RESULTS.md](./PERF_TEST_RESULTS.md) for baseline comparisons
3. Review [TILE_SIZE_OPTIMIZATION_RESULTS.md](./TILE_SIZE_OPTIMIZATION_RESULTS.md) for optimization examples

---

## 📝 Notes

- All benchmarks in this directory use the Florence v1.6.2 text transformer model as reference
- Performance numbers are for ARM64 Windows platform with NEON optimizations
- QUInt16 performance has been significantly improved from 598ms to 127ms (4.72x speedup)
- QUInt8 remains the recommended quantization for best performance (55ms)

---

## 🤝 Contributing

When adding new scripts or documentation to this directory:
1. Update this README with a description and usage example
2. Use consistent naming: `<verb>_<noun>.py` or `<action>_<target>.ps1`
3. Include docstrings and command-line help
4. Add results to corresponding `.md` documentation files
