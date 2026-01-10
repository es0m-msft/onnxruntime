# QUInt16×QUInt8 QGEMM Integration Notes

> **Note**: This document uses placeholders for local paths. See `.localpaths.txt` in the ONNX Runtime source directory for your environment-specific paths:
> - `ORTSOURCEDIR` - ONNX Runtime source directory
> - `MODELDIR` - Florence model directory
> - `MODELPATH` - Full path to model .onnx file
> - `MODELCONFIG` - Full path to model .json config file

## Environment
- Running on Windows
- Default shell is Git Bash (not PowerShell)
- To use PowerShell, invoke it explicitly: `powershell -NoProfile -Command "..."`
- Path formats: Git Bash uses `/` (e.g., `/c/git/repo`), Windows uses `\` (e.g., `C:\git\repo`)

## Quick Start: Automated Development Workflow

**For rapid development and testing, use these single-line automation scripts:**

### 1. Build, Deploy, and Test in One Command
```powershell
# After making code changes, rebuild everything and test
${ORTSOURCEDIR}\test_changes.ps1
```

### 2. Build and Deploy Only
```powershell
# Rebuild C++, rebuild Python wheel, install in venv
${ORTSOURCEDIR}\build_and_deploy.ps1

# Optional: Skip specific steps
${ORTSOURCEDIR}\build_and_deploy.ps1 -SkipBuild    # Only wheel and install
${ORTSOURCEDIR}\build_and_deploy.ps1 -SkipWheel    # Only C++ build
${ORTSOURCEDIR}\build_and_deploy.ps1 -SkipInstall  # Build but don't install
```

### 3. Compare Model Performance
```powershell
# Compare original vs fixed model variants
${ORTSOURCEDIR}\tools\python\perf_analysis\compare_models.ps1
```

### 4. Diagnose QDQ Fusion Issues
```bash
# Analyze QDQ fusion in quantized models
cd ${MODELDIR}
${ORTSOURCEDIR}\.venv_quant_test\Scripts\python.exe ${ORTSOURCEDIR}\tools\python\perf_analysis\diagnose_qdq_fusion.py embimg_c.quant.onnx
```

**See the [QUInt16 Performance Investigation Plan](C:\Users\ersomme\.claude\plans\eager-toasting-pearl.md) for detailed script documentation and usage.**

---

## Building ONNX Runtime ARM64 with QUInt16×QUInt8 Support

### Full Build Command (Initial Build)
```powershell
cd ${ORTSOURCEDIR}
python tools/ci_build/build.py --config Release --build_dir build_arm64_u16u8 --arm64 --build_wheel --parallel --skip_tests
```

### Incremental Build Commands (After Initial Build)

**1. Rebuild C++ only (after code changes):**
```powershell
cd ${ORTSOURCEDIR}\build_arm64_u16u8\Release
cmake --build . --config Release --parallel
```
This recompiles only changed C++ files without regenerating CMake cache. Much faster for iterative development.

**2. Rebuild Python wheel only (from latest build):**
```powershell
cd ${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release
python ../../../setup.py bdist_wheel
```
This regenerates the wheel from current compiled binaries without recompiling C++ code. The wheel will be created in `dist\onnxruntime-1.24.0-cp312-cp312-win_arm64.whl` (relative to the current directory).

### Build Artifacts
- Wheel location: `${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release\dist\onnxruntime-1.24.0-cp312-cp312-win_arm64.whl`
- Build time: ~15-20 minutes on ARM64 Windows

## Installing the Built Wheel

### Using Virtual Environment
```powershell
cd ${ORTSOURCEDIR}
.venv_quant_test\Scripts\Activate.ps1
pip install --force-reinstall "${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release\dist\onnxruntime-1.24.0-cp312-cp312-win_arm64.whl"
```

## Running Quantization Tests on Florence Model

### Model Location
- Base model: `${MODELPATH}`
- Config file: `${MODELCONFIG}`

### Run Command
```powershell
cd ${MODELDIR}
${ORTSOURCEDIR}\.venv_quant_test\Scripts\Activate.ps1
python ${ORTSOURCEDIR}\tools\python\perf_analysis\quantize_and_evaluate_with_profiling.py --model model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx --config model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.json
```

### Important Notes
1. **Use PowerShell** for activation scripts on Windows
2. **Scripts location**: Performance analysis scripts are in `tools/python/perf_analysis/`
3. **Opset version**: Model can stay at opset 14 or 21 (both have same schema issue)
4. **UTF-8 fix**: Replace Unicode checkmarks (✓) with [OK] in Python scripts for Windows console compatibility

## Performance Testing with onnxruntime_perf_test

### Official Performance Testing Tool

The build includes `onnxruntime_perf_test.exe` which provides accurate, low-level performance measurements.

**Location**: `${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release\onnxruntime_perf_test.exe`

### Basic Usage

```powershell
cd ${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release

# Basic performance test
.\onnxruntime_perf_test.exe -m times -r 20 -I ${MODELPATH}
```

### Command Options

- `-m times` - Measure execution time
- `-r 20` - Number of runs (20 iterations)
- `-I <model_path>` - Input model file
- `-e cpu` - Use CPU execution provider (default)
- `-o 99` - Graph optimization level (0=disable, 1=basic, 2=extended, 99=all)

### Comparing Quantization Approaches

To benchmark FP32, Dynamic Quant, QDQ QUInt8, and QDQ QUInt16:

```powershell
$PERF_TEST = "${ORTSOURCEDIR}\build_arm64_u16u8\Release\Release\onnxruntime_perf_test.exe"

# FP32 Baseline
& $PERF_TEST -m times -r 20 -I "${MODELPATH}"

# Dynamic Quantization (if available)
& $PERF_TEST -m times -r 20 -I "${MODELDIR}\model_dynamic_quant.onnx"

# QDQ QUInt8
& $PERF_TEST -m times -r 20 -I "${ORTSOURCEDIR}\quantized_models\florence_quint8_qdq.onnx"

# QDQ QUInt16
& $PERF_TEST -m times -r 20 -I "${ORTSOURCEDIR}\quantized_models\florence_quint16_qdq_proper.onnx"
```

### Example Output

```
Model: florence_quint16_qdq_proper.onnx
Total time: 12,645.23 ms
Iterations: 20
Average: 632.26 ms
Min: 524.18 ms
Max: 798.42 ms
```

### Performance Testing Best Practices

1. **Thermal Control**: Run tests in isolation with cool-down periods
2. **Multiple Runs**: Use at least 20 iterations (`-r 20`)
3. **Graph Optimization**: Ensure extended optimization is enabled (default with `-o 99`)
4. **System Load**: Close other applications for consistent results
5. **Cool-down**: Wait 30-60 seconds between tests to prevent thermal throttling

## Key Integration Changes

### Files Modified (relative to ORTSOURCEDIR)
1. `onnxruntime/core/mlas/inc/mlas.h` - Added U16U8 GEMM interface
2. `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp` - NEW: MLAS dispatcher implementation
3. `cmake/onnxruntime_mlas.cmake` - Added qgemm_u16u8.cpp to ARM64 builds
4. `onnxruntime/core/providers/cpu/quantization/quantize_linear_matmul.cc` - Added uint16_t kernel registration (opset 10-20 and 21)
5. `onnxruntime/core/providers/cpu/cpu_execution_provider.cc` - Registered uint16_t kernels
6. `onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc` - Enabled 16-bit QDQ fusion

### Kernel Support
- **Opset 10-20**: Versioned kernel for backward compatibility with existing models
- **Opset 21**: Full kernel with scale type constraints
- **Platform**: ARM64 NEON only (Windows ARM64, Linux ARM64)
- **Types**: QUInt16 activations × QUInt8 weights → QUInt16 output

## Architecture

### Three-Layer Integration
1. **MLAS Layer** (`qgemm_u16u8.cpp`)
   - Calls ARM64 NEON assembly kernel `MlasGemmU16U8KernelNeon`
   - Handles row/column sum computation for zero-point corrections
   - Processes matrices in 4x16 tiles (StrideM=4, StrideN=16, PackedK=8)

2. **Operator Layer** (`quantize_linear_matmul.cc`)
   - Detects QUInt16×QUInt8 case via `a->IsDataType<uint16_t>()`
   - Sets up `MLAS_GEMM_U16U8_DATA_PARAMS` structures
   - Handles requantization via `MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR`

3. **Graph Optimization Layer** (`qdq_selector_action_transformer.cc`)
   - Fuses DQ→MatMul→Q patterns into QLinearMatMul at runtime
   - Enabled via `allow_16bit = true` in MatMulQDQRules

## Output Types
- **u8x8 GEMM**: uint8_t activations → uint8_t output
- **i8xi8 GEMM**: int8_t activations → int8_t output
- **u16xu8 GEMM**: uint16_t activations → uint16_t output

**Output type matches activation type, not fixed to 16-bit.**

## Common Issues

### Issue: Unicode encoding error
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```
**Solution**: Replace UTF-8 characters in Python scripts (if needed):
```powershell
cd ${ORTSOURCEDIR}\tools\python\perf_analysis
(Get-Content script_name.py -Encoding UTF8) -replace '✓', '[OK]' | Set-Content script_name.py -Encoding UTF8
```

### Issue: ModuleNotFoundError for onnxruntime.capi
```
ModuleNotFoundError: No module named 'onnxruntime.capi'
```
**Solution**: Python is importing source directory instead of installed package. Run from different directory or remove source from PYTHONPATH.

## Performance Expectations

### Before Integration
- FP32 baseline: ~47.7 ms
- QDQ QUInt16×QUInt8: **123.6 ms** (2.6x slower - using FP32 MatMul!)
- Dynamic QUInt8: 20.7 ms (2.3x faster)

### After Integration (Expected)
- QDQ QUInt16×QUInt8: **~20-30 ms** (similar to Dynamic QUInt8)
- Using direct NEON kernel instead of DQ→FP32 MatMul→Q path

---

### Testing Commands

**Current Working State** (Without fusion):
```powershell
cd ${MODELDIR}
${ORTSOURCEDIR}\.venv_quant_test\Scripts\Activate.ps1

# Quantize model (opset 14 works fine)
python ${ORTSOURCEDIR}\tools\python\perf_analysis\quantize_and_evaluate_with_profiling.py --model model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx --config model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.json

# Test with disabled optimizations (works!)
python ${ORTSOURCEDIR}\tools\python\perf_analysis\test_graph_optimization.py
```

**After Schema Fix** (With fusion):
```powershell
# Same commands, but model should load with ORT_ENABLE_EXTENDED
# and show performance improvement from ~123ms to ~20-30ms
```

## Mixed-Precision Quantization Results (2026-01-06)

### Performance Summary
- **FP32 Baseline**: 78.0 ms
- **QUInt8**: 68.2 ms (1.14x faster than FP32)
- **QUInt16**: 464.3 ms (5.95x slower than FP32)
- **Mixed-Precision (no LUT)**: **66.07 ms** - 🏆 WINNER (1.18x faster than FP32, 1.03x faster than QUInt8)
- **Mixed-Precision (with LUT)**: 77.23 ms (lookup table overhead too high)

### Key Files
- `tools/python/perf_analysis/quantize_mixed_precision.py` - Mixed-precision quantization script
- `MIXED_PRECISION_RESULTS.md` - Detailed analysis and results
- `${MODELDIR}/evaluate_model.py` - Performance evaluation script

---

## TensorQuantOverrides Fix (2026-01-06)

### Problem
Current TensorQuantOverrides implementation for mixed_u16 quantization only finds 73 MatMul activation tensors when it should find approximately 5,000.

**Impact**:
- Only 2,035 QLinearMatMul ops created (expected: ~5,995)
- 3,960 FP32 MatMul fallbacks (expected: 0)
- Performance: 868ms (catastrophic, expected: 100-150ms)
- Accuracy: 3.634e-03 (17% better than QUInt8, maintained after fix)

### Solution
Detailed implementation plan: [TENSORQUANT_OVERRIDES_FIX_PLAN.md](./TENSORQUANT_OVERRIDES_FIX_PLAN.md)

**Two-Stage Quantization Approach**:
1. **Stage 1**: Quantize FP32 model to QUInt8 to create all Q/DQ pairs
2. **Stage 2**: Analyze quantized model to find ALL MatMul activation tensors
3. **Stage 3**: Re-quantize FP32 model with TensorQuantOverrides for those tensors

**Root Cause**: The `find_matmul_activation_tensors()` function searches the FP32 model, but many MatMul patterns only exist after quantization (e.g., shared Q/DQ nodes, fused operations).

**Expected Results After Fix**:
- 5,995 QLinearMatMul ops (all MatMuls converted)
- 0 FP32 MatMul fallbacks
- Performance: 100-150ms (4-5x speedup from current broken state)
- Accuracy: 3.634e-03 maintained (17% better than QUInt8)

---

## Vanilla UInt16 Quantization Performance Investigation (2026-01-06)

### Mission Statement

**PRIMARY GOAL**: Enable vanilla QUInt16×QUInt8 quantized models (as specified in `florence_v1_6_2_d3_tulrv6_multi_text_transformer.json`) to run as fast as FP32 or dynamic quantization on ARM64 CPU **WITHOUT modifying the quantized model itself**.

The model structure is correct, all operations are quantized, and QLinearMatMul kernels are being invoked. The problem is purely a runtime performance issue at the kernel execution or session configuration level.

### Status: Partial Success with NEON Optimization

**Vanilla QUInt16×QUInt8 Quantization Results** (using `quint16_full` config):

| Metric | Original | After NEON Opt | vs FP32 | Status |
|--------|----------|----------------|---------|--------|
| All operations quantized | 11,445 QLinearMatMul ops | 11,445 QLinearMatMul ops | 100% coverage | ✅ SUCCESS |
| FP32 fallbacks | 0 operations | 0 operations | No fallbacks | ✅ SUCCESS |
| **Performance** | **~~422ms~~** | **198ms** | **3.6x slower** | ⚠️ **IMPROVED** |
| **Speedup achieved** | baseline | **2.1x faster** | - | ✅ **Partial success** |
| Accuracy | 5.047e-02 | 5.057e-02 | 10x worse than QUInt8 | ⚠️ Secondary issue |

**FP32 Baseline** (for comparison):
- Performance: 54.9ms (stable measurement)
- Earlier measurements: 86.8ms (with system variance)
- QUInt16 NEON optimized is **3.6x slower** than stable FP32 baseline

**Dynamic Quantization** (best performer):
- Performance: ~25-30ms
- Best accuracy: 1.24e-03 L2 error
- ~2x faster than FP32

### Root Cause and Optimization

**Investigation Results** (see [FINAL_QUANTIZATION_COMPARISON.md](./FINAL_QUANTIZATION_COMPARISON.md) for detailed analysis):

✅ **Confirmed**: NEON kernels ARE being invoked correctly
- All 11,445 QLinearMatMul operations execute correctly
- Correct data types (uint16 × uint8) confirmed
- ARM64 NEON kernel `MlasGemmU16U8KernelNeon` is working

❌ **Bottleneck identified**: Scalar sum computation in dispatcher
- **File**: `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp`
- Per-operation cost: 649,000 scalar additions for zero-point corrections
- Model-wide: 7.4 billion scalar operations
- Accounts for ~200ms of the original 422ms inference time

### Optimization Solution: NEON-Accelerated Row Sum

**Remaining bottlenecks**:
1. Column sum computation: ~100ms (50% of runtime) - strided access prevents NEON optimization
2. ReduceMean operator: ~30ms (15% of runtime) - no optimized uint16 kernel
3. Q/DQ overhead: ~8ms (4% of runtime) - reference implementations

**Path to FP32 parity**: Additional 2x speedup possible through column sum optimization, NEON ReduceMean, and NEON Q/DQ kernels.

See [FINAL_QUANTIZATION_COMPARISON.md](./FINAL_QUANTIZATION_COMPARISON.md) for detailed operator-level analysis and performance breakdowns.

### Summary

**Investigation Outcome**:
- ✅ Confirmed NEON kernels are correctly invoked (no fallbacks)
- ✅ Identified root cause: Scalar sum computation bottleneck (7.4B operations)
- ✅ Implemented NEON row sum optimization: **2.1x speedup** (422ms → 198ms)
- ⚠️ Mission partially achieved: Still 3.6x slower than FP32
- 📊 Remaining bottlenecks identified with clear optimization path

**Key Lesson**: Memory access patterns matter more than SIMD width. Contiguous access + NEON = 8x speedup, but strided access + NEON = performance regression.

**Recommendation**: For production use, prefer **Mixed-Precision** approach (QUInt16 MatMul + QUInt8 everything else) which achieves better performance (66ms) with same accuracy as QUInt8.

**For detailed operator-level analysis, performance breakdowns, and comparison tables**, see [FINAL_QUANTIZATION_COMPARISON.md](./FINAL_QUANTIZATION_COMPARISON.md#quint16-neon-optimization).

### Files and Scripts

- **Diagnostic Script**: `tools/python/perf_analysis/diagnose_uint16_kernel_usage.py`
- **Test Model**: `quantized_models\florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx`
- **Evaluation Script**: `tools/python/perf_analysis/quantize_and_evaluate_with_profiling.py`
- **Dispatcher Code (Modified)**: `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp`
- **NEON Kernel**: `onnxruntime/core/mlas/lib/arm64/QgemmU16U8KernelNeon.asm`

---

## ROOT CAUSE IDENTIFIED (2026-01-06, Updated 2026-01-08)

### Critical Discovery: Scalar Sum Computation Bottleneck

### Why uint8×uint8 is Faster

The QUInt8×QUInt8 implementation likely:
1. Has optimized SIMD sum computation, or
2. Has a dedicated dispatcher and multithreaded handling
3. Has pre-computed sums stored in the model

### The Fix (Partially Implemented - 2026-01-10)

**Vectorize sum computation using ARM64 NEON SIMD instructions**


### Implementation Status

**File**: `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp`

**Completed**:
1. ✅ NEON intrinsics includes (`<arm_neon.h>`) added
2. ✅ Implemented Column-by-Column Packing Layout (BIsPacked)
4. ✅ Correctness maintained for non-multiple-of-8 dimensions (tail handling)

**Testing Completed**:
1. ✅ Numerical correctness verified (sums match scalar version)
2. ✅ Full Florence model evaluation performed
3. ✅ Compared with FP32 and QUInt8 baselines

### Success Metrics

**Target Performance** (after fix):
- QLinearMatMul: < 0.5ms per operation (matching or beating FP32)
- Total inference time: < 100ms (competitive with dynquant)
- **ACHIEVE PRIMARY MISSION**: Vanilla uint16 runs as fast as FP32!

**Stretch Goal**:
- QLinearMatMul: < 0.4ms per operation (better than FP32)
- Total inference time: < 50ms (matching uint8 quantized model)

**Full Analysis**: See `tools/python/perf_analysis/QUINT8_VS_QUINT16_ANALYSISOPTION_C_RESULTS.md`

### Current Status (2026-01-08)

- [x] Implemented NEON column sum computation (4 columns at once) → 127ms → 60-80ms
- [x] **PERFORMANCE RESULT**: 4.72x speedup achieved (598ms → 127ms) ✅
- [ ] **NEXT**: Optional - Implement Option B (vectorize column sums for additional 2x speedup)


**Further Optimization Options**:
2. **Option C** (Optimal, 2-3x): Move column sums to B packing phase → 127ms → 50-60ms (match QUInt8)

**Full Results**: See `tools/python/perf_analysis/TILE_SIZE_OPTIMIZATION_RESULTS.md`

### Next Steps

#### 0. Helper Script for Development Workflow

**Script: `run_quantization.ps1`** - Invoke quantization with proper environment
```powershell
# Usage: .\run_quantization.ps1 -QuantType mixed
param(
    [ValidateSet("fp32", "dynamic", "quint8", "quint16", "mixed")]
    [string]$QuantType = "quint8",
    [switch]$UseLUT
)

$OrtSourceDir = "C:\d\onnxruntime"
$ModelDir = "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer"
$VenvPython = "$OrtSourceDir\.venv_quant_test\Scripts\python.exe"
$ScriptDir = "$OrtSourceDir\tools\python\perf_analysis"

cd $ModelDir

switch ($QuantType) {
    "mixed" {
        $Args = "--model model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx --output model_mixed_precision.onnx"
        if ($UseLUT) {
            $Args += " --use-lookup-table"
        }
        & $VenvPython "$ScriptDir\quantize_mixed_precision.py" $Args
    }
    default {
        & $VenvPython "$ScriptDir\quantize_and_evaluate_with_profiling.py" --model model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx --config model/florence_v1_6_2_d3_tulrv6_multi_text_transformer.json --quant-type $QuantType
    }
}
```

**Note**: Scripts are now located in `tools/python/perf_analysis/` and can be invoked directly without copying.

#### 1. Code Review: Shared Code Analysis
Use the code review agent to analyze common code patterns between:
- `tools/python/perf_analysis/quantize_and_evaluate_with_profiling.py`
- `tools/python/perf_analysis/quantize_mixed_precision.py`

Focus on:
- DummyCalibrationDataReader class (duplicated)
- Model loading and session setup patterns
- Profiling and evaluation code
- Quantization configuration handling

#### 2. Script Consolidation
Combine both scripts into a unified `quantize_and_evaluate_with_profiling.py` with:
- Single DummyCalibrationDataReader implementation
- Unified quantization interface supporting all types:
  - `--quant-type fp32` - Run FP32 baseline
  - `--quant-type dynamic` - Dynamic quantization
  - `--quant-type quint8` - Static QUInt8×QUInt8
  - `--quant-type quint16` - Static QUInt16×QUInt8
  - `--quant-type mixed` - Mixed-precision (QUInt16 MatMul + QUInt8 everything else)
- Optional flags:
  - `--use-lookup-table` - Enable LUT optimization for mixed-precision
  - `--num-runs` - Number of benchmark runs
  - `--warmup-runs` - Number of warmup runs
- Integrated profiling and evaluation

### Remaining Bottlenecks

**2. ReduceMean Operator (~30ms, 15% of runtime)**
- **Problem**: No optimized uint16 kernel exists
- **Solution**: Implement NEON-accelerated ReduceMean for uint16 data type

**3. Q/DQ Overhead (~8ms, 4% of runtime)**
- **Problem**: uint16 quantization/dequantization uses reference implementation
- **Solution**: Implement NEON-accelerated Q/DQ kernels for uint16

### Lessons Learned

1. **Memory Access Patterns Matter More Than SIMD Width**
   - Contiguous access + NEON = 8x speedup ✅
   - Strided access + NEON = performance regression ❌
   - Cache-friendly algorithms > raw vectorization

2. **Profile Before Optimizing**
   - Originally thought NEON kernel wasn't invoked
   - Profiling revealed scalar sums were the bottleneck
   - Targeted optimization delivered 2x improvement

3. **Test Incrementally**
   - Broken column sum NEON made things worse
   - Incremental testing caught the regression
   - Row sum only = safe partial optimization

### Current Status

**PRIMARY MISSION**: Enable vanilla QDQ model with QUInt16×QUInt8 to **meet or outperform FP32 performance** on CPU without modifying the model, stretch goal: meet uint8 quantization performance (<50ms).

**Current Performance**:
- FP32 Baseline: **54.9ms**
- Vanilla QUInt16 (NEON optimized): **127ms** (3.6x slower)

---

## Next Targets: Achieving FP32 Parity

### Analysis Framework

**Tool**: `analyze_u16u8_model_nodes.py`

Comprehensive model analysis script that:
1. Maps all nodes with input/output types and counts occurrences
2. Identifies operations that would benefit most from optimization
3. Detects Q/DQ and DQ/Q roundtrip patterns that could be elided
4. Analyzes accuracy-limiting factors and generates prioritized optimization plan

**Usage**:
```bash
cd C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer
C:\d\onnxruntime\.venv_quant_test\Scripts\python.exe C:\d\onnxruntime\analyze_u16u8_model_nodes.py
```

**Status**: ✅ **COMPLETED** - Analysis run on 2026-01-07

### Analysis Results (2026-01-07)

**Model**: `florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_full_qdq.onnx`

**Summary Statistics**:
- Total nodes: 2,289
- Unique node configurations: 25
- Analysis output: `quantized_models/u16u8_model_analysis.txt`

**Top Node Configurations**:

| Configuration | Count | Impact |
|---------------|-------|--------|
| QuantizeLinear(float32→uint16) | 650 | 11.6% runtime (MEDIUM priority) |
| DequantizeLinear(unknown→float32) | 650 | 6.3% runtime (MEDIUM priority) |
| Add(unknown→float32) | 195 | Operations after DQ |
| DequantizeLinear(uint16→float32) | 141 | Additional DQ overhead |
| **MatMul(unknown→float32)** | **109** | **Fused to QLinearMatMul at runtime** |
| Mul(unknown→float32) | 87 | Operations after DQ |
| DequantizeLinear(uint8→float32) | 85 | Weight DQ operations |

**✅ DISCOVERY: MatMul Fusion Status Verified**

The analysis revealed **109 MatMul operations** in the quantized QDQ model file. Follow-up investigation confirmed:

**QDQ Model File** (before optimization):
- 109 MatMul operations with DQ→MatMul→Q patterns
- All 109 have correct structure for fusion (verified by `investigate_unfused_matmuls.py`)

**Runtime Graph** (after ORT_ENABLE_EXTENDED optimization):
- ✅ All 109 MatMuls successfully fused to QLinearMatMul
- ✅ 34% reduction in Q/DQ node count (1,754 → 1,159 nodes)
- ✅ uint16 NEON kernels ARE being invoked correctly

**Implication**: Performance issue (198ms vs 54.9ms target) is NOT due to fusion failure or FP32 fallbacks. It's due to kernel performance bottlenecks:
- Column sum computation: ~100ms (50% of runtime)
- ReduceMean uint16: ~30ms (15% of runtime)
- Q/DQ operations: ~8ms (4% of runtime)

**Q/DQ Roundtrip Patterns**:
- Found **650 Q→DQ roundtrip patterns**
- Opportunity: Eliminate both Q and DQ operations through direct type conversion
- Potential saving: ~10-15ms if eliminated

**Prioritized Optimization Targets** (from analysis):
1. **Priority 1**: NEON ReduceMean for uint16 (~30ms improvement)
2. **Priority 2**: Column sum optimization in GEMM (~50-70ms improvement)
3. **Priority 3**: NEON Q/DQ kernels (~8-15ms improvement)
4. **Priority 4**: Elide 650 Q→DQ roundtrips
5. **Priority 5**: Optimize quantization parameters for accuracy
6. **Priority 6**: Implement per-channel quantization

**Status**: ✅ **RESOLVED** - Fusion is working correctly!

### Priority 1: Performance Optimizations (Target: Match FP32 at 54.9ms)

#### 1.1 Implement NEON-Optimized ReduceMean for uint16

**Implementation approach**:
- **Option A**: Transpose B matrix once (one-time cost, makes columns contiguous)
- **Option B**: Compute column sums inside GEMM kernel (amortize cost)
- **Option C**: Cache column sums for repeated GEMM operations
- **Recommended**: Option B (most efficient, no extra memory)

#### 1.3 Implement NEON Q/DQ Kernels for uint16
**Rationale**: Combined 18% of original runtime (~8ms in optimized build)
- Current: Reference implementations for QuantizeLinear/DequantizeLinear
- Target: NEON-accelerated quantization/dequantization
- **Estimated Impact**: 5-8ms improvement
- **Complexity**: Low-Medium
- **Files**: `onnxruntime/core/providers/cpu/math/quantize_linear.cc`

**Implementation approach**:
- Use NEON for parallel scale/zero-point operations
- Vectorize clamping and rounding operations
- Handle both scalar and per-channel quantization

**Combined Estimated Impact**: 75-103ms improvement → **Target: 93-123ms total** (within range of FP32!)

**Full Analysis**: See `tools/python/perf_analysis/OPTION_C_RESULTS.md`

** Next steps: **

  To achieve the target <100ms performance, we could proceed with Option A (threading + dispatcher integration):
  - Expected additional speedup: 1.4-1.6x (163ms → 90-100ms)
  - Benefits: Multi-core utilization, standard MLAS infrastructure
  - Timeline: 4-5 days


### Priority 2: Graph Optimizations (Reduce Q/DQ Overhead)

#### 2.1 Detect and Elide Q/DQ Roundtrips
**Rationale**: Reduce unnecessary quantization/dequantization cycles
- **Pattern 1**: DQ → Q (dequantize then immediately quantize)
  - Opportunity: Direct uint→uint cast or bypass
- **Pattern 2**: Q → DQ (quantize then immediately dequantize)
  - Opportunity: Direct type conversion or bypass
- **Pattern 3**: DQ → Op → Q (unsupported quantized ops)
  - Opportunity: Implement native quantized operator

**Estimated Impact**: Eliminate redundant Q/DQ pairs (identified by analysis script)
**Complexity**: Medium
**Files**: `onnxruntime/core/optimizer/qdq_transformer/*`

#### 2.2 Operator Fusion for uint16 Operations
**Rationale**: Reduce Q/DQ overhead through graph transformations
- Identify common DQ → Op → Q patterns
- Implement fused quantized operators where beneficial
- Focus on high-frequency operations (Add, Mul, Softmax with uint16)

**Estimated Impact**: Variable (depends on pattern frequency)
**Complexity**: Medium-High
**Files**: `onnxruntime/core/optimizer/qdq_transformer/selectors_actions/*`


**Phase 2: Accuracy Improvements** (Estimated: 1-2 weeks) --> see accuracy_improvement_plan.md

**Success Criteria**:
- ✅ **Performance**: Meet or outperform FP32 (≤54.9ms)
- ✅ **Accuracy**: Match QUInt8 accuracy (≤4.35e-03 L2 error)
- ✅ **Compatibility**: Vanilla QDQ model without modifications, tested on embimg_c.quant.onnx

### Tracking and Documentation

**Progress Tracking**: Update this section with completion status
**Detailed Results**: Document in [FINAL_QUANTIZATION_COMPARISON.md](./FINAL_QUANTIZATION_COMPARISON.md)
**Analysis Results**: Save output from `analyze_u16u8_model_nodes.py` to track improvement

---

## Optimization Progress Summary (2026-01-07)

### Completed Optimizations ✅

**1. NEON Row Sum Optimization** (2026-01-06)
**2. NEON ReduceMean Optimization** (2026-01-07)
**Details**: See [FINAL_QUANTIZATION_COMPARISON.md](./FINAL_QUANTIZATION_COMPARISON.md#neon-reducemean-implementation-2026-01-07)

#### 3. NEON DequantizeLinear Optimization (2026-01-07)
**Status**: ✅ **COMPLETED**
- **Impact**: DequantizeLinear reduced from reference implementation to NEON-optimized
- **Files**: `onnxruntime/core/mlas/lib/dequantize.cpp` (lines 329-412)
