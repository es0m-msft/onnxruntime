# File Organization - Performance Analysis Tools

**Date Reorganized**: 2026-01-09

This directory contains all performance analysis tools, scripts, and documentation for QUInt16×QUInt8 quantization optimization work.

---

## Directory Structure

```
tools/python/perf_analysis/
├── README.md                               # Main documentation (comprehensive guide)
├── FILE_ORGANIZATION.md                    # This file
│
├── 📊 DOCUMENTATION (Performance Analysis Results)
│   ├── OPTIMIZATION_SUMMARY.md             # Complete optimization journey (598ms → 127ms)
│   ├── QUINT8_VS_QUINT16_ANALYSIS.md      # Root cause analysis (300+ lines)
│   ├── TILE_SIZE_OPTIMIZATION_RESULTS.md  # Option A implementation results
│   └── PERF_TEST_RESULTS.md               # Official benchmark comparisons
│
├── 🔧 BENCHMARKING SCRIPTS
│   ├── run_perf_test_comparison.ps1       # PowerShell: Official perf_test runner
│   ├── run_thermal_controlled_benchmark.ps1 # PowerShell: Thermal-controlled tests
│   ├── benchmark_all_quantizations.py      # Python: Compare all quant types
│   └── benchmark_isolated.py               # Python: Isolated process benchmarks
│
├── 🎯 QUANTIZATION SCRIPTS
│   ├── quantize_and_evaluate_with_profiling.py  # Enhanced with profiling
│   ├── quantize_and_evaluate.py            # Basic quantization
│   ├── quantize_mixed_precision.py         # Mixed-precision (QU16+QU8)
│   ├── quantize_quint8.py                  # Pure QUInt8
│   └── quantize_florence_properly.py       # Florence-specific
│
├── 🔍 MODEL ANALYSIS SCRIPTS
│   ├── analyze_u16u8_model_nodes.py        # Model structure analysis
│   ├── investigate_unfused_matmuls.py      # Fusion analysis
│   ├── diagnose_uint16_kernel_usage.py     # Kernel verification
│   ├── profile_runtime_graph.py            # Runtime profiling
│   └── identify_ort_elision_points.py      # Q/DQ elision opportunities
│
├── 🧪 TESTING SCRIPTS
│   └── test_graph_optimization.py          # Graph optimization verification
│
└── 🔧 DEPRECATED (Schema Patches - No Longer Needed)
    ├── patch_onnx_schema.py
    ├── patch_onnx_schema_old.py
    └── patch_qdq_opset13.py
```

---

## Files Moved (2026-01-09)

### From `C:\d\onnxruntime\runtime_analysis\` → `tools/python/perf_analysis/`

**Documentation**:
- ✅ `PERF_TEST_RESULTS.md`
- ✅ `QUINT8_VS_QUINT16_ANALYSIS.md`
- ✅ `TILE_SIZE_OPTIMIZATION_RESULTS.md`
- ✅ `OPTIMIZATION_SUMMARY.md`

### From `C:\d\onnxruntime\` (root) → `tools/python/perf_analysis/`

**Scripts**:
- ✅ `run_perf_test_comparison.ps1`
- ✅ `run_thermal_controlled_benchmark.ps1`
- ✅ `quantize_quint8.py`
- ✅ `quantize_florence_properly.py`
- ✅ `benchmark_all_quantizations.py`
- ✅ `benchmark_isolated.py`

---

## Why This Organization?

### Before (Scattered)
```
C:\d\onnxruntime\
├── quantize_quint8.py                    # Root directory (messy)
├── run_perf_test_comparison.ps1          # Root directory (messy)
├── benchmark_all_quantizations.py        # Root directory (messy)
├── runtime_analysis\
│   ├── PERF_TEST_RESULTS.md              # Separate analysis directory
│   └── OPTIMIZATION_SUMMARY.md           # Separate analysis directory
└── tools\python\perf_analysis\
    ├── quantize_and_evaluate.py          # Some tools here
    └── diagnose_uint16_kernel_usage.py   # Some tools here
```

### After (Organized)
```
C:\d\onnxruntime\
└── tools\python\perf_analysis\
    ├── *.md                              # All documentation in one place
    ├── *.py                              # All Python scripts together
    └── *.ps1                             # All PowerShell scripts together
```

### Benefits

1. **Single Source of Truth**: All performance analysis tools in one directory
2. **Discoverability**: Easy to find related scripts and documentation
3. **Maintainability**: Clear ownership and organization
4. **Consistency**: Follows ORT tools organization pattern
5. **Documentation**: Comprehensive README with usage examples

---

## Quick Access

### Most Used Files

**For New Users**:
1. Start with [README.md](./README.md) - comprehensive guide
2. Read [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md) - complete story
3. Review [QUINT8_VS_QUINT16_ANALYSIS.md](./QUINT8_VS_QUINT16_ANALYSIS.md) - technical deep dive

**For Benchmarking**:
1. `run_perf_test_comparison.ps1` - Official tool comparison
2. `benchmark_all_quantizations.py` - Python-based comparison

**For Quantization**:
1. `quantize_and_evaluate_with_profiling.py` - Primary tool (with profiling)
2. `quantize_quint8.py` - QUInt8 only
3. `quantize_mixed_precision.py` - Mixed precision

**For Analysis**:
1. `analyze_u16u8_model_nodes.py` - Model structure
2. `diagnose_uint16_kernel_usage.py` - Kernel verification

---

## Related Directories

### Core Code
- `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp` - MLAS dispatcher (tile size modified here)
- `onnxruntime/core/mlas/lib/arm64/QgemmU16U8KernelNeon.asm` - NEON kernel

### Build Outputs
- `build_arm64_u16u8/Release/Release/` - Compiled binaries
- `build_arm64_u16u8/Release/Release/dist/` - Python wheel

### Quantized Models
- `quantized_models/` - Generated quantized models
  - `florence_quint8_qdq.onnx` - QUInt8 model
  - `florence_quint16_qdq_proper.onnx` - QUInt16 model

### Test Results (Historical)
- `runtime_analysis/` - Older analysis results (kept for reference)
  - Contains earlier investigation documents
  - Historical benchmark data
  - Large ONNX models from intermediate tests

---

## Migration Notes

### For Scripts Referencing Old Paths

If you have scripts referencing the old locations, update them:

**Old**:
```python
from C:\d\onnxruntime\quantize_quint8 import ...
```

**New**:
```python
from tools.python.perf_analysis.quantize_quint8 import ...
```

**Old** (PowerShell):
```powershell
.\run_perf_test_comparison.ps1
```

**New** (PowerShell):
```powershell
cd tools\python\perf_analysis
.\run_perf_test_comparison.ps1
```

### Documentation References Updated

The following files have been updated to reference new paths:
- ✅ `CLAUDE.md` - Main integration guide
- ✅ `tools/python/perf_analysis/README.md` - This directory's main doc
- ✅ All `.md` files in this directory (cross-references)

---

## Maintenance

### Adding New Files

When adding new scripts or documentation:

1. **Place in correct category**:
   - Documentation: `*.md` files
   - Python scripts: `*.py` files
   - PowerShell scripts: `*.ps1` files

2. **Update README.md**:
   - Add description under appropriate section
   - Include usage example
   - Update quick start if relevant

3. **Follow naming conventions**:
   - Python: `<verb>_<noun>.py` (e.g., `quantize_model.py`)
   - PowerShell: `<action>_<target>.ps1` (e.g., `run_benchmark.ps1`)
   - Documentation: `<TOPIC>_<TYPE>.md` (e.g., `PERFORMANCE_RESULTS.md`)

### Deprecating Files

When deprecating scripts:
1. Add "DEPRECATED" to filename or move to `deprecated/` subdirectory
2. Update README.md to mark as deprecated
3. Document replacement in deprecation notice

---

## Summary

All performance analysis tools are now organized in `tools/python/perf_analysis/`:
- ✅ **4 documentation files** (complete performance story)
- ✅ **16 Python scripts** (quantization, analysis, testing)
- ✅ **2 PowerShell scripts** (benchmarking)
- ✅ **1 comprehensive README** (usage guide)

**Everything in one place, easy to find, well-documented!**
