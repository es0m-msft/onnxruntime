# Option A Implementation Plan: QUInt16 Dispatcher Integration

**Date**: 2026-01-10
**Status**: In Progress - Day 1
**Goal**: Integrate QUInt16 with standard MLAS dispatcher infrastructure for 2-3x performance improvement

---

## Current Architecture Analysis

### Discovery: Template Infrastructure Already Exists!

The QUInt16 implementation has **two parallel code paths**:

#### Path 1: Template-Based Infrastructure (EXISTS but UNUSED)
**File**: `qgemm_kernel_neon_u16u8.cpp`

```cpp
struct MLAS_GEMM_U16U8_KERNEL_NEON { ... };  // ✅ Kernel structure
const MLAS_GEMM_QUANT_DISPATCH MlasGemmU16U8DispatchNeon = {
    MlasGemmQuantOperation<MLAS_GEMM_U16U8_KERNEL_NEON>,       // ✅ Template operation
    MlasGemmQuantPackedOperation<MLAS_GEMM_U16U8_KERNEL_NEON>, // ✅ Packed operation
    MlasGemmQuantCopyPackB<MLAS_GEMM_U16U8_KERNEL_NEON>,       // ❌ SCALAR column sums
    ...
};
```

**Problem**: `MlasGemmQuantCopyPackA/B` use **scalar loops** for sum computation:
- Row sums (qgemm_kernel_neon_u16u8.cpp:150-153): SCALAR
- Column sums (qgemm_kernel_neon_u16u8.cpp:210-213): SCALAR

#### Path 2: Custom Bypass (CURRENTLY USED)
**File**: `qgemm_u16u8.cpp`

```cpp
void MlasGemmU16U8Batch(...) {
    // ❌ No threading - ThreadPool parameter ignored (line 214)

    // ✅ Row sums: NEON-optimized
    for (size_t m = 0; m < M; m++) {
        AllRowSums[m] = ComputeRowSumNeon(A + m * lda, K);  // Lines 255-260
    }

    // ❌ Column sums: SCALAR BOTTLENECK
    for (size_t n = 0; n < N; n++) {
        int32_t col_sum = 0;
        for (size_t k = 0; k < K; k++) {
            col_sum += static_cast<int32_t>(B[k * ldb + n]);  // Lines 271-278
        }
        AllColumnSums[n] = col_sum;
    }

    // Custom tile loop - bypasses standard dispatcher
    for (size_t m = 0; m < M; m += StrideM) { ... }
}
```

**Why this is used**: `quantize_linear_matmul.cc` explicitly calls the custom path:
```cpp
if (is_u16u8) {
    MlasGemmU16U8Batch(Shape, DataParams, BatchN, threadpool);  // CUSTOM!
} else {
    MlasGemmBatch(Shape, DataParams, BatchN, threadpool);       // STANDARD
}
```

---

## Performance Impact

### Current Bottlenecks

| Bottleneck | Impact | Status |
|------------|--------|--------|
| **No Threading** | 1.5-2x slowdown | ThreadPool ignored |
| **Scalar Column Sums** | 30-50ms overhead | 589,824 scalar ops per MatMul |
| **Custom Tile Loop** | Missing optimizations | Bypasses standard dispatcher |

**Typical dimensions**: M=77, K=768, N=768
- Column sums: 768 × 768 = 589,824 scalar additions **per MatMul**
- Florence model: 145 MatMuls × 589,824 = **85.5 million scalar operations**

### Target Performance (After Option A)

- **Current**: 234-266ms average (fixed model), 79-80% CPU
- **Target**: <150ms average, >85% CPU
- **Expected**: 2-3x speedup from threading + column sum optimization

---

## Implementation Plan

### Day 1: Optimize Template-Based Sum Computation ⏳ IN PROGRESS

**Files to modify**: `onnxruntime/core/mlas/lib/qgemm_kernel_neon_u16u8.cpp`

#### Task 1.1: Add NEON Row Sum Computation

**Current** (lines 119-166):
```cpp
template<>
void MlasGemmQuantCopyPackA<MLAS_GEMM_U16U8_KERNEL_NEON>(...) {
    for (size_t m = 0; m < CountM; m++) {
        int32_t RowSum = 0;
        for (k = 0; k < CountK; k++) {
            uint16_t a0 = A16[k] ^ BitFlipValue;
            D[k] = a0;
            RowSum += a0;  // ❌ SCALAR!
        }
        *RowSumBuffer++ = RowSum;
    }
}
```

**Target** (based on qgemm_u16u8.cpp:50-100 `ComputeRowSumNeon`):
```cpp
// Process 8 uint16 elements at a time
uint32x4_t sum_vec_lo = vdupq_n_u32(0);
uint32x4_t sum_vec_hi = vdupq_n_u32(0);

for (; k + 8 <= CountK; k += 8) {
    uint16x8_t data = vld1q_u16(A16 + k);

    // Apply bit flip if signed
    if (AIsSigned) {
        data = veorq_u16(data, vdupq_n_u16(0x8000));
    }

    // Store packed data
    vst1q_u16(D + k, data);

    // Widen to uint32 and accumulate
    uint32x4_t data_lo = vmovl_u16(vget_low_u16(data));
    uint32x4_t data_hi = vmovl_u16(vget_high_u16(data));
    sum_vec_lo = vaddq_u32(sum_vec_lo, data_lo);
    sum_vec_hi = vaddq_u32(sum_vec_hi, data_hi);
}

// Horizontal sum + handle remaining elements
```

**Expected speedup**: 8x faster row sum computation

#### Task 1.2: Add NEON Column Sum Computation

**Current** (lines 178-227):
```cpp
template<>
void MlasGemmQuantCopyPackB<MLAS_GEMM_U16U8_KERNEL_NEON>(...) {
    for (size_t n = 0; n < CountN; n++) {
        int32_t ColumnSum = 0;
        for (k = 0; k < CountK; k++) {
            uint8_t b0 = b[0] ^ BitFlipValue;
            D[k] = b0;
            ColumnSum += b0;  // ❌ SCALAR!
            b += ldb;
        }
        *ColumnSumBuffer++ = ColumnSum;
    }
}
```

**Target** (based on QUInt8's `MlasGemmU8X8CopyPackBProcessNeon` pattern):
```cpp
// Process 8 uint8 elements at a time
uint32x4_t ColumnSums[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

for (k = 0; k < CountK; k++) {
    uint8x8_t BytesRow = veor_u8(vld1_u8(b), BitFlipVector);
    vst1_u8(D + k * 8, BytesRow);

    // Widen uint8 -> uint16 -> uint32 and accumulate
    uint16x8_t WordsRow = vmovl_u8(BytesRow);
    ColumnSums[0] = vaddq_u32(ColumnSums[0], vmovl_u16(vget_low_u16(WordsRow)));
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_high_u16(WordsRow));

    b += ldb;
}

// Store column sums
vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
```

**Expected speedup**: 8-16x faster column sum computation

**Note**: Column data is strided (b += ldb), but NEON still provides 4-8x speedup over scalar.

---

### Day 2: Register Dispatcher and Switch Paths

#### Task 2.1: Verify Dispatcher Registration

**File**: `onnxruntime/core/mlas/lib/platform.cpp`

Check if `MlasGemmU16U8DispatchNeon` is registered:
```cpp
// Look for ARM64 platform initialization
#if defined(MLAS_TARGET_ARM64)
    // Check if U16U8 dispatcher is registered in GemmU16U8Dispatch
    // If not, add registration
#endif
```

**Expected**: Dispatcher might already be registered but unused.

#### Task 2.2: Switch to Standard Dispatcher

**File**: `onnxruntime/core/providers/cpu/quantization/quantize_linear_matmul.cc`

**Current** (lines 127-203):
```cpp
if (is_u16u8) {
    // U16U8 custom path
    MlasGemmU16U8Batch(Shape, DataParams, BatchN, threadpool);
} else {
    // Standard path
    MlasGemmBatch(Shape, DataParams, BatchN, threadpool);
}
```

**Target**:
```cpp
// Use standard dispatcher for ALL quantization types
MlasGemmBatch(Shape, DataParams, BatchN, threadpool);
```

**This change automatically enables**:
- ✅ Threading via `MlasTrySimpleParallel`
- ✅ SIMD column sums (from Day 1 changes)
- ✅ Standard work partitioning
- ✅ All future optimizations to standard dispatcher

---

### Day 3: Cleanup and Validation

#### Task 3.1: Delete Custom Bypass

**File**: `onnxruntime/core/mlas/lib/qgemm_u16u8.cpp`

**Action**: Delete entire file (no longer needed)

**Files to update**:
- `cmake/onnxruntime_mlas.cmake` - Remove qgemm_u16u8.cpp from build
- `onnxruntime/core/mlas/inc/mlas.h` - Remove MlasGemmU16U8Batch declaration

#### Task 3.2: Build and Test

```powershell
# Rebuild C++ code
cd C:\d\onnxruntime\build_arm64_u16u8\Release
cmake --build . --config Release --parallel

# Rebuild Python wheel
cd Release
python ../../../setup.py bdist_wheel

# Install in venv
C:\d\onnxruntime\.venv_quant_test\Scripts\python.exe -m pip install --force-reinstall dist\onnxruntime-*.whl
```

#### Task 3.3: Validate Correctness

```powershell
# Run fixed model (should still work correctly)
C:\d\onnxruntime\build_arm64_u16u8\Release\Release\onnxruntime_perf_test.exe `
    -I "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\embimg_c.quant.fixed2.onnx" `
    -m times -r 5
```

**Expected**: No crashes, numerical results match previous runs

---

### Day 4: Performance Testing

#### Task 4.1: Benchmark Single Model

```powershell
# Quick test (5 runs)
C:\d\onnxruntime\test_changes.ps1 -Runs 5
```

**Expected metrics**:
- Latency: **<150ms** (from 234ms baseline) - **1.6x speedup minimum**
- CPU Usage: **>85%** (from 79% baseline)

#### Task 4.2: Comprehensive Comparison

```powershell
# Full benchmark (20 runs)
C:\d\onnxruntime\tools\python\perf_analysis\compare_models.ps1 -Runs 20
```

**Target metrics**:

| Metric | Before (Baseline) | After (Option A) | Improvement |
|--------|-------------------|------------------|-------------|
| **Latency** | 234ms | <150ms | 1.6x faster |
| **CPU Usage** | 79% | >85% | Better utilization |
| **QLinearMatMul/op** | ~1.6ms | <1.0ms | 1.6x faster |

**Stretch goal**: <100ms latency (2.3x speedup)

---

### Day 5: Documentation and Validation

#### Task 5.1: Update Documentation

**Files to update**:
1. `CLAUDE.md` - Update with Option A completion status
2. `QUINT8_VS_QUINT16_ARCHITECTURE.md` - Add "After Option A" section
3. `tools/python/perf_analysis/OPTIMIZATION_SUMMARY.md` - Document changes

#### Task 5.2: Validate Against QUInt8 Baseline

```bash
# Compare QUInt16 (after Option A) vs QUInt8 baseline
# Target: Within 2-3x of QUInt8 (theoretical limit due to 2x data width)
```

**QUInt8 baseline**: 55-68ms
**Theoretical QUInt16 minimum**: 110-136ms (2x due to uint16 vs uint8)
**Actual target**: <150ms (within acceptable range)

---

## Success Criteria

### Phase 1: Implementation Complete ✅ when:
- ✅ NEON row sums implemented in CopyPackA
- ✅ NEON column sums implemented in CopyPackB
- ✅ Standard dispatcher path used (qgemm_u16u8.cpp deleted)
- ✅ Code compiles without errors
- ✅ Model runs without crashes

### Phase 2: Performance Validated ✅ when:
- ✅ Latency <150ms (1.6x improvement minimum)
- ✅ CPU usage >85%
- ✅ All correctness tests pass
- ✅ Performance stable across multiple runs

### Phase 3: Production Ready ✅ when:
- ✅ Documentation updated
- ✅ Benchmarks published
- ✅ No regressions in existing tests
- ✅ Code reviewed and merged

---

## Risk Mitigation

### Risk 1: NEON Implementation Bugs
**Mitigation**:
- Copy proven patterns from QUInt8 implementation
- Add validation that NEON sums match scalar reference
- Test with multiple model sizes

### Risk 2: Performance Regression
**Mitigation**:
- Keep qgemm_u16u8.cpp temporarily for comparison
- Benchmark before and after each change
- Have rollback plan if performance degrades

### Risk 3: Threading Issues
**Mitigation**:
- Standard dispatcher is battle-tested (used by QUInt8)
- Test with different thread counts
- Monitor for race conditions

---

## Notes

### Why Column Sum Optimization Matters

**Typical Florence MatMul**: M=77, K=768, N=768
- Column sums: 768 columns × 768 elements = 589,824 operations **per MatMul**
- 145 MatMuls in model: 589,824 × 145 = **85.5 million operations**
- Current: Scalar code ~8-16 cycles/operation = **680-1360 million cycles**
- With NEON: ~1-2 cycles/operation = **85-170 million cycles**
- **Savings: 500-1200 million cycles = 166-400ms @ 3GHz**

This alone could achieve the target speedup!

### Why Threading Matters

**Current**: Single-threaded execution on 8-core CPU
- CPU usage: 79% (one core maxed, others idle)
- Effective performance: 1/8 of potential

**After threading**:
- CPU usage: >90% (all cores utilized)
- Expected speedup: 1.5-2x from parallelization
- Work partitioned across M and N dimensions

### Combined Expected Speedup

- Column sum optimization: 1.5-2x
- Threading: 1.5-2x
- **Combined: 2.25-4x speedup**
- **Target: 234ms → 58-104ms**
- **Conservative estimate: <150ms (1.6x)**

---

## File Reference

### Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `qgemm_kernel_neon_u16u8.cpp` | Add NEON row sums | 119-166 |
| `qgemm_kernel_neon_u16u8.cpp` | Add NEON column sums | 178-227 |
| `quantize_linear_matmul.cc` | Use standard dispatcher | 127-203 |
| `platform.cpp` | Verify dispatcher registration | TBD |
| `cmake/onnxruntime_mlas.cmake` | Remove qgemm_u16u8.cpp | TBD |
| `mlas.h` | Remove MlasGemmU16U8Batch | TBD |

### Files to Delete

| File | Reason |
|------|--------|
| `qgemm_u16u8.cpp` | Custom bypass no longer needed |

### Reference Files (Don't Modify)

| File | Purpose |
|------|---------|
| `qgemm_kernel_neon.cpp` | QUInt8 reference for NEON patterns |
| `qgemm.h` | Template infrastructure (already works) |
| `qgemm.cpp` | Standard dispatcher (already works) |

---

## Current Status

- [x] Investigation phase complete
- [x] Architecture analysis complete
- [x] Implementation plan written
- [ ] **⏳ IN PROGRESS: Day 1 - NEON sum optimization**
- [ ] Day 2 - Dispatcher registration
- [ ] Day 3 - Cleanup and validation
- [ ] Day 4 - Performance testing
- [ ] Day 5 - Documentation

**Next step**: Implement NEON row sum computation in MlasGemmQuantCopyPackA
