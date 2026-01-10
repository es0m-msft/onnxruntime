# Next Optimization Strategy: Beyond 117.6ms

**Date**: 2026-01-10
**Current Performance**: 117.6ms average, 85% CPU, 2.25x speedup
**Target**: 100-110ms (approaching theoretical 2x limit of QUInt8)

---

## Current State Summary

### Completed Optimizations ✅

| Optimization | Performance | CPU | Improvement | Status |
|-------------|-------------|-----|-------------|--------|
| **Baseline** | 265ms | 79% | - | - |
| **Option C** (B Matrix Packing) | 163ms | 77-80% | 1.6x | ✅ Complete |
| **Option A** (Threading) | 132ms | 78% | 2.0x | ✅ Complete |
| **Option 2** (PrePack API) | **117.6ms** | **85%** | **2.25x** | ✅ **Current** |
| **Option 1** (M×N Grid) | 147.5ms | 78% | 1.8x | ❌ Rejected |

### Theoretical Limit

**QUInt8 Performance**: 55-68ms
**Theoretical QUInt16 Minimum**: 110-136ms (2x slower due to 2x data width)
**Current**: 117.6ms
**Distance to Theoretical**: **7.6ms (6.5% above minimum)**

We're already very close to optimal! Further gains will be incremental.

---

## Rejected Optimizations

### ❌ Option 1: N-Dimension Partitioning
**Why Rejected**: 25% regression due to cache locality destruction
**Lesson**: For row-major matrices, NEVER partition along N-dimension

---

## Remaining Optimization Opportunities

### Priority 1: Tile Size Tuning (LOW EFFORT, MEDIUM IMPACT)

**Current Tile Sizes**:
```cpp
constexpr size_t StrideM = 24;  // Process 24 rows at a time
constexpr size_t StrideN = 128; // Process 128 columns at a time
```

**Hypothesis**: These values may not be optimal for ARM64 cache hierarchy.

**ARM64 Cache Hierarchy** (typical):
- L1 Data Cache: 64 KB per core
- L2 Cache: 512 KB - 2 MB per core
- L3 Cache: 4-32 MB shared

**Current Tile Memory Footprint**:
- Tile A: 24 × K × 2 bytes (uint16) = 24 × 768 × 2 = 36,864 bytes
- Tile B (packed): 128 × K × 1 byte (uint8) = 128 × 768 = 98,304 bytes
- Tile C: 24 × 128 × 4 bytes (int32) = 12,288 bytes
- **Total per tile: ~147 KB**

This exceeds L1 cache (64 KB) but fits in L2 cache (512 KB+).

**Optimization Strategy**:

1. **Experiment with smaller tiles** to fit in L1 cache:
   - Option A: StrideM=16, StrideN=96
     - Tile A: 24,576 bytes
     - Tile B: 73,728 bytes
     - Tile C: 6,144 bytes
     - **Total: ~104 KB** (still exceeds L1)

   - Option B: StrideM=12, StrideN=64
     - Tile A: 18,432 bytes
     - Tile B: 49,152 bytes
     - Tile C: 3,072 bytes
     - **Total: ~71 KB** (fits in L1!)

2. **Experiment with larger tiles** for better amortization:
   - Option C: StrideM=32, StrideN=128
   - Option D: StrideM=24, StrideN=256

**Expected Impact**: 5-10% improvement (117ms → 105-112ms)

**Implementation Effort**: LOW (simple constant changes)

**Testing Required**:
- Test each configuration with 20 runs
- Use thermal cooldown between tests
- Measure both average and P90 latency

---

### Priority 2: Prefetching Optimization (MEDIUM EFFORT, LOW-MEDIUM IMPACT)

**Current**: No explicit software prefetching
**Opportunity**: Prefetch next tile data while processing current tile

**Strategy**:

1. **Prefetch next M-block** during current M-block processing:
   ```cpp
   for (size_t m_block = m_block_start; m_block < m_block_end; m_block++) {
       // Prefetch next M-block's data
       if (m_block + 1 < m_block_end) {
           const size_t next_m = (m_block + 1) * StrideM;
           __builtin_prefetch(&A[next_m * lda], 0, 3);  // Prefetch for read, high temporal locality
       }

       // Process current tile
       for (size_t n = 0; n < N; n += StrideN) {
           // ... kernel processing ...
       }
   }
   ```

2. **Prefetch next N-tile** during current N-tile processing:
   ```cpp
   for (size_t n = 0; n < N; n += StrideN) {
       // Prefetch next N-tile's B data
       if (n + StrideN < N) {
           const size_t next_n = n + StrideN;
           __builtin_prefetch(&PackedB[next_n * K], 0, 3);
       }

       // Process current tile
       MlasGemmU16U8KernelNeon(...);
   }
   ```

**Expected Impact**: 3-5% improvement (117ms → 111-114ms)

**Risk**: Excessive prefetching can pollute cache and hurt performance

**Implementation Effort**: MEDIUM (requires careful tuning)

---

### Priority 3: Work Stealing / Dynamic Load Balancing (MEDIUM EFFORT, LOW IMPACT)

**Current Problem**: Thread 3 processes only 5 rows (M=77, last block), while threads 0-2 process 24 rows each.

**Work Imbalance**: 4.8:1 ratio (18,432 vs 3,840 elements)

**Optimization Strategy**:

Instead of static partitioning, use **work stealing**:
```cpp
std::atomic<size_t> next_m_block(0);

MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm, [&](ptrdiff_t thread_id) {
    while (true) {
        size_t m_block = next_m_block.fetch_add(1, std::memory_order_relaxed);
        if (m_block >= BlockedM) break;

        // Process this M-block
        // ...
    }
});
```

**Expected Impact**: 2-3% improvement (117ms → 114-116ms)

**Trade-off**: Adds atomic operations overhead, may not be worth it

**Implementation Effort**: MEDIUM

---

### Priority 4: Column Sum Optimization (LOW EFFORT, LOW IMPACT)

**Current**: Column sums precomputed during PrePack and stored in buffer.

**Potential Optimization**: Verify that column sum extraction is optimal.

**Investigation**:
```cpp
// Current implementation (lines 483-490)
const int32_t* PackedColumnSumBuffer = reinterpret_cast<const int32_t*>(B);
std::copy_n(PackedColumnSumBuffer, N, AllColumnSums.data());
```

**Questions**:
1. Is `std::copy_n` the fastest method?
2. Would `memcpy` be faster?
3. Can we avoid copying and use pointer directly?

**Expected Impact**: <1% improvement (negligible)

**Recommendation**: LOW priority, not worth effort

---

### Priority 5: NEON Kernel Micro-optimization (HIGH EFFORT, LOW IMPACT)

**Current**: Assembly kernel `MlasGemmU16U8KernelNeon` processes 4×16 tiles

**Optimization Areas**:
1. Register allocation optimization
2. Instruction scheduling
3. Loop unrolling
4. NEON instruction selection

**Example Optimization**:
```asm
; Current: Load-Process-Store pattern
vld1.16 {q0-q1}, [r0]!     ; Load A
vld1.8  {q2-q3}, [r1]!     ; Load B
; ... processing ...
vst1.32 {q8-q9}, [r2]!     ; Store C

; Optimized: Interleave loads with processing
vld1.16 {q0-q1}, [r0]!     ; Load A
vld1.8  {q2-q3}, [r1]!     ; Load B
vld1.16 {q4-q5}, [r0]!     ; Prefetch next A
; ... process q0-q3 while q4-q5 loads ...
```

**Expected Impact**: 5-10% improvement (117ms → 105-112ms)

**Implementation Effort**: HIGH (requires ARM64 assembly expertise)

**Risk**: Bugs in assembly are hard to debug

---

## Recommended Execution Plan

### Phase 1: Quick Wins (1-2 days)

**Priority 1: Tile Size Tuning**
1. Test 4-6 different tile size configurations
2. Run each with 20 iterations and thermal cooldown
3. Select best configuration
4. Document results

**Expected**: 5-10% improvement (117ms → 105-112ms)
**Effort**: Low
**Risk**: Low

### Phase 2: Medium Effort (2-3 days)

**Priority 2: Prefetching**
1. Add prefetch hints for next M-block
2. Add prefetch hints for next N-tile
3. Benchmark and tune prefetch distance
4. Document results

**Expected**: 3-5% improvement on top of Phase 1
**Effort**: Medium
**Risk**: Medium (can hurt if overdone)

### Phase 3: Diminishing Returns (1 week+)

**Priority 5: NEON Kernel Optimization**
1. Analyze current kernel with profiling tools
2. Identify specific bottlenecks
3. Implement targeted optimizations
4. Extensive testing for correctness

**Expected**: 5-10% improvement
**Effort**: High
**Risk**: High (assembly bugs)

---

## Success Criteria

### Minimum Success
- **Target**: 110ms (within 10ms of theoretical minimum)
- **CPU**: >85%
- **Correctness**: All tests pass

### Stretch Goal
- **Target**: 105ms (within 5ms of theoretical minimum)
- **CPU**: >90%
- **Stability**: <5ms variance across runs

---

## Benchmarking Protocol

### Standard Benchmark Command
```powershell
# 2-minute thermal cooldown
sleep 120

cd C:\d\onnxruntime\build_arm64_u16u8\Release\Release

.\onnxruntime_perf_test.exe `
    -I "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\embimg_c.quant.onnx" `
    -m times -r 20
```

### Comparison Template
| Configuration | Avg Latency | CPU | P50 | P90 | vs Baseline |
|--------------|-------------|-----|-----|-----|-------------|
| Option 2 (baseline) | 117.6ms | 85% | 120ms | 158ms | - |
| Tile Config A | ? | ? | ? | ? | ? |
| Tile Config B | ? | ? | ? | ? | ? |

---

## Risk Assessment

### Low Risk Optimizations ✅
- Tile size tuning: Easy to revert, low complexity
- Column sum verification: Read-only analysis

### Medium Risk Optimizations ⚠️
- Prefetching: Can hurt if overdone, needs careful tuning
- Work stealing: Atomic operations overhead

### High Risk Optimizations ❌
- NEON kernel optimization: Assembly bugs hard to debug
- Cache prefetch tuning: System-specific, may not generalize

---

## Alternative: Accept Current Performance

**Current**: 117.6ms (7.6ms above theoretical minimum)
**Achievement**: 2.25x speedup, 85% CPU utilization
**Status**: Production ready, exceeds <150ms target

**Question**: Is 5-10% additional improvement worth 1-2 weeks of effort?

**Recommendation**:
- ✅ **YES** if target is to match QUInt8 performance (110ms)
- ❌ **NO** if current performance is acceptable for production

---

## Conclusion

**Next Steps**:
1. **Start with Tile Size Tuning** (Phase 1)
   - Low effort, medium impact
   - Quick validation of approach
   - Can be completed in 1-2 days

2. **If successful**, proceed to Prefetching (Phase 2)
   - Build on tile size improvements
   - Moderate additional effort

3. **Reevaluate** before committing to NEON optimization (Phase 3)
   - High effort, diminishing returns
   - Consider cost/benefit at that point

**Decision Point**: Should we proceed with Phase 1 (Tile Size Tuning)?

---

**Date**: 2026-01-10
**Current Branch**: option-2-prepack-precompute
**Baseline**: 117.6ms, 85% CPU, 2.25x speedup
**Status**: Awaiting direction for next optimization phase
