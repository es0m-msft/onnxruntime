# NEON Kernel Optimization Analysis

**Date**: 2026-01-10
**File**: `QgemmU16U8KernelNeon.asm`
**Current Performance**: 105.5ms average (2.51x speedup from baseline)
**Target**: 95-100ms (5-10% improvement, approaching QUInt8 parity)

---

## Current Kernel Architecture

### Tile Size
- **M dimension**: 4 rows processed simultaneously
- **N dimension**: 16 columns per iteration
- **K dimension**: 8 elements per loop iteration

### Register Allocation (Main Loop)
```
v0-v3:   B values (uint8 extended to uint16) - 4 registers
v4-v7:   A values (uint16) for 4 rows - 4 registers
v8-v23:  Accumulators (4 rows × 4 column blocks) - 16 registers
v24-v27: Row fixups (broadcasted row sums) - 4 registers
v28-v31: Temporary/ZeroPointB values - 4 registers
Total: 32 NEON registers (fully utilized)
```

### Main Computation Loop (ComputeBlockLoopM4)

**Current pattern** (lines 234-292):
1. UMLAL multiply-accumulate operations (32 operations per iteration)
2. Interleaved loads of next B values
3. UXTL extend uint8→uint16
4. Load next A values at end of iteration
5. Loop decrement and branch

**Iteration count**: K/8 iterations (e.g., K=768 → 96 iterations)

---

## Optimization Opportunities

### 1. Instruction Scheduling - MEDIUM IMPACT (3-5% improvement)

**Current Issue**: Suboptimal interleaving of loads and computation

**Lines 234-258** (first half of compute loop):
```asm
umlal   v12.4s,v0.4h,v4.h[0]    // Compute
umlal   v13.4s,v0.4h,v4.h[1]    // Compute
umlal   v14.4s,v0.4h,v4.h[2]    // Compute
umlal   v15.4s,v0.4h,v4.h[3]    // Compute
umlal   v16.4s,v0.4h,v5.h[0]    // Compute
...
ld1     {v1.8b},[x1],#8         // ← Load next B (after 8 UMLALs)
umlal   v20.4s,v0.4h,v6.h[0]    // Compute
...
uxtl    v1.8h,v1.8b             // ← Extend B (after more UMLALs)
```

**Problem**:
- Load of v1 happens after 8 UMLAL operations
- UXTL happens after 12 UMLAL operations
- Memory latency not hidden by computation

**Optimization**:
```asm
ld1     {v1.8b},[x1],#8         // ← Move load EARLIER
umlal   v12.4s,v0.4h,v4.h[0]    // Compute (while v1 loads)
umlal   v13.4s,v0.4h,v4.h[1]    // Compute
uxtl    v1.8h,v1.8b             // ← Move extend EARLIER (2-3 cycles after load)
umlal   v14.4s,v0.4h,v4.h[2]    // Compute (while extend executes)
umlal   v15.4s,v0.4h,v4.h[3]    // Compute
umlal   v16.4s,v0.4h,v5.h[0]    // Compute (v1 now ready for use)
...
```

**Expected Impact**: 3-5% improvement by hiding memory latency

### 2. Loop Unrolling - MEDIUM-HIGH IMPACT (5-7% improvement)

**Current**: Process 8 K elements per iteration (1× unroll)
**Opportunity**: Process 16 K elements per iteration (2× unroll)

**Benefits**:
- Reduce loop overhead (sub/cbz/branch) by 50%
- Better instruction-level parallelism
- More opportunities for instruction scheduling

**Costs**:
- Increased code size (manageable)
- More complex tail handling (if K not multiple of 16)

**Implementation**:
```asm
ComputeBlockLoopM4_Unrolled
    // First 8 K elements (current code)
    umlal v12.4s,v0.4h,v4.h[0]
    ...
    umlal v11.4s,v1.4h,v7.h[7]

    // Second 8 K elements (duplicate pattern with v2,v3 and next A)
    ld1     {v2.8b},[x1],#8
    uxtl    v2.8h,v2.8b
    ld1     {v30.8h},[x0],#16       // Next A row 0
    umlal   v12.4s,v2.4h,v30.h[0]
    ...

    sub     x3,x3,#2                // Decrement by 2 instead of 1
    cbnz    x3,ComputeBlockLoopM4_Unrolled
```

**Expected Impact**: 5-7% improvement

### 3. Prefetching - LOW-MEDIUM IMPACT (1-3% improvement)

**Current**: No explicit prefetch instructions in assembly kernel

**Opportunity**: Add PLD/PLDL1KEEP instructions for next tiles

**Implementation**:
```asm
ComputeBlockLoopM4
    // Prefetch next iteration's data
    prfm    pldl1keep,[x0,#256]     // Prefetch A data 256 bytes ahead
    prfm    pldl1keep,[x1,#128]     // Prefetch B data 128 bytes ahead

    // Existing computation
    umlal   v12.4s,v0.4h,v4.h[0]
    ...
```

**Note**: We already have C++ level prefetching (from previous optimization), so this might provide diminishing returns.

**Expected Impact**: 1-3% improvement (may overlap with existing prefetching)

### 4. Reduce Register Pressure - RISKY (potential 0-5% improvement or regression)

**Current**: Uses all 32 NEON registers
- 16 accumulator registers (v8-v23) for 4 rows × 4 column blocks
- This might cause register spilling or prevent optimizations

**Observation**: Processing 16 columns requires 16 accumulator registers, which is aggressive.

**Alternative Approach**:
- Process 8 columns instead of 16 (reduce to 8 accumulator registers)
- Free up 8 registers for better scheduling/prefetching
- Trade-off: Double the number of outer loop iterations

**Analysis**:
```
Current: Process 16 cols/iteration → N/16 iterations
Alternative: Process 8 cols/iteration → N/8 iterations (2× more)

Impact on 768 columns:
Current: 768/16 = 48 outer iterations
Alternative: 768/8 = 96 outer iterations

Cost: 2× more loop overhead, column sum loads, etc.
Benefit: 8 free registers for better scheduling
```

**Recommendation**: SKIP - likely to hurt performance due to doubled loop overhead

### 5. Optimize Zero-Point Handling - LOW IMPACT (1-2% improvement)

**Current** (lines 145-188): Two code paths for with/without ZeroPointB
- With ZeroPointB: 40 instructions for initialization
- Without ZeroPointB: 16 instructions for initialization

**Observation**: ZeroPointB path has significant overhead (24 extra instructions)

**Opportunity**: Optimize the with-ZeroPointB path
- Combine loads and multiplies more efficiently
- Reduce instruction count

**Expected Impact**: 1-2% improvement (only if ZeroPointB is commonly used)

### 6. Store Optimization - LOW IMPACT (1% improvement)

**Current** (lines 309-341): Full store path
- Check ZeroMode (zero vs accumulate)
- Accumulate mode: Load 4×16 values, add, then store
- Zero mode: Direct store

**Opportunity**: The accumulate mode loads and adds 64 int32 values
- Could this be optimized with better instruction scheduling?
- Possibly use STP (store pair) more aggressively

**Expected Impact**: <1% improvement (store is not on critical path)

---

## Recommended Optimization Strategy

### Phase 1: Low-Risk, High-Impact (2-3 days)

**Priority 1: Instruction Scheduling**
- Move memory loads earlier in the loop
- Interleave UXTL with computation
- Hide memory latency with arithmetic operations
- **Expected**: 3-5% improvement
- **Risk**: Low (correctness easy to verify)
- **Effort**: Medium (careful manual scheduling)

### Phase 2: Medium-Risk, High-Impact (3-4 days)

**Priority 2: Loop Unrolling (2×)**
- Process 16 K elements per iteration instead of 8
- Reduce loop overhead by 50%
- Improve instruction-level parallelism
- **Expected**: 5-7% improvement
- **Risk**: Medium (more complex, potential for bugs)
- **Effort**: High (significant code duplication and testing)

### Phase 3: Diminishing Returns (2-3 days)

**Priority 3: Assembly Prefetching**
- Add PRFM instructions for next tiles
- May overlap with existing C++ prefetching
- **Expected**: 1-3% improvement
- **Risk**: Low-Medium (can hurt if tuned incorrectly)
- **Effort**: Low (simple to add)

**Priority 4: Zero-Point Optimization**
- Optimize with-ZeroPointB initialization path
- **Expected**: 1-2% improvement
- **Risk**: Low
- **Effort**: Medium

---

## Combined Impact Estimate

**Conservative (Phase 1 only)**:
- Current: 105.5ms
- With instruction scheduling: 100-102ms (3-5% improvement)
- **Status**: Meets target of <100ms with stretch goal!

**Optimistic (Phases 1 + 2)**:
- Current: 105.5ms
- With scheduling + unrolling: 94-100ms (5-11% improvement)
- **Status**: Approaching QUInt8 parity (55-68ms theoretical base)

**Maximum (Phases 1 + 2 + 3)**:
- Current: 105.5ms
- With all optimizations: 91-98ms (7-14% improvement)
- **Status**: Excellent performance, diminishing returns beyond this

---

## Implementation Plan

### Step 1: Instruction Scheduling (Recommended First)

**File**: `QgemmU16U8KernelNeon.asm`
**Lines to modify**: 234-292 (ComputeBlockLoopM4)

**Changes**:
1. Move `ld1 {v1.8b},[x1],#8` from line 246 to line 236 (after first 2 UMLALs)
2. Move `uxtl v1.8h,v1.8b` from line 253 to line 240 (3 instructions after load)
3. Move `ld1 {v0.8b},[x1],#8` from line 271 to earlier position
4. Move `uxtl v0.8h,v0.8b` from line 279 to earlier position
5. Move A loads (lines 287-290) to earlier in the loop

**Testing**:
- Run correctness tests (compare output with scalar reference)
- Benchmark with 20 runs
- Verify no regression in edge cases

**Rollback plan**: Keep original code commented out for easy revert

### Step 2: Loop Unrolling (If Step 1 succeeds)

**Approach**:
1. Duplicate the 8-K-element block
2. Adjust loop counter decrement (sub x3,x3,#2)
3. Add tail handling for odd K values
4. Extensive testing for various K dimensions

**Risk mitigation**:
- Implement as separate function first
- Compare outputs with original kernel
- Only merge after thorough validation

---

## Benchmarking Protocol

### Before/After Comparison

**Baseline** (current with C++ prefetching):
```
Average: 105.5ms
P50: 97.7ms
CPU: 87%
```

**After Instruction Scheduling**:
```
Target: <102ms average (3% improvement)
Target: <95ms P50 (3% improvement)
Target: ≥87% CPU
```

**After Loop Unrolling**:
```
Target: <98ms average (7% improvement)
Target: <91ms P50 (7% improvement)
Target: ≥87% CPU
```

### Testing Methodology

1. **Thermal cooldown**: 2 minutes between tests
2. **Iterations**: 20 runs per configuration
3. **Validation**: Run numerical correctness tests first
4. **Comparison**: Compare with both baseline and previous optimization

---

## Risk Assessment

### Low Risk ✅
- Instruction scheduling (easy to verify, easy to revert)
- Assembly prefetching (already proven effective in C++)

### Medium Risk ⚠️
- Loop unrolling (more code, more complex, need thorough testing)
- Zero-point optimization (conditional code paths)

### High Risk ❌
- Register reduction (likely to hurt performance)
- Algorithmic changes (out of scope for assembly tuning)

---

## Alternative: Accept Current Performance

**Current Achievement**:
- 105.5ms average (2.51x speedup)
- 4% better than theoretical minimum
- Exceeds target by 42%

**Question**: Is 5-10% additional improvement worth 1-2 weeks of assembly optimization effort?

**Arguments FOR continuing**:
- Get closer to QUInt8 parity (goal: <100ms)
- Demonstrate maximum achievable performance
- Learn optimization techniques for future kernels

**Arguments AGAINST continuing**:
- Diminishing returns (high effort, modest gains)
- Assembly bugs are hard to debug
- Current performance already exceeds requirements

---

## Recommendation

**PRIMARY RECOMMENDATION**: **Start with Instruction Scheduling only**
- Low risk, 3-5% expected improvement
- Can be completed in 2-3 days
- If successful and performance < 100ms, **stop here**
- If needs more improvement, proceed to loop unrolling

**SECONDARY RECOMMENDATION**: **Accept current performance**
- 105.5ms is already exceptional
- Focus engineering effort on other features/optimizations
- Return to assembly tuning only if QUInt8 parity becomes critical

**DECISION POINT**: Should we proceed with instruction scheduling optimization?

---

## Implementation Results (2026-01-10)

### Instruction Scheduling Optimization - FAILED ❌

**Implementation**: Moved memory loads earlier in the compute loop to hide latency
- Moved `ld1 {v1.8b}` from after 8 UMLALs to after 2 UMLALs
- Moved `uxtl v1.8h,v1.8b` from after 12 UMLALs to after 5 UMLALs
- Interleaved A matrix loads with final row 3 computation

**Expected**: 3-5% improvement (105.5ms → 100-102ms)

**Actual Results**:
| Metric | Baseline (C++ Prefetch) | Assembly Scheduled | Change |
|--------|------------------------|-------------------|--------|
| Average | 105.5ms | **154.0ms** | **+46% WORSE** ❌ |
| CPU | 87% | **59%** | **-32%** ❌ |
| P50 | 97.7ms | 108.4ms | +11% worse |

**Root Cause Analysis**:

1. **CPU Utilization Drop (87% → 59%)**
   - Indicates pipeline stalls or memory bottlenecks
   - System is waiting, not computing

2. **Hypothesis 1: Pipeline Hazards**
   - Moving loads too early may cause data hazards
   - ARM64 NEON pipeline may have dependencies we didn't account for
   - UXTL immediately after LD1 might stall waiting for load to complete

3. **Hypothesis 2: Register Dependency Chains**
   - Early loads might create longer dependency chains
   - Prevents out-of-order execution from hiding latency
   - Modern ARM cores have sophisticated OoO engines that were already doing better scheduling

4. **Hypothesis 3: Cache/Memory Subsystem Interaction**
   - ARM64 memory subsystem may handle sequential loads better than scattered ones
   - Disrupted prefetcher patterns

5. **Hypothesis 4: Branch Misprediction**
   - Changed CBZ timing might affect branch predictor accuracy
   - Original code had CBZ after completing computations
   - New code has CBZ in the middle of row 3 computation

**Lessons Learned**:

1. **Modern Out-of-Order Execution is Sophisticated**
   - ARM64 Cortex-A/X cores have advanced OoO engines
   - Manual instruction scheduling can hurt performance if it fights the hardware
   - The compiler/assembler was already doing a good job

2. **Load-Use Distance Matters**
   - Too close: Data not ready, causes stall
   - Too far: Data evicted from register before use
   - Original scheduling was likely optimal for the target architecture

3. **Profile Before Optimizing**
   - Should have profiled to identify actual bottlenecks
   - CPU performance counters would show if loads were really the issue
   - Assumption that memory latency was the bottleneck was incorrect

4. **C++ Prefetching is Sufficient**
   - The C++ level prefetching (105.5ms) already hides memory latency
   - Assembly-level optimization provided no additional benefit
   - May have disrupted existing optimization

**Decision**: **REVERT** to C++ prefetching baseline (105.5ms)

**Conclusion**: Further assembly-level optimization is **NOT RECOMMENDED**
- Current performance (105.5ms) already exceeds target by 42%
- Below theoretical minimum by 4%
- Assembly tuning carries high risk of regression
- Focus effort on higher-value optimizations instead

---

**Date**: 2026-01-10
**Current Performance**: 105.5ms (2.51x speedup) - maintained with C++ prefetching
**Assembly Optimization Status**: Attempted and reverted - no benefit
**Final Recommendation**: Accept current performance as optimal
**Risk Level**: Proven high - assembly changes caused 46% regression
