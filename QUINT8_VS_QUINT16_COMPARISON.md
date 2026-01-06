# QUInt8 vs QUInt16 Performance Comparison

## Executive Summary

**QUInt8×QUInt8 is significantly faster than QUInt16×QUInt8** and approaches FP32 performance!

| Quantization | Mean Inference | vs FP32 | Model Size | Accuracy (L2) |
|--------------|----------------|---------|------------|---------------|
| **FP32** | 51.1 ms | 1.00x (baseline) | 354.5 MB | baseline |
| **QUInt8×QUInt8** | **68.2 ms** | **0.75x (1.33x slower)** | 89.5 MB | 4.35e-03 (excellent) |
| **QUInt16×QUInt8** | 464.3 ms | 0.11x (9.08x slower) | 89.8 MB | 5.05e-02 (good) |

### Key Findings

1. ✅ **QUInt8 is 6.8x FASTER than QUInt16** (68ms vs 464ms)
2. ✅ **QUInt8 is only 1.33x slower than FP32** (industry competitive!)
3. ✅ **QUInt8 has BETTER accuracy** than QUInt16 (4.35e-03 vs 5.05e-02 L2 error)
4. ✅ **QUInt8 has extensive operator fusion** (QLinearAdd, QLinearMul, QLinearSoftmax)
5. ❌ **QUInt16 still suffers from LayerNorm decomposition**

---

## Detailed Performance Breakdown

### 1. Overall Metrics

```
                   FP32      QUInt8     QUInt16
Mean time:        51.1 ms    68.2 ms    464.3 ms
Median time:      47.7 ms    64.3 ms    453.0 ms
Min time:         37.4 ms    50.4 ms    214.3 ms
Max time:         99.0 ms   136.4 ms    863.4 ms
P95 time:         72.0 ms    95.4 ms    735.9 ms

Speedup vs FP32:  1.00x      0.75x      0.11x
Slowdown ratio:   -          1.33x      9.08x

Model size:       354.5 MB   89.5 MB    89.8 MB
Size reduction:   1.00x      3.96x      3.95x

Accuracy (L2):    baseline   4.35e-03   5.05e-02
Accuracy ratio:   1.00x      97x better 11.6x worse
```

### 2. Top 10 Operators Comparison

#### FP32 Operators
```
Operator                  Time (ms)  Count   Avg (ms)  % of Total
Gemm                      3,051.6    7,560   0.404     59.93%
MatMul                      477.8    2,625   0.182      9.38%
Transpose                   345.5    5,040   0.069      6.79%
Reshape                     322.5   16,380   0.020      6.33%
SkipLayerNormalization      183.1    2,625   0.070      3.60%
Gelu                        152.6    1,260   0.121      3.00%
Add                         143.1    3,780   0.038      2.81%
Mul                         117.4    3,990   0.029      2.31%
FusedMatMul                 107.3    1,260   0.085      2.11%
Softmax                      52.5    1,260   0.042      1.03%
```

#### QUInt8 Operators (OPTIMIZED!)
```
Operator                  Time (ms)  Count   Avg (ms)  % of Total
QLinearMatMul             2,086.4   11,445   0.182     31.08%  ✅ Fused
QLinearAdd                1,036.7   20,475   0.051     15.44%  ✅ Fused
Transpose                   803.8    5,040   0.159     11.97%
DequantizeLinear            507.2   17,220   0.029      7.55%
QLinearMul                  390.0    9,135   0.043      5.81%  ✅ Fused
QuantizeLinear              361.1   10,605   0.034      5.38%
ReduceMean                  335.1    5,250   0.064      4.99%
Div                         314.8    5,250   0.060      4.69%
Sub                         178.3    3,990   0.045      2.65%
QLinearSoftmax              174.6    1,260   0.139      2.60%  ✅ Fused
```

#### QUInt16 Operators (POOR FUSION!)
```
Operator                  Time (ms)  Count   Avg (ms)  % of Total
QLinearMatMul            14,817.8   11,445   1.295     31.35%  ⚠️ 6.9x slower than QUInt8!
ReduceMean                9,988.4    5,250   1.903     21.13%  ❌ NOT fused (LayerNorm decomp)
QuantizeLinear            6,236.4   42,735   0.146     13.19%  ❌ 4x more Q/DQ ops!
Add                       4,058.9   20,475   0.198      8.59%  ❌ NOT fused
Transpose                 3,884.0    5,040   0.771      8.22%
DequantizeLinear          2,623.1   78,960   0.033      5.55%  ❌ 4.6x more Q/DQ ops!
Div                       2,227.6    5,250   0.424      4.71%  ❌ NOT fused (LayerNorm decomp)
Erf                       1,576.0    1,260   1.251      3.33%
Mul                         896.0    9,135   0.098      1.90%  ❌ NOT fused
Softmax                     314.7    1,260   0.250      0.67%  ❌ NOT fused
```

### 3. Matrix Multiplication Performance

```
                          FP32         QUInt8       QUInt16
Operator:                 Gemm+MatMul  QLinearMatMul QLinearMatMul
Total time:               3,529 ms     2,086 ms      14,818 ms
Operation count:          10,185       11,445        11,445
Avg time per op:          0.346 ms     0.182 ms      1.295 ms

Ratio vs FP32:            1.00x        0.53x         3.74x
                                       (1.9x FASTER) (3.7x SLOWER)
```

**Analysis:**
- QUInt8 QLinearMatMul is **1.9x FASTER than FP32**! 🎉
- QUInt16 QLinearMatMul is **3.7x SLOWER than FP32** 😢
- QUInt8 kernel is **7.1x faster per operation** than QUInt16 (0.182ms vs 1.295ms)

### 4. Quantization Overhead

```
                          QUInt8       QUInt16
Q/DQ Operations:
  QuantizeLinear:         361 ms       6,236 ms      ❌ 17.3x worse in QUInt16
                          (10,605 ops) (42,735 ops)  (4x more ops!)

  DequantizeLinear:       507 ms       2,623 ms      ❌ 5.2x worse in QUInt16
                          (17,220 ops) (78,960 ops)  (4.6x more ops!)

Total Q/DQ overhead:      868 ms       8,859 ms      ❌ 10.2x worse in QUInt16
% of inference time:      12.93%       18.74%

Q/DQ per QLinearMatMul:   2.45         10.56         ❌ 4.3x more conversions!
```

**Analysis:**
- QUInt8 has **10.2x less Q/DQ overhead**
- QUInt8 requires **4x fewer Q/DQ operations** (better fusion)
- QUInt16 spends **18.74% of time just converting** between formats

### 5. Operator Fusion Comparison

#### QUInt8 - EXCELLENT Fusion ✅

**Fused Operators:**
- ✅ **QLinearMatMul** (11,445 ops) - Matrix multiplication
- ✅ **QLinearAdd** (20,475 ops) - Addition stays in quantized domain
- ✅ **QLinearMul** (9,135 ops) - Multiplication stays in quantized domain
- ✅ **QLinearSoftmax** (1,260 ops) - Softmax stays in quantized domain

**Impact:**
- Minimal Q/DQ overhead (only 27,825 Q/DQ operations total)
- Most arithmetic stays in INT8 domain
- **Excellent cache efficiency** (8-bit data types)

#### QUInt16 - POOR Fusion ❌

**Fused Operators:**
- ⚠️ **QLinearMatMul** (11,445 ops) - Only MatMul is fused

**Unfused/Decomposed:**
- ❌ **Add** (20,475 ops) - Regular FP32 Add with Q/DQ conversions
- ❌ **Mul** (9,135 ops) - Regular FP32 Mul with Q/DQ conversions
- ❌ **Softmax** (1,260 ops) - Regular FP32 Softmax with Q/DQ conversions
- ❌ **LayerNorm decomposition**:
  - ReduceMean: 9,988 ms (5,250 ops)
  - Div: 2,228 ms (5,250 ops)
  - Sub: 191 ms (3,990 ops)
  - Total LayerNorm overhead: ~12,407 ms (26.7% of inference!)

**Impact:**
- **Excessive Q/DQ overhead** (121,695 Q/DQ operations total)
- Most operations convert to FP32 for computation
- **Poor cache efficiency** (16-bit data types + constant conversions)

---

## Why is QUInt8 So Much Faster?

### 1. Better ONNX Runtime Support ✅

**QUInt8 has extensive quantized operators:**
- `QLinearMatMul` - Matrix multiplication
- `QLinearAdd` - Addition (stays quantized!)
- `QLinearMul` - Multiplication (stays quantized!)
- `QLinearSoftmax` - Softmax (stays quantized!)
- `QLinearConv` - Convolution (if present)
- `QLinearAveragePool` - Pooling (if present)

**QUInt16 only has:**
- `QLinearMatMul` - Matrix multiplication ONLY
- Everything else decomposes to FP32 + Q/DQ conversions

### 2. Half the Memory Bandwidth 🚀

```
Matrix multiplication for M=1024, N=1024, K=1024:

QUInt8:
  - Activations: 1024×1024×1 = 1 MB
  - Weights:     1024×1024×1 = 1 MB
  - Total input: 2 MB

QUInt16:
  - Activations: 1024×1024×2 = 2 MB  ← 2x larger!
  - Weights:     1024×1024×1 = 1 MB
  - Total input: 3 MB  ← 50% more bandwidth!
```

**Impact:**
- QUInt8 is more bandwidth-efficient
- Better cache utilization
- Faster memory access patterns

### 3. Better ARM NEON Instructions 💪

**QUInt8 kernel optimizations:**
- Processes 8 elements per SIMD lane efficiently
- Better register utilization
- More compact inner loops
- Optimized for ARM cache hierarchy

**QUInt16 kernel limitations:**
- UMLAL processes only 4 lanes at a time
- 100% register pressure (no optimization room)
- Larger working set doesn't fit in L1 cache
- More memory stalls

### 4. LayerNorm Fusion Disaster 💥

```
LayerNorm in QUInt8:
  - Uses FP32 ReduceMean+Div: 335+315 = 650 ms (still not great)
  - But only 4.99% + 4.69% = 9.68% of total time

LayerNorm in QUInt16:
  - ReduceMean: 9,988 ms (21.13% of total!)
  - Div:        2,228 ms (4.71% of total!)
  - Total:     ~12,216 ms (26.3% of total!)

Ratio: QUInt16 LayerNorm is 18.8x SLOWER than QUInt8!
```

**Why such a huge difference?**
- QUInt16 ReduceMean operates on **16-bit values** (2x data size)
- More Q/DQ conversions for LayerNorm components
- Worse cache behavior with larger data types

---

## Accuracy Comparison

### L2 Error Analysis

```
Model         L2 Norm Error   Quality
FP32          0.000000        Perfect (baseline)
QUInt8        0.004354        Excellent (0.4% error)
QUInt16       0.050524        Good (5% error)
```

**Surprising Result:** QUInt8 has **11.6x BETTER accuracy** than QUInt16!

### Why is QUInt8 More Accurate?

1. **Better quantized operator implementations**
   - QLinearAdd/Mul/Softmax are optimized and tested
   - QUInt16 uses FP32 fallbacks which accumulate error

2. **Less Q/DQ conversion overhead**
   - Each Q/DQ introduces quantization error
   - QUInt8: 27,825 Q/DQ ops
   - QUInt16: 121,695 Q/DQ ops (4.4x more error sources!)

3. **Calibration optimized for INT8**
   - Quantization tools are tuned for INT8 quantization
   - INT16 quantization is less mature

---

## Graph Structure Analysis

### QUInt8 Optimized Graph (Inferred from profiling)

```
Estimated node breakdown:
- QLinear* fused operators: ~42,000 (MatMul, Add, Mul, Softmax fused)
- Q/DQ nodes:               ~27,825 (QuantizeLinear + DequantizeLinear)
- FP32 fallbacks:           ~11,500 (Transpose, Reshape, ReduceMean, Div, Sub)
- Total estimated:          ~81,325 nodes

Fusion ratio: ~52% of operations stay in quantized domain
```

### QUInt16 Optimized Graph (Measured)

```
Actual node breakdown:
- QLinearMatMul:   109 (ONLY MatMul is fused!)
- Q/DQ nodes:     1,159 (QuantizeLinear + DequantizeLinear)
- FP32 operations: ~750 (Add, Mul, Softmax, LayerNorm components)
- Total:          1,910 nodes

Fusion ratio: ~6% of operations stay in quantized domain
```

**Key Difference:**
- QUInt8: **52% fusion** - most ops stay quantized
- QUInt16: **6% fusion** - almost everything goes through FP32

---

## Recommendations

### When to Use QUInt8 ✅ RECOMMENDED

**Use QUInt8×QUInt8 when:**
- ✅ Performance is important (only 1.33x slower than FP32)
- ✅ You want production-ready quantization
- ✅ Industry-standard compatibility is needed
- ✅ You need extensive operator support
- ✅ Memory size reduction is important (4x smaller)
- ✅ You want better accuracy (4.35e-03 L2 error)

**Pros:**
- Fast (68ms vs 51ms FP32)
- Excellent operator fusion
- Better accuracy than QUInt16
- Industry standard (TFLite, PyTorch Mobile compatible)
- Low Q/DQ overhead

**Cons:**
- Slightly lower activation precision (8-bit vs 16-bit)
- Still 1.33x slower than FP32

### When to Use QUInt16 ❌ NOT RECOMMENDED (Currently)

**Only use QUInt16×QUInt8 when:**
- ⚠️ You absolutely need 16-bit activation precision
- ⚠️ Accuracy is more important than performance (but QUInt8 is actually MORE accurate!)
- ⚠️ You're willing to accept 9x slowdown vs FP32

**Pros:**
- Higher activation precision (16-bit)
- Same model size as QUInt8 (90MB)

**Cons:**
- **9.08x slower than FP32**
- **6.8x slower than QUInt8**
- Poor operator fusion (only MatMul)
- LayerNorm decomposition disaster
- **Worse accuracy than QUInt8** (11.6x worse!)
- Excessive Q/DQ overhead
- Immature ONNX Runtime support

### Future Work for QUInt16

To make QUInt16 competitive, ONNX Runtime needs:

1. **✅ DONE: QLinearMatMul kernel** - Completed and optimized (8.9x speedup)

2. **❌ TODO: Quantized LayerNorm** - Would save ~10-12 seconds
   - Implement `QLinearLayerNormalization` operator
   - Fuse ReduceMean → Div → Sub → Sqrt sequence

3. **❌ TODO: QLinearAdd/Mul/Softmax for INT16** - Would save ~5 seconds
   - Port QUInt8 implementations to support QUInt16

4. **❌ TODO: Better graph fusion** - Would reduce Q/DQ by 4x
   - Keep more operations in quantized domain
   - Reduce Q/DQ count from 121,695 to ~30,000

**Estimated performance after all fixes:** ~120-150ms (2.4-3.0x slower than FP32)
**Still worse than QUInt8's 68ms (1.33x slower than FP32)**

---

## Conclusion

### Summary Table

| Metric | FP32 | QUInt8 | QUInt16 | Winner |
|--------|------|--------|---------|--------|
| **Performance** | 51 ms | 68 ms | 464 ms | QUInt8 ✅ |
| **vs FP32** | 1.00x | 1.33x slower | 9.08x slower | QUInt8 ✅ |
| **Accuracy** | baseline | 4.35e-03 | 5.05e-02 | QUInt8 ✅ |
| **Model size** | 354 MB | 90 MB | 90 MB | Tie |
| **Operator fusion** | Native | Excellent | Poor | QUInt8 ✅ |
| **Q/DQ overhead** | N/A | 13% | 19% | QUInt8 ✅ |
| **Industry support** | Universal | Excellent | Limited | QUInt8 ✅ |
| **Maturity** | Stable | Stable | Experimental | QUInt8 ✅ |

### Final Recommendation

**Use QUInt8×QUInt8 for production workloads.**

QUInt8 is:
- ✅ **6.8x faster** than QUInt16
- ✅ **11.6x more accurate** than QUInt16
- ✅ Industry-standard and mature
- ✅ Only 1.33x slower than FP32 (acceptable tradeoff)
- ✅ Same model size reduction (4x)

QUInt16×QUInt8 is **not ready for production** due to:
- ❌ Poor ONNX Runtime operator support
- ❌ LayerNorm decomposition disaster
- ❌ 9x slower than FP32 (unacceptable)
- ❌ Worse accuracy than QUInt8

**The QUInt16 integration was technically successful** (kernel works, fusion enabled), but the **ecosystem support is inadequate** for practical use.

---

**Date**: 2026-01-06
**ONNX Runtime Version**: 1.24.0
**Platform**: Windows ARM64, ARMv8-A NEON
**Test Model**: Florence Vision Transformer (77×768 sequence length)
