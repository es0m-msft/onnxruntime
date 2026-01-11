# QUInt16xQUInt8 QGEMM x64 Implementation Plan

## Overview

This document describes the implementation plan for porting the ARM64 NEON QUInt16xQUInt8 QGEMM kernel to x64 with AVX2 and AVX512 variants.

## Architecture Summary

### ARM64 NEON Reference (Current Implementation)
- **File**: `onnxruntime/core/mlas/lib/arm64/QgemmU16U8KernelNeon.asm`
- **Tile size**: 4 rows x 16 columns (M=4, N=16)
- **Algorithm**: UMLAL (unsigned multiply-accumulate long) for uint16×uint16→uint32
- **Key operation**: Extend uint8 to uint16, then uint16×uint16 multiply-accumulate

### x64 Target Implementations

#### AVX2 Variant (256-bit, ymm registers)
- **Tile size**: 6 rows x 16 columns (M=6, N=16)
- **Registers**: ymm0-ymm15 (256-bit each)
- **Key instructions**:
  - `vpmovzxbw ymm, xmm` - Zero extend 16×uint8 → 16×uint16
  - `vpmaddwd ymm, ymm, ymm` - Multiply 16×uint16 pairs, add horizontally → 8×uint32
  - `vpaddd ymm, ymm, ymm` - 32-bit add for accumulation

#### AVX512 Variant (512-bit, zmm registers)
- **Tile size**: 6 rows x 48 columns (M=6, N=48)
- **Registers**: zmm0-zmm31 (512-bit each)
- **Key instructions**:
  - `vpmovzxbw zmm, ymm` - Zero extend 32×uint8 → 32×uint16
  - `vpmaddwd zmm, zmm, zmm` - Multiply 32×uint16 pairs, add horizontally → 16×uint32
  - `vpaddd zmm, zmm, zmm` - 32-bit add for accumulation

## File Structure

### New Files to Create

1. **`onnxruntime/core/mlas/lib/amd64/QgemmU16U8CopyPackAAvx2.asm`**
   - CopyPackA routine for U16U8 (copies and zero-extends activations, computes row sums)
   - Similar to QgemmU8U8CopyPackAAvx2.asm but handles uint16 inputs

2. **`onnxruntime/core/mlas/lib/amd64/QgemmU16U8KernelAvx2.asm`**
   - Main GEMM kernel for AVX2
   - Processes uint16 activations × uint8 weights → int32 accumulator

3. **`onnxruntime/core/mlas/lib/amd64/QgemmU16U8KernelAvx512Core.asm`**
   - Main GEMM kernel for AVX512
   - Conditional compilation via `MLAS_USE_AVX512` define

### Files to Modify

1. **`onnxruntime/core/mlas/lib/qgemm_u16u8.cpp`**
   - Add x64 platform detection (`#if defined(MLAS_TARGET_AMD64)`)
   - Add extern declarations for x64 kernels
   - Implement x64-specific helper functions (AVX2/AVX512 row/column sums)
   - Platform dispatch in MlasGemmU16U8Batch

2. **`cmake/onnxruntime_mlas.cmake`**
   - Add new assembly files to x64 build
   - Add compile flags for AVX512 variant (`-DMLAS_USE_AVX512`)

## Algorithm Details

### Data Flow

```
Input:
  A: M×K matrix of uint16_t (activations)
  B: K×N matrix of uint8_t (weights)

Output:
  C: M×N matrix of int32_t (accumulator)

Computation:
  C[i,j] = Σ (A[i,k] - ZeroPointA) × (B[k,j] - ZeroPointB)
         = Σ A[i,k]×B[k,j] - ZeroPointA×Σ B[k,j] - ZeroPointB×Σ A[i,k] + K×ZeroPointA×ZeroPointB
```

### CopyPackA (uint16 activations → packed format)

```asm
; AVX2: Process 8 uint16 elements per iteration
; Input: uint16_t array
; Output: Packed uint16_t array + row sums

ProcessNextColumnLoopM4:
    vmovdqu ymm4, [rdx]           ; Load 16×uint16 (256 bits)
    vmovdqu [rcx], ymm4           ; Store to packed buffer
    vpmaddwd ymm5, ymm4, ymm_ones ; Horizontal sum: 16×u16 → 8×u32
    vpaddd ymm0, ymm0, ymm5       ; Accumulate row sum
```

### CopyPackB (uint8 weights → packed format)

```asm
; AVX2: Process 16 uint8 elements per iteration
; Input: uint8_t array (row-major)
; Output: Packed uint8_t array + column sums

ProcessNextColumnN16:
    vmovdqu xmm2, [rdx]           ; Load 16×uint8
    vmovdqu xmm3, [rdx+r8]        ; Load next row
    vpunpcklbw xmm4, xmm2, xmm3   ; Interleave rows
    vpunpckhbw xmm3, xmm2, xmm3
    vmovdqu [rcx], xmm4           ; Store interleaved
    vmovdqu [rcx+16], xmm3
    ; Compute column sums...
```

### Main Kernel (multiply-accumulate)

```asm
; AVX2 U16U8 GEMM kernel
; Key difference from U8U8: A is already uint16, no extension needed

ComputeBlock:
    ; Load B: extend uint8 → uint16
    vpmovzxbw ymm0, [rdx]         ; 16×u8 → 16×u16 (256 bits)

    ; Load A: already uint16
    vmovdqu ymm1, [rcx]           ; 16×u16 (256 bits)

    ; Multiply and accumulate: u16×u16 → u32
    vpmaddwd ymm2, ymm0, ymm1     ; 16×u16 × 16×u16 → 8×u32
    vpaddd ymm_acc, ymm_acc, ymm2 ; Accumulate
```

## Register Allocation

### AVX2 Kernel

| Register | Purpose |
|----------|---------|
| ymm0-ymm3 | B matrix values (extended to u16) |
| ymm4-ymm7 | A matrix values (u16) |
| ymm8-ymm13 | Accumulators (M=6 rows) |
| ymm14 | Row fixups |
| ymm15 | Scratch/constants |

### AVX512 Kernel

| Register | Purpose |
|----------|---------|
| zmm0-zmm2 | B matrix values (3 blocks of 16 columns) |
| zmm3 | Broadcast A value |
| zmm4 | Scratch |
| zmm13 | Constant [0x0001] for vpmaddwd |
| zmm14-zmm31 | Accumulators (6 rows × 3 column blocks) |
| k1 | Mask for partial column writes |

## Build Configuration

### CMake Changes

```cmake
# In onnxruntime_mlas.cmake, x64 section:
elseif(onnxruntime_target_platform STREQUAL "x64")
  # Add new files after existing QGEMM entries
  ${MLAS_SRC_DIR}/amd64/QgemmU16U8CopyPackAAvx2.asm
  ${MLAS_SRC_DIR}/amd64/QgemmU16U8KernelAvx2.asm
  ${MLAS_SRC_DIR}/amd64/QgemmU16U8KernelAvx512Core.asm
```

### Compile-time AVX512 Selection

```asm
; In assembly file header:
IFDEF MLAS_USE_AVX512
; Use zmm registers and AVX512 instructions
ELSE
; Fallback to AVX2 implementation
ENDIF
```

### Runtime CPU Detection

The implementation uses runtime CPU detection via `GetMlasPlatform().Avx512Supported_`:

```cpp
// In MlasGemmU16U8Batch() x64 section:
if (GetMlasPlatform().Avx512Supported_) {
    // Use AVX512 kernel for better throughput (6 rows × 48 cols)
    RowsHandled = MlasGemmU16U8KernelAvx512Core(...);
} else {
    // Fallback to AVX2 kernel (4 rows × 16 cols)
    RowsHandled = MlasGemmU16U8KernelAvx2(...);
}
```

The `Avx512Supported_` flag is set during MLAS platform initialization when:
- AVX512BW (bit 30 of CPUID.7.0.EBX) is supported
- AVX512DQ (bit 17 of CPUID.7.0.EBX) is supported
- AVX512VL (bit 31 of CPUID.7.0.EBX) is supported

## Testing Strategy

### Correctness Testing
1. Compare x64 output with ARM64 reference implementation
2. Test edge cases: M=1, N=1, K=1, partial tiles
3. Verify zero-point handling matches expected values

### Performance Testing (Under Emulation)
1. Build x64 wheel without `--arm64` flag
2. Run under ARM64 emulation (will be slow but functional)
3. Compare relative timing with FP32 baseline

### Build Commands

```powershell
# Build x64 variant (from ARM64 device, will cross-compile)
python tools/ci_build/build.py --config Release --build_dir build_x64_u16u8 --build_wheel --parallel --skip_tests

# Note: Do NOT use --arm64 flag for x64 build
```

## Implementation Order

1. **Phase 1: CopyPackA/CopyPackB routines (AVX2)**
   - Simpler routines, establish basic MASM structure
   - Test packing and sum computation independently

2. **Phase 2: Main kernel (AVX2)**
   - Core multiply-accumulate logic
   - Full tile processing and edge case handling

3. **Phase 3: AVX512 variant**
   - Copy AVX2 structure, expand to zmm registers
   - Add conditional compilation for AVX512 features

4. **Phase 4: Integration and testing**
   - Update C++ dispatcher
   - CMake integration
   - Full system testing

## References

### Existing MLAS Code
- `QgemmU8U8KernelAvx2.asm` - U8×U8 kernel reference
- `QgemmU8X8KernelAvx512Core.asm` - AVX512 kernel patterns
- `QgemmU16U8KernelNeon.asm` - Algorithm reference

### Intel Intrinsics
- `vpmovzxbw` - Zero extend byte to word
- `vpmaddwd` - Multiply and add packed integers
- `vpaddd` - Add packed doubleword integers
- `vpbroadcastd` - Broadcast 32-bit value

---

*Created: 2026-01-11*
*Project: QUInt16×QUInt8 QGEMM x64 Implementation*
