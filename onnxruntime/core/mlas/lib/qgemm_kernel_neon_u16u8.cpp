/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_neon_u16u8.cpp

Abstract:

    This module implements QGEMM kernel for QUInt16 activations × QUInt8 weights
    on ARM64 NEON.

--*/

#include "mlasi.h"
#include "qgemm.h"

#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)

//
// Define the prototypes of the NEON routines written in assembly text.
//

extern "C" {
    size_t
    MLASCALL
    MlasGemmU16U8KernelNeon(
        const uint16_t* A,
        const uint8_t* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumBuffer,
        const int32_t* ColumnSumBuffer,
        const int32_t* ZeroPointB,
        bool ZeroMode
        );
}

//
// Kernel structure for QUInt16 × QUInt8 mixed precision QGEMM.
//

struct MLAS_GEMM_U16U8_KERNEL_NEON
{
    typedef uint16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint16_t OffsetAType;
    typedef uint8_t OffsetBType;

    // Optimal strides for ARM64 NEON
    // PackedK = 8: process 8 elements at a time (fits in vector registers)
    // M = 4: process 4 rows at a time (optimal register pressure)
    // N = 128: good balance for cache line utilization
    // K = 128: sufficient depth for amortizing overhead
    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 4, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 4, 128, 128 };
};

constexpr size_t MLAS_GEMM_U16U8_KERNEL_NEON::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U16U8_KERNEL_NEON::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U16U8_KERNEL_NEON::PackedStrides;

//
// Zero point fixup for A (uint16 activations).
// No transformation needed for unsigned types.
//

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_U16U8_KERNEL_NEON>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    // QInt16: need to flip sign bit
    if (AIsSigned) {
        ZeroPointA = (uint16_t)(ZeroPointA ^ 0x8000);
    }

    return ZeroPointA;
}

//
// Zero point fixup for B (uint8 weights).
//

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U16U8_KERNEL_NEON>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    // QInt8: need to flip sign bit
    if (BIsSigned) {
        ZeroPointB = (uint8_t)(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

//
// Pack A matrix (uint16 activations) for optimal cache and SIMD access.
//
// Layout: [M, K] -> [M, K_packed] where K_packed = (K + PackedK - 1) / PackedK * PackedK
// Packing ensures:
// 1. Aligned access for SIMD loads
// 2. Contiguous memory access within inner loops
// 3. Zero-padding for incomplete blocks
//

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U16U8_KERNEL_NEON>(
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    const size_t AlignedCountK = (CountK + MLAS_GEMM_U16U8_KERNEL_NEON::PackedK - 1) &
                                 ~(MLAS_GEMM_U16U8_KERNEL_NEON::PackedK - 1);

    const uint16x8_t BitFlipVector = vdupq_n_u16(AIsSigned ? 0x8000 : 0);

    // Cast input to uint16 (A matrix is uint16)
    const uint16_t* A16 = reinterpret_cast<const uint16_t*>(A);
    const size_t lda16 = lda / sizeof(uint16_t);

    //
    // Process each row of matrix A using NEON SIMD.
    //
    // This processes 8 uint16 elements at a time, providing ~8x speedup
    // over the previous scalar implementation.
    //

    for (size_t m = 0; m < CountM; m++) {

        // NEON-accelerated row sum computation
        uint32x4_t sum_vec_lo = vdupq_n_u32(0);
        uint32x4_t sum_vec_hi = vdupq_n_u32(0);

        // Process 8 uint16 elements at a time
        size_t k;
        for (k = 0; k + 8 <= CountK; k += 8) {
            // Load 8 uint16 values
            uint16x8_t data = vld1q_u16(A16 + k);

            // Apply bit flip if signed
            if (AIsSigned) {
                data = veorq_u16(data, BitFlipVector);
            }

            // Store packed data
            vst1q_u16(D + k, data);

            // Widen to 2x uint32x4_t and accumulate
            uint32x4_t data_lo = vmovl_u16(vget_low_u16(data));
            uint32x4_t data_hi = vmovl_high_u16(data);

            sum_vec_lo = vaddq_u32(sum_vec_lo, data_lo);
            sum_vec_hi = vaddq_u32(sum_vec_hi, data_hi);
        }

        // Horizontal sum of vector accumulators
        sum_vec_lo = vaddq_u32(sum_vec_lo, sum_vec_hi);
        uint32x2_t sum_pair = vadd_u32(vget_low_u32(sum_vec_lo), vget_high_u32(sum_vec_lo));
        int32_t RowSum = static_cast<int32_t>(vget_lane_u32(sum_pair, 0) + vget_lane_u32(sum_pair, 1));

        // Handle remaining elements (< 8)
        for (; k < CountK; k++) {
            uint16_t a0 = A16[k];
            if (AIsSigned) {
                a0 ^= 0x8000;
            }
            D[k] = a0;
            RowSum += a0;
        }

        // Zero-pad to aligned boundary
        for (; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *RowSumBuffer++ = RowSum;

        A16 += lda16;
        D += AlignedCountK;
    }
}

//
// Pack B matrix (uint8 weights) for optimal cache and SIMD access.
//
// Layout: Transpose [K, N] -> [N, K_packed]
// Benefits:
// 1. Column-major to row-major transformation for better cache locality
// 2. Aligned and zero-padded for SIMD operations
// 3. Contiguous access pattern in inner kernel loop
//

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U16U8_KERNEL_NEON>(
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U16U8_KERNEL_NEON::PackedK - 1) &
        ~(MLAS_GEMM_U16U8_KERNEL_NEON::PackedK - 1);

    const uint8x16_t BitFlipVector = vdupq_n_u8(BIsSigned ? 0x80 : 0);

    //
    // Process each column of matrix B using NEON SIMD.
    //
    // Note: Column data is strided (not contiguous) by ldb bytes between elements.
    // Even with strided access, NEON provides 4-8x speedup over scalar code.
    //

    for (size_t n = 0; n < CountN; n++) {

        const uint8_t* b = B;

        // NEON-accelerated column sum computation
        uint32x4_t sum_vec = vdupq_n_u32(0);

        // Process 16 uint8 elements at a time
        size_t k;
        for (k = 0; k + 16 <= CountK; k += 16) {
            // Load 16 uint8 values from strided locations
            // This is slower than contiguous access but still faster than scalar
            uint8_t vals[16];
            for (size_t i = 0; i < 16; i++) {
                vals[i] = b[i * ldb];
            }

            // Load into NEON vector and apply bit flip
            uint8x16_t data = vld1q_u8(vals);
            if (BIsSigned) {
                data = veorq_u8(data, BitFlipVector);
            }

            // Store packed data (now contiguous)
            vst1q_u8(D + k, data);

            // Widen uint8 -> uint16 -> uint32 and accumulate
            uint16x8_t data_lo = vmovl_u8(vget_low_u8(data));
            uint16x8_t data_hi = vmovl_high_u8(data);

            uint32x4_t data_u32_0 = vmovl_u16(vget_low_u16(data_lo));
            uint32x4_t data_u32_1 = vmovl_high_u16(data_lo);
            uint32x4_t data_u32_2 = vmovl_u16(vget_low_u16(data_hi));
            uint32x4_t data_u32_3 = vmovl_high_u16(data_hi);

            sum_vec = vaddq_u32(sum_vec, data_u32_0);
            sum_vec = vaddq_u32(sum_vec, data_u32_1);
            sum_vec = vaddq_u32(sum_vec, data_u32_2);
            sum_vec = vaddq_u32(sum_vec, data_u32_3);

            b += 16 * ldb;
        }

        // Horizontal sum of vector accumulator
        uint32x2_t sum_pair = vadd_u32(vget_low_u32(sum_vec), vget_high_u32(sum_vec));
        int32_t ColumnSum = static_cast<int32_t>(vget_lane_u32(sum_pair, 0) + vget_lane_u32(sum_pair, 1));

        // Handle remaining elements (< 16)
        for (; k < CountK; k++) {
            uint8_t b0 = b[0];
            if (BIsSigned) {
                b0 ^= 0x80;
            }
            D[k] = b0;
            ColumnSum += b0;
            b += ldb;
        }

        // Zero-pad to aligned boundary
        for (; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *ColumnSumBuffer++ = ColumnSum;

        B += 1;
        D += AlignedCountK;
    }
}

//
// QGEMM kernel dispatcher.
//
// This function performs the core matrix multiplication:
// C (int32) = A (uint16) × B (uint8)
//
// The kernel processes blocks of the matrix using the assembly-optimized
// NEON implementation, with proper handling of zero-points and accumulation.
//

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U16U8_KERNEL_NEON>(
    const MLAS_GEMM_U16U8_KERNEL_NEON::PackedAType* A,
    const MLAS_GEMM_U16U8_KERNEL_NEON::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasGemmU16U8KernelNeon(
        A,
        B,
        C,
        PackedCountK,
        CountM,
        CountN,
        ldc,
        RowSumBuffer,
        ColumnSumBuffer,
        ZeroPointB,
        ZeroMode
        );
}

//
// Explicit template instantiations to ensure code generation
//

template void MlasGemmQuantOperation<MLAS_GEMM_U16U8_KERNEL_NEON>(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS*, const MLAS_GEMM_QUANT_DATA_PARAMS*,
    size_t, size_t, size_t, size_t);

template void MlasGemmQuantPackedOperation<MLAS_GEMM_U16U8_KERNEL_NEON>(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS*, const MLAS_GEMM_QUANT_DATA_PARAMS*,
    size_t, size_t, size_t, size_t);

template void MlasGemmQuantCopyPackB<MLAS_GEMM_U16U8_KERNEL_NEON>(
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedBType*, const uint8_t*, size_t, size_t, size_t,
    int32_t*, bool);

//
// Quantized GEMM operation entry point.
//
// This is the main entry point called by higher-level APIs.
// It handles:
// 1. Matrix packing and preparation
// 2. Zero-point compensation
// 3. Dispatching to the optimized kernel
// 4. Post-processing and requantization
//

extern const MLAS_GEMM_QUANT_DISPATCH MlasGemmU16U8DispatchNeon = {
    MlasGemmQuantOperation<MLAS_GEMM_U16U8_KERNEL_NEON>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U16U8_KERNEL_NEON>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U16U8_KERNEL_NEON>,
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedK,
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedStrides.K,
    MLAS_GEMM_U16U8_KERNEL_NEON::PackedStrides.M  // StrideM: rows processed per iteration
};

#endif  // MLAS_TARGET_ARM64 || MLAS_TARGET_ARM64EC
