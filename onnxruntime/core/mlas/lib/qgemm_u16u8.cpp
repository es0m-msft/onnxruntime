/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_u16u8.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM) for QUInt16 activations × QUInt8 weights.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// External kernel declarations
//

#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)
extern "C" {
    size_t MLASCALL MlasGemmU16U8KernelNeon(
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
#endif

//
// U16U8 GEMM operation implementation
//

void
MLASCALL
MlasGemmU16U8Batch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U16U8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine implements QUInt16 x QUInt8 quantized matrix/matrix multiply.

    The operation performed is:
        C = (A - ZeroPointA) * (B - ZeroPointB)

    Where:
        A is uint16_t activations
        B is uint8_t weights
        C is int32_t accumulation

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    DataParams - Supplies the array of structures containing the GEMM data layout.

    BatchN - Supplies the number of GEMM operations to perform.

    ThreadPool - Supplies the thread pool object to use for parallelization.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)

    // ThreadPool is reserved for future parallel implementation
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;

    // Process each batch
    for (size_t batch = 0; batch < BatchN; batch++) {
        const auto* params = &DataParams[batch];

        const uint16_t* A = params->A;
        const uint8_t* B = params->B;
        int32_t* C = params->C;

        const size_t lda = params->lda;
        const size_t ldb = params->ldb;
        const size_t ldc = params->ldc;

        const int32_t ZeroPointA = static_cast<int32_t>(params->ZeroPointA);
        const uint8_t* ZeroPointBPtr = params->ZeroPointB;
        const bool PerColumnZeroPoints = params->PerColumnZeroPoints;

        // Allocate temporary buffers for row/column sums
        // These are used to compute the zero-point corrections:
        // (A - ZeroPointA) * (B - ZeroPointB) = A*B - A*ZeroPointB - B*ZeroPointA + ZeroPointA*ZeroPointB

        constexpr size_t StrideM = 4;  // Process 4 rows at a time (from kernel)
        constexpr size_t StrideN = 16; // Process 16 columns at a time (from kernel)
        constexpr size_t PackedK = 8;  // K dimension packing (8 elements per block)

        const size_t PackedCountK = (K + PackedK - 1) / PackedK;

        std::vector<int32_t> RowSumBuffer(StrideM);
        std::vector<int32_t> ColumnSumBuffer(StrideN);
        std::vector<int32_t> ZeroPointBBuffer(StrideN);

        // PERFORMANCE FIX: Pre-compute ALL row sums once (was being recomputed for every column tile)
        std::vector<int32_t> AllRowSums(M);
        for (size_t m = 0; m < M; m++) {
            int32_t row_sum = 0;
            for (size_t k = 0; k < K; k++) {
                row_sum += static_cast<int32_t>(A[m * lda + k]);
            }
            AllRowSums[m] = row_sum;
        }

        // PERFORMANCE FIX: Pre-compute ALL column sums once (was being recomputed for every row tile)
        std::vector<int32_t> AllColumnSums(N);
        for (size_t n = 0; n < N; n++) {
            int32_t col_sum = 0;
            for (size_t k = 0; k < K; k++) {
                col_sum += static_cast<int32_t>(B[k * ldb + n]);
            }
            AllColumnSums[n] = col_sum;
        }

        // Process in tiles
        for (size_t m = 0; m < M; m += StrideM) {
            const size_t CountM = std::min(M - m, StrideM);

            // Copy pre-computed row sums for this tile
            for (size_t mm = 0; mm < CountM; mm++) {
                RowSumBuffer[mm] = AllRowSums[m + mm];
            }

            for (size_t n = 0; n < N; n += StrideN) {
                const size_t CountN = std::min(N - n, StrideN);

                // Copy pre-computed column sums for this tile
                for (size_t nn = 0; nn < CountN; nn++) {
                    ColumnSumBuffer[nn] = AllColumnSums[n + nn];
                }

                // Setup zero point B buffer
                const int32_t* ZeroPointBParam = nullptr;
                if (ZeroPointBPtr != nullptr) {
                    if (PerColumnZeroPoints) {
                        for (size_t nn = 0; nn < CountN; nn++) {
                            ZeroPointBBuffer[nn] = -static_cast<int32_t>(ZeroPointBPtr[n + nn]);
                        }
                        ZeroPointBParam = ZeroPointBBuffer.data();
                    } else {
                        const int32_t zpb = -static_cast<int32_t>(ZeroPointBPtr[0]);
                        std::fill(ZeroPointBBuffer.begin(), ZeroPointBBuffer.begin() + CountN, zpb);
                        ZeroPointBParam = ZeroPointBBuffer.data();
                    }
                }

                // Apply zero point corrections to column sums
                for (size_t nn = 0; nn < CountN; nn++) {
                    ColumnSumBuffer[nn] = -ColumnSumBuffer[nn] * ZeroPointA;
                }

                // Apply zero point B correction to row sums
                if (ZeroPointBPtr != nullptr) {
                    for (size_t mm = 0; mm < CountM; mm++) {
                        if (PerColumnZeroPoints) {
                            // Will be handled inside kernel
                            RowSumBuffer[mm] -= static_cast<int32_t>(K) * ZeroPointA;
                        } else {
                            const int32_t zpb = static_cast<int32_t>(ZeroPointBPtr[0]);
                            RowSumBuffer[mm] = -RowSumBuffer[mm] * zpb
                                             - static_cast<int32_t>(K) * ZeroPointA;
                        }
                    }
                } else {
                    for (size_t mm = 0; mm < CountM; mm++) {
                        RowSumBuffer[mm] -= static_cast<int32_t>(K) * ZeroPointA;
                    }
                }

                // Call the ARM64 NEON kernel
                bool ZeroMode = true; // Always write, don't accumulate

                size_t RowsHandled = MlasGemmU16U8KernelNeon(
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    PackedCountK,
                    CountM,
                    CountN,
                    ldc,
                    RowSumBuffer.data(),
                    ColumnSumBuffer.data(),
                    ZeroPointBParam,
                    ZeroMode
                );

                // Apply output post-processing if specified
                if (params->OutputProcessor != nullptr) {
                    params->OutputProcessor->Process(
                        C,
                        m,
                        n,
                        RowsHandled,
                        CountN,
                        ldc
                    );
                }
            }
        }
    }

#else
    // U16U8 GEMM is currently only implemented for ARM64
    MLAS_UNREFERENCED_PARAMETER(Shape);
    MLAS_UNREFERENCED_PARAMETER(DataParams);
    MLAS_UNREFERENCED_PARAMETER(BatchN);
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    throw std::runtime_error("QUInt16 x QUInt8 GEMM is only supported on ARM64 platforms");
#endif
}
