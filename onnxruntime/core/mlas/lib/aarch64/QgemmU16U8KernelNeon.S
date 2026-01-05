/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU16U8KernelNeon.S

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM) for QUInt16 activations × QUInt8 weights.

--*/

#include "asmmacro.h"

//
// Stack frame layout
//

#define GemmU16U8KernelFrame_SavedRegisters         (8 * 8)
#define GemmU16U8KernelFrame_ColumnSumBuffer        (8 * 8)
#define GemmU16U8KernelFrame_ZeroPointB             (8 * 8 + 8)
#define GemmU16U8KernelFrame_ZeroMode               (8 * 8 + 16)

        .text

/*++

Routine Description:

    This routine is the inner kernel to compute matrix multiplication for a
    set of rows with QUInt16 activations and QUInt8 weights.

Arguments:

    x0 - A (uint16_t*)
    x1 - B (uint8_t*)
    x2 - C (int32_t*)
    x3 - PackedCountK
    x4 - CountM
    x5 - CountN
    x6 - ldc
    x7 - RowSumBuffer
    [sp, #GemmU16U8KernelFrame_ColumnSumBuffer] - ColumnSumBuffer
    [sp, #GemmU16U8KernelFrame_ZeroPointB] - ZeroPointB
    [sp, #GemmU16U8KernelFrame_ZeroMode] - ZeroMode

Return Value:

    x0 - Number of rows handled

--*/

        FUNCTION_ENTRY MlasGemmU16U8KernelNeon

        stp     d8, d9, [sp, #-64]!
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]

        ldr     x8, [sp, #GemmU16U8KernelFrame_ColumnSumBuffer]
        ldr     x9, [sp, #GemmU16U8KernelFrame_ZeroPointB]
        ldrb    w13, [sp, #GemmU16U8KernelFrame_ZeroMode]
        mov     x14, x0
        ld1     {v27.4s}, [x7]
        mov     x15, x3
        dup     v24.4s, v27.s[0]
        dup     v25.4s, v27.s[1]
        cmp     x4, #4
        b.lo    .LProcessRowsLessThan4
        dup     v26.4s, v27.s[2]
        dup     v27.4s, v27.s[3]

.LProcessNextColumnLoopM4:
        ld1     {v0.8b}, [x1], #8
        mov     x0, x14
        ldp     q8, q9, [x8], #32
        ldp     q10, q11, [x8], #32
        mov     x3, x15
        uxtl    v0.8h, v0.8b

        cbz     x9, .LSkipScaleByZeroPointBM4

        ldp     q28, q29, [x9], #32
        ldp     q30, q31, [x9], #32

        mul     v12.4s, v24.4s, v28.4s
        mul     v13.4s, v24.4s, v29.4s
        mul     v14.4s, v24.4s, v30.4s
        mul     v15.4s, v24.4s, v31.4s

        mul     v16.4s, v25.4s, v28.4s
        mul     v17.4s, v25.4s, v29.4s
        mul     v18.4s, v25.4s, v30.4s
        mul     v19.4s, v25.4s, v31.4s

        mul     v20.4s, v26.4s, v28.4s
        mul     v21.4s, v26.4s, v29.4s
        mul     v22.4s, v26.4s, v30.4s
        mul     v23.4s, v26.4s, v31.4s

        mul     v28.4s, v27.4s, v28.4s
        mul     v29.4s, v27.4s, v29.4s
        mul     v30.4s, v27.4s, v30.4s
        mul     v31.4s, v27.4s, v31.4s

        add     v12.4s, v8.4s, v12.4s
        add     v13.4s, v9.4s, v13.4s
        add     v14.4s, v10.4s, v14.4s
        add     v15.4s, v11.4s, v15.4s

        add     v16.4s, v8.4s, v16.4s
        add     v17.4s, v9.4s, v17.4s
        add     v18.4s, v10.4s, v18.4s
        add     v19.4s, v11.4s, v19.4s

        add     v20.4s, v8.4s, v20.4s
        add     v21.4s, v9.4s, v21.4s
        add     v22.4s, v10.4s, v22.4s
        add     v23.4s, v11.4s, v23.4s

        add     v8.4s, v8.4s, v28.4s
        add     v9.4s, v9.4s, v29.4s
        add     v10.4s, v10.4s, v30.4s
        add     v11.4s, v11.4s, v31.4s

        ldp     q4, q5, [x0], #32
        ldp     q6, q7, [x0], #32

        b       .LComputeBlockLoopM4

.LSkipScaleByZeroPointBM4:
        add     v12.4s, v8.4s, v24.4s
        add     v13.4s, v9.4s, v24.4s
        add     v14.4s, v10.4s, v24.4s
        add     v15.4s, v11.4s, v24.4s

        add     v16.4s, v8.4s, v25.4s
        add     v17.4s, v9.4s, v25.4s
        add     v18.4s, v10.4s, v25.4s
        add     v19.4s, v11.4s, v25.4s

        add     v20.4s, v8.4s, v26.4s
        add     v21.4s, v9.4s, v26.4s
        add     v22.4s, v10.4s, v26.4s
        add     v23.4s, v11.4s, v26.4s

        add     v8.4s, v8.4s, v27.4s
        add     v9.4s, v9.4s, v27.4s
        add     v10.4s, v10.4s, v27.4s
        add     v11.4s, v11.4s, v27.4s

        ldp     q4, q5, [x0], #32
        ldp     q6, q7, [x0], #32

.LComputeBlockLoopM4:
        umlal   v12.4s, v0.4h, v4.h[0]
        umlal   v13.4s, v0.4h, v4.h[1]
        umlal   v14.4s, v0.4h, v4.h[2]
        umlal   v15.4s, v0.4h, v4.h[3]

        umlal   v16.4s, v0.4h, v5.h[0]
        umlal   v17.4s, v0.4h, v5.h[1]
        umlal   v18.4s, v0.4h, v5.h[2]
        umlal   v19.4s, v0.4h, v5.h[3]

        ld1     {v1.8b}, [x1], #8

        umlal   v20.4s, v0.4h, v6.h[0]
        umlal   v21.4s, v0.4h, v6.h[1]
        umlal   v22.4s, v0.4h, v6.h[2]
        umlal   v23.4s, v0.4h, v6.h[3]

        uxtl    v1.8h, v1.8b

        umlal   v8.4s, v0.4h, v7.h[0]
        umlal   v9.4s, v0.4h, v7.h[1]
        umlal   v10.4s, v0.4h, v7.h[2]
        umlal   v11.4s, v0.4h, v7.h[3]

        umlal   v12.4s, v1.4h, v4.h[4]
        umlal   v13.4s, v1.4h, v4.h[5]
        umlal   v14.4s, v1.4h, v4.h[6]
        umlal   v15.4s, v1.4h, v4.h[7]

        umlal   v16.4s, v1.4h, v5.h[4]
        umlal   v17.4s, v1.4h, v5.h[5]
        umlal   v18.4s, v1.4h, v5.h[6]
        umlal   v19.4s, v1.4h, v5.h[7]

        ld1     {v0.8b}, [x1], #8

        umlal   v20.4s, v1.4h, v6.h[4]
        umlal   v21.4s, v1.4h, v6.h[5]
        umlal   v22.4s, v1.4h, v6.h[6]
        umlal   v23.4s, v1.4h, v6.h[7]

        sub     x3, x3, #1
        uxtl    v0.8h, v0.8b
        cbz     x3, .LComputeBlockLoopFinishM4

        umlal   v8.4s, v1.4h, v7.h[4]
        umlal   v9.4s, v1.4h, v7.h[5]
        umlal   v10.4s, v1.4h, v7.h[6]
        umlal   v11.4s, v1.4h, v7.h[7]

        ldp     q4, q5, [x0], #32
        ldp     q6, q7, [x0], #32

        b       .LComputeBlockLoopM4

.LComputeBlockLoopFinishM4:
        umlal   v8.4s, v1.4h, v7.h[4]
        umlal   v9.4s, v1.4h, v7.h[5]
        umlal   v10.4s, v1.4h, v7.h[6]
        umlal   v11.4s, v1.4h, v7.h[7]

        add     x10, x2, x6, lsl #2
        add     x11, x10, x6, lsl #2
        add     x12, x11, x6, lsl #2

        subs    x5, x5, #16
        b.lo    .LStoreOutputPartialM4

        cbnz    x13, .LSkipAccumulateOutputM4

        ldp     q28, q29, [x2]
        ldp     q30, q31, [x2, #32]
        add     v12.4s, v12.4s, v28.4s
        add     v13.4s, v13.4s, v29.4s
        add     v14.4s, v14.4s, v30.4s
        add     v15.4s, v15.4s, v31.4s

        ldp     q28, q29, [x10]
        ldp     q30, q31, [x10, #32]
        add     v16.4s, v16.4s, v28.4s
        add     v17.4s, v17.4s, v29.4s
        add     v18.4s, v18.4s, v30.4s
        add     v19.4s, v19.4s, v31.4s

        ldp     q28, q29, [x11]
        ldp     q30, q31, [x11, #32]
        add     v20.4s, v20.4s, v28.4s
        add     v21.4s, v21.4s, v29.4s
        add     v22.4s, v22.4s, v30.4s
        add     v23.4s, v23.4s, v31.4s

        ldp     q28, q29, [x12]
        ldp     q30, q31, [x12, #32]
        add     v8.4s, v8.4s, v28.4s
        add     v9.4s, v9.4s, v29.4s
        add     v10.4s, v10.4s, v30.4s
        add     v11.4s, v11.4s, v31.4s

.LSkipAccumulateOutputM4:
        stp     q12, q13, [x2], #32
        stp     q14, q15, [x2], #32
        stp     q16, q17, [x10]
        stp     q18, q19, [x10, #32]
        stp     q20, q21, [x11]
        stp     q22, q23, [x11, #32]
        stp     q8, q9, [x12]
        stp     q10, q11, [x12, #32]

        cbnz    x5, .LProcessNextColumnLoopM4

.LExitKernelM4:
        mov     x0, #4

        ldp     d14, d15, [sp, #48]
        ldp     d12, d13, [sp, #32]
        ldp     d10, d11, [sp, #16]
        ldp     d8, d9, [sp], #64
        ret

.LStoreOutputPartialM4:
        add     x5, x5, #16

.LStoreOutputPartialLoopM4:
        st1     {v12.s}[0], [x2], #4
        st1     {v16.s}[0], [x10], #4
        st1     {v20.s}[0], [x11], #4
        st1     {v8.s}[0], [x12], #4
        ext     v12.16b, v12.16b, v13.16b, #4
        ext     v16.16b, v16.16b, v17.16b, #4
        ext     v20.16b, v20.16b, v21.16b, #4
        ext     v8.16b, v8.16b, v9.16b, #4
        subs    x5, x5, #1
        b.hi    .LStoreOutputPartialLoopM4
        b       .LExitKernelM4

.LProcessRowsLessThan4:
        cmp     x4, #2
        b.lo    .LProcessOneRow
        mov     x0, #2
        b       .LExitKernelM4

.LProcessOneRow:
        mov     x0, #1
        b       .LExitKernelM4

        .end
