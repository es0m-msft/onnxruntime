;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU16U8KernelAvx512Core.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM) for QUInt16 activations x QUInt8 weights.
;
;   This implementation uses AVX512 core (BW/DQ/VL) instructions for maximum
;   throughput on modern x64 processors.
;
;   Key features:
;   - 512-bit zmm registers (32 x uint16 per register)
;   - Process 6 rows x 48 columns per iteration
;   - Mask registers for partial writes
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

;
; Stack frame layout for the U16U8 AVX512 kernel.
;

GemmU16U8KernelAvx512Frame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR15 QWORD ?
        SavedR14 QWORD ?
        SavedR13 QWORD ?
        SavedR12 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountM QWORD ?
        CountN QWORD ?
        ldc QWORD ?
        RowSumBuffer QWORD ?
        ColumnSumBuffer QWORD ?
        ZeroPointB QWORD ?
        ZeroMode QWORD ?

GemmU16U8KernelAvx512Frame ENDS

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate a single cell of the
;   output block for U16U8.
;
; Arguments:
;
;   AccumReg - Supplies the register to accumulate into.
;
;   Mult1Reg - Supplies the first multiplication operand register (A values, uint16).
;
;   Mult2Reg - Supplies the second multiplication operand register (B values, uint16).
;
; Implicit Arguments:
;
;   zmm4 - Supplies a scratch register for intermediate results.
;

MultiplyAccumulateCellU16U8 MACRO AccumReg, Mult1Reg, Mult2Reg

        vpmaddwd zmm4,Mult1Reg,Mult2Reg
        vpaddd  AccumReg,AccumReg,zmm4

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate each row of the output
;   block.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce (16, 32, or 48).
;
;   RowCount - Supplies the number of rows to produce (1-6).
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r14 - Supplies the stride in bytes between packed blocks of matrix B.
;
;   zmm14-zmm31 - Supplies the block accumulators.
;

ComputeBlockU16U8 MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

IF ColumnCount GE 48
        vpmovzxbw zmm0,YMMWORD PTR [rdx+VectorOffset]
        vpmovzxbw zmm1,YMMWORD PTR [rdx+r14+VectorOffset]
        vpmovzxbw zmm2,YMMWORD PTR [rdx+r14*2+VectorOffset]
ELSEIF ColumnCount GE 32
        vpmovzxbw zmm1,YMMWORD PTR [rdx+VectorOffset]
        vpmovzxbw zmm2,YMMWORD PTR [rdx+r14+VectorOffset]
ELSE
        vpmovzxbw zmm2,YMMWORD PTR [rdx+VectorOffset]
ENDIF
        ; Row 0
        EmitIfCountGE RowCount, 1, <vpbroadcastd zmm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm26,zmm3,zmm0>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm20,zmm3,zmm1>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm14,zmm3,zmm2>
        ; Row 1
        EmitIfCountGE RowCount, 2, <vpbroadcastd zmm3,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm27,zmm3,zmm0>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm21,zmm3,zmm1>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm15,zmm3,zmm2>
        ; Row 2
        EmitIfCountGE RowCount, 3, <vpbroadcastd zmm3,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm28,zmm3,zmm0>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm22,zmm3,zmm1>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm16,zmm3,zmm2>
        ; Row 3
        EmitIfCountGE RowCount, 4, <vpbroadcastd zmm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm29,zmm3,zmm0>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm23,zmm3,zmm1>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm17,zmm3,zmm2>
        ; Row 4
        EmitIfCountGE RowCount, 5, <vpbroadcastd zmm3,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm30,zmm3,zmm0>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm24,zmm3,zmm1>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm18,zmm3,zmm2>
        ; Row 5
        EmitIfCountGE RowCount, 6, <vpbroadcastd zmm3,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <MultiplyAccumulateCellU16U8 zmm31,zmm3,zmm0>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <MultiplyAccumulateCellU16U8 zmm25,zmm3,zmm1>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <MultiplyAccumulateCellU16U8 zmm19,zmm3,zmm2>

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple times
;   and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;

ComputeBlockLoopU16U8 MACRO ColumnCount, RowCount

        LOCAL   ComputeBlockBy1Loop

        mov     rsi,r9                      ; reload row length remaining

ComputeBlockBy1Loop:
        ComputeBlockU16U8 ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 pair of uint16
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32                      ; advance matrix B by 32 uint8
        sub     rsi,4
        jnz     ComputeBlockBy1Loop

        ENDM

;
; Macro Description:
;
;   This macro generates code to produce an output block for a set of columns
;   and rows.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;

ProduceOutputBlockU16U8 MACRO ColumnCount, RowCount

        LOCAL   SkipScaleByZeroPointB
        LOCAL   AccumulatorsInitialized
        LOCAL   ExitProduceOutputBlock

;
; Initialize the accumulators with the row and column sums.
;

IF ColumnCount GE 32
IF ColumnCount GE 48
        vmovdqu32 zmm2,ZMMWORD PTR [r12]
        vmovdqu32 zmm1,ZMMWORD PTR [r12+64]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+128]
ELSE
        vmovdqu32 zmm1,ZMMWORD PTR [r12]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+64]
ENDIF
        add_immed r12,ColumnCount*4         ; advance ColumnSumBuffer by N columns
ELSE
        vmovdqu32 zmm0,ZMMWORD PTR [r12]
ENDIF
        test    r13,r13                     ; per column zero points?
        jz      SkipScaleByZeroPointB
        ; Handle per-column zero points (similar to U8X8 kernel)
        jmp     AccumulatorsInitialized

SkipScaleByZeroPointB:
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd zmm14,zmm0,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpaddd zmm20,zmm1,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <vpaddd zmm26,zmm2,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd zmm15,zmm0,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpaddd zmm21,zmm1,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <vpaddd zmm27,zmm2,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd zmm16,zmm0,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpaddd zmm22,zmm1,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <vpaddd zmm28,zmm2,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd zmm17,zmm0,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpaddd zmm23,zmm1,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <vpaddd zmm29,zmm2,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd zmm18,zmm0,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpaddd zmm24,zmm1,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <vpaddd zmm30,zmm2,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd zmm19,zmm0,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpaddd zmm25,zmm1,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <vpaddd zmm31,zmm2,DWORD BCST [r11+20]>

AccumulatorsInitialized:

;
; Iterate over the length of a matrix A row to produce the output accumulators.
;

IF RowCount GT 3
        lea     rbx,[r9*2+r9]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoopU16U8 ColumnCount, RowCount

ExitProduceOutputBlock:
IF RowCount GT 3
        lea     rbx,[rax*2+rax]
        add     rbx,r8                      ; compute matrix C plus 3 rows
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute matrix multiplication for a fixed set
;   of rows using AVX512.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;

ProcessCountMU16U8 MACRO RowCount

        LOCAL   ProcessNextColumnLoop32xN
        LOCAL   Output32xNBlock
        LOCAL   SkipAccumulateOutput32xNBlock
        LOCAL   Output16xNBlock
        LOCAL   Output16xNBlockWithMask
        LOCAL   SkipAccumulateOutput16xNBlockWithMask
        LOCAL   ProcessRemainingCountN
        LOCAL   ProcessNextColumnLoop48xN
        LOCAL   SkipAccumulateOutput48xNBlock

        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop32xN:
        ProduceOutputBlockU16U8 32, RowCount
        add     rdx,r14                     ; advance matrix B by packed block stride

Output32xNBlock:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput32xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm20,zmm20,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm21,zmm21,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm22,zmm22,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm23,zmm23,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm25,zmm25,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput32xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm20>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm21>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm22>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm23>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm25>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16

Output16xNBlock:
        sub     rbp,16
        jae     Output16xNBlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     esi,1
        shl     esi,cl
        dec     esi
        kmovw   k1,esi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16xNBlockWithMask:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlockWithMask
        EmitIfCountGE RowCount, 1, <vpaddd zmm14{k1},zmm14,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm15{k1},zmm15,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm16{k1},zmm16,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm17{k1},zmm17,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm18{k1},zmm18,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm19{k1},zmm19,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput16xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8]{k1},zmm14>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax]{k1},zmm15>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2]{k1},zmm16>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm17>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm18>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm19>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        ja      ProcessNextColumnLoop32xN
        test    rbp,rbp
        jnz     ProcessRemainingCountN
        mov     eax,RowCount
        jmp     ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlockU16U8 16, RowCount
        jmp     Output16xNBlock

ProcessNextColumnLoop48xN:
        ProduceOutputBlockU16U8 48, RowCount
        lea     rdx,[rdx+r14*2]             ; advance matrix B by packed block stride
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput48xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm26,zmm26,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm27,zmm27,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm28,zmm28,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm29,zmm29,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm30,zmm30,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm31,zmm31,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput48xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm26>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm27>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm28>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm29>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm30>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm31>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16
        jmp     Output32xNBlock

        ENDM

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows using AVX512.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A (uint16_t). The matrix data has
;       been packed using MlasGemmU16U8CopyPackAAvx2.
;
;   B (rdx) - Supplies the address of matrix B (uint8_t). The matrix data has
;       been packed using MlasGemmU8U8CopyPackBAvx2.
;
;   C (r8) - Supplies the address of matrix C (int32_t output).
;
;   PackedCountK (r9) - Supplies the number of packed columns from matrix A and
;       the number of packed rows from matrix B to iterate over.
;
;   CountM - Supplies the maximum number of rows that can be processed.
;
;   CountN - Supplies the number of columns from matrix B and matrix C.
;
;   ldc - Supplies the first dimension of matrix C.
;
;   RowSumBuffer - Supplies the sum of each row from matrix A.
;
;   ColumnSumBuffer - Supplies the sum of each column from matrix B.
;
;   ZeroPointB - Optionally supplies the per-column zero point offsets.
;
;   ZeroMode - Supplies true if output must be zero initialized.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

        NESTED_ENTRY MlasGemmU16U8KernelAvx512Core, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        push_reg r14
        push_reg r15
        alloc_stack (GemmU16U8KernelAvx512Frame.SavedR15)
        save_xmm128 xmm6,GemmU16U8KernelAvx512Frame.SavedXmm6
        save_xmm128 xmm7,GemmU16U8KernelAvx512Frame.SavedXmm7
        save_xmm128 xmm8,GemmU16U8KernelAvx512Frame.SavedXmm8
        save_xmm128 xmm9,GemmU16U8KernelAvx512Frame.SavedXmm9
        save_xmm128 xmm10,GemmU16U8KernelAvx512Frame.SavedXmm10
        save_xmm128 xmm11,GemmU16U8KernelAvx512Frame.SavedXmm11
        save_xmm128 xmm12,GemmU16U8KernelAvx512Frame.SavedXmm12
        save_xmm128 xmm13,GemmU16U8KernelAvx512Frame.SavedXmm13
        save_xmm128 xmm14,GemmU16U8KernelAvx512Frame.SavedXmm14
        save_xmm128 xmm15,GemmU16U8KernelAvx512Frame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbx,GemmU16U8KernelAvx512Frame.CountM[rsp]
        mov     rbp,GemmU16U8KernelAvx512Frame.CountN[rsp]
        mov     rax,GemmU16U8KernelAvx512Frame.ldc[rsp]
        shl     rax,2                       ; convert ldc to bytes
        shl     r9,2                        ; convert to row length (uint16 pairs)
        movzx   r10d,BYTE PTR GemmU16U8KernelAvx512Frame.ZeroMode[rsp]
        mov     r11,GemmU16U8KernelAvx512Frame.RowSumBuffer[rsp]
        mov     r12,GemmU16U8KernelAvx512Frame.ColumnSumBuffer[rsp]
        mov     r13,GemmU16U8KernelAvx512Frame.ZeroPointB[rsp]
        mov     esi,-1
        kmovw   k1,esi                      ; update mask to write all columns
        ; Compute matrix B packed stride (each block processes 16 columns)
        lea     r14,[r9*8]                  ; stride = PackedCountK * 8 (U16U8 B packing)

;
; Process CountM rows of the matrices.
;

        cmp     rbx,5
        ja      ProcessCountM6
        je      ProcessCountM5
        cmp     rbx,3
        ja      ProcessCountM4
        je      ProcessCountM3
        cmp     rbx,1
        ja      ProcessCountM2

ProcessCountM1:
        ProcessCountMU16U8 1

ProcessCountM2:
        ProcessCountMU16U8 2

ProcessCountM3:
        ProcessCountMU16U8 3

ProcessCountM4:
        ProcessCountMU16U8 4

ProcessCountM5:
        ProcessCountMU16U8 5

ProcessCountM6:
        ProcessCountMU16U8 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,GemmU16U8KernelAvx512Frame.SavedXmm6[rsp]
        movaps  xmm7,GemmU16U8KernelAvx512Frame.SavedXmm7[rsp]
        movaps  xmm8,GemmU16U8KernelAvx512Frame.SavedXmm8[rsp]
        movaps  xmm9,GemmU16U8KernelAvx512Frame.SavedXmm9[rsp]
        movaps  xmm10,GemmU16U8KernelAvx512Frame.SavedXmm10[rsp]
        movaps  xmm11,GemmU16U8KernelAvx512Frame.SavedXmm11[rsp]
        movaps  xmm12,GemmU16U8KernelAvx512Frame.SavedXmm12[rsp]
        movaps  xmm13,GemmU16U8KernelAvx512Frame.SavedXmm13[rsp]
        movaps  xmm14,GemmU16U8KernelAvx512Frame.SavedXmm14[rsp]
        movaps  xmm15,GemmU16U8KernelAvx512Frame.SavedXmm15[rsp]
        add     rsp,(GemmU16U8KernelAvx512Frame.SavedR15)

        BEGIN_EPILOGUE

        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasGemmU16U8KernelAvx512Core, _TEXT

        END
