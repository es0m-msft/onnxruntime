#!/usr/bin/env python3
"""Patch ONNX QLinearMatMul schema to add uint16 support"""

import re

file_path = "cmake/external/onnx/onnx/defs/math/defs.cc"

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace T1 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(float8e4m3fn)",''',
    '''        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",'''
)

# Find and replace T2 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(float8e4m3fn)",''',
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",'''
)

# Find and replace T3 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T3",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(float8e4m3fn)",''',
    '''        .TypeConstraint(
            "T3",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",'''
)

with open(file_path, 'w') as f:
    f.write(content)

print("Successfully patched ONNX QLinearMatMul schema to add uint16 support")
