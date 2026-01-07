#!/usr/bin/env python3
"""Patch ONNX QLinearMatMul opset 10 schema to add uint16 support"""

import re

file_path = "cmake/external/onnx/onnx/defs/math/old.cc"

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace T1 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input a and its zero point data type to 8-bit integer tensor.")''',
    '''        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)", "tensor(uint16)"},
            "Constrain input a and its zero point data type to 8-bit or 16-bit integer tensor.")'''
)

# Find and replace T2 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input b and its zero point data type to 8-bit integer tensor.")''',
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)", "tensor(uint16)"},
            "Constrain input b and its zero point data type to 8-bit or 16-bit integer tensor.")'''
)

# Find and replace T3 type constraint
content = content.replace(
    '''        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain output y and its zero point data type to 8-bit integer tensor.")''',
    '''        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)", "tensor(uint16)"},
            "Constrain output y and its zero point data type to 8-bit or 16-bit integer tensor.")'''
)

with open(file_path, 'w') as f:
    f.write(content)

print("Successfully patched ONNX QLinearMatMul opset 10 schema to add uint16 support")
