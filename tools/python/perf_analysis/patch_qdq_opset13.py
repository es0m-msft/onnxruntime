#!/usr/bin/env python3
"""Patch ONNX Q/DQ opset 13 schemas to add uint16 support"""

import re

file_path = "cmake/external/onnx/onnx/defs/quantization/old.cc"

with open(file_path, 'r') as f:
    content = f.read()

# Patch DequantizeLinear opset 13
content = content.replace(
    '''        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")''',
    '''        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(uint16)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/16-bit/32-bit integer tensor.")'''
)

# Patch QuantizeLinear opset 13
content = content.replace(
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")''',
    '''        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)", "tensor(uint16)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit or 16-bit integer tensor.")'''
)

with open(file_path, 'w') as f:
    f.write(content)

print("Successfully patched ONNX Q/DQ opset 13 schemas to add uint16 support")
