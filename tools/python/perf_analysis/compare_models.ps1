# Compare Model Variants - Performance Comparison Script
#
# Compares embimg_c.quant.onnx vs embimg_c.quant.fixed2.onnx
#
# Usage: .\compare_models.ps1 [-Runs 20]

param(
    [int]$Runs = 20
)

$PERF_TEST = "C:\d\onnxruntime\build_arm64_u16u8\Release\Release\onnxruntime_perf_test.exe"
$MODEL_DIR = "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer"

Write-Host "=== Testing Original Model (embimg_c.quant.onnx) ===" -ForegroundColor Cyan
& $PERF_TEST -I "$MODEL_DIR\embimg_c.quant.onnx" -m times -r $Runs | Select-String "Average|P50|P90|CPU"

Start-Sleep -Seconds 5

Write-Host "`n=== Testing Fixed Model (embimg_c.quant.fixed2.onnx) ===" -ForegroundColor Cyan
& $PERF_TEST -I "$MODEL_DIR\embimg_c.quant.fixed2.onnx" -m times -r $Runs | Select-String "Average|P50|P90|CPU"

Write-Host "`n=== Comparison Complete ===" -ForegroundColor Green
