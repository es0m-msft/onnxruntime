# Performance comparison using onnxruntime_perf_test.exe
# Compares FP32, QUInt8, and QUInt16 quantization

$ErrorActionPreference = "Stop"

$PERF_TEST = "C:\d\onnxruntime\build_arm64_u16u8\Release\Release\onnxruntime_perf_test.exe"

# Check if perf_test exists
if (-not (Test-Path $PERF_TEST)) {
    Write-Host "[ERROR] onnxruntime_perf_test.exe not found at: $PERF_TEST" -ForegroundColor Red
    Write-Host "Please build ONNX Runtime first." -ForegroundColor Yellow
    exit 1
}

# Models to test
$models = @(
    @{
        Name = "FP32_Baseline"
        Path = "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\model\florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx"
        Description = "Float32 baseline"
    },
    @{
        Name = "QUInt8_QDQ"
        Path = "C:\d\onnxruntime\quantized_models\florence_quint8_qdq.onnx"
        Description = "QUInt8x8 QDQ quantization"
    },
    @{
        Name = "QUInt16_QDQ"
        Path = "C:\d\onnxruntime\quantized_models\florence_quint16_qdq_proper.onnx"
        Description = "QUInt16x8 QDQ quantization (with cleanup)"
    }
)

Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host "ONNX RUNTIME PERFORMANCE TEST COMPARISON" -ForegroundColor Cyan
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host ""
Write-Host "Tool: onnxruntime_perf_test.exe" -ForegroundColor Yellow
Write-Host "Runs: 20 iterations per model" -ForegroundColor Yellow
Write-Host ""

$results = @()
$modelNum = 1

foreach ($model in $models) {
    Write-Host ""
    Write-Host "=" * 100 -ForegroundColor Cyan
    Write-Host "TEST $modelNum/$($models.Count): $($model.Name)" -ForegroundColor Cyan
    Write-Host "=" * 100 -ForegroundColor Cyan
    Write-Host "Description: $($model.Description)" -ForegroundColor Gray
    Write-Host "Model: $($model.Path)" -ForegroundColor Gray
    Write-Host ""

    # Check if model exists
    if (-not (Test-Path $model.Path)) {
        Write-Host "[ERROR] Model not found!" -ForegroundColor Red
        Write-Host "Skipping..." -ForegroundColor Yellow
        continue
    }

    # Get model size
    $modelSize = (Get-Item $model.Path).Length / 1MB
    Write-Host "Size: $([math]::Round($modelSize, 2)) MB" -ForegroundColor Gray
    Write-Host ""

    # Run performance test
    Write-Host "Running performance test..." -ForegroundColor Yellow
    Write-Host "Command: onnxruntime_perf_test.exe -m times -r 20 -I `"$($model.Path)`"" -ForegroundColor DarkGray
    Write-Host ""

    try {
        $output = & $PERF_TEST -m times -r 20 -I $model.Path 2>&1 | Out-String

        Write-Host $output

        # Parse results
        if ($output -match "Average:\s+(\d+\.?\d*)\s*ms") {
            $avgTime = [double]$Matches[1]
            $results += @{
                Name = $model.Name
                Description = $model.Description
                Path = $model.Path
                Size_MB = $modelSize
                Average_ms = $avgTime
            }
            Write-Host "[OK] Test completed successfully" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Could not parse average time from output" -ForegroundColor Yellow
        }

    } catch {
        Write-Host "[ERROR] Test failed: $_" -ForegroundColor Red
    }

    # Cool down before next test (except for last one)
    if ($modelNum -lt $models.Count) {
        Write-Host ""
        Write-Host "Cooling down for 30 seconds..." -ForegroundColor Yellow

        for ($i = 30; $i -gt 0; $i--) {
            Write-Host -NoNewline "`r  Remaining: $i seconds  "
            Start-Sleep -Seconds 1
        }

        Write-Host ""
        Write-Host "[OK] Cool-down complete" -ForegroundColor Green
    }

    $modelNum++
}

# Print comparison table
Write-Host ""
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host "PERFORMANCE COMPARISON SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host ""

if ($results.Count -gt 0) {
    Write-Host ("=" * 100)
    Write-Host ("{0,-20} {1,-40} {2,-12} {3,-12} {4,-12}" -f "Model", "Description", "Size (MB)", "Avg (ms)", "vs FP32")
    Write-Host ("-" * 100)

    $fp32Time = ($results | Where-Object { $_.Name -like "*FP32*" }).Average_ms

    foreach ($result in $results) {
        $speedup = if ($fp32Time -and $result.Average_ms) {
            $ratio = $fp32Time / $result.Average_ms
            "{0:F2}x" -f $ratio
        } else {
            "N/A"
        }

        Write-Host ("{0,-20} {1,-40} {2,-12:F2} {3,-12:F2} {4,-12}" -f `
            $result.Name, `
            $result.Description, `
            $result.Size_MB, `
            $result.Average_ms, `
            $speedup)
    }

    Write-Host ("=" * 100)

    # Save results to JSON
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = "C:\d\onnxruntime\runtime_analysis\perf_test_results_$timestamp.json"

    $results | ConvertTo-Json | Out-File -FilePath $outputFile -Encoding UTF8

    Write-Host ""
    Write-Host "Results saved to: $outputFile" -ForegroundColor Green

} else {
    Write-Host "[WARNING] No results to display" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host "BENCHMARK COMPLETE" -ForegroundColor Cyan
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host ""
