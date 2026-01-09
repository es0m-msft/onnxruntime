# Thermally-controlled benchmark runner
# Runs each model in a separate process with cool-down periods

$ErrorActionPreference = "Stop"

$PYTHON = "C:\d\onnxruntime\.venv_quant_test\Scripts\python.exe"
$SCRIPT_DIR = "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer"
$BENCHMARK_SCRIPT = "C:\d\onnxruntime\benchmark_isolated.py"

# Models to benchmark
$models = @(
    @{
        Name = "FP32_Baseline"
        Path = "C:\d\models\florence_v1_6_2_d3_tulrv6_multi_text_transformer\model\florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx"
        EnableQDQ = $false
    },
    @{
        Name = "QUInt16_Proper"
        Path = "C:\d\onnxruntime\quantized_models\florence_quint16_qdq_proper.onnx"
        EnableQDQ = $true
    },
    @{
        Name = "QUInt16_Old"
        Path = "C:\d\onnxruntime\quantized_models\florence_v1_6_2_d3_tulrv6_multi_text_transformer_quint16_qdq.onnx"
        EnableQDQ = $false
    }
)

Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host "THERMALLY-CONTROLLED BENCHMARK SUITE" -ForegroundColor Cyan
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host ""
Write-Host "Strategy:" -ForegroundColor Yellow
Write-Host "  - Run each model in separate Python process" -ForegroundColor Yellow
Write-Host "  - 60 second cool-down between models" -ForegroundColor Yellow
Write-Host "  - 30 benchmark runs per model (after 10 warmup)" -ForegroundColor Yellow
Write-Host "  - Results saved to isolated JSON files" -ForegroundColor Yellow
Write-Host ""

# Copy benchmark script to working directory
Copy-Item $BENCHMARK_SCRIPT $SCRIPT_DIR -Force
Write-Host "[OK] Benchmark script copied" -ForegroundColor Green
Write-Host ""

$results = @()
$modelNum = 1

foreach ($model in $models) {
    Write-Host ""
    Write-Host "=" * 100 -ForegroundColor Cyan
    Write-Host "MODEL $modelNum/$($models.Count): $($model.Name)" -ForegroundColor Cyan
    Write-Host "=" * 100 -ForegroundColor Cyan
    Write-Host ""

    # Check if model exists
    if (-not (Test-Path $model.Path)) {
        Write-Host "[ERROR] Model not found: $($model.Path)" -ForegroundColor Red
        Write-Host "Skipping..." -ForegroundColor Yellow
        continue
    }

    Write-Host "[INFO] Starting benchmark..." -ForegroundColor Yellow
    Write-Host "Path: $($model.Path)" -ForegroundColor Gray

    # Build command
    $qdqFlag = if ($model.EnableQDQ) { "1" } else { "0" }
    $args = @(
        "benchmark_isolated.py",
        $model.Name,
        $model.Path,
        $qdqFlag
    )

    # Run benchmark in isolated process
    try {
        Push-Location $SCRIPT_DIR
        & $PYTHON @args

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "[OK] Benchmark completed successfully" -ForegroundColor Green

            # Find the most recent result file
            $resultPattern = "$($model.Name.Replace(' ', '_'))_*.json"
            $resultDir = "C:\d\onnxruntime\runtime_analysis\isolated_benchmarks"
            $resultFile = Get-ChildItem -Path $resultDir -Filter $resultPattern |
                          Sort-Object LastWriteTime -Descending |
                          Select-Object -First 1

            if ($resultFile) {
                $results += @{
                    Model = $model.Name
                    ResultFile = $resultFile.FullName
                }
            }

        } else {
            Write-Host "[ERROR] Benchmark failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        }

    } catch {
        Write-Host "[ERROR] Exception: $_" -ForegroundColor Red
    } finally {
        Pop-Location
    }

    # Cool down before next model (except for last one)
    if ($modelNum -lt $models.Count) {
        Write-Host ""
        Write-Host "Cooling down for 60 seconds before next model..." -ForegroundColor Yellow

        for ($i = 60; $i -gt 0; $i--) {
            Write-Host -NoNewline "`r  Remaining: $i seconds  "
            Start-Sleep -Seconds 1
        }

        Write-Host ""
        Write-Host "[OK] Cool-down complete" -ForegroundColor Green
    }

    $modelNum++
}

# Print summary
Write-Host ""
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host "BENCHMARK SUITE COMPLETE" -ForegroundColor Cyan
Write-Host "=" * 100 -ForegroundColor Cyan
Write-Host ""

if ($results.Count -gt 0) {
    Write-Host "Results files:" -ForegroundColor Yellow
    foreach ($result in $results) {
        Write-Host "  - $($result.Model): $($result.ResultFile)" -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "[INFO] To compare results, run:" -ForegroundColor Yellow
    Write-Host "  python compare_isolated_results.py" -ForegroundColor Cyan
} else {
    Write-Host "[WARNING] No results generated!" -ForegroundColor Yellow
}

Write-Host ""
