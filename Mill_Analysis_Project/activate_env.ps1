# Mill Analysis Project - Quick Environment Activation
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "MILL ANALYSIS PROJECT - QUICK ACTIVATION" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$ENV_NAME = "mill_analysis_env"

# Check if environment exists
if (-not (Test-Path $ENV_NAME)) {
    Write-Host "ERROR: Virtual environment '$ENV_NAME' not found" -ForegroundColor Red
    Write-Host
    Write-Host "Please run 'setup_env.ps1' first to create the environment" -ForegroundColor Yellow
    Write-Host
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate environment
Write-Host "Activating virtual environment '$ENV_NAME'..." -ForegroundColor Yellow
try {
    & "$ENV_NAME\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Activation failed"
    }
} catch {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to enable script execution:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host
Write-Host "✓ Environment activated successfully!" -ForegroundColor Green
Write-Host "Python location: $env:VIRTUAL_ENV" -ForegroundColor Cyan
Write-Host

# Quick dependency check
Write-Host "Checking core dependencies..." -ForegroundColor Yellow
try {
    python -c "import open3d, numpy, matplotlib, scipy; print('✓ All core dependencies available')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency check failed"
    }
    Write-Host "✓ All dependencies verified" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Some dependencies missing or broken" -ForegroundColor Yellow
    Write-Host "Run 'setup_env.ps1' to reinstall dependencies" -ForegroundColor Yellow
}

Write-Host
Write-Host "================================================================" -ForegroundColor Green
Write-Host "ENVIRONMENT READY FOR MILL ANALYSIS (CPU-ONLY MODE)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  python src/run_pipeline.py              # Standard CPU processing"
Write-Host "  python src/run_heatmap_analysis.py      # Heatmap analysis"
Write-Host "  python src/main_processor.py            # Complete pipeline"
Write-Host
Write-Host "Type 'deactivate' to exit the environment" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Green
