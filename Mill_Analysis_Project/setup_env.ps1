# Mill Analysis Project - PowerShell Environment Setup (Clean Version)
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "MILL ANALYSIS PROJECT - ENVIRONMENT SETUP (CLEAN VERSION)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host

# Set environment name and Python version
$ENV_NAME = "mill_analysis_env"
$PYTHON_VERSION = "3.11"

# Check if Python is installed
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+\.\d+)") {
        $detectedVersion = $matches[1]
        Write-Host "Python $detectedVersion detected" -ForegroundColor Green
    } else {
        throw "Invalid Python version format"
    }
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ and add it to your PATH" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
Write-Host
Write-Host "[2/6] Checking for existing environment..." -ForegroundColor Yellow
if (Test-Path $ENV_NAME) {
    Write-Host "Virtual environment '$ENV_NAME' already exists" -ForegroundColor Green
    $envExists = $true
} else {
    Write-Host "Virtual environment '$ENV_NAME' not found" -ForegroundColor Yellow
    $envExists = $false
}

# Create virtual environment if needed
if (-not $envExists) {
    Write-Host
    Write-Host "[3/6] Creating virtual environment '$ENV_NAME'..." -ForegroundColor Yellow
    python -m venv $ENV_NAME
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created successfully" -ForegroundColor Green
} else {
    Write-Host "[3/6] Using existing virtual environment" -ForegroundColor Green
}

# Activate virtual environment
Write-Host
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
& "$ENV_NAME\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to enable script execution:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green

# Upgrade pip
Write-Host
Write-Host "[5/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

# Install requirements
Write-Host
Write-Host "[6/6] Installing requirements..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    Write-Host "Installing from requirements.txt..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install some requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "All requirements installed successfully" -ForegroundColor Green
} elseif (Test-Path "requirements_clean.txt") {
    Write-Host "Installing from requirements_clean.txt..." -ForegroundColor Cyan
    python -m pip install -r requirements_clean.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install some requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "All requirements installed successfully" -ForegroundColor Green
} else {
    Write-Host "WARNING: No requirements file found" -ForegroundColor Yellow
    Write-Host "Installing essential packages manually..." -ForegroundColor Yellow
    python -m pip install open3d numpy==1.24.3 matplotlib==3.7.2 scipy==1.9.0 "pye57>=0.3.9" "plotly>=5.14.0"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install essential packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Essential packages installed" -ForegroundColor Green
}

# Test installations
Write-Host
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "TESTING INSTALLATION" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$tests = @(
    @("Open3D", "import open3d as o3d; print('✓ Open3D version:', o3d.__version__)"),
    @("NumPy", "import numpy as np; print('✓ NumPy version:', np.__version__)"),
    @("Matplotlib", "import matplotlib; print('✓ Matplotlib version:', matplotlib.__version__)"),
    @("SciPy", "import scipy; print('✓ SciPy version:', scipy.__version__)"),
    @("PyE57", "import pye57; print('✓ PyE57 available')"),
    @("Plotly", "import plotly; print('✓ Plotly version:', plotly.__version__)")
)

$allTestsPassed = $true
foreach ($test in $tests) {
    Write-Host "Testing $($test[0])..." -ForegroundColor Yellow
    python -c $test[1] 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ $($test[0]) test failed" -ForegroundColor Red
        $allTestsPassed = $false
    }
}

if (-not $allTestsPassed) {
    Write-Host "ERROR: Some dependency tests failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ All dependencies working correctly!" -ForegroundColor Green

# Success message
Write-Host
Write-Host "================================================================" -ForegroundColor Green
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Environment: $ENV_NAME" -ForegroundColor Cyan
Write-Host "Python: $env:VIRTUAL_ENV" -ForegroundColor Cyan
Write-Host
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. The environment is now activated and ready to use"
Write-Host "2. Run your mill analysis: python src/run_pipeline.py"
Write-Host "3. Run heatmap analysis: python src/run_heatmap_analysis.py"
Write-Host "4. Run complete processor: python src/main_processor.py"
Write-Host
Write-Host "FUTURE SESSIONS:" -ForegroundColor Yellow
Write-Host "- Run this script again to activate and check the environment"
Write-Host "- Use 'deactivate' to exit the environment"
Write-Host
Write-Host "Environment ready for 3D point cloud processing!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host
Write-Host "Press Enter to continue..."
Read-Host
