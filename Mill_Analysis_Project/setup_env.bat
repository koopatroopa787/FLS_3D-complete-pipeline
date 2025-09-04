@echo off
echo ================================================================
echo MILL ANALYSIS PROJECT - ENVIRONMENT SETUP (CLEAN VERSION)
echo ================================================================

:: Set environment name and Python version
set ENV_NAME=mill_analysis_env
set PYTHON_VERSION=3.11

echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and add it to your PATH
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set CURRENT_PYTHON_VERSION=%%i
echo Python %CURRENT_PYTHON_VERSION% detected

echo.
echo [2/6] Checking for existing environment...
if exist "%ENV_NAME%" (
    echo Virtual environment '%ENV_NAME%' already exists
    set ENV_EXISTS=1
) else (
    echo Virtual environment '%ENV_NAME%' not found
    set ENV_EXISTS=0
)

:: Create virtual environment if needed
if %ENV_EXISTS%==0 (
    echo.
    echo [3/6] Creating virtual environment '%ENV_NAME%'...
    python -m venv %ENV_NAME%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
) else (
    echo [3/6] Using existing virtual environment
)

:: Activate virtual environment
echo.
echo [4/6] Activating virtual environment...
call %ENV_NAME%\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated: %VIRTUAL_ENV%

:: Upgrade pip
echo.
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

:: Install requirements
echo.
echo [6/6] Installing requirements...
if exist "requirements.txt" (
    echo Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install some requirements
        pause
        exit /b 1
    )
    echo All requirements installed successfully
) else if exist "requirements_clean.txt" (
    echo Installing from requirements_clean.txt...
    python -m pip install -r requirements_clean.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install some requirements
        pause
        exit /b 1
    )
    echo All requirements installed successfully
) else (
    echo WARNING: No requirements file found
    echo Installing essential packages manually...
    python -m pip install open3d numpy>=1.22.0 matplotlib>=3.7.0 scipy>=1.9.0 pye57>=0.3.9 plotly>=5.14.0
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install essential packages
        pause
        exit /b 1
    )
    echo Essential packages installed
)

:: Test installation
echo.
echo ================================================================
echo TESTING INSTALLATION
echo ================================================================
echo Testing core dependencies...

python -c "import open3d as o3d; print('✓ Open3D version:', o3d.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Open3D test failed
    goto :test_failed
)

python -c "import numpy as np; print('✓ NumPy version:', np.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ✗ NumPy test failed
    goto :test_failed
)

python -c "import matplotlib; print('✓ Matplotlib version:', matplotlib.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Matplotlib test failed
    goto :test_failed
)

python -c "import scipy; print('✓ SciPy version:', scipy.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ✗ SciPy test failed
    goto :test_failed
)

echo ✓ All core dependencies working correctly!
goto :success

:test_failed
echo ERROR: Some dependencies failed to load
pause
exit /b 1

:success
echo.
echo ================================================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ================================================================
echo Environment: %ENV_NAME%
echo Python: %VIRTUAL_ENV%
echo.
echo NEXT STEPS:
echo 1. The environment is now activated and ready to use
echo 2. Run your mill analysis: python src/run_pipeline.py
echo 3. Run heatmap analysis: python src/run_heatmap_analysis.py
echo 4. Run complete processor: python src/main_processor.py
echo.
echo FUTURE SESSIONS:
echo - Run this script again to activate and check the environment
echo - Use 'deactivate' to exit the environment
echo.
echo Environment ready for 3D point cloud processing!
echo ================================================================
echo.
echo Press any key to continue or close this window...
pause >nul
