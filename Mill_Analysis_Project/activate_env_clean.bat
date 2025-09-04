@echo off
echo ================================================================
echo MILL ANALYSIS PROJECT - ENVIRONMENT ACTIVATION (CLEAN VERSION)
echo ================================================================

:: Set environment name
set ENV_NAME=mill_analysis_env

:: Check if environment exists
if not exist "%ENV_NAME%" (
    echo ERROR: Virtual environment '%ENV_NAME%' not found
    echo.
    echo Please run 'setup_env.bat' first to create the environment
    echo.
    pause
    exit /b 1
)

:: Activate environment
echo Activating virtual environment '%ENV_NAME%'...
call %ENV_NAME%\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Verify activation
echo.
echo Environment activated successfully!
echo Python location: %VIRTUAL_ENV%
echo.

:: Quick dependency check
echo Checking core dependencies...
python -c "import open3d, numpy, matplotlib; print('✓ All core dependencies available')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies missing
    echo Run 'setup_env.bat' to install/update dependencies
    echo.
)

echo.
echo ================================================================
echo ENVIRONMENT READY FOR MILL ANALYSIS (CPU-ONLY MODE)
echo ================================================================
echo.
echo Available commands:
echo   python src/run_pipeline.py              # Standard CPU processing
echo   python src/run_heatmap_analysis.py      # Heatmap analysis
echo   python src/main_processor.py            # Complete pipeline
echo.
echo Type 'deactivate' to exit the environment
echo ================================================================
