@echo off
echo ============================================
echo Mill Analysis Project - Clean Open3D Implementation
echo ============================================
echo.
echo Activating virtual environment (Python 3.11.3)...
call mill_analysis_env\Scripts\activate.bat
echo.
echo Environment activated with clean dependencies!
echo.
echo Available commands:
echo   - python test_clean_installation.py   (verify clean setup)
echo   - python src\data_loader.py           (test data loading template)
echo   - python src\config.py                (test configuration)
echo.
echo Clean libraries installed:
echo   - Open3D 0.19.0 (3D point cloud processing)
echo   - NumPy (numerical computing)
echo   - Matplotlib (basic visualization)
echo   - typing-extensions (type hints)
echo.
echo Ready for clean implementation from scratch!
echo To deactivate: type 'deactivate'
echo.
cmd /k