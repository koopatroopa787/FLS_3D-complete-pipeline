@echo off
echo ================================================================
echo MILL ANALYSIS PROJECT - GITHUB PREPARATION TOOL
echo ================================================================
echo.

REM Activate the virtual environment if it exists
if exist "mill_analysis_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call mill_analysis_env\Scripts\activate.bat
    echo.
)

REM Run the GitHub preparation tool
echo Running GitHub preparation analysis...
echo.
python github_prep.py

echo.
echo ================================================================
echo GitHub preparation complete!
echo ================================================================
pause
