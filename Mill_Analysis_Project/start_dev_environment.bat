@echo off
title Mill Analysis Project - Python 3.10 Environment
color 0A
echo.
echo ========================================
echo    MILL ANALYSIS PROJECT ENVIRONMENT
echo ========================================
echo.
echo Initializing Python 3.10 development environment...

REM Change to the project directory
cd /d "C:\Mill_Analysis_Project"

REM Activate the virtual environment
call "C:\Users\Koopatroopa787\AppData\Local\AnthropicClaude\app-0.12.129\venv310\Scripts\activate.bat"

REM Display environment information
echo.
echo Environment Status:
echo ------------------
python --version
pip --version
echo Working Directory: %CD%
echo.

REM Check if requirements.txt exists and offer to install
if exist "requirements.txt" (
    echo requirements.txt found!
    set /p install_req="Install requirements? (y/n): "
    if /i "%install_req%"=="y" (
        echo Installing requirements...
        pip install -r requirements.txt
    )
    echo.
)

REM Display helpful commands
echo Quick Commands:
echo --------------
echo pip install package_name    - Install a Python package
echo python script.py           - Run a Python script
echo jupyter notebook           - Start Jupyter Notebook
echo deactivate                 - Exit virtual environment
echo cls                        - Clear screen
echo.
echo Ready for AI/ML development!
echo ========================================
echo.

REM Keep the command prompt open with the activated environment
cmd /k
