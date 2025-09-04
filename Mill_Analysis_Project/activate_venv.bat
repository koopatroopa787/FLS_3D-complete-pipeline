@echo off
echo Activating Python 3.10 Virtual Environment for Mill Analysis Project...
echo.

REM Change to the project directory
cd /d "C:\Mill_Analysis_Project"

REM Activate the virtual environment
call "C:\Users\Koopatroopa787\AppData\Local\AnthropicClaude\app-0.12.129\venv310\Scripts\activate.bat"

REM Display confirmation
echo.
echo Virtual environment activated successfully!
echo Python version:
python --version
echo.
echo You are now in: %CD%
echo Ready for development!
echo.
echo To deactivate the environment, type: deactivate
echo.

REM Keep the command prompt open
cmd /k
