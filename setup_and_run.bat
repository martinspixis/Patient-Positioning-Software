@echo off
echo Checking Python installation...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.10 from python.org
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Found Python version %PYTHON_VERSION%

:: Create and activate virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/upgrade required packages
echo Installing/upgrading required packages...
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt

:: Run the application
echo.
echo Starting Patient Positioning System...
python -m patient_positioning.main

:: Deactivate virtual environment
deactivate

pause