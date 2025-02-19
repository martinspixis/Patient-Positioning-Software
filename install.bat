@echo off
echo Installing Patient Positioning System...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.10 from python.org
    pause
    exit /b 1
)

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/upgrade required packages
echo Installing required packages...
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt

echo.
echo Installation complete!
echo You can now run the program using setup_and_run.bat
pause