@echo off
REM Setup Python environment for Stream Diffusion RS
REM This script creates a virtual environment and installs required dependencies

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing Python dependencies...
pip install -r scripts\requirements.txt

echo.
echo Setup complete! To activate the virtual environment, run:
echo call venv\Scripts\activate.bat