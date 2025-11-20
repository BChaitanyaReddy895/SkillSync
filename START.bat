@echo off
REM Quick fix and run script for SkillSync

echo ========================================
echo SkillSync - Quick Start Script
echo ========================================
echo.

REM Change to the correct directory
cd /d "%~dp0"

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo Option 1: Install tf-keras compatibility layer (RECOMMENDED)
echo Option 2: Remove TensorFlow and use PyTorch only
echo Option 3: Skip and run anyway
echo.

set /p choice="Enter your choice (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo Installing tf-keras...
    pip install tf-keras
    if errorlevel 1 (
        echo Warning: tf-keras installation had issues
    ) else (
        echo tf-keras installed successfully!
    )
)

if "%choice%"=="2" (
    echo.
    echo Removing TensorFlow...
    pip uninstall tensorflow tensorflow-intel -y
    echo Installing PyTorch CPU...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Starting SkillSync application...
echo.
echo The app will be available at:
echo   - Local: http://localhost:7860
echo   - Network: http://0.0.0.0:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
