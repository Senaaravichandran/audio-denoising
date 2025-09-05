@echo off
title AudioClarity - System Verification
color 0D

echo.
echo  =============================================
echo    AudioClarity - System Verification
echo    Check installation and dependencies
echo  =============================================
echo.

:: System Requirements Check
echo [1/10] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found
    echo Please install Python 3.8+ from https://python.org
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo ✓ Python %%i found
)

echo [2/10] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found
    echo Please install Node.js from https://nodejs.org/
) else (
    for /f %%i in ('node --version 2^>^&1') do echo ✓ Node.js %%i found
)

echo [3/10] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found
) else (
    for /f %%i in ('npm --version 2^>^&1') do echo ✓ npm %%i found
)

echo [4/10] Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ FFmpeg not found (optional for video processing)
) else (
    echo ✓ FFmpeg found
)

echo [5/10] Checking Python virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo ✓ Virtual environment exists
    for /f "tokens=2" %%i in ('".venv\Scripts\python.exe" --version 2^>^&1') do echo   Python %%i in venv
) else (
    echo ❌ Virtual environment not found
    echo Run start.bat to create virtual environment
)

echo [6/10] Checking Python dependencies...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>nul
    if %errorlevel% neq 0 (
        echo ❌ PyTorch not installed in virtual environment
    )
    
    ".venv\Scripts\python.exe" -c "import torchaudio; print('✓ TorchAudio available')" 2>nul
    if %errorlevel% neq 0 (
        echo ❌ TorchAudio not installed
    )
    
    ".venv\Scripts\python.exe" -c "import librosa; print('✓ Librosa available')" 2>nul
    if %errorlevel% neq 0 (
        echo ❌ Librosa not installed
    )
    
    ".venv\Scripts\python.exe" -c "import numpy; print('✓ NumPy available')" 2>nul
    if %errorlevel% neq 0 (
        echo ❌ NumPy not installed
    )
) else (
    echo ⏭️ Skipping Python dependency check (no venv)
)

echo [7/10] Checking Node.js dependencies...
if exist "node_modules\package.json" (
    echo ✓ Node modules installed
) else if exist "node_modules\" (
    echo ✓ Node modules directory exists
) else (
    echo ❌ Node modules not installed
    echo Run: npm install
)

echo [8/10] Checking project structure...
if exist "ml\" (
    echo ✓ ML directory exists
) else (
    echo ❌ ML directory missing
)

if exist "server\" (
    echo ✓ Server directory exists
) else (
    echo ❌ Server directory missing
)

if exist "client\" (
    echo ✓ Client directory exists
) else (
    echo ❌ Client directory missing
)

echo [9/10] Checking AI model...
if exist "checkpoints\dccrn_latest.pth" (
    echo ✓ DCCRN model found
    for %%i in ("checkpoints\dccrn_latest.pth") do echo   Size: %%~zi bytes
) else (
    echo ⚠️ No trained model found
    echo Run train-model.bat to train a model
)

echo [10/10] Checking training data...
if exist "data\clean\" (
    set clean_count=0
    for %%f in ("data\clean\*.wav") do set /a clean_count+=1
    echo ✓ Clean data directory exists (%%clean_count%% WAV files)
) else (
    echo ⚠️ Clean training data directory not found (data\clean\)
)

if exist "data\noisy\" (
    set noisy_count=0
    for %%f in ("data\noisy\*.wav") do set /a noisy_count+=1
    echo ✓ Noisy data directory exists (%%noisy_count%% WAV files)
) else (
    echo ⚠️ Noisy training data directory not found (data\noisy\)
)

echo.
echo =============================================
echo   System Verification Complete
echo.
echo   📊 Recommendations:
if not exist ".venv\Scripts\python.exe" echo   - Run start.bat to set up environment
if not exist "node_modules\" echo   - Run npm install for Node.js dependencies
if not exist "checkpoints\dccrn_latest.pth" echo   - Run train-model.bat to train AI model
if not exist "data\clean\" echo   - Create data\clean\ with clean audio files
if not exist "data\noisy\" echo   - Create data\noisy\ with noisy audio files
echo.
echo   🚀 Ready to start: start.bat or dev-start.bat
echo =============================================
echo.
pause
