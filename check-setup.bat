@echo off
title AudioClarity - Quick Setup Check
chcp 65001 >nul

echo.
echo  =============================================
echo    AudioClarity - Quick Setup Check
echo  =============================================
echo.

echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found
) else (
    echo [OK] Python found
)

echo Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Node.js not found
) else (
    echo [OK] Node.js found
)

echo Checking virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment exists
) else (
    echo [!] Virtual environment not found
)

echo Checking AI model...
if exist "checkpoints\dccrn_latest.pth" (
    echo [OK] DCCRN model found
) else (
    echo [!] No trained model found
)

echo Checking training data...
if exist "data\clean\" (
    echo [OK] Clean data directory exists
) else (
    echo [!] No clean training data
)

if exist "data\noisy\" (
    echo [OK] Noisy data directory exists
) else (
    echo [!] No noisy training data
)

echo.
echo Setup check complete!
echo.
pause
