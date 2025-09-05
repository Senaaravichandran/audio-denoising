@echo off
title AudioClarity - Quick Development Start
color 0B

echo.
echo  =============================================
echo    AudioClarity - Quick Development Start
echo    (Assumes dependencies already installed)
echo  =============================================
echo.

:: Quick check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run start.bat first to set up the environment
    pause
    exit /b 1
)

:: Quick check for Node modules
if not exist "node_modules\" (
    echo ERROR: Node modules not found
    echo Please run start.bat first to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies found - starting server...
echo.

:: Check if model exists
if exist "checkpoints\dccrn_latest.pth" (
    echo ✓ DCCRN model found
    echo.
    echo =============================================
    echo   🚀 AudioClarity Development Server
    echo   
    echo   🌐 Local: http://localhost:5000
    echo   🤖 AI Model: Ready (CPU Optimized)
    echo   ⚡ Mode: Development
    echo =============================================
    echo.
) else (
    echo ⚠️ No DCCRN model found - limited functionality
    echo.
    echo =============================================
    echo   🚀 AudioClarity Development Server
    echo   
    echo   🌐 Local: http://localhost:5000
    echo   ❌ AI Model: Not Available
    echo   ⚡ Mode: Development (Limited)
    echo =============================================
    echo.
)

:: Start the development server
npm run dev
