@echo off
title AudioClarity - CPU-Optimized AI Audio Enhancement
color 0A

echo.
echo  =============================================
echo    AudioClarity - CPU-Optimized AI Enhancement
echo    Version 2.0 - Universal Compatibility
echo  =============================================
echo.

:: Check if Python is installed
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

:: Check if Node.js is installed
echo [2/7] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found

:: Check if npm is available
echo [3/7] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm is not available
    echo Please ensure npm is installed with Node.js
    pause
    exit /b 1
)
echo [OK] npm found

:: Check if FFmpeg is available (optional for video processing)
echo [4/7] Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg not found - video processing will not be available
    echo Install FFmpeg from https://ffmpeg.org/download.html for video support
    echo.
) else (
    echo [OK] FFmpeg found
)

:: Check and setup Python virtual environment
echo [5/7] Setting up Python virtual environment...
if not exist ".venv\" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

:: Install Python dependencies
echo [6/7] Installing Python dependencies...
echo Installing CPU-optimized PyTorch and audio processing libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r ml\requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Some Python packages may not be installed
    echo The application should still work with core functionality
)
echo [OK] Python dependencies ready

:: Install Node.js dependencies
echo [7/7] Installing Node.js dependencies...
npm install >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)
echo [OK] Node.js dependencies ready

:: Start the application
echo.
echo =============================================
echo   AudioClarity is starting...
echo   
echo   🌐 Web Application:
echo   http://localhost:5000
echo   
echo   🔗 Local Network Access:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr "IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        echo   http://%%b:5000
    )
)
echo   
echo   📡 Server Status: Initializing...
echo =============================================
echo.

:: Check if we have a trained model
if exist "checkpoints\dccrn_latest.pth" (
    echo [OK] DCCRN model found - starting server with AI enhancement
    echo.
    echo =============================================
    echo   🚀 AudioClarity is READY! (CPU Optimized)
    echo   
    echo   🌐 Access your application at:
    echo   👉 http://localhost:5000
    echo   
    echo   📱 Mobile/Network Access:
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
        for /f "tokens=1" %%b in ("%%a") do (
            echo   👉 http://%%b:5000
        )
    )
    echo   
    echo   ⚡ Server: Node.js + TypeScript
    echo   🤖 AI Model: DCCRN (CPU Optimized)
    echo   💾 Storage: SQLite + File System
    echo   📡 WebSocket: Real-time Updates
    echo   🎯 Performance: Universal CPU Compatibility
    echo =============================================
    echo.
    echo Starting server...
    npm run dev
) else (
    echo.
    echo WARNING: No trained DCCRN model found
    echo Looking for: checkpoints\dccrn_latest.pth
    echo.
    echo You have two options:
    echo   1. Train a new model (recommended for first run)
    echo   2. Start server without AI model (limited functionality)
    echo.
    set /p train_choice="Do you want to train the model now? (y/n): "
    if /i "%train_choice%"=="y" (
        echo.
        echo =============================================
        echo   🧠 Training CPU-Optimized DCCRN Model
        echo =============================================
        echo.
        echo Starting fast training process...
        echo This will take approximately 10-30 minutes depending on your CPU
        echo.
        
        :: Use the virtual environment Python for training
        ".venv\Scripts\python.exe" ml\training\train_fast.py
        if %errorlevel% neq 0 (
            echo.
            echo Training failed or was interrupted
            echo You can try again later or start without a model
            pause
            exit /b 1
        )
        
        echo.
        echo [SUCCESS] Training completed successfully!
        echo Starting server with new model...
        echo.
        echo =============================================
        echo   🚀 AudioClarity is READY! (Newly Trained)
        echo   
        echo   🌐 Access your application at:
        echo   👉 http://localhost:5000
        echo   
        echo   ⚡ Server: Node.js + TypeScript  
        echo   🤖 AI Model: DCCRN (CPU Optimized)
        echo   💾 Storage: SQLite + File System
        echo   📡 WebSocket: Real-time Updates
        echo   🎯 Performance: Universal CPU Compatibility
        echo =============================================
        echo.
        npm run dev
    ) else (
        echo.
        echo Starting server without trained model...
        echo Note: Audio enhancement will not work until you train a model
        echo You can train later by running: python ml\training\train_fast.py
        echo.
        echo =============================================
        echo   ⚠️  AudioClarity (Limited Mode)
        echo   
        echo   🌐 Access your application at:
        echo   👉 http://localhost:5000
        echo   
        echo   ❌ AI Model: Not Available
        echo   ⚡ Server: Node.js + TypeScript
        echo   💾 Storage: SQLite + File System
        echo   🎯 Performance: Universal CPU Compatibility
        echo =============================================
        echo.
        npm run dev
    )
)

echo.
echo =============================================
echo   🎉 Thank you for using AudioClarity!
echo   CPU-Optimized AI Audio Enhancement
echo   
echo   💡 Keep this window open while using the app
echo   🔄 Press Ctrl+C to stop the server
echo   📖 Documentation: README.md
echo   🆘 Support: Check OPTIMIZATION_REPORT.md
echo =============================================
echo.
pause
