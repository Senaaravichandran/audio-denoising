@echo off
title AudioClarity - Model Training
color 0E

echo.
echo  =============================================
echo    AudioClarity - DCCRN Model Training
echo    CPU-Optimized Fast Training Pipeline
echo  =============================================
echo.

:: Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run start.bat first to set up the environment
    pause
    exit /b 1
)

:: Check for training data
if not exist "data\clean\" (
    echo ERROR: Training data not found
    echo.
    echo Please create the following directory structure:
    echo   data\
    echo     clean\     (clean audio files - .wav format)
    echo     noisy\     (noisy audio files - .wav format)
    echo.
    echo Files in clean\ and noisy\ should have matching names
    echo Example: data\clean\sample001.wav and data\noisy\sample001.wav
    pause
    exit /b 1
)

if not exist "data\noisy\" (
    echo ERROR: Noisy training data directory not found
    echo Please create data\noisy\ directory with noisy audio files
    pause
    exit /b 1
)

:: Count training files
set clean_count=0
set noisy_count=0

for %%f in ("data\clean\*.wav") do set /a clean_count+=1
for %%f in ("data\noisy\*.wav") do set /a noisy_count+=1

if %clean_count%==0 (
    echo ERROR: No WAV files found in data\clean\
    echo Please add clean audio files (.wav format)
    pause
    exit /b 1
)

if %noisy_count%==0 (
    echo ERROR: No WAV files found in data\noisy\
    echo Please add noisy audio files (.wav format)
    pause
    exit /b 1
)

echo ✓ Found %clean_count% clean files and %noisy_count% noisy files
echo.

:: Training options
echo Training Options:
echo   1. Fast Training (1000 files, 3 epochs) - ~10-30 minutes
echo   2. Full Training (all files, 5 epochs) - longer duration
echo   3. Custom Training (specify parameters)
echo.
set /p training_mode="Select training mode (1/2/3): "

if "%training_mode%"=="1" (
    echo.
    echo =============================================
    echo   🧠 Fast Training Mode
    echo   Files: Up to 1000 pairs
    echo   Epochs: 3
    echo   Estimated time: 10-30 minutes
    echo =============================================
    echo.
    ".venv\Scripts\python.exe" ml\training\train_fast.py
) else if "%training_mode%"=="2" (
    echo.
    echo =============================================
    echo   🧠 Full Training Mode
    echo   Files: All available pairs
    echo   Epochs: 5
    echo   Estimated time: 30+ minutes
    echo =============================================
    echo.
    ".venv\Scripts\python.exe" ml\training\train.py
) else if "%training_mode%"=="3" (
    echo.
    set /p max_files="Maximum files to use (default 1000): "
    set /p num_epochs="Number of epochs (default 3): "
    
    if "%max_files%"=="" set max_files=1000
    if "%num_epochs%"=="" set num_epochs=3
    
    echo.
    echo =============================================
    echo   🧠 Custom Training Mode
    echo   Files: Up to %max_files% pairs
    echo   Epochs: %num_epochs%
    echo =============================================
    echo.
    ".venv\Scripts\python.exe" -c "
import sys
sys.path.append('ml/training')
from train_fast import FastTrainer
import yaml

config = {'model': {'encoder_layers': 3, 'hidden_dim': 64, 'lstm_layers': 1}}
with open('config_temp.yaml', 'w') as f:
    yaml.dump(config, f)

trainer = FastTrainer('config_temp.yaml')
trainer.train(num_epochs=%num_epochs%, max_files=%max_files%)
"
) else (
    echo Invalid selection. Please run the script again.
    pause
    exit /b 1
)

echo.
if %errorlevel%==0 (
    echo =============================================
    echo   ✅ Training completed successfully!
    echo   
    echo   📁 Model saved: checkpoints\dccrn_latest.pth
    echo   🎯 Ready for audio enhancement
    echo   🚀 Start server with: start.bat or dev-start.bat
    echo =============================================
) else (
    echo =============================================
    echo   ❌ Training failed or was interrupted
    echo   
    echo   💡 Possible solutions:
    echo   - Check training data format (WAV files)
    echo   - Ensure file names match between clean/noisy
    echo   - Try with fewer files or epochs
    echo   - Check error messages above
    echo =============================================
)

echo.
pause
