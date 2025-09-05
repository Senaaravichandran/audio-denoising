# AudioClarity - Batch Files Guide

## Windows Batch Files

AudioClarity includes several batch files to make setup and usage easier on Windows:

### 🚀 Main Files

**`start.bat`** - Complete setup and start
- Checks all system requirements
- Creates Python virtual environment
- Installs CPU-optimized PyTorch and dependencies
- Installs Node.js packages
- Offers to train model if needed
- Starts the AudioClarity server

**`dev-start.bat`** - Quick development start
- Fast startup for developers
- Assumes dependencies are already installed
- Immediately starts the server

### 🧠 Training Files

**`train-model.bat`** - AI model training
- Guided training process with options:
  - Fast training (1000 files, 3 epochs, ~10-30 min)
  - Full training (all files, 5 epochs, longer)
  - Custom training (specify parameters)
- Automatic progress monitoring
- Uses CPU-optimized training

### 🔍 Utility Files

**`check-setup.bat`** - Quick system check
- Simple verification of key components
- Fast overview of system status

**`verify-setup.bat`** - Comprehensive system verification
- Detailed check of all requirements
- Dependency verification
- Recommendations for fixes

## Usage Instructions

### First Time Setup
1. Run `start.bat` - this will set up everything
2. If you have training data, it will offer to train the model
3. The server will start automatically

### Development Workflow
1. Use `dev-start.bat` for quick server starts
2. Use `train-model.bat` when you want to retrain
3. Use `check-setup.bat` to verify everything is working

### Troubleshooting
1. Run `verify-setup.bat` to see detailed system status
2. Check the recommendations it provides
3. Re-run `start.bat` if needed

## Requirements

Before running any batch files, ensure you have:
- Python 3.8+ installed and in PATH
- Node.js 18+ installed and in PATH
- Sufficient disk space (10GB+)
- Training data in `data/clean/` and `data/noisy/` (for training)

## Notes

- All batch files use CPU-optimized configurations
- Virtual environment is automatically created in `.venv/`
- PyTorch CPU version is installed for universal compatibility
- Training is optimized for systems without GPU requirements
