#!/bin/bash

# AudioClarity - CPU-Optimized AI Audio Enhancement
# Universal Startup Script for macOS/Linux

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Clear screen and show header
clear
echo -e "${CYAN}"
echo "============================================="
echo "  AudioClarity - CPU-Optimized AI Enhancement"
echo "  Version 2.0 - Universal Compatibility"
echo "============================================="
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get IP address
get_ip_address() {
    if command_exists "ip"; then
        ip route get 8.8.8.8 | awk '{print $7; exit}' 2>/dev/null
    elif command_exists "ifconfig"; then
        ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1
    else
        echo "localhost"
    fi
}

# Check Python installation
echo -e "${BLUE}[1/7] Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}[OK] Python3 found${NC}"
elif command_exists python; then
    PYTHON_VERSION=$(python -c 'import sys; print(sys.version_info[0])')
    if [ "$PYTHON_VERSION" = "3" ]; then
        PYTHON_CMD="python"
        echo -e "${GREEN}[OK] Python found${NC}"
    else
        echo -e "${RED}ERROR: Python 3.8+ is required${NC}"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
else
    echo -e "${RED}ERROR: Python is not installed${NC}"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check Node.js installation
echo -e "${BLUE}[2/7] Checking Node.js installation...${NC}"
if command_exists node; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge "16" ]; then
        echo -e "${GREEN}[OK] Node.js found (v$(node -v))${NC}"
    else
        echo -e "${YELLOW}WARNING: Node.js 18+ recommended (found v$(node -v))${NC}"
    fi
else
    echo -e "${RED}ERROR: Node.js is not installed${NC}"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check npm installation
echo -e "${BLUE}[3/7] Checking npm installation...${NC}"
if command_exists npm; then
    echo -e "${GREEN}[OK] npm found${NC}"
else
    echo -e "${RED}ERROR: npm is not available${NC}"
    echo "Please ensure npm is installed with Node.js"
    exit 1
fi

# Check FFmpeg installation (optional)
echo -e "${BLUE}[4/7] Checking FFmpeg installation...${NC}"
if command_exists ffmpeg; then
    echo -e "${GREEN}[OK] FFmpeg found${NC}"
else
    echo -e "${YELLOW}WARNING: FFmpeg not found - video processing will not be available${NC}"
    echo "Install FFmpeg from https://ffmpeg.org/download.html for video support"
fi

# Setup Python virtual environment
echo -e "${BLUE}[5/7] Setting up Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
else
    echo -e "${GREEN}[OK] Virtual environment exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo -e "${BLUE}[6/7] Installing Python dependencies...${NC}"
echo "Installing CPU-optimized PyTorch and audio processing libraries..."

# Upgrade pip
.venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1

# Install PyTorch CPU version
.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install PyTorch${NC}"
    exit 1
fi

# Install other ML requirements
if [ -f "ml/requirements.txt" ]; then
    .venv/bin/python -m pip install -r ml/requirements.txt >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}WARNING: Some Python packages may not be installed${NC}"
        echo "The application should still work with core functionality"
    fi
fi

echo -e "${GREEN}[OK] Python dependencies ready${NC}"

# Install Node.js dependencies
echo -e "${BLUE}[7/7] Installing Node.js dependencies...${NC}"
npm install >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install Node.js dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Node.js dependencies ready${NC}"

echo
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}  AudioClarity is starting...${NC}"
echo
echo -e "${GREEN}  🌐 Web Application:${NC}"
echo -e "${GREEN}  http://localhost:5000${NC}"
echo
echo -e "${GREEN}  🔗 Local Network Access:${NC}"
IP_ADDRESS=$(get_ip_address)
echo -e "${GREEN}  http://$IP_ADDRESS:5000${NC}"
echo
echo -e "${BLUE}  📡 Server Status: Initializing...${NC}"
echo -e "${CYAN}=============================================${NC}"
echo

# Check if we have a trained model
if [ -f "checkpoints/dccrn_latest.pth" ]; then
    echo -e "${GREEN}[OK] DCCRN model found - starting server with AI enhancement${NC}"
    echo
    echo -e "${CYAN}=============================================${NC}"
    echo -e "${GREEN}  🚀 AudioClarity is READY! (CPU Optimized)${NC}"
    echo
    echo -e "${GREEN}  🌐 Access your application at:${NC}"
    echo -e "${GREEN}  👉 http://localhost:5000${NC}"
    echo
    echo -e "${GREEN}  📱 Mobile/Network Access:${NC}"
    echo -e "${GREEN}  👉 http://$IP_ADDRESS:5000${NC}"
    echo
    echo -e "${BLUE}  ⚡ Server: Node.js + TypeScript${NC}"
    echo -e "${PURPLE}  🤖 AI Model: DCCRN (CPU Optimized)${NC}"
    echo -e "${YELLOW}  💾 Storage: SQLite + File System${NC}"
    echo -e "${CYAN}  📡 WebSocket: Real-time Updates${NC}"
    echo -e "${GREEN}  🎯 Performance: Universal CPU Compatibility${NC}"
    echo -e "${CYAN}=============================================${NC}"
    echo
    echo "Starting server..."
    npm run dev
else
    echo
    echo -e "${YELLOW}WARNING: No trained DCCRN model found${NC}"
    echo "Looking for: checkpoints/dccrn_latest.pth"
    echo
    echo "You have two options:"
    echo "  1. Train a new model (recommended for first run)"
    echo "  2. Start server without AI model (limited functionality)"
    echo
    read -p "Do you want to train the model now? (y/n): " train_choice
    
    if [[ $train_choice =~ ^[Yy]$ ]]; then
        echo
        echo -e "${CYAN}=============================================${NC}"
        echo -e "${PURPLE}  🧠 Training CPU-Optimized DCCRN Model${NC}"
        echo -e "${CYAN}=============================================${NC}"
        echo
        echo "Starting fast training process..."
        echo "This will take approximately 10-30 minutes depending on your CPU"
        echo
        
        # Use the virtual environment Python for training
        .venv/bin/python ml/training/train_fast.py
        if [ $? -ne 0 ]; then
            echo
            echo "Training failed or was interrupted"
            echo "You can try again later or start without a model"
            exit 1
        fi
        
        echo
        echo -e "${GREEN}[SUCCESS] Training completed successfully!${NC}"
        echo "Starting server with new model..."
        echo
        echo -e "${CYAN}=============================================${NC}"
        echo -e "${GREEN}  🚀 AudioClarity is READY! (Newly Trained)${NC}"
        echo
        echo -e "${GREEN}  🌐 Access your application at:${NC}"
        echo -e "${GREEN}  👉 http://localhost:5000${NC}"
        echo
        echo -e "${BLUE}  ⚡ Server: Node.js + TypeScript${NC}"
        echo -e "${PURPLE}  🤖 AI Model: DCCRN (CPU Optimized)${NC}"
        echo -e "${YELLOW}  💾 Storage: SQLite + File System${NC}"
        echo -e "${CYAN}  📡 WebSocket: Real-time Updates${NC}"
        echo -e "${GREEN}  🎯 Performance: Universal CPU Compatibility${NC}"
        echo -e "${CYAN}=============================================${NC}"
        echo
        npm run dev
    else
        echo
        echo "Starting server without trained model..."
        echo "Note: Audio enhancement will not work until you train a model"
        echo "You can train later by running: python ml/training/train_fast.py"
        echo
        echo -e "${CYAN}=============================================${NC}"
        echo -e "${YELLOW}  ⚠️  AudioClarity (Limited Mode)${NC}"
        echo
        echo -e "${GREEN}  🌐 Access your application at:${NC}"
        echo -e "${GREEN}  👉 http://localhost:5000${NC}"
        echo
        echo -e "${RED}  ❌ AI Model: Not Available${NC}"
        echo -e "${BLUE}  ⚡ Server: Node.js + TypeScript${NC}"
        echo -e "${YELLOW}  💾 Storage: SQLite + File System${NC}"
        echo -e "${GREEN}  🎯 Performance: Universal CPU Compatibility${NC}"
        echo -e "${CYAN}=============================================${NC}"
        echo
        npm run dev
    fi
fi

echo
echo -e "${CYAN}=============================================${NC}"
echo -e "${GREEN}  🎉 Thank you for using AudioClarity!${NC}"
echo -e "${PURPLE}  CPU-Optimized AI Audio Enhancement${NC}"
echo
echo -e "${BLUE}  💡 Keep this terminal open while using the app${NC}"
echo -e "${YELLOW}  🔄 Press Ctrl+C to stop the server${NC}"
echo -e "${CYAN}  📖 Documentation: README.md${NC}"
echo -e "${GREEN}  🆘 Support: Check OPTIMIZATION_REPORT.md${NC}"
echo -e "${CYAN}=============================================${NC}"
echo

# Keep the script running
read -p "Press Enter to exit..."
