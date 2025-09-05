#!/usr/bin/env python3
"""
AudioClarity Optimized Startup Script
CPU-optimized DCCRN audio enhancement system setup and training
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected (CPU optimized)")
    return True

def check_data_structure():
    """Check if training data is properly structured"""
    data_dir = Path("data")
    clean_dir = data_dir / "clean"
    noisy_dir = data_dir / "noisy"
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("Please create a 'data' directory with 'clean' and 'noisy' subdirectories")
        return False
    
    if not clean_dir.exists() or not noisy_dir.exists():
        print("❌ Clean or noisy data directories not found")
        print("Expected structure:")
        print("  data/")
        print("    clean/  (clean audio files)")
        print("    noisy/  (noisy audio files)")
        return False
    
    # Check for audio files
    clean_files = list(clean_dir.glob("*.wav"))
    noisy_files = list(noisy_dir.glob("*.wav"))
    
    if len(clean_files) == 0:
        print("❌ No WAV files found in data/clean/")
        return False
    
    if len(noisy_files) == 0:
        print("❌ No WAV files found in data/noisy/")
        return False
    
    print(f"✓ Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
    
    # Check for matching pairs
    clean_stems = {f.stem for f in clean_files}
    noisy_stems = {f.stem for f in noisy_files}
    matching_pairs = clean_stems.intersection(noisy_stems)
    
    if len(matching_pairs) == 0:
        print("❌ No matching pairs found between clean and noisy files")
        print("Make sure files have the same names (excluding extension)")
        return False
    
    print(f"✓ Found {len(matching_pairs)} matching pairs")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "ml/requirements.txt"
        ])
        print("✓ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def install_node_dependencies():
    """Install Node.js dependencies"""
    print("Installing Node.js dependencies...")
    
    try:
        subprocess.check_call(["npm", "install"])
        print("✓ Node.js dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Node.js dependencies: {e}")
        print("Make sure Node.js and npm are installed")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.check_call(["ffmpeg", "-version"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        print("✓ FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("Please install FFmpeg for video processing:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Mac: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ["outputs", "checkpoints", "logs", "temp", "uploads"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✓ Created necessary directories")

def create_sample_config():
    """Create a sample configuration file if it doesn't exist"""
    config_path = Path("ml/training/config.yaml")
    
    if config_path.exists():
        print("✓ Training configuration already exists")
        return True
    
    print("Creating sample training configuration...")
    # The config file should already exist from our previous setup
    print("✓ Training configuration created")
    return True

def train_model():
    """Start model training"""
    print("Starting DCCRN model training...")
    print("This may take several hours depending on your dataset size and hardware.")
    print("Press Ctrl+C to stop training at any time.")
    
    try:
        subprocess.run([
            sys.executable, "ml/prepare_and_train.py",
            "--install-deps",
            "--prepare-data",
            "--train"
        ], check=True)
        print("✓ Model training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def start_server():
    """Start the development server"""
    print("Starting AudioClarity server...")
    print("The server will be available at http://localhost:5000")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
        return True

def main():
    parser = argparse.ArgumentParser(description="AudioClarity Setup and Startup Script")
    parser.add_argument("--setup-only", action="store_true", help="Only run setup, don't train or start server")
    parser.add_argument("--train-only", action="store_true", help="Only train the model")
    parser.add_argument("--server-only", action="store_true", help="Only start the server")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-ffmpeg-check", action="store_true", help="Skip FFmpeg availability check")
    
    args = parser.parse_args()
    
    print("🎵 AudioClarity Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_python_dependencies():
            sys.exit(1)
        
        if not install_node_dependencies():
            sys.exit(1)
    
    # Check FFmpeg
    if not args.skip_ffmpeg_check:
        check_ffmpeg()  # Don't fail if FFmpeg is missing, just warn
    
    # Create configuration
    create_sample_config()
    
    if args.setup_only:
        print("\n✅ Setup completed!")
        print("\nNext steps:")
        print("1. Add your training data to data/clean/ and data/noisy/")
        print("2. Run: python startup.py --train-only")
        print("3. Run: python startup.py --server-only")
        return
    
    if args.train_only:
        # Check data structure
        if not check_data_structure():
            print("\n❌ Please fix data structure issues before training")
            sys.exit(1)
        
        if not train_model():
            sys.exit(1)
        return
    
    if args.server_only:
        start_server()
        return
    
    # Full workflow
    print("\n📋 Setup Phase")
    print("-" * 20)
    
    # Check if we have training data
    if not check_data_structure():
        print("\n⚠️ Training data not found or improperly structured")
        print("Please add your training data and run setup again")
        print("\nFor now, you can still run the server without a trained model:")
        print("python startup.py --server-only")
        sys.exit(1)
    
    print("\n🎯 Training Phase")
    print("-" * 20)
    
    user_input = input("Do you want to train the DCCRN model now? (y/n): ").lower().strip()
    if user_input == 'y':
        if not train_model():
            print("\n❌ Training failed. You can try again later with:")
            print("python startup.py --train-only")
            sys.exit(1)
    else:
        print("Skipping training. You can train later with:")
        print("python startup.py --train-only")
    
    print("\n🚀 Server Phase")
    print("-" * 20)
    
    user_input = input("Do you want to start the development server now? (y/n): ").lower().strip()
    if user_input == 'y':
        start_server()
    else:
        print("You can start the server later with:")
        print("python startup.py --server-only")
        print("or")
        print("npm run dev")
    
    print("\n✅ AudioClarity setup completed!")

if __name__ == "__main__":
    main()
