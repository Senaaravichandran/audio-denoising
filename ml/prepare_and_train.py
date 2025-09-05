#!/usr/bin/env python3
"""
Data preparation and training script for DCCRN
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import random


def check_data_structure(data_dir):
    """Check if data directory has the expected structure"""
    data_path = Path(data_dir)
    clean_dir = data_path / "clean"
    noisy_dir = data_path / "noisy"
    
    if not clean_dir.exists():
        print(f"Error: Clean data directory not found: {clean_dir}")
        return False
    
    if not noisy_dir.exists():
        print(f"Error: Noisy data directory not found: {noisy_dir}")
        return False
    
    # Check for audio files
    clean_files = list(clean_dir.glob("*.wav"))
    noisy_files = list(noisy_dir.glob("*.wav"))
    
    if len(clean_files) == 0:
        print(f"Error: No WAV files found in {clean_dir}")
        return False
    
    if len(noisy_files) == 0:
        print(f"Error: No WAV files found in {noisy_dir}")
        return False
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
    
    # Check for matching pairs
    clean_stems = {f.stem for f in clean_files}
    noisy_stems = {f.stem for f in noisy_files}
    
    matching_pairs = clean_stems.intersection(noisy_stems)
    
    if len(matching_pairs) == 0:
        print("Error: No matching pairs found between clean and noisy files")
        print("Make sure clean and noisy files have the same names (excluding extension)")
        return False
    
    print(f"Found {len(matching_pairs)} matching pairs")
    return True


def create_data_splits(data_dir, train_ratio=0.8, val_ratio=0.2):
    """Create train/validation splits"""
    data_path = Path(data_dir)
    clean_dir = data_path / "clean"
    noisy_dir = data_path / "noisy"
    
    # Get all clean files
    clean_files = list(clean_dir.glob("*.wav"))
    
    # Find matching pairs
    pairs = []
    for clean_file in clean_files:
        noisy_file = noisy_dir / clean_file.name
        if noisy_file.exists():
            pairs.append((clean_file, noisy_file))
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Split data
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    
    # Create split directories
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        clean_split_dir = data_path / f"clean_{split_name}"
        noisy_split_dir = data_path / f"noisy_{split_name}"
        
        clean_split_dir.mkdir(exist_ok=True)
        noisy_split_dir.mkdir(exist_ok=True)
        
        # Copy files (or create symlinks on Unix systems)
        for clean_file, noisy_file in split_pairs:
            clean_dest = clean_split_dir / clean_file.name
            noisy_dest = noisy_split_dir / noisy_file.name
            
            # Copy files
            shutil.copy2(clean_file, clean_dest)
            shutil.copy2(noisy_file, noisy_dest)
    
    print(f"Created data splits:")
    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")
    
    return len(train_pairs), len(val_pairs)


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "ml/requirements.txt"
        ])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def start_training(config_path="ml/training/config.yaml", resume=None):
    """Start training process"""
    print("Starting training...")
    
    cmd = [sys.executable, "ml/training/train.py", "--config", config_path]
    
    if resume:
        cmd.extend(["--resume", resume])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare data and train DCCRN model")
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Path to data directory containing clean/ and noisy/ folders")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install Python dependencies")
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Prepare training/validation data splits")
    parser.add_argument("--train", action="store_true", 
                       help="Start training")
    parser.add_argument("--config", type=str, default="ml/training/config.yaml",
                       help="Path to training config file")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    
    args = parser.parse_args()
    
    if not any([args.install_deps, args.prepare_data, args.train]):
        print("No action specified. Use --help for options.")
        return
    
    # Install dependencies
    if args.install_deps:
        if not install_dependencies():
            return
    
    # Prepare data
    if args.prepare_data:
        print(f"Checking data structure in: {args.data_dir}")
        
        if not check_data_structure(args.data_dir):
            print("Data structure check failed. Please fix the issues and try again.")
            return
        
        print("Creating train/validation splits...")
        val_ratio = 1.0 - args.train_ratio
        create_data_splits(args.data_dir, args.train_ratio, val_ratio)
    
    # Start training
    if args.train:
        if not start_training(args.config, args.resume):
            print("Training failed!")
            return
    
    print("Script completed successfully!")


if __name__ == "__main__":
    main()
