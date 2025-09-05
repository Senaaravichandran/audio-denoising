"""
Quick Data Preparation Script
Creates minimal noisy data from clean audio for fast training
Optimized for systems with limited storage
"""

import os
import sys
import numpy as np
import torchaudio
from pathlib import Path
import random

def add_synthetic_noise(clean_audio, noise_level=0.1):
    """Add synthetic noise to clean audio"""
    # Generate white noise
    noise = torch.randn_like(clean_audio) * noise_level
    
    # Add some colored noise (pink noise simulation)
    if len(clean_audio.shape) > 1:
        for i in range(clean_audio.shape[0]):
            # Simple pink noise approximation
            pink_filter = torch.exp(-torch.arange(clean_audio.shape[-1], dtype=torch.float32) * 0.0001)
            colored_noise = torch.fft.ifft(torch.fft.fft(noise[i]) * pink_filter).real
            noise[i] = colored_noise * noise_level * 0.5
    
    # Mix clean audio with noise
    noisy_audio = clean_audio + noise
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val * 0.95
    
    return noisy_audio

def prepare_minimal_dataset(clean_dir, noisy_dir, max_files=50):
    """Prepare minimal training dataset"""
    clean_path = Path(clean_dir)
    noisy_path = Path(noisy_dir)
    
    # Create noisy directory if it doesn't exist
    noisy_path.mkdir(exist_ok=True)
    
    # Get clean audio files (limited number)
    clean_files = sorted(list(clean_path.glob("*.wav")))[:max_files]
    
    print(f"Processing {len(clean_files)} files...")
    
    processed = 0
    for clean_file in clean_files:
        try:
            # Load clean audio
            waveform, sample_rate = torchaudio.load(clean_file)
            
            # Resample to 16kHz if needed (to save space and processing time)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Trim to maximum 2 seconds to save space
            max_length = 32000  # 2 seconds at 16kHz
            if waveform.shape[-1] > max_length:
                waveform = waveform[:, :max_length]
            
            # Add synthetic noise with varying levels
            noise_level = random.uniform(0.05, 0.2)
            noisy_waveform = add_synthetic_noise(waveform, noise_level)
            
            # Save noisy version
            noisy_file = noisy_path / clean_file.name
            torchaudio.save(str(noisy_file), noisy_waveform, sample_rate)
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(clean_files)} files")
                
        except Exception as e:
            print(f"Error processing {clean_file}: {e}")
            continue
    
    print(f"Dataset preparation completed! Processed {processed} files.")
    return processed

def main():
    """Main function"""
    # Get the correct paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # AudioClarity directory
    clean_dir = base_dir / "data" / "clean"
    noisy_dir = base_dir / "data" / "noisy"
    
    print("Quick Data Preparation for Fast Training")
    print("=" * 50)
    print(f"Clean audio directory: {clean_dir}")
    print(f"Noisy audio directory: {noisy_dir}")
    
    if not clean_dir.exists():
        print(f"Error: Clean audio directory not found: {clean_dir}")
        return
    
    # Check available space and limit files accordingly
    try:
        import shutil
        free_space = shutil.disk_usage(str(base_dir))[2] / (1024**3)  # GB
        print(f"Available storage: {free_space:.1f} GB")
        
        # Limit files based on available space (rough estimate: 1MB per file pair)
        max_files = min(50, int(free_space * 500))  # Conservative estimate
        print(f"Will process maximum {max_files} files")
        
    except:
        max_files = 25  # Very conservative default
        print(f"Cannot determine disk space, using {max_files} files")
    
    # Prepare dataset
    processed_count = prepare_minimal_dataset(clean_dir, noisy_dir, max_files)
    
    if processed_count > 0:
        print(f"\nDataset ready for training!")
        print(f"Clean files: {clean_dir}")
        print(f"Noisy files: {noisy_dir}")
        print(f"Processed files: {processed_count}")
        print("\nYou can now run fast training with:")
        print("python ml/training/fast_train.py")
    else:
        print("No files were processed. Please check your audio data.")

if __name__ == "__main__":
    import torch
    import torchaudio
    main()
