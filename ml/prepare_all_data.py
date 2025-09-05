#!/usr/bin/env python3
"""
Comprehensive data preparation for DCCRN training using ALL available data
Optimized for Intel i3-7020U with 8GB RAM and 10GB storage
"""
import os
import glob
import numpy as np
import torchaudio
import torch
import random
from pathlib import Path
import json
from tqdm import tqdm

def create_noise_augmentation():
    """Create various types of noise for data augmentation"""
    def white_noise(signal, noise_factor):
        noise = torch.randn_like(signal) * noise_factor
        return signal + noise
    
    def pink_noise(signal, noise_factor):
        # Simplified pink noise generation
        noise = torch.randn_like(signal)
        # Apply simple filtering to approximate pink noise
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        # Simple moving average approximation for pink noise
        noise = torch.nn.functional.conv1d(
            noise.unsqueeze(0).unsqueeze(0), 
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]).view(1, 1, -1),
            padding=2
        ).squeeze()[:signal.shape[0]]
        return signal + noise * noise_factor
    
    def brown_noise(signal, noise_factor):
        # Brown noise (integrated white noise)
        noise = torch.randn_like(signal)
        noise = torch.cumsum(noise, dim=0)
        noise = noise / torch.std(noise) * noise_factor
        return signal + noise
    
    def environmental_noise(signal, noise_factor):
        # Simulate environmental noise patterns
        t = torch.linspace(0, 1, signal.shape[0])
        # Fan noise (low frequency hum)
        fan_noise = torch.sin(2 * np.pi * 60 * t) * 0.3
        # Traffic noise (varied frequencies)
        traffic_noise = torch.sin(2 * np.pi * 120 * t) * 0.2 + torch.sin(2 * np.pi * 80 * t) * 0.1
        # Keyboard typing (short bursts)
        typing_noise = torch.zeros_like(signal)
        for _ in range(random.randint(5, 15)):
            start = random.randint(0, max(1, signal.shape[0] - 1000))
            end = min(start + random.randint(100, 500), signal.shape[0])
            typing_noise[start:end] = torch.randn(end - start) * 0.5
        
        combined_noise = (fan_noise + traffic_noise + typing_noise) * noise_factor
        return signal + combined_noise
    
    return [white_noise, pink_noise, brown_noise, environmental_noise]

def prepare_comprehensive_dataset():
    """Prepare comprehensive dataset using all available clean files"""
    
    print("🎵 Starting comprehensive data preparation for DCCRN...")
    print(f"💾 Available storage: ~10GB - optimizing for space efficiency")
    
    # Paths
    clean_dir = Path("data/clean")
    noisy_dir = Path("data/noisy")
    
    # Create noisy directory if it doesn't exist
    noisy_dir.mkdir(exist_ok=True)
    
    # Get all clean files
    clean_files = list(clean_dir.glob("*.wav"))
    total_files = len(clean_files)
    
    print(f"📁 Found {total_files} clean audio files")
    print(f"🔄 Will create {total_files * 4} noisy variants (4 noise types per file)")
    
    # Initialize noise functions
    noise_functions = create_noise_augmentation()
    noise_names = ['white', 'pink', 'brown', 'environmental']
    
    # Process files in batches to manage memory
    batch_size = 50  # Process 50 files at a time to manage RAM
    processed_count = 0
    
    # Create metadata for training
    metadata = {
        'total_pairs': 0,
        'sample_rate': 16000,
        'duration_stats': [],
        'noise_types': noise_names
    }
    
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = clean_files[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        print(f"   Files {batch_start + 1} to {batch_end} of {total_files}")
        
        for clean_file in tqdm(batch_files, desc="Processing files"):
            try:
                # Load clean audio
                waveform, sample_rate = torchaudio.load(clean_file)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Normalize audio
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                
                # Store duration info
                duration = waveform.shape[1] / 16000
                metadata['duration_stats'].append(duration)
                
                # Create 4 different noisy versions
                for noise_idx, (noise_func, noise_name) in enumerate(zip(noise_functions, noise_names)):
                    # Random noise level between 0.05 and 0.3
                    noise_factor = random.uniform(0.05, 0.3)
                    
                    # Apply noise
                    noisy_waveform = noise_func(waveform.squeeze(), noise_factor)
                    
                    # Ensure same length
                    if noisy_waveform.shape[0] != waveform.shape[1]:
                        min_len = min(noisy_waveform.shape[0], waveform.shape[1])
                        noisy_waveform = noisy_waveform[:min_len]
                        waveform = waveform[:, :min_len]
                    
                    # Normalize noisy audio
                    noisy_waveform = noisy_waveform / (torch.max(torch.abs(noisy_waveform)) + 1e-8)
                    
                    # Save noisy file
                    base_name = clean_file.stem
                    noisy_filename = f"{base_name}_{noise_name}.wav"
                    noisy_path = noisy_dir / noisy_filename
                    
                    # Save with compression to save space
                    torchaudio.save(
                        str(noisy_path), 
                        noisy_waveform.unsqueeze(0), 
                        16000,
                        encoding="PCM_S",
                        bits_per_sample=16  # Use 16-bit instead of 32-bit to save space
                    )
                    
                    metadata['total_pairs'] += 1
                
                processed_count += 1
                
                # Memory cleanup every 10 files
                if processed_count % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                print(f"⚠️  Error processing {clean_file}: {e}")
                continue
    
    # Calculate statistics
    metadata['avg_duration'] = np.mean(metadata['duration_stats'])
    metadata['total_duration_hours'] = sum(metadata['duration_stats']) / 3600
    
    # Save metadata
    with open('data/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Comprehensive dataset preparation complete!")
    print(f"📊 Statistics:")
    print(f"   • Total audio pairs: {metadata['total_pairs']:,}")
    print(f"   • Average duration: {metadata['avg_duration']:.2f} seconds")
    print(f"   • Total audio: {metadata['total_duration_hours']:.1f} hours")
    print(f"   • Sample rate: {metadata['sample_rate']} Hz")
    print(f"   • Noise types: {', '.join(metadata['noise_types'])}")
    
    return metadata

if __name__ == "__main__":
    try:
        metadata = prepare_comprehensive_dataset()
        print(f"\n🎯 Ready for training with {metadata['total_pairs']:,} audio pairs!")
        print("Next step: Run 'python ml/training/train.py' to start training")
        
    except KeyboardInterrupt:
        print("\n⏹️  Data preparation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        raise
