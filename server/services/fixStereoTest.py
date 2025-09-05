#!/usr/bin/env python3
"""
Quick test to fix stereo audio processing
"""
import torch
import torchaudio
import sys
import os

def fix_stereo_and_test(input_path, output_path):
    """Test stereo conversion fix"""
    print("🚨 TESTING STEREO FIX...")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(input_path)
    print(f"   Original shape: {waveform.shape}")
    
    # Fix stereo conversion
    if waveform.shape[0] == 2:  # Stereo
        print("   Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"   New shape: {waveform.shape}")
    
    # Save test
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"   Saved to: {output_path}")
    print("✅ Stereo fix test completed!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fixStereoTest.py <input> <output>")
        sys.exit(1)
    
    fix_stereo_and_test(sys.argv[1], sys.argv[2])
