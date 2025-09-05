"""
DCCRN Ultra-Aggressive Audio Enhancement Service - FRESH VERSION
Bridge between Node.js server and Python DCCRN model
"""

import sys
import os
import json
import argparse
import tempfile
from pathlib import Path
import torch
import torchaudio
from pydub import AudioSegment

# Add ml directory to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.models.dccrn import DCCRN
from ml.inference import DCCRNInference

class DCCRNServiceUltra:
    """Ultra-Aggressive Service for DCCRN audio enhancement"""
    
    def __init__(self, model_path: str = None):
        """Initialize DCCRN service"""
        
        if model_path is None:
            # Default to latest checkpoint
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'dccrn_latest.pth')
        
        self.model_path = model_path
        
        # Model configuration matching our trained model
        self.config = {
            'n_fft': 512,
            'hop_length': 256,
            'win_length': 512,
            'encoder_layers': 3,  # Fast training config
            'hidden_dim': 64,     # Fast training config  
            'lstm_layers': 1,     # Fast training config
            'use_clstm': True,
            'kernel_size': [5, 2],
            'stride': [2, 1],
            'use_cbn': True,
            'masking_mode': 'E',
            'causal': False
        }
        
        # Initialize inference
        self.inference = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained DCCRN model"""
        try:
            self.inference = DCCRNInference(self.model_path, config=self.config, fast_mode=True)
            print(f"[SUCCESS] DCCRN model loaded successfully from {self.model_path}")
            print(f"   Parameters: {sum(p.numel() for p in self.inference.model.parameters()):,}")
            print(f"   Device: {self.inference.device}")
            print(f"   Mode: ULTRA-AGGRESSIVE denoising enabled")
        except Exception as e:
            print(f"[ERROR] Failed to load DCCRN model: {e}")
            raise
    
    def _convert_to_wav_if_needed(self, input_path: str) -> str:
        """
        Convert audio file to WAV format if needed.
        Returns the path to the WAV file (original or converted).
        """
        file_ext = Path(input_path).suffix.lower()
        
        # If already WAV, return as-is
        if file_ext == '.wav':
            return input_path
                
        # For other formats, convert to temporary WAV file
        try:
            print(f"[PROCESSING] Converting {file_ext} to WAV format...")
            
            # Load audio with pydub
            audio = AudioSegment.from_file(input_path)
            
            # Convert to WAV in temp directory
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Export as WAV
            audio.export(temp_wav_path, format='wav')
            
            print(f"   Converted to: {temp_wav_path}")
            print(f"   Duration: {len(audio) / 1000:.2f}s")
            print(f"   Sample rate: {audio.frame_rate} Hz")
            print(f"   Channels: {audio.channels}")
            
            return temp_wav_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert {file_ext} to WAV: {str(e)}")
    
    def _apply_ultra_aggressive_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply ultra-aggressive 6-stage denoising for dramatically cleaner audio
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Dramatically enhanced audio tensor
        """
        print(f"[PROCESSING] Applying ULTRA-AGGRESSIVE 6-stage denoising with strength {strength}")
        
        # Stage 1: Initial DCCRN noise reduction with maximum strength
        print("   🚀 Stage 1/6: Initial AI noise reduction...")
        enhanced = self.inference.enhance_audio(waveform, min(strength * 1.5, 1.0))
        
        # Stage 2: Apply DCCRN again with different parameters for deeper cleaning
        print("   🔧 Stage 2/6: Deep AI enhancement pass...")
        enhanced = self.inference.enhance_audio(enhanced, min(strength * 1.2, 1.0))
        
        # Stage 3: Advanced spectral denoising with ultra-aggressive parameters
        print("   🎯 Stage 3/6: Advanced spectral noise reduction...")
        enhanced = self._apply_advanced_spectral_denoising(enhanced, strength)
        
        # Stage 4: Targeted noise type removal (ultra-aggressive)
        print("   ⚡ Stage 4/6: Targeted noise pattern removal...")
        enhanced = self._apply_targeted_noise_removal(enhanced, strength)
        
        # Stage 5: Voice enhancement and clarity boost
        print("   🎤 Stage 5/6: Voice clarity enhancement...")
        enhanced = self._apply_voice_enhancement(enhanced, strength)
        
        # Stage 6: Final quality enhancement and dynamic range optimization
        print("   ✨ Stage 6/6: Final quality enhancement...")
        enhanced = self._apply_final_quality_enhancement(enhanced, strength)
        
        print(f"   🎉 ULTRA-AGGRESSIVE 6-stage denoising complete - audio dramatically enhanced!")
        return enhanced
    
    def _apply_advanced_spectral_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply ultra-aggressive spectral denoising"""
        print(f"      → Ultra-aggressive spectral analysis and noise reduction...")
        
        # Use larger FFT for better frequency resolution
        n_fft = 1024  # Higher resolution for better noise detection
        hop_length = 256
        win_length = 1024
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         win_length=win_length, return_complex=True, window=torch.hann_window(win_length))
        
        # Calculate magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Ultra-aggressive noise floor estimation
        noise_floor_temporal = torch.quantile(magnitude, 0.05, dim=-1, keepdim=True)  # Very aggressive
        noise_floor_spectral = torch.quantile(magnitude, 0.03, dim=-2, keepdim=True)  # Ultra-aggressive
        noise_floor = torch.minimum(noise_floor_temporal, noise_floor_spectral)
        
        # Create ultra-aggressive noise gate
        gate_threshold = noise_floor * (0.05 + 0.15 * (1 - strength))  # Ultra-aggressive threshold
        
        # Smooth gating to avoid artifacts
        gate_mask = torch.sigmoid((magnitude - gate_threshold) / (gate_threshold * 0.02))
        
        # Apply ultra-aggressive suppression for low-energy regions
        energy_threshold = torch.quantile(magnitude, 0.1)  # Lower threshold
        low_energy_mask = magnitude < energy_threshold
        gate_mask = torch.where(low_energy_mask, gate_mask * 0.1, gate_mask)  # Ultra-strong suppression
        
        # Apply the ultra-aggressive gate
        cleaned_magnitude = magnitude * gate_mask
        
        # Reconstruct with preserved phase
        cleaned_stft = cleaned_magnitude * torch.exp(1j * phase)
        
        # Convert back to time domain
        cleaned_audio = torch.istft(cleaned_stft, n_fft=n_fft, hop_length=hop_length, 
                                   win_length=win_length, length=waveform.shape[-1],
                                   window=torch.hann_window(win_length))
        
        # Restore original shape
        if waveform.dim() == 2:
            cleaned_audio = cleaned_audio.unsqueeze(0)
            
        return cleaned_audio
    
    def _apply_targeted_noise_removal(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """Remove specific noise patterns with ultra-aggressive approach"""
        print(f"      → Ultra-aggressive removal of hums, clicks, and electrical interference...")
        
        # Setup for targeted noise removal
        sample_rate = 16000  # DCCRN model sample rate
        n_fft = 512
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Remove electrical hum and harmonics (comprehensive list)
        freq_bins = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Extended target frequencies for ultra-aggressive removal
        target_freqs = [50, 60, 100, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600]  # Hz
        
        for target_freq in target_freqs:
            # Find frequency bins close to target
            freq_mask = torch.abs(freq_bins - target_freq) < 10  # Wider window for better removal
            if freq_mask.any():
                suppression = 0.02 + 0.05 * (1 - strength)  # Ultra-aggressive suppression
                magnitude[freq_mask, :] *= suppression
        
        # Remove high-frequency noise and clicks (ultra-aggressive)
        high_freq_cutoff = sample_rate * 0.35  # Lower cutoff (35% of Nyquist)
        high_freq_mask = freq_bins > high_freq_cutoff
        magnitude[high_freq_mask, :] *= (0.05 + 0.2 * (1 - strength))  # Ultra-strong suppression
        
        # Detect and remove transient clicks/pops (ultra-aggressive)
        magnitude_diff = torch.diff(magnitude, dim=-1)
        spike_threshold = torch.quantile(torch.abs(magnitude_diff), 0.90)  # Lower threshold
        spike_mask = torch.abs(magnitude_diff) > spike_threshold * (1.5 - strength)
        
        # Pad the spike mask to match original size
        spike_mask_padded = torch.cat([spike_mask, torch.zeros_like(spike_mask[:, :1])], dim=-1)
        magnitude = torch.where(spike_mask_padded, magnitude * 0.2, magnitude)
        
        # Reconstruct
        cleaned_stft = magnitude * torch.exp(1j * phase)
        cleaned_audio = torch.istft(cleaned_stft, n_fft=n_fft, hop_length=hop_length, 
                                   length=waveform.shape[-1], window=torch.hann_window(n_fft))
        
        if waveform.dim() == 2:
            cleaned_audio = cleaned_audio.unsqueeze(0)
            
        return cleaned_audio
    
    def _apply_voice_enhancement(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """Ultra-aggressive voice enhancement while suppressing non-voice content"""
        print(f"      → Ultra-boosting voice clarity and presence...")
        
        sample_rate = 16000
        n_fft = 512
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        freq_bins = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Voice frequency enhancement (ultra-aggressive boosting)
        enhancement_factor = torch.ones_like(magnitude)
        
        # Ultra-boost fundamental voice frequencies (80-300Hz)
        fundamental_mask = (freq_bins >= 80) & (freq_bins <= 300)
        enhancement_factor[fundamental_mask, :] *= (1.0 + 0.5 * strength)  # Stronger boost
        
        # Ultra-boost voice clarity range (1kHz - 4kHz)
        clarity_mask = (freq_bins >= 1000) & (freq_bins <= 4000)
        enhancement_factor[clarity_mask, :] *= (1.0 + 0.6 * strength)  # Even stronger boost
        
        # Ultra-suppress very low frequencies (below 60Hz - often noise)
        low_freq_mask = freq_bins < 60
        enhancement_factor[low_freq_mask, :] *= (0.1 + 0.2 * (1 - strength))  # Ultra-strong suppression
        
        # Ultra-suppress very high frequencies (above 10kHz - often noise)
        very_high_mask = freq_bins > 10000
        enhancement_factor[very_high_mask, :] *= (0.1 + 0.2 * (1 - strength))  # Ultra-strong suppression
        
        # Apply enhancement
        enhanced_magnitude = magnitude * enhancement_factor
        
        # Apply dynamic range compression to voice frequencies
        voice_mask = (freq_bins >= 80) & (freq_bins <= 8000)
        voice_content = enhanced_magnitude[voice_mask, :]
        if voice_content.numel() > 0:
            # Stronger compression for more even voice levels
            compressed_voice = torch.sign(voice_content) * torch.pow(torch.abs(voice_content), 0.75)
            enhanced_magnitude[voice_mask, :] = compressed_voice
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length, 
                                    length=waveform.shape[-1], window=torch.hann_window(n_fft))
        
        if waveform.dim() == 2:
            enhanced_audio = enhanced_audio.unsqueeze(0)
            
        return enhanced_audio
    
    def _apply_final_quality_enhancement(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply final quality enhancement for maximum clarity"""
        print(f"      → Final ultra-quality enhancement and dynamic range optimization...")
        
        sample_rate = 16000
        n_fft = 512
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Apply final ultra-aggressive noise gating
        noise_floor = torch.quantile(magnitude, 0.02, dim=-1, keepdim=True)  # Ultra-low quantile
        gate_threshold = noise_floor * (0.05 + 0.1 * (1 - strength))  # Ultra-aggressive threshold
        
        # Create ultra-smooth gate to avoid artifacts
        gate_mask = torch.sigmoid((magnitude - gate_threshold) / (gate_threshold * 0.01))
        
        # Apply ultra-aggressive suppression for very quiet regions
        ultra_quiet_mask = magnitude < (noise_floor * 0.25)
        gate_mask = torch.where(ultra_quiet_mask, gate_mask * 0.05, gate_mask)  # Ultra-strong suppression
        
        # Apply final gating
        final_magnitude = magnitude * gate_mask
        
        # Ultra-dynamic range compression for consistency
        compressed_magnitude = torch.sign(final_magnitude) * torch.pow(torch.abs(final_magnitude), 0.85)
        
        # Reconstruct
        final_stft = compressed_magnitude * torch.exp(1j * phase)
        final_audio = torch.istft(final_stft, n_fft=n_fft, hop_length=hop_length, 
                                 length=waveform.shape[-1], window=torch.hann_window(n_fft))
        
        if waveform.dim() == 2:
            final_audio = final_audio.unsqueeze(0)
            
        return final_audio
    
    def enhance_audio(self, input_path: str, output_path: str, denoising_strength: float = 1.0) -> dict:
        """
        Enhance audio file using ultra-aggressive 6-stage DCCRN denoising
        """
        try:
            # Convert to absolute paths
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            print(f"[PROCESSING] Starting ULTRA-AGGRESSIVE denoising: {Path(input_path).name}")
            print(f"   Absolute input path: {input_path}")
            print(f"   File exists: {os.path.exists(input_path)}")
            print(f"   File size: {os.path.getsize(input_path)} bytes")
            print(f"   Denoising strength: {denoising_strength} (ULTRA-AGGRESSIVE MODE)")
            
            # Convert to WAV if needed
            wav_path = self._convert_to_wav_if_needed(input_path)
            temp_wav_created = wav_path != input_path
            
            try:
                # Load input audio from WAV file
                waveform, sample_rate = torchaudio.load(wav_path)
                
                print(f"   Loaded audio - shape: {waveform.shape}, sample rate: {sample_rate} Hz")
                
                # 🚨 CRITICAL FIX: Convert stereo to mono BEFORE any processing
                if waveform.shape[0] == 2:  # Stereo audio
                    print("   🚨 FIXING: Converting stereo to mono for DCCRN compatibility...")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    print(f"   ✅ FIXED: Converted to mono - new shape: {waveform.shape}")
                elif waveform.shape[0] > 2:  # Multi-channel audio
                    print(f"   🚨 FIXING: Converting {waveform.shape[0]}-channel audio to mono...")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    print(f"   ✅ FIXED: Converted to mono - new shape: {waveform.shape}")
                
                # Ensure correct sample rate for DCCRN model (16kHz)
                target_sample_rate = 16000
                if sample_rate != target_sample_rate:
                    print(f"   Resampling from {sample_rate}Hz to {target_sample_rate}Hz...")
                    resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                    waveform = resampler(waveform)
                    sample_rate = target_sample_rate
                
                print(f"   Input shape: {waveform.shape}")
                print(f"   Sample rate: {sample_rate} Hz")
                print(f"   Duration: {waveform.shape[-1] / sample_rate:.2f}s")
                
                # Apply ultra-aggressive 6-stage processing
                enhanced_audio = self._apply_ultra_aggressive_denoising(waveform, denoising_strength)
                
                print(f"   Output shape: {enhanced_audio.shape}")
                
                # Create output directory if needed
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save enhanced audio
                torchaudio.save(output_path, enhanced_audio, sample_rate)
                
                # Get file sizes for comparison
                input_size = Path(input_path).stat().st_size
                output_size = Path(output_path).stat().st_size
                
                result = {
                    'success': True,
                    'input_path': input_path,
                    'output_path': output_path,
                    'metadata': {
                        'input_duration': waveform.shape[-1] / sample_rate,
                        'sample_rate': sample_rate,
                        'channels': enhanced_audio.shape[0],
                        'input_size_kb': input_size / 1024,
                        'output_size_kb': output_size / 1024,
                        'denoising_strength': denoising_strength,
                        'model_parameters': sum(p.numel() for p in self.inference.model.parameters()),
                        'device': str(self.inference.device),
                        'processing_type': 'ULTRA_AGGRESSIVE_6_STAGE'
                    }
                }
                
                print(f"[SUCCESS] ULTRA-AGGRESSIVE 6-Stage Enhancement completed:")
                print(f"   Duration: {result['metadata']['input_duration']:.2f}s")
                print(f"   Size: {result['metadata']['input_size_kb']:.1f}KB -> {result['metadata']['output_size_kb']:.1f}KB")
                print(f"   Processing: 6-stage ultra-aggressive noise reduction applied")
                print(f"   Saved: {output_path}")
                
                return result
                
            finally:
                # Clean up temporary WAV file if created
                if temp_wav_created and os.path.exists(wav_path):
                    try:
                        os.unlink(wav_path)
                        print(f"   Cleaned up temporary file: {wav_path}")
                    except:
                        pass  # Ignore cleanup errors
                        
        except Exception as e:
            error_msg = f"DCCRN ultra-aggressive enhancement failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': input_path if 'input_path' in locals() else None,
                'output_path': output_path if 'output_path' in locals() else None
            }

def main():
    """Command line interface for DCCRN Ultra-Aggressive service"""
    parser = argparse.ArgumentParser(description='DCCRN Ultra-Aggressive Audio Enhancement Service')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path') 
    parser.add_argument('--model', '-m', help='Model checkpoint path')
    parser.add_argument('--strength', '-s', type=float, default=1.0, help='Denoising strength (0.0-1.0)')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    
    args = parser.parse_args()
    
    # Validate denoising strength
    if not 0.0 <= args.strength <= 1.0:
        print("❌ Denoising strength must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize service
    try:
        service = DCCRNServiceUltra(args.model)
    except Exception as e:
        if args.json:
            print(json.dumps({'success': False, 'error': f'Model initialization failed: {e}'}))
        else:
            print(f"[ERROR] Model initialization failed: {e}")
        sys.exit(1)
    
    # Enhance audio
    result = service.enhance_audio(args.input, args.output, args.strength)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result['success']:
            print(f"\n[SUCCESS] Ultra-aggressive audio enhancement completed successfully!")
        else:
            print(f"\n[ERROR] Audio enhancement failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
