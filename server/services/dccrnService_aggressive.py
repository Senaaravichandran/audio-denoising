"""
DCCRN Audio Enhancement Service with Aggressive Multi-Stage Denoising
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

class DCCRNService:
    """Service for aggressive DCCRN audio enhancement"""
    
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
        
        # Load model
        self._load_model()

    def _load_model(self):
        """Load the trained DCCRN model"""
        try:
            self.inference = DCCRNInference(self.model_path, config=self.config, fast_mode=True)
            print(f"[SUCCESS] DCCRN model loaded successfully from {self.model_path}")
            print(f"   Parameters: {sum(p.numel() for p in self.inference.model.parameters()):,}")
            print(f"   Device: {self.inference.device}")
            print(f"   Mode: AGGRESSIVE denoising enabled")
        except Exception as e:
            print(f"[ERROR] Failed to load DCCRN model: {e}")
            raise

    def _convert_to_wav_if_needed(self, input_path: str) -> str:
        """Convert audio file to WAV format if needed"""
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.wav':
            print(f"   Input is already WAV format")
            return input_path
            
        print(f"[CONVERSION] Converting {file_ext} to WAV...")
        
        try:
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
    
    def _apply_enhanced_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply aggressive multi-stage denoising for dramatically cleaner audio
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Dramatically enhanced audio tensor
        """
        print(f"[PROCESSING] Applying AGGRESSIVE multi-stage denoising with strength {strength}")
        
        # Stage 1: Initial DCCRN noise reduction
        print("   Stage 1/5: Initial AI noise reduction...")
        enhanced = self.inference.enhance_audio(waveform, min(strength * 1.2, 1.0))
        
        # Stage 2: Apply DCCRN again for deeper cleaning
        print("   Stage 2/5: Deep AI enhancement pass...")
        enhanced = self.inference.enhance_audio(enhanced, min(strength * 0.8, 1.0))
        
        # Stage 3: Advanced spectral denoising
        print("   Stage 3/5: Advanced spectral noise reduction...")
        enhanced = self._apply_advanced_spectral_denoising(enhanced, strength)
        
        # Stage 4: Targeted noise type removal
        print("   Stage 4/5: Targeted noise pattern removal...")
        enhanced = self._apply_targeted_noise_removal(enhanced, strength)
        
        # Stage 5: Final voice enhancement and cleanup
        print("   Stage 5/5: Voice clarity enhancement...")
        enhanced = self._apply_voice_enhancement(enhanced, strength)
        
        print(f"   Multi-stage aggressive denoising complete - audio should be dramatically cleaner!")
        return enhanced
    
    def _apply_advanced_spectral_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply advanced spectral denoising with aggressive noise reduction
        """
        print(f"      → Advanced spectral analysis and noise reduction...")
        
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
        
        # Advanced noise floor estimation using multiple methods
        noise_floor_temporal = torch.quantile(magnitude, 0.15, dim=-1, keepdim=True)  # Time-based
        noise_floor_spectral = torch.quantile(magnitude, 0.10, dim=-2, keepdim=True)  # Frequency-based
        noise_floor = torch.minimum(noise_floor_temporal, noise_floor_spectral)
        
        # Create aggressive noise gate with smooth transitions
        gate_threshold = noise_floor * (0.3 + 0.4 * (1 - strength))  # More aggressive for higher strength
        
        # Smooth gating to avoid artifacts
        gate_mask = torch.sigmoid((magnitude - gate_threshold) / (gate_threshold * 0.1))
        
        # Apply additional suppression for low-energy regions
        energy_threshold = torch.quantile(magnitude, 0.25)
        low_energy_mask = magnitude < energy_threshold
        gate_mask = torch.where(low_energy_mask, gate_mask * 0.5, gate_mask)
        
        # Apply the aggressive gate
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
        """
        Remove specific noise patterns: hums, clicks, electrical interference
        """
        print(f"      → Removing hums, clicks, and electrical interference...")
        
        # Setup for targeted noise removal
        sample_rate = 16000  # DCCRN model sample rate
        n_fft = 512
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Remove electrical hum (50Hz, 60Hz and harmonics)
        freq_bins = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Target frequencies for removal (hums and electrical noise)
        target_freqs = [50, 60, 100, 120, 150, 180, 240, 300]  # Hz
        
        for target_freq in target_freqs:
            # Find frequency bins close to target
            freq_mask = torch.abs(freq_bins - target_freq) < 5  # 5Hz window
            if freq_mask.any():
                suppression = 0.1 + 0.2 * (1 - strength)  # More aggressive suppression
                magnitude[freq_mask, :] *= suppression
        
        # Remove high-frequency noise and clicks
        high_freq_cutoff = sample_rate * 0.45  # 45% of Nyquist
        high_freq_mask = freq_bins > high_freq_cutoff
        magnitude[high_freq_mask, :] *= (0.3 + 0.5 * (1 - strength))
        
        # Detect and remove transient clicks/pops
        # Look for sudden energy spikes across frequencies
        magnitude_diff = torch.diff(magnitude, dim=-1)
        spike_threshold = torch.quantile(torch.abs(magnitude_diff), 0.95)
        spike_mask = torch.abs(magnitude_diff) > spike_threshold * (2 - strength)
        
        # Pad the spike mask to match original size
        spike_mask_padded = torch.cat([spike_mask, torch.zeros_like(spike_mask[:, :1])], dim=-1)
        magnitude = torch.where(spike_mask_padded, magnitude * 0.5, magnitude)
        
        # Reconstruct
        cleaned_stft = magnitude * torch.exp(1j * phase)
        cleaned_audio = torch.istft(cleaned_stft, n_fft=n_fft, hop_length=hop_length, 
                                   length=waveform.shape[-1], window=torch.hann_window(n_fft))
        
        if waveform.dim() == 2:
            cleaned_audio = cleaned_audio.unsqueeze(0)
            
        return cleaned_audio
    
    def _apply_voice_enhancement(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Enhance voice frequencies while suppressing non-voice content
        """
        print(f"      → Enhancing voice clarity and presence...")
        
        sample_rate = 16000
        n_fft = 512
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        freq_bins = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Voice frequency enhancement (typical human speech: 80Hz - 8kHz)
        voice_low = 80
        voice_high = 8000
        voice_mask = (freq_bins >= voice_low) & (freq_bins <= voice_high)
        
        # Create voice enhancement curve
        enhancement_factor = torch.ones_like(magnitude)
        
        # Boost fundamental voice frequencies (80-300Hz)
        fundamental_mask = (freq_bins >= 80) & (freq_bins <= 300)
        enhancement_factor[fundamental_mask, :] *= (1.0 + 0.3 * strength)
        
        # Boost voice clarity range (1kHz - 4kHz)
        clarity_mask = (freq_bins >= 1000) & (freq_bins <= 4000)
        enhancement_factor[clarity_mask, :] *= (1.0 + 0.4 * strength)
        
        # Suppress very low frequencies (below 60Hz - often noise)
        low_freq_mask = freq_bins < 60
        enhancement_factor[low_freq_mask, :] *= (0.2 + 0.3 * (1 - strength))
        
        # Suppress very high frequencies (above 10kHz - often noise)
        very_high_mask = freq_bins > 10000
        enhancement_factor[very_high_mask, :] *= (0.3 + 0.4 * (1 - strength))
        
        # Apply enhancement
        enhanced_magnitude = magnitude * enhancement_factor
        
        # Apply dynamic range compression to voice frequencies
        voice_content = enhanced_magnitude[voice_mask, :]
        if voice_content.numel() > 0:
            # Gentle compression to even out voice levels
            compressed_voice = torch.sign(voice_content) * torch.pow(torch.abs(voice_content), 0.8)
            enhanced_magnitude[voice_mask, :] = compressed_voice
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length, 
                                    length=waveform.shape[-1], window=torch.hann_window(n_fft))
        
        if waveform.dim() == 2:
            enhanced_audio = enhanced_audio.unsqueeze(0)
            
        return enhanced_audio
    
    def _process_chunked_aggressive(self, waveform: torch.Tensor, strength: float, sample_rate: int) -> torch.Tensor:
        """
        Process long audio files in smaller chunks with aggressive denoising
        """
        print(f"   Processing in high-quality chunks for maximum noise reduction...")
        
        # Smaller chunks for better quality (5 seconds per chunk with 0.5 second overlap)
        chunk_duration = 5  # seconds
        overlap_duration = 0.5  # seconds
        
        chunk_size = chunk_duration * sample_rate
        overlap_size = int(overlap_duration * sample_rate)
        hop_size = chunk_size - overlap_size
        
        total_samples = waveform.shape[-1]
        num_chunks = (total_samples - overlap_size) // hop_size + 1
        
        enhanced_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * hop_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Extract chunk
            chunk = waveform[..., start_idx:end_idx]
            
            print(f"      Processing chunk {i+1}/{num_chunks} with aggressive denoising...")
            # Apply aggressive multi-stage denoising to each chunk
            enhanced_chunk = self._apply_enhanced_denoising(chunk, strength)
            
            # Handle overlap for smooth transitions
            if i > 0 and len(enhanced_chunks) > 0:
                # Apply crossfade to overlap region
                overlap_samples = overlap_size
                if overlap_samples > 0 and enhanced_chunk.shape[-1] >= overlap_samples:
                    fade_in = torch.linspace(0, 1, overlap_samples)
                    fade_out = torch.linspace(1, 0, overlap_samples)
                    
                    # Crossfade overlapping parts
                    if enhanced_chunks[-1].shape[-1] >= overlap_samples:
                        enhanced_chunks[-1][..., -overlap_samples:] *= fade_out
                        enhanced_chunk[..., :overlap_samples] *= fade_in
                        enhanced_chunks[-1][..., -overlap_samples:] += enhanced_chunk[..., :overlap_samples]
                        
                        # Add non-overlapping part
                        enhanced_chunks.append(enhanced_chunk[..., overlap_samples:])
                    else:
                        enhanced_chunks.append(enhanced_chunk)
                else:
                    enhanced_chunks.append(enhanced_chunk)
            else:
                enhanced_chunks.append(enhanced_chunk)
            
            # Progress update
            progress = (i + 1) / num_chunks * 100
            print(f"      Chunk {i+1}/{num_chunks} completed ({progress:.1f}% done)")
        
        # Concatenate all chunks
        enhanced_waveform = torch.cat(enhanced_chunks, dim=-1)
        
        print(f"   High-quality chunked processing complete: {enhanced_waveform.shape}")
        return enhanced_waveform

    def enhance_audio(self, input_path: str, output_path: str, denoising_strength: float = 1.0) -> dict:
        """
        Enhance audio file using aggressive multi-stage DCCRN denoising
        """
        try:
            # Convert to absolute paths
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            print(f"[PROCESSING] Starting AGGRESSIVE denoising: {Path(input_path).name}")
            print(f"   Absolute input path: {input_path}")
            print(f"   File exists: {os.path.exists(input_path)}")
            print(f"   File size: {os.path.getsize(input_path)} bytes")
            print(f"   Denoising strength: {denoising_strength} (AGGRESSIVE MODE)")
            
            # Convert to WAV if needed
            wav_path = self._convert_to_wav_if_needed(input_path)
            temp_wav_created = wav_path != input_path
            
            try:
                # Load input audio from WAV file
                waveform, sample_rate = torchaudio.load(wav_path)
                
                print(f"   Input shape: {waveform.shape}")
                print(f"   Sample rate: {sample_rate} Hz")
                print(f"   Duration: {waveform.shape[-1] / sample_rate:.2f}s")
                
                # Apply aggressive processing for long audio files  
                duration_seconds = waveform.shape[-1] / sample_rate
                if duration_seconds > 20:  # Use chunking for files longer than 20 seconds
                    print(f"   Using high-quality chunked processing for {duration_seconds:.1f}s audio")
                    enhanced_audio = self._process_chunked_aggressive(waveform, denoising_strength, sample_rate)
                else:
                    # Direct aggressive processing for shorter files
                    enhanced_audio = self._apply_enhanced_denoising(waveform, denoising_strength)
                
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
                        'processing_type': 'AGGRESSIVE_MULTI_STAGE'
                    }
                }
                
                print(f"[SUCCESS] AGGRESSIVE Enhancement completed:")
                print(f"   Duration: {result['metadata']['input_duration']:.2f}s")
                print(f"   Size: {result['metadata']['input_size_kb']:.1f}KB -> {result['metadata']['output_size_kb']:.1f}KB")
                print(f"   Processing: 5-stage aggressive noise reduction applied")
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
            error_msg = f"DCCRN aggressive enhancement failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': input_path if 'input_path' in locals() else None,
                'output_path': output_path if 'output_path' in locals() else None
            }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='DCCRN Aggressive Audio Enhancement Service')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path')
    parser.add_argument('--strength', '-s', type=float, default=0.9, help='Denoising strength (0.0-1.0) - default 0.9 for aggressive mode')
    parser.add_argument('--model', '-m', help='Path to DCCRN model checkpoint')
    
    args = parser.parse_args()
    
    # Initialize service
    service = DCCRNService(model_path=args.model)
    
    # Process audio
    result = service.enhance_audio(args.input, args.output, args.strength)
    
    # Output result as JSON for Node.js integration
    print(json.dumps(result, indent=2))
    
    return 0 if result['success'] else 1

if __name__ == '__main__':
    exit(main())
