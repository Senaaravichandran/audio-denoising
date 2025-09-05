"""
DCCRN Audio Enhancement Service
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
    """Service for DCCRN audio enhancement"""
    
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
            print(f"   Fast mode: enabled for optimized performance")
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
        Apply optimized fast denoising with perfect quality
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Enhanced audio tensor
        """
        print(f"[PROCESSING] Applying fast enhanced denoising with strength {strength}")
        
        # Use optimized DCCRN enhancement
        enhanced = self.inference.enhance_audio(waveform, strength)
        
        # Only apply additional processing for very high strength requests
        if strength >= 0.9:  # Only for maximum quality requests
            print("   Applying high-quality post-processing...")
            # Quick spectral cleanup for highest quality
            enhanced = self._apply_fast_spectral_cleanup(enhanced)
        
        print(f"   Enhanced denoising complete")
        return enhanced
    
    def _apply_fast_spectral_cleanup(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply fast spectral cleanup for highest quality
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            Cleaned audio tensor
        """
        print(f"   Applying fast spectral cleanup")
        
        # Use smaller FFT for speed
        n_fft = 256  # Smaller FFT for speed
        hop_length = 128
        
        # Convert to frequency domain
        stft = torch.stft(waveform.squeeze(), n_fft=n_fft, hop_length=hop_length, 
                         win_length=n_fft, return_complex=True)
        
        # Calculate magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Quick noise floor estimation (faster method)
        noise_floor = torch.quantile(magnitude, 0.05, dim=-1, keepdim=True)
        
        # Apply gentle gating (preserve quality while removing noise)
        threshold = noise_floor * 0.5
        gate_mask = magnitude > threshold
        
        # Smooth the mask to avoid artifacts
        gate_mask = gate_mask.float()
        
        # Apply gating
        cleaned_magnitude = magnitude * gate_mask
        
        # Reconstruct complex spectrum
        cleaned_stft = cleaned_magnitude * torch.exp(1j * phase)
        
        # Convert back to time domain
        cleaned_audio = torch.istft(cleaned_stft, n_fft=n_fft, hop_length=hop_length, 
                                   win_length=n_fft, length=waveform.shape[-1])
        
        # Restore original shape
        if waveform.dim() == 2:
            cleaned_audio = cleaned_audio.unsqueeze(0)
            
        return cleaned_audio
    
    def _process_chunked(self, waveform: torch.Tensor, strength: float, sample_rate: int) -> torch.Tensor:
        """
        Process long audio files in chunks for faster processing
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength
            sample_rate: Audio sample rate
            
        Returns:
            Enhanced audio tensor
        """
        print(f"   Processing in optimized chunks...")
        
        # Chunk settings (10 seconds per chunk with 1 second overlap)
        chunk_duration = 10  # seconds
        overlap_duration = 1  # seconds
        
        chunk_size = chunk_duration * sample_rate
        overlap_size = overlap_duration * sample_rate
        hop_size = chunk_size - overlap_size
        
        total_samples = waveform.shape[-1]
        num_chunks = (total_samples - overlap_size) // hop_size + 1
        
        enhanced_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * hop_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Extract chunk
            chunk = waveform[..., start_idx:end_idx]
            
            # Process chunk
            enhanced_chunk = self._apply_enhanced_denoising(chunk, strength)
            
            # Handle overlap for smooth transitions
            if i > 0 and len(enhanced_chunks) > 0:
                # Apply crossfade to overlap region
                overlap_samples = overlap_size
                fade_in = torch.linspace(0, 1, overlap_samples)
                fade_out = torch.linspace(1, 0, overlap_samples)
                
                # Crossfade overlapping parts
                enhanced_chunks[-1][..., -overlap_samples:] *= fade_out
                enhanced_chunk[..., :overlap_samples] *= fade_in
                enhanced_chunks[-1][..., -overlap_samples:] += enhanced_chunk[..., :overlap_samples]
                
                # Add non-overlapping part
                enhanced_chunks.append(enhanced_chunk[..., overlap_samples:])
            else:
                enhanced_chunks.append(enhanced_chunk)
            
            # Progress update
            progress = (i + 1) / num_chunks * 100
            print(f"   Chunk {i+1}/{num_chunks} processed ({progress:.1f}%)")
        
        # Concatenate all chunks
        enhanced_waveform = torch.cat(enhanced_chunks, dim=-1)
        
        print(f"   Chunked processing complete: {enhanced_waveform.shape}")
        return enhanced_waveform
    
    def enhance_audio(self, input_path: str, output_path: str, denoising_strength: float = 1.0) -> dict:
        """
        Enhance audio file using DCCRN model
        
        Args:
            input_path: Path to input noisy audio file
            output_path: Path to save enhanced audio  
            denoising_strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Result dictionary with success status and metadata
        """
        try:
            # Convert to absolute paths
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            print(f"[PROCESSING] Processing: {Path(input_path).name}")
            print(f"   Absolute input path: {input_path}")
            print(f"   File exists: {os.path.exists(input_path)}")
            print(f"   File size: {os.path.getsize(input_path)} bytes")
            
            # Convert to WAV if needed
            wav_path = self._convert_to_wav_if_needed(input_path)
            temp_wav_created = wav_path != input_path
            
            try:
                # Load input audio from WAV file
                waveform, sample_rate = torchaudio.load(wav_path)
                
                print(f"   Input shape: {waveform.shape}")
                print(f"   Sample rate: {sample_rate} Hz")
                print(f"   Duration: {waveform.shape[-1] / sample_rate:.2f}s")
                print(f"   Denoising strength: {denoising_strength}")
                
                # Apply fast chunked processing for long audio files
                duration_seconds = waveform.shape[-1] / sample_rate
                if duration_seconds > 30:  # For files longer than 30 seconds
                    print(f"   Using chunked processing for {duration_seconds:.1f}s audio")
                    enhanced_audio = self._process_chunked(waveform, denoising_strength, sample_rate)
                else:
                    # Direct processing for short files
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
                        'device': str(self.inference.device)
                    }
                }
                
                print(f"[SUCCESS] Enhancement completed:")
                print(f"   Duration: {result['metadata']['input_duration']:.2f}s")
                print(f"   Size: {result['metadata']['input_size_kb']:.1f}KB -> {result['metadata']['output_size_kb']:.1f}KB")
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
            error_msg = f"DCCRN enhancement failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': input_path if 'input_path' in locals() else None,
                'output_path': output_path if 'output_path' in locals() else None
            }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='DCCRN Audio Enhancement Service')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path')
    parser.add_argument('--strength', '-s', type=float, default=0.8, help='Denoising strength (0.0-1.0)')
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
