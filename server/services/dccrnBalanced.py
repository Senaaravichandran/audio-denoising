"""
DCCRN Balanced Audio Enhancement Service - SPEECH-PRESERVING VERSION
Effective denoising while preserving natural speech quality
"""

import sys
import os
import json
import argparse
import tempfile
import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment

# Add ML module to path - fix path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
ml_path = os.path.join(project_root, 'ml')
sys.path.insert(0, os.path.abspath(project_root))
sys.path.insert(0, os.path.abspath(ml_path))

try:
    from ml.inference import DCCRNInference
except ImportError as e:
    print(f"Error importing DCCRNInference: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {os.path.abspath(project_root)}")
    print(f"ML path: {os.path.abspath(ml_path)}")
    sys.exit(1)

class DCCRNBalancedService:
    """Balanced DCCRN Audio Enhancement Service - preserves speech while removing noise"""
    
    def __init__(self, model_path: str = None):
        # Default model path
        if model_path is None:
            # Get model path relative to this script
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
            print(f"   Mode: BALANCED denoising - preserves speech quality")
        except Exception as e:
            print(f"[ERROR] Failed to load DCCRN model: {e}")
            raise
    
    def _convert_to_wav_if_needed(self, input_path: str) -> str:
        """Convert audio file to WAV format if needed"""
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.wav':
            return input_path
                
        try:
            print(f"[PROCESSING] Converting {file_ext} to WAV format...")
            
            audio = AudioSegment.from_file(input_path)
            
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            audio.export(temp_wav_path, format='wav')
            
            print(f"   Converted to: {temp_wav_path}")
            print(f"   Duration: {len(audio) / 1000:.2f}s")
            print(f"   Sample rate: {audio.frame_rate} Hz")
            print(f"   Channels: {audio.channels}")
            
            return temp_wav_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert {file_ext} to WAV: {str(e)}")
    
    def _apply_balanced_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply balanced 3-stage denoising that preserves speech while removing noise
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Enhanced audio tensor with preserved speech
        """
        print(f"[PROCESSING] Applying BALANCED 3-stage denoising with strength {strength}")
        
        # Stage 1: Primary DCCRN denoising with controlled strength
        print("   🎯 Stage 1/3: AI-powered noise reduction...")
        # Use moderate strength to preserve speech dynamics
        dccrn_strength = min(strength * 0.8, 0.9)  # Cap at 0.9 to preserve speech
        enhanced = self.inference.enhance_audio(waveform, dccrn_strength)
        
        # Stage 2: Gentle spectral cleanup - preserve speech frequencies
        print("   🔧 Stage 2/3: Gentle spectral enhancement...")
        enhanced = self._apply_speech_preserving_spectral_cleanup(enhanced, strength)
        
        # Stage 3: Voice clarity enhancement without over-processing
        print("   ✨ Stage 3/3: Voice clarity optimization...")
        enhanced = self._apply_voice_clarity_enhancement(enhanced, strength)
        
        print(f"   🎉 BALANCED denoising complete - natural speech preserved!")
        return enhanced
    
    def _apply_speech_preserving_spectral_cleanup(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply gentle spectral cleanup that preserves speech frequencies
        """
        print(f"      → Gentle spectral cleanup preserving speech...")
        
        # Ensure waveform is 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Use smaller FFT for better time resolution (preserves speech dynamics)
        n_fft = 512
        hop_length = 256
        win_length = 512
        
        # Convert to frequency domain
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                         win_length=win_length, return_complex=True, window=torch.hann_window(win_length))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Conservative noise floor estimation (preserve quiet speech)
        noise_floor = torch.quantile(magnitude, 0.15, dim=-1, keepdim=True)  # Less aggressive
        
        # Gentle noise gate that preserves speech dynamics
        gate_threshold = noise_floor * (0.3 + 0.4 * (1 - strength))  # More conservative
        
        # Smooth gating to avoid speech artifacts
        gate_mask = torch.sigmoid((magnitude - gate_threshold) / (gate_threshold * 0.2))  # Smoother transition
        
        # Apply gentle suppression only to very low-energy regions
        suppression_mask = torch.where(
            magnitude < gate_threshold * 0.5,  # Only suppress very quiet noise
            gate_mask * (0.1 + 0.2 * (1 - strength)),  # Gentle suppression
            torch.ones_like(magnitude)  # Preserve everything else
        )
        
        # Apply mask
        enhanced_magnitude = magnitude * suppression_mask
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_waveform = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, window=torch.hann_window(win_length),
                                       length=waveform.shape[-1])
        
        # Ensure output dimension matches input
        if enhanced_waveform.dim() == 1:
            enhanced_waveform = enhanced_waveform.unsqueeze(0)
        
        return enhanced_waveform
    
    def _apply_voice_clarity_enhancement(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply gentle voice clarity enhancement without over-processing
        """
        print(f"      → Enhancing voice clarity naturally...")
        
        # Ensure waveform is 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        sample_rate = 16000
        n_fft = 512
        hop_length = 256
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True, window=torch.hann_window(n_fft))
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        freq_bins = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Define speech frequency ranges
        speech_fundamental = (80, 300)    # Fundamental frequency range
        speech_formants = (300, 3400)     # Main speech formants
        speech_clarity = (3400, 8000)     # Clarity and consonants
        
        # Create gentle enhancement for speech frequencies
        enhancement_mask = torch.ones_like(magnitude)
        
        # Gentle boost for speech clarity (high frequencies)
        clarity_mask = (freq_bins >= speech_clarity[0]) & (freq_bins <= speech_clarity[1])
        if clarity_mask.any():
            clarity_boost = 1.0 + (strength * 0.1)  # Very gentle boost
            enhancement_mask[clarity_mask] *= clarity_boost
        
        # Gentle boost for speech formants
        formant_mask = (freq_bins >= speech_formants[0]) & (freq_bins <= speech_formants[1])
        if formant_mask.any():
            formant_boost = 1.0 + (strength * 0.05)  # Very gentle boost
            enhancement_mask[formant_mask] *= formant_boost
        
        # Apply enhancement
        enhanced_magnitude = magnitude * enhancement_mask
        
        # Gentle dynamic range compression to improve clarity
        # Apply soft compression only to very loud parts
        loud_threshold = torch.quantile(enhanced_magnitude, 0.95)
        compression_mask = enhanced_magnitude > loud_threshold
        if compression_mask.any():
            compression_ratio = 0.9 + (0.1 * (1 - strength))  # Gentle compression
            enhanced_magnitude = torch.where(
                compression_mask,
                loud_threshold + (enhanced_magnitude - loud_threshold) * compression_ratio,
                enhanced_magnitude
            )
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_waveform = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                                       window=torch.hann_window(n_fft), length=waveform.shape[-1])
        
        # Ensure output dimension matches input
        if enhanced_waveform.dim() == 1:
            enhanced_waveform = enhanced_waveform.unsqueeze(0)
        
        return enhanced_waveform
    
    def _apply_fast_processing_for_long_audio(self, waveform: torch.Tensor, strength: float, sample_rate: int) -> torch.Tensor:
        """
        Apply fast processing for long audio files using efficient chunking
        """
        duration = waveform.shape[-1] / sample_rate
        print(f"   Using optimized chunked processing for {duration:.1f}s audio")
        
        # Use larger chunks for efficiency (10-second chunks)
        chunk_duration = 10  # seconds
        chunk_length = chunk_duration * sample_rate
        
        # Split audio into overlapping chunks for smooth processing
        overlap = 0.1  # 10% overlap
        overlap_length = int(chunk_length * overlap)
        
        chunks = []
        for i in range(0, waveform.shape[-1], chunk_length - overlap_length):
            end_idx = min(i + chunk_length, waveform.shape[-1])
            chunk = waveform[:, i:end_idx]
            chunks.append(chunk)
            
            progress = min(100, (i / waveform.shape[-1]) * 100)
            if i > 0:  # Don't print for first chunk
                print(f"      Processing chunk {len(chunks)}/{((waveform.shape[-1] - 1) // (chunk_length - overlap_length)) + 1} ({progress:.0f}%)")
        
        print(f"   Processing {len(chunks)} chunks with optimized denoising...")
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Apply balanced denoising to each chunk
            enhanced_chunk = self._apply_balanced_denoising(chunk, strength)
            enhanced_chunks.append(enhanced_chunk)
        
        # Merge chunks with overlap handling
        if len(enhanced_chunks) == 1:
            return enhanced_chunks[0]
        
        # Merge with cross-fade to avoid artifacts
        merged = enhanced_chunks[0]
        for i in range(1, len(enhanced_chunks)):
            # Calculate overlap region
            if merged.shape[-1] > overlap_length:
                # Cross-fade in overlap region
                fade_start = merged.shape[-1] - overlap_length
                fade_region = torch.linspace(1, 0, overlap_length)
                merged[:, fade_start:] *= fade_region
                
                # Add next chunk with fade-in
                next_chunk = enhanced_chunks[i]
                if next_chunk.shape[-1] >= overlap_length:
                    fade_in_region = torch.linspace(0, 1, overlap_length)
                    next_chunk[:, :overlap_length] *= fade_in_region
                    
                    # Merge
                    merged[:, fade_start:] += next_chunk[:, :overlap_length]
                    if next_chunk.shape[-1] > overlap_length:
                        merged = torch.cat([merged, next_chunk[:, overlap_length:]], dim=-1)
                else:
                    merged = torch.cat([merged, next_chunk], dim=-1)
            else:
                merged = torch.cat([merged, enhanced_chunks[i]], dim=-1)
        
        return merged
    
    def enhance_audio(self, input_path: str, output_path: str, denoising_strength: float = 0.7) -> dict:
        """
        Main enhancement function with balanced processing
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            denoising_strength: Denoising strength (0.0 to 1.0) - default 0.7 for balanced results
            
        Returns:
            Dict with processing results
        """
        try:
            # Convert paths to absolute
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            print(f"[PROCESSING] Starting BALANCED denoising: {Path(input_path).name}")
            print(f"   Absolute input path: {input_path}")
            print(f"   File exists: {os.path.exists(input_path)}")
            print(f"   File size: {os.path.getsize(input_path)} bytes")
            print(f"   Denoising strength: {denoising_strength} (BALANCED MODE - preserves speech)")
            
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
                
                # Choose processing method based on duration
                duration_seconds = waveform.shape[-1] / sample_rate
                if duration_seconds > 15:  # Use chunking for files longer than 15 seconds
                    enhanced_audio = self._apply_fast_processing_for_long_audio(waveform, denoising_strength, sample_rate)
                else:
                    # Direct processing for shorter files
                    enhanced_audio = self._apply_balanced_denoising(waveform, denoising_strength)
                
                print(f"   Output shape: {enhanced_audio.shape}")
                
                # Create output directory if needed
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save enhanced audio
                torchaudio.save(output_path, enhanced_audio, sample_rate)
                
                # Get file sizes
                original_size = os.path.getsize(input_path)
                enhanced_size = os.path.getsize(output_path)
                
                print(f"[SUCCESS] BALANCED Enhancement completed:")
                print(f"   Duration: {duration_seconds:.2f}s")
                print(f"   Size: {original_size / 1024:.1f}KB -> {enhanced_size / 1024:.1f}KB")
                print(f"   Processing: 3-stage balanced noise reduction applied")
                print(f"   Saved: {output_path}")
                
                return {
                    'success': True,
                    'input_path': input_path,
                    'output_path': output_path,
                    'duration': duration_seconds,
                    'original_size': original_size,
                    'enhanced_size': enhanced_size,
                    'processing_mode': 'balanced'
                }
                
            finally:
                # Clean up temporary file if created
                if temp_wav_created:
                    try:
                        os.unlink(wav_path)
                        print(f"   Cleaned up temporary file: {wav_path}")
                    except:
                        pass
                        
        except Exception as e:
            error_msg = f"DCCRN balanced enhancement failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': input_path if 'input_path' in locals() else None,
                'output_path': output_path if 'output_path' in locals() else None
            }

def main():
    """Command line interface for DCCRN Balanced service"""
    parser = argparse.ArgumentParser(description='DCCRN Balanced Audio Enhancement Service')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path') 
    parser.add_argument('--model', '-m', help='Model checkpoint path')
    parser.add_argument('--strength', '-s', type=float, default=0.7, help='Denoising strength (0.0-1.0, default 0.7)')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    
    args = parser.parse_args()
    
    # Validate denoising strength
    if not 0.0 <= args.strength <= 1.0:
        print("❌ Denoising strength must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize service
    try:
        service = DCCRNBalancedService(args.model)
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
            print(f"\n[SUCCESS] Balanced audio enhancement completed successfully!")
        else:
            print(f"\n[ERROR] Audio enhancement failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
