"""
DCCRN Fast Audio Enhancement Service - SPEED-OPTIMIZED VERSION
Quick and effective denoising with minimal processing time
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

# Add ML module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.inference import DCCRNInference

class DCCRNFastService:
    """Fast DCCRN Audio Enhancement Service - optimized for speed"""
    
    def __init__(self, model_path: str = None):
        # Default model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'dccrn_latest.pth')
        
        self.model_path = model_path
        
        # Model configuration matching our trained model
        self.config = {
            'n_fft': 512,
            'hop_length': 256,
            'win_length': 512,
            'encoder_layers': 3,
            'hidden_dim': 64,
            'lstm_layers': 1,
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
            print(f"   Mode: FAST denoising - optimized for speed")
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
    
    def _apply_fast_denoising(self, waveform: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Apply fast single-stage denoising optimized for speed
        
        Args:
            waveform: Input audio tensor
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Enhanced audio tensor
        """
        print(f"[PROCESSING] Applying FAST single-stage denoising with strength {strength}")
        
        # Single stage: Primary DCCRN denoising with optimized strength
        print("   ⚡ Stage 1/1: Fast AI noise reduction...")
        
        # Use moderate strength for good balance of speed and quality
        dccrn_strength = min(strength * 0.75, 0.85)  # Conservative for speech preservation
        enhanced = self.inference.enhance_audio(waveform, dccrn_strength)
        
        print(f"   ⚡ FAST denoising complete - processed in minimal time!")
        return enhanced
    
    def _apply_fast_chunked_processing(self, waveform: torch.Tensor, strength: float, sample_rate: int) -> torch.Tensor:
        """
        Apply very fast processing for long audio files using large chunks
        """
        duration = waveform.shape[-1] / sample_rate
        print(f"   Using FAST chunked processing for {duration:.1f}s audio")
        
        # Use very large chunks for maximum speed (20-second chunks)
        chunk_duration = 20  # seconds
        chunk_length = chunk_duration * sample_rate
        
        # Minimal overlap for speed
        overlap = 0.05  # 5% overlap
        overlap_length = int(chunk_length * overlap)
        
        chunks = []
        for i in range(0, waveform.shape[-1], chunk_length - overlap_length):
            end_idx = min(i + chunk_length, waveform.shape[-1])
            chunk = waveform[:, i:end_idx]
            chunks.append(chunk)
        
        print(f"   Processing {len(chunks)} chunks with FAST denoising...")
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Apply fast denoising to each chunk
            enhanced_chunk = self._apply_fast_denoising(chunk, strength)
            enhanced_chunks.append(enhanced_chunk)
            
            if len(chunks) > 1:  # Show progress for multi-chunk processing
                progress = ((i + 1) / len(chunks)) * 100
                print(f"      Chunk {i+1}/{len(chunks)} processed ({progress:.0f}%)")
        
        # Simple concatenation for speed (minimal overlap processing)
        if len(enhanced_chunks) == 1:
            return enhanced_chunks[0]
        
        # Quick merge with minimal overlap processing
        merged = enhanced_chunks[0]
        for i in range(1, len(enhanced_chunks)):
            if merged.shape[-1] > overlap_length:
                # Simple linear fade for speed
                fade_length = min(overlap_length, enhanced_chunks[i].shape[-1])
                if fade_length > 0:
                    # Quick fade
                    fade_out = torch.linspace(1, 0, fade_length)
                    fade_in = torch.linspace(0, 1, fade_length)
                    
                    merged[:, -fade_length:] *= fade_out
                    enhanced_chunks[i][:, :fade_length] *= fade_in
                    merged[:, -fade_length:] += enhanced_chunks[i][:, :fade_length]
                    
                    if enhanced_chunks[i].shape[-1] > fade_length:
                        merged = torch.cat([merged, enhanced_chunks[i][:, fade_length:]], dim=-1)
                else:
                    merged = torch.cat([merged, enhanced_chunks[i]], dim=-1)
            else:
                merged = torch.cat([merged, enhanced_chunks[i]], dim=-1)
        
        return merged
    
    def enhance_audio(self, input_path: str, output_path: str, denoising_strength: float = 0.6) -> dict:
        """
        Main enhancement function optimized for speed
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            denoising_strength: Denoising strength (0.0 to 1.0) - default 0.6 for fast processing
            
        Returns:
            Dict with processing results
        """
        try:
            # Convert paths to absolute
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            print(f"[PROCESSING] Starting FAST denoising: {Path(input_path).name}")
            print(f"   Absolute input path: {input_path}")
            print(f"   File exists: {os.path.exists(input_path)}")
            print(f"   File size: {os.path.getsize(input_path)} bytes")
            print(f"   Denoising strength: {denoising_strength} (FAST MODE - optimized for speed)")
            
            # Convert to WAV if needed
            wav_path = self._convert_to_wav_if_needed(input_path)
            temp_wav_created = wav_path != input_path
            
            try:
                # Load input audio from WAV file
                waveform, sample_rate = torchaudio.load(wav_path)
                
                print(f"   Loaded audio - shape: {waveform.shape}, sample rate: {sample_rate} Hz")
                
                # Convert stereo to mono BEFORE any processing
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
                
                # Choose processing method based on duration (higher threshold for fast mode)
                duration_seconds = waveform.shape[-1] / sample_rate
                if duration_seconds > 30:  # Use chunking for files longer than 30 seconds
                    enhanced_audio = self._apply_fast_chunked_processing(waveform, denoising_strength, sample_rate)
                else:
                    # Direct fast processing for shorter files
                    enhanced_audio = self._apply_fast_denoising(waveform, denoising_strength)
                
                print(f"   Output shape: {enhanced_audio.shape}")
                
                # Create output directory if needed
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save enhanced audio
                torchaudio.save(output_path, enhanced_audio, sample_rate)
                
                # Get file sizes
                original_size = os.path.getsize(input_path)
                enhanced_size = os.path.getsize(output_path)
                
                print(f"[SUCCESS] FAST Enhancement completed:")
                print(f"   Duration: {duration_seconds:.2f}s")
                print(f"   Size: {original_size / 1024:.1f}KB -> {enhanced_size / 1024:.1f}KB")
                print(f"   Processing: Single-stage fast noise reduction applied")
                print(f"   Saved: {output_path}")
                
                return {
                    'success': True,
                    'input_path': input_path,
                    'output_path': output_path,
                    'duration': duration_seconds,
                    'original_size': original_size,
                    'enhanced_size': enhanced_size,
                    'processing_mode': 'fast'
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
            error_msg = f"DCCRN fast enhancement failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'input_path': input_path if 'input_path' in locals() else None,
                'output_path': output_path if 'output_path' in locals() else None
            }

def main():
    """Command line interface for DCCRN Fast service"""
    parser = argparse.ArgumentParser(description='DCCRN Fast Audio Enhancement Service')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path') 
    parser.add_argument('--model', '-m', help='Model checkpoint path')
    parser.add_argument('--strength', '-s', type=float, default=0.6, help='Denoising strength (0.0-1.0, default 0.6)')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    
    args = parser.parse_args()
    
    # Validate denoising strength
    if not 0.0 <= args.strength <= 1.0:
        print("❌ Denoising strength must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize service
    try:
        service = DCCRNFastService(args.model)
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
            print(f"\n[SUCCESS] Fast audio enhancement completed successfully!")
        else:
            print(f"\n[ERROR] Audio enhancement failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
